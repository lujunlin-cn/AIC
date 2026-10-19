#!/usr/bin/env python3
"""R84 public native-crop probe: small utility head + coordinate consistency.

The input contract is the frozen R70 [129,2305] candidate feature.  The new
head is intentionally small and adds an auxiliary soft-argmax coordinate loss
to discourage utility scores whose ordering is correct only by chance.  This
is a public RetargetVid/LIVE diagnostic; official test data is not read.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


class CoordHead(nn.Module):
    def __init__(self, d: int = 2305):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(), nn.Linear(128, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def source_macro(values, sources):
    per = {}
    for v, s in zip(values, sources):
        per.setdefault(str(s), []).append(float(v))
    return float(np.mean([np.mean(v) for v in per.values()])), len(per)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--checkpoint", type=Path, default=None)
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--seed", type=int, default=20261019)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    xs, ys, srcs, names = [], [], [], []
    for p in sorted(glob.glob(str(args.shards / "*.npz"))):
        z = np.load(p, allow_pickle=True)
        xs.append(z["x"].astype(np.float32))
        ys.append(z["y"].astype(np.float32))
        srcs.extend(z["source"].tolist())
        names.extend(z["name"].tolist())
    x = np.concatenate(xs)
    y = np.concatenate(ys)
    srcs = np.asarray(srcs, dtype=object)
    names = np.asarray(names, dtype=object)
    tr = np.array([str(n).endswith("_train") for n in names])
    te = ~tr
    # Normalize per feature using train rows only.
    mu = x[tr].reshape(-1, x.shape[-1]).mean(0)
    sd = x[tr].reshape(-1, x.shape[-1]).std(0) + 1e-5
    x = (x - mu[None, None, :]) / sd[None, None, :]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy(x).to(dev)
    yt = torch.from_numpy(y).to(dev)
    ti = torch.from_numpy(np.flatnonzero(tr)).to(dev)
    model = CoordHead(x.shape[-1]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=4e-4, weight_decay=1e-4)
    gen = torch.Generator(device=dev).manual_seed(args.seed + 5)
    pos = torch.linspace(0, 1, y.shape[1], device=dev)[None, :]
    tau = 0.08
    for _ in range(args.steps):
        take = ti[torch.randint(len(ti), (min(256, len(ti)),), generator=gen, device=dev)]
        logits = model(xt[take])
        target = yt[take]
        mse = nn.functional.mse_loss(logits, target)
        target_dist = torch.softmax(target / tau, dim=1)
        pred_dist = torch.softmax(logits / tau, dim=1)
        kl = nn.functional.kl_div(torch.log(pred_dist.clamp_min(1e-6)), target_dist, reduction="batchmean")
        target_c = (target_dist * pos).sum(1)
        pred_c = (pred_dist * pos).sum(1)
        coord = nn.functional.smooth_l1_loss(pred_c, target_c)
        loss = mse + 0.30 * kl + 0.50 * coord
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(xt).cpu().numpy()
    # Utility argmax and center controls on the held-out source pool.
    picked = np.argmax(pred[te], axis=1)
    center = y.shape[1] // 2
    row_iou = y[te, picked]
    center_iou = y[te, center]
    picked_macro, nsrc = source_macro(row_iou, srcs[te])
    center_macro, _ = source_macro(center_iou, srcs[te])
    per_domain = {}
    for dom in sorted(set(str(n) for n in names[te])):
        mask = np.asarray([str(n) == dom for n in names[te]])
        per_domain[dom] = {
            "rows": int(mask.sum()),
            "sources": int(len(set(srcs[te][mask]))),
            "row_iou": float(np.mean(row_iou[mask])),
            "center_row_iou": float(np.mean(center_iou[mask])),
        }
    # Coordinate error uses the soft target distribution from the GT utility.
    ydev = y[te]
    weights = np.exp((ydev - ydev.max(1, keepdims=True)) / tau)
    weights /= np.maximum(weights.sum(1, keepdims=True), 1e-8)
    target_c = (weights * np.linspace(0, 1, y.shape[1])[None, :]).sum(1)
    predw = np.exp((pred[te] - pred[te].max(1, keepdims=True)) / tau)
    predw /= np.maximum(predw.sum(1, keepdims=True), 1e-8)
    pred_c = (predw * np.linspace(0, 1, y.shape[1])[None, :]).sum(1)
    out = {
        "protocol": "R84_COORD_CONSISTENCY_PUBLIC_NATIVE_CROP",
        "rows": int(len(x)),
        "train_rows": int(tr.sum()),
        "dev_rows": int(te.sum()),
        "dev_sources": int(nsrc),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "device": str(dev),
        "steps": args.steps,
        "dev_source_macro_iou": picked_macro,
        "dev_center_source_macro_iou": center_macro,
        "delta": picked_macro - center_macro,
        "dev_row_iou": float(np.mean(row_iou)),
        "dev_center_row_iou": float(np.mean(center_iou)),
        "soft_coordinate_mae": float(np.mean(np.abs(pred_c - target_c))),
        "per_domain": per_domain,
        "public_test_labels_used": False,
        "comparison": {"r70_full_readout_dev_source_macro_delta": 0.07006},
    }
    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "mu": mu, "sd": sd}, args.checkpoint)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
