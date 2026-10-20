#!/usr/bin/env python3
"""PHD2 label-contract probe on cached DINOv3 CLS features.

This is a source-level diagnostic: the architecture, cache, split and fixed
85% frame decision stay fixed while only union-binary versus soft-vote targets
change.  It does not use PHD2 testing.csv or official test information.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def load_data(root: Path):
    rows = [json.loads(x) for x in (root / "manifest.jsonl").read_text().splitlines() if x.strip()]
    xs, yu, ys, srcs = [], [], [], []
    for r in rows:
        z = np.load(r["path"])
        xs.append(z["pooled"].astype(np.float32))
        yu.append(z["union"].astype(np.float32))
        ys.append(z["soft"].astype(np.float32))
        srcs.append(r["src"])
    return np.stack(xs), np.stack(yu), np.stack(ys), np.asarray(srcs)


def fixed85_source_macro(scores, labels, srcs):
    per = defaultdict(list)
    for i, src in enumerate(srcs):
        n = len(scores[i]); k = max(1, int(round(.85 * n)))
        keep = np.zeros(n, dtype=np.float32)
        keep[np.argsort(-scores[i])[:k]] = 1
        gt = labels[i] > .5
        f = 2 * float((keep * gt).sum()) / max(1.0, float(keep.sum() + gt.sum()))
        per[str(src)].append(f)
    vals = [float(np.mean(v)) for v in per.values()]
    return float(np.mean(vals)), len(vals)


def average_precision(scores, labels):
    s, y = scores.reshape(-1), labels.reshape(-1) > .5
    order = np.argsort(-s); yy = y[order]
    n = int(yy.sum())
    if not n:
        return float("nan")
    prec = np.cumsum(yy) / np.arange(1, len(yy) + 1)
    return float((prec * yy).sum() / n)


def run(mode, Xtr, Yu_tr, Ys_tr, Xv, Yu_v, Ys_v, train_src, val_src, steps, seed, device):
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    ytr = Yu_tr if mode == "union" else Ys_tr

    class TCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Conv1d(768, 128, 1)
            self.convs = torch.nn.ModuleList([torch.nn.Conv1d(128, 128, 3, padding=d, dilation=d)
                                              for d in (1, 2, 4)])
            self.out = torch.nn.Conv1d(128, 1, 1)
        def forward(self, x):
            h = self.proj(x.transpose(1, 2))
            for c in self.convs:
                h = torch.nn.functional.gelu(c(h) + h)
            return self.out(h).squeeze(1)

    model = TCN().to(device).float()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    xt = torch.from_numpy(Xtr).float().to(device); yt = torch.from_numpy(ytr).float().to(device)
    xv = torch.from_numpy(Xv).float().to(device)
    rng = np.random.default_rng(seed)
    for step in range(steps):
        idx = rng.integers(0, len(xt), size=min(64, len(xt)))
        pred = model(xt[idx])
        if mode == "union":
            loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, yt[idx])
        else:
            loss = torch.nn.functional.mse_loss(torch.sigmoid(pred), yt[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.inference_mode():
        pv = torch.sigmoid(model(xv)).cpu().numpy()
    f, nsrc = fixed85_source_macro(pv, Yu_v, val_src)
    return {"mode": mode, "seed": seed, "steps": steps, "params": sum(p.numel() for p in model.parameters()),
            "val_sources": nsrc, "val_fixed85_source_macro_f": f,
            "val_ap_union": average_precision(pv, Yu_v), "train_loss_last": float(loss.detach().cpu())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=20261020)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    X, Yu, Ys, src = load_data(args.cache)
    uniq = np.array(sorted(set(src)))
    rng = np.random.default_rng(args.seed); rng.shuffle(uniq)
    nval = max(1, int(round(.06 * len(uniq))))
    val_set = set(uniq[:nval]); va = np.asarray([s in val_set for s in src])
    tr = ~va
    results = {"protocol": "same DINOv3 CLS/TCN/cache/split/steps; only target contract changes",
               "rows": int(len(src)), "sources": int(len(uniq)), "train_sources": int(tr.sum()),
               "val_sources": int(va.sum()), "seed": args.seed, "steps": args.steps, "results": []}
    for mode in ("union", "soft"):
        results["results"].append(run(mode, X[tr], Yu[tr], Ys[tr], X[va], Yu[va], Ys[va], src[tr], src[va],
                                       args.steps, args.seed, args.device))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
