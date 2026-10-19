#!/usr/bin/env python3
"""R100: VistaHL-inspired spatial utility context probe.

The R70 shards already contain 129 legal candidate features for each native
crop keyframe.  This experiment keeps that current-frame high-resolution
candidate tensor fixed and adds a low-dimensional summary of neighbouring
keyframes from the same source.  It is a controlled proxy for a low-resolution
temporal branch: it does not claim to be a new encoder or an end-to-end
deployment component.

Compared models have identical parameter capacity.  The control receives a
zero context vector; the interaction model conditions each candidate on a
same-source context through a rank-32 Hadamard product.  A source-permuted
context is evaluated as a negative control.  Labels are public native crop
IoU only; no official test labels are read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def load_shards(root: Path):
    arrays = []
    for p in sorted(root.glob("shard_*.npz")):
        with np.load(p, allow_pickle=True) as z:
            arrays.append({k: z[k] for k in z.files})
    if not arrays:
        raise FileNotFoundError(root)
    keys = arrays[0].keys()
    out = {k: np.concatenate([a[k] for a in arrays], axis=0) for k in keys}
    return out


def make_context(x, source, ratio, split):
    # Context is an average of neighbouring keyframe candidate pools.  The
    # source/ratio/split key prevents leakage between train/dev or geometries.
    pooled = x.mean(axis=1).astype(np.float32)
    groups = {}
    for i, key in enumerate(zip(source.astype(str), ratio.astype(str), split.astype(str))):
        groups.setdefault(key, []).append(i)
    g = np.zeros_like(pooled, dtype=np.float32)
    for ids in groups.values():
        if len(ids) < 2:
            continue
        for pos, i in enumerate(ids):
            nb = ids[max(0, pos - 2):pos] + ids[pos + 1:min(len(ids), pos + 3)]
            if nb:
                g[i] = pooled[np.asarray(nb)].mean(axis=0)
    return g


def source_unit(source, ratio, split, name):
    return np.asarray([f"{n}|{s}|{r}" for n, s, r in zip(name.astype(str), source.astype(str), ratio.astype(str))])


def evaluate(pred, y, units, mask):
    chosen = pred.argmax(axis=1)
    row_iou = y[np.arange(len(y)), chosen]
    per = {}
    for i in np.flatnonzero(mask):
        per.setdefault(str(units[i]), []).append(float(row_iou[i]))
    per = {k: float(np.mean(v)) for k, v in per.items()}
    return {"rows": int(mask.sum()), "units": len(per),
            "macro_keyframe_iou": float(np.mean(list(per.values()))),
            "per_unit": per}


def paired_bootstrap(base, new, seed=20261019, n=10000):
    ids = sorted(set(base) & set(new))
    d = np.asarray([new[k] - base[k] for k in ids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    b = np.asarray([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"units": len(d), "mean": float(d.mean()),
            "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))],
            "better": int((d > 0).sum()), "worse": int((d < 0).sum())}


class UtilityNet:
    @staticmethod
    def build(mode: str, rank=32):
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__(); self.mode = mode; self.rank = rank
                self.cur = nn.Sequential(nn.LayerNorm(2305), nn.Linear(2305, 128), nn.GELU())
                self.ctx = nn.Sequential(nn.LayerNorm(2305), nn.Linear(2305, 128), nn.GELU())
                self.u = nn.Linear(128, rank); self.v = nn.Linear(128, rank)
                self.head = nn.Sequential(nn.Linear(128 + rank, 96), nn.GELU(), nn.Linear(96, 1))
            def forward(self, x, context):
                # x: [B,129,2305], context: [B,2305]
                x = x.float(); context = context.float()
                h = self.cur(x)
                g = self.ctx(context)
                if self.mode == "control":
                    inter = h.new_zeros((*h.shape[:-1], self.rank))
                elif self.mode == "interaction":
                    inter = self.u(h) * self.v(g).unsqueeze(1)
                else:
                    raise ValueError(self.mode)
                return self.head(__import__('torch').cat([h, inter], -1)).squeeze(-1)
        return Net()


def predict_batches(net, X, C, device, batch=8):
    import torch
    out = []
    with torch.inference_mode():
        for i in range(0, len(X), batch):
            out.append(net(torch.from_numpy(X[i:i + batch]).to(device),
                           torch.from_numpy(C[i:i + batch]).to(device)).float().cpu().numpy())
    return np.concatenate(out, axis=0)


def train_one(X, C, Y, train_mask, units, mode, seed, steps, device, batch):
    import torch
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = UtilityNet.build(mode).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-3)
    ids = np.flatnonzero(train_mask)
    Xt = torch.from_numpy(X); Ct = torch.from_numpy(C); Yt = torch.from_numpy(Y)
    best = (-1., None); hist = []
    for step in range(1, steps + 1):
        sel = ids[rng.integers(0, len(ids), batch)]
        xb = Xt[sel].to(device); cb = Ct[sel].to(device); yb = Yt[sel].to(device)
        pred = net(xb, cb)
        loss = ((pred - yb) ** 2).mean()
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.); opt.step()
        if step % 100 == 0 or step == steps:
            net.eval()
            allp = predict_batches(net, X, C, device, batch)
            m = evaluate(allp, Y, units, ~train_mask)
            hist.append({"step": step, "loss": float(loss.detach().cpu()),
                         "dev_macro_keyframe_iou": m["macro_keyframe_iou"]})
            if m["macro_keyframe_iou"] > best[0]:
                best = (m["macro_keyframe_iou"],
                        {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
            net.train()
    net.load_state_dict(best[1]); net.eval()
    allp = predict_batches(net, X, C, device, batch)
    return net, allp, {"mode": mode, "params": int(sum(p.numel() for p in net.parameters())),
                       "train": evaluate(allp, Y, units, train_mask),
                       "dev": evaluate(allp, Y, units, ~train_mask), "history": hist}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True); ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=8); ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args(); a.output.mkdir(parents=True, exist_ok=False)
    import torch
    z = load_shards(a.shards)
    X = z["x"].astype(np.float16); Y = z["y"].astype(np.float32)
    source = z["source"].astype(str); ratio = z["ratio"].astype(str); name = z["name"].astype(str)
    split = name.copy(); train_mask = np.isin(split, ["rv_train", "live_train"])
    units = source_unit(source, ratio, split, name)
    C = make_context(X, source, ratio, split).astype(np.float16)
    models = {}
    for mode in ("control", "interaction"):
        net, pred, met = train_one(X, C, Y, train_mask, units, mode, a.seed, a.steps, a.device, a.batch)
        models[mode] = met
        torch.save({"state_dict": net.state_dict(), "mode": mode,
                    "params": met["params"]}, a.output / f"{mode}.pt")
    base = models["control"]["dev"]["per_unit"]
    main = {"interaction": paired_bootstrap(base, models["interaction"]["dev"]["per_unit"])}
    # Permute complete context rows only among held-out units.
    net = UtilityNet.build("interaction").to(a.device)
    net.load_state_dict(torch.load(a.output / "interaction.pt", map_location="cpu", weights_only=True)["state_dict"]); net.eval()
    idx = np.flatnonzero(~train_mask); perm = idx.copy(); np.random.default_rng(a.seed).shuffle(perm)
    cp = C.copy(); cp[idx] = C[perm]
    pperm = predict_batches(net, X, cp, a.device, a.batch)
    pm = evaluate(pperm, Y, units, ~train_mask)
    main["interaction_permuted_context"] = paired_bootstrap(base, pm["per_unit"])
    out = {"experiment_id": "R100_SPATIAL_CONTEXT_UTILITY", "protocol": "R70 native candidate utility + neighbouring keyframe pooled context; source-disjoint RV/LIVE split", "official_test_labels_used": False, "audio": "excluded_due_to_project_negative_transfer_evidence", "shards_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(a.shards.glob("shard_*.npz"))}, "rows": int(len(Y)), "train_rows": int(train_mask.sum()), "dev_rows": int((~train_mask).sum()), "units": int(len(set(units))), "models": models, "paired_dev": main, "permuted_dev": pm, "decision": {"official_package": False, "note": "proxy for high-resolution current candidate plus low-resolution temporal context; not a new end-to-end encoder"}}
    (a.output / "metrics.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({"models": {k: {"dev": v["dev"]} for k, v in models.items()}, "paired_dev": main}, indent=2))


if __name__ == "__main__":
    main()
