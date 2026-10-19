#!/usr/bin/env python3
"""Source-held-out probe for a low-capacity interval policy.

This is a public PHD2/GIF diagnostic. It does not read AIC test labels.
The policy is deliberately constrained to a contiguous 85% window on the
8-slot fragment, so it cannot win by injecting arbitrary frame order.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn


def source_fold(src: str, folds: int) -> int:
    return int(hashlib.sha256(src.encode()).hexdigest()[:8], 16) % folds


def binary_f(y: np.ndarray, sel: np.ndarray) -> float:
    m = float(y.sum())
    if len(sel) == 0 and m == 0:
        return 1.0
    if len(sel) == 0 or m == 0:
        return 0.0
    return float(2.0 * y[sel].sum() / (len(sel) + m))


class FrameRank(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d + 1, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t[..., None]], dim=-1)).squeeze(-1)


class IntervalPolicy(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.frame = nn.Sequential(nn.Linear(d + 1, 32), nn.GELU())
        self.out = nn.Sequential(nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 2))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.frame(torch.cat([x, t[..., None]], dim=-1))
        pooled = torch.cat([h.mean(1), h.amax(1)], dim=-1)
        return self.out(pooled)


def load_rows(labels: Path, feat_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    xs, ys, ts, srcs = [], [], [], []
    for line in labels.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        feat = feat_root / f"{row['video_id']}.npz"
        if not feat.exists():
            continue
        z = np.load(feat)
        x = z["mean"].astype(np.float32)
        t = z["t"].astype(np.float32)
        y = np.asarray([f["hard"] for f in row["frames"]], dtype=np.float32)
        n = min(len(y), x.shape[0], len(t))
        if n < 4:
            continue
        xs.append(x[:n])
        ts.append(t[:n])
        ys.append(y[:n])
        srcs.append(str(row["src"]))
    if not xs:
        raise RuntimeError("no rows with matching features")
    nslots = min(x.shape[0] for x in xs)
    return (
        np.stack([x[:nslots] for x in xs]),
        np.stack([y[:nslots] for y in ys]),
        np.stack([t[:nslots] for t in ts]),
        srcs,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--features", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--steps", type=int, default=1800)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    x, y, t, srcs = load_rows(args.labels, args.features)
    n, slots, d = x.shape
    folds = np.asarray([source_fold(s, args.folds) for s in srcs])
    tr = folds != args.folds - 1
    te = ~tr
    mu = x[tr].reshape(-1, d).mean(0)
    sd = x[tr].reshape(-1, d).std(0) + 1e-5
    x = (x - mu[None, None, :]) / sd[None, None, :]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    xt = torch.from_numpy(x).to(device)
    tt = torch.from_numpy(t).to(device)
    yt = torch.from_numpy(y).to(device)
    tr_idx = torch.from_numpy(np.flatnonzero(tr)).to(device)
    # 85% contiguous window, with a tie-safe pair of legal starts.
    k = max(1, int(round(0.85 * slots)))
    starts = np.arange(slots - k + 1, dtype=np.int64)
    if len(starts) == 1:
        starts = np.asarray([0], dtype=np.int64)
    target = []
    for yy in y:
        scores = [binary_f(yy, np.arange(s, s + k)) for s in starts]
        target.append(int(np.argmax(scores)))
    target = torch.tensor(target, dtype=torch.long, device=device)
    model = IntervalPolicy(d).to(device)
    frame_model = FrameRank(d).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    fopt = torch.optim.AdamW(frame_model.parameters(), lr=3e-4, weight_decay=1e-4)
    model.train()
    frame_model.train()
    pos_weight = torch.tensor([(yt[tr_idx].sum().clamp_min(1.0) / (yt[tr_idx].numel() - yt[tr_idx].sum()).clamp_min(1.0)).item()], device=device)
    g = torch.Generator(device=device).manual_seed(args.seed + 17)
    for step in range(args.steps):
        take = tr_idx[torch.randint(len(tr_idx), (min(256, len(tr_idx)),), generator=g, device=device)]
        loss = nn.functional.cross_entropy(model(xt[take], tt[take]), target[take])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        floss = nn.functional.binary_cross_entropy_with_logits(
            frame_model(xt[take], tt[take]), yt[take], pos_weight=pos_weight
        )
        fopt.zero_grad(set_to_none=True)
        floss.backward()
        fopt.step()
    model.eval()
    frame_model.eval()
    with torch.no_grad():
        pred = model(xt, tt).argmax(1).cpu().numpy()
        frame_scores = frame_model(xt, tt).cpu().numpy()
    rows = {"center": [], "frame_rank": [], "policy": [], "oracle_interval": [], "n_positive": []}
    for i in np.flatnonzero(te):
        yy = y[i]
        # Center tie is the first legal window. Frame rank is a separate
        # unconstrained readout control using the frozen mean feature mean.
        center = np.arange(starts[0], starts[0] + k)
        pick = np.argsort(-frame_scores[i], kind="stable")[:k]
        policy = np.arange(starts[pred[i]], starts[pred[i]] + k)
        oracle_s = max(starts, key=lambda s: binary_f(yy, np.arange(s, s + k)))
        rows["center"].append(binary_f(yy, center))
        rows["frame_rank"].append(binary_f(yy, pick))
        rows["policy"].append(binary_f(yy, policy))
        rows["oracle_interval"].append(binary_f(yy, np.arange(oracle_s, oracle_s + k)))
        rows["n_positive"].append(int(yy.sum()))
    summary = {
        "protocol": "R83_INTERVAL_POLICY_PUBLIC_PHD2",
        "rows_loaded": int(n),
        "sources_loaded": int(len(set(srcs))),
        "heldout_rows": int(te.sum()),
        "heldout_sources": int(len(set(np.asarray(srcs)[te]))),
        "slots": int(slots),
        "keep": float(k / slots),
        "starts": starts.tolist(),
        "parameter_count": int(sum(p.numel() for p in model.parameters())),
        "device": str(device),
        "seed": args.seed,
        "metrics": {name: float(np.mean(vals)) for name, vals in rows.items() if name != "n_positive"},
        "positive_rate": float(np.mean(np.asarray(rows["n_positive"]) > 0)),
        "limitations": [
            "PHD2/GIF labels are public proxy labels, not AIC official GT",
            "the eight-slot contiguous policy is a mechanism probe, not a deployable package",
            "frame_rank control is a small BCE head, not the historical trained head",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
