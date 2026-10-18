#!/usr/bin/env python3
"""Nested source-level ensemble probe for the direct deployed-box utility.

This probe keeps the R78 label contract fixed:
  target = native IoU of the audited B3 argmax candidate.
It trains several small heads on the same inner source split, selects each
checkpoint only on the inner tune sources, and evaluates mean/uncertainty
aggregates on the untouched R70 dev sources.  No official data are read.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Direct(nn.Module):
    def __init__(self, dim: int = 136):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 256), nn.GELU(), nn.LayerNorm(256),
            nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def rankcorr(a: np.ndarray, b: np.ndarray) -> float:
    ra = np.argsort(np.argsort(a, kind="mergesort"), kind="mergesort")
    rb = np.argsort(np.argsort(b, kind="mergesort"), kind="mergesort")
    if np.std(ra) < 1e-8 or np.std(rb) < 1e-8:
        return 0.0
    return float(np.corrcoef(ra, rb)[0, 1])


def source_macro(pred: np.ndarray, target: np.ndarray, source: np.ndarray, mask: np.ndarray) -> float:
    vals = []
    for s in sorted(set(source[mask])):
        q = mask & (source == s)
        if q.sum() >= 3:
            vals.append(rankcorr(pred[q], target[q]))
    return float(np.mean(vals)) if vals else 0.0


def make_features(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    z = np.load(root / "labels.npz", allow_pickle=True)
    y = z["y"].astype(np.float32)
    source = z["source"].astype(str)
    name = z["name"].astype(str)
    idx = z["idx"].astype(np.float32)
    logits = np.load(root / "topsoft_logits.npy").astype(np.float32)
    top = np.sort(logits, axis=1)
    exp = np.exp(logits - logits.max(1, keepdims=True))
    prob = exp / exp.sum(1, keepdims=True)
    stats = np.stack(
        [logits.max(1), top[:, -1] - top[:, -2], logits.std(1), logits.mean(1),
         -(prob * np.log(np.maximum(prob, 1e-12))).sum(1),
         np.argmax(logits, 1) / 128.0],
        axis=1,
    )
    # R78 nested contract: 129 topsoft logits + exact B3 argmax position + six stats.
    x = np.concatenate([logits, idx[:, None] / 128.0, stats], axis=1).astype(np.float32)
    if x.shape[1] != 136:
        raise ValueError(f"unexpected feature dimension {x.shape}")
    return x, y, source, name, idx


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--ensemble-seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = ap.parse_args()

    x, y, source, name, _ = make_features(args.root)
    external = np.char.endswith(name, "_dev")
    train_sources = sorted(set(source[~external]))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(train_sources)
    cut = int(0.8 * len(train_sources))
    inner_sources = set(train_sources[:cut])
    tune_sources = set(train_sources[cut:])
    inner = (~external) & np.isin(source, list(inner_sources))
    tune = (~external) & np.isin(source, list(tune_sources))

    mu = x[inner].mean(0)
    sd = np.maximum(x[inner].std(0), 1e-4)
    xn = ((x - mu) / sd).astype(np.float32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    xt = torch.from_numpy(xn[inner]).to(device)
    yt = torch.from_numpy(y[inner]).to(device)
    predictions = []
    histories = []
    checkpoints = []
    for seed in args.ensemble_seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        model = Direct(x.shape[1]).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        best_loss = float("inf")
        best_state = None
        hist = []
        for step in range(args.steps):
            ids = torch.randint(0, len(xt), (min(args.batch, len(xt)),), generator=generator, device=device)
            pred = model(xt[ids])
            loss = F.smooth_l1_loss(pred, yt[ids])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            if step % 100 == 0 or step == args.steps - 1:
                with torch.inference_mode():
                    q = model(torch.from_numpy(xn[tune]).to(device)).cpu().numpy()
                mse = float(np.mean((q - y[tune]) ** 2))
                hist.append({"step": step, "tune_mse": mse, "tune_row_spearman": rankcorr(q, y[tune])})
                if mse < best_loss:
                    best_loss = mse
                    best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        if best_state is None:
            raise RuntimeError("no checkpoint selected")
        model.load_state_dict(best_state)
        model.eval()
        with torch.inference_mode():
            p = model(torch.from_numpy(xn).to(device)).cpu().numpy()
        predictions.append(p)
        histories.append({"seed": seed, "best_tune_mse": best_loss, "history": hist})
        checkpoints.append(best_state)

    pred = np.stack(predictions, axis=0)
    mean = pred.mean(0)
    std = pred.std(0)
    result = {
        "variant": "direct_output_utility_ensemble_nested",
        "feature_dim": int(x.shape[1]),
        "head_params": int(sum(p.numel() for p in Direct(x.shape[1]).parameters())),
        "ensemble_seeds": list(args.ensemble_seeds),
        "inner_sources": len(inner_sources),
        "tune_sources": len(tune_sources),
        "external_sources": len(set(source[external])),
        "external": {},
        "tune": {},
        "training": {"steps": args.steps, "batch": args.batch, "device": device, "seed_split": args.seed},
    }
    for label, p in [("single_seed0", pred[0]), ("mean", mean),
                     ("mean_minus_0.25std", mean - 0.25 * std),
                     ("mean_minus_0.5std", mean - 0.5 * std)]:
        result["external"][label] = {
            "mse": float(np.mean((p[external] - y[external]) ** 2)),
            "mae": float(np.mean(np.abs(p[external] - y[external]))),
            "row_spearman": rankcorr(p[external], y[external]),
            "source_macro_spearman": source_macro(p, y, source, external),
        }
        result["tune"][label] = {
            "mse": float(np.mean((p[tune] - y[tune]) ** 2)),
            "row_spearman": rankcorr(p[tune], y[tune]),
            "source_macro_spearman": source_macro(p, y, source, tune),
        }
    result["uncertainty"] = {
        "external_mean_std": float(std[external].mean()),
        "external_p95_std": float(np.quantile(std[external], 0.95)),
        "tune_mean_std": float(std[tune].mean()),
    }
    result["histories"] = histories
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    np.savez_compressed(args.out / "predictions.npz", pred=pred, y=y, source=source, name=name, external=external, mu=mu, sd=sd)
    for seed, state in zip(args.ensemble_seeds, checkpoints):
        torch.save({"state_dict": state, "mu": mu, "sd": sd,
                    "variant": "direct_output_utility_ensemble_nested",
                    "config": {"feature_dim": int(x.shape[1]), "seed": seed}},
                   args.out / f"head_seed{seed}.pt")
    print(json.dumps(result["external"], indent=2))


if __name__ == "__main__":
    main()
