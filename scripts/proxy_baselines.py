#!/usr/bin/env python3
"""Evaluate simple TVSum temporal proxy baselines on a locked split.

The script never uses a validation video's positive count to choose a budget.
Budgets are derived from the training split target rate, and the binary target
definition is fixed by ``aic.train.TVSUM_TARGET_THRESHOLD``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from aic.features import load_feature_cache
from aic.train import TVSUM_LABEL_PROTOCOL, TVSUM_TARGET_THRESHOLD, video_proxy_metrics


def records(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def as_metrics(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray, video_id: str) -> dict:
    scores = np.asarray(scores, dtype=np.float32)
    # Convert probabilities to logits so the shared metric owns thresholding.
    p = np.clip(scores, 1e-6, 1 - 1e-6)
    logits = torch.from_numpy(np.log(p / (1 - p)))
    return video_proxy_metrics(logits, torch.from_numpy(labels), torch.from_numpy(mask), .5, video_id)


def aggregate(rows: list[dict]) -> dict:
    vals = [row["f1"] for row in rows if row["f1"] is not None]
    return {
        "video_macro_f1": float(np.mean(vals)) if vals else None,
        "video_median_f1": float(np.median(vals)) if vals else None,
        "video_std_f1": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        "mean_prediction_rate": float(np.mean([r["prediction_rate"] for r in rows])),
        "mean_target_rate": float(np.mean([r["target_rate"] for r in rows])),
        "empty_prediction_rate": float(np.mean([r["empty_prediction"] for r in rows])),
        "mean_spearman": float(np.mean([r["spearman"] for r in rows if r["spearman"] is not None])),
        "videos": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--seed", type=int, default=20260925)
    args = ap.parse_args()
    train = records(args.train); val = records(args.val)
    train_items = [load_feature_cache(r["path"]) for r in train]
    val_items = [(str(r.get("video_id", Path(r["path"]).stem)), load_feature_cache(r["path"])) for r in val]
    train_labels = np.concatenate([x["labels"][x["mask"]] for x in train_items])
    target_rate = float(np.mean(train_labels >= TVSUM_TARGET_THRESHOLD))
    train_mean = float(np.mean(train_labels))
    # Ridge fit is a fixed, low-capacity linear reference on frozen features.
    x = np.concatenate([item["features"][item["mask"]] for item in train_items]).astype(np.float64)
    y = train_labels.astype(np.float64)
    mu, sd = x.mean(0), x.std(0) + 1e-5
    xs = (x - mu) / sd
    reg = 1.0
    coef = np.linalg.solve(xs.T @ xs + reg * np.eye(xs.shape[1]), xs.T @ (y - y.mean()))
    rng = np.random.default_rng(args.seed)
    methods: dict[str, list[dict]] = {}
    for name in ("all_negative", "all_positive", "constant_train_mean", "uniform_budget", "random_budget", "linear_ridge"):
        rows = []
        for video_id, item in val_items:
            labels, mask, n = item["labels"], item["mask"], len(item["labels"])
            if name == "all_negative": scores = np.zeros(n, np.float32)
            elif name == "all_positive": scores = np.ones(n, np.float32)
            elif name == "constant_train_mean": scores = np.full(n, train_mean, np.float32)
            elif name in ("uniform_budget", "random_budget"):
                scores = np.zeros(n, np.float32); k = int(round(target_rate * int(mask.sum())))
                valid = np.flatnonzero(mask)
                selected = np.linspace(0, len(valid) - 1, max(0, k), dtype=int) if name == "uniform_budget" else rng.choice(valid, size=min(k, len(valid)), replace=False)
                scores[valid[selected] if name == "uniform_budget" else selected] = 1.0
            else:
                z = ((item["features"] - mu) / sd) @ coef + y.mean()
                scores = np.clip(z, 0.0, 1.0).astype(np.float32)
            rows.append(as_metrics(scores, labels, mask, video_id))
        methods[name] = aggregate(rows)
    result = {"protocol": TVSUM_LABEL_PROTOCOL, "target_threshold": TVSUM_TARGET_THRESHOLD,
              "budget_source": "train_target_rate_only", "train_target_rate": target_rate,
              "train_mean_continuous": train_mean, "methods": methods}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({k: {m: v[m] for m in ("video_macro_f1", "video_median_f1", "video_std_f1", "mean_prediction_rate", "empty_prediction_rate", "mean_spearman")} for k,v in methods.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
