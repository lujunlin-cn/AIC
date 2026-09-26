#!/usr/bin/env python3
"""Summarize matched clip probes with development-only baselines and paired CIs."""
import argparse
import json
from pathlib import Path

import numpy as np
from aic.features import FeatureCacheDataset
from scripts.probe_videomae_qvh import metrics


def paired_bootstrap(delta, seed=20260926):
    delta = np.asarray(delta, dtype=float)
    rng = np.random.default_rng(seed)
    draws = delta[rng.integers(0, len(delta), size=(10000, len(delta)))].mean(axis=1)
    return {"mean_delta": float(delta.mean()),
            "ci95": np.quantile(draws, [.025, .975]).tolist(),
            "bootstrap_samples": 10000, "paired_videos": len(delta),
            "interpretation": "exploratory DEV-selected checkpoints; not held-out evidence"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads((args.run / "metrics.json").read_text())
    train = FeatureCacheDataset(args.run / "videomaev2/train.jsonl")
    val = FeatureCacheDataset(args.run / "videomaev2/val.jsonl")
    grid = np.linspace(0, 1, 1000)
    prior = np.mean([np.interp(grid, np.linspace(0, 1, len(train[i]["labels"])), train[i]["labels"])
                     for i in range(len(train))], axis=0)
    baselines = {"all_positive": [], "train_mean_normalized_timeline": []}
    for i in range(len(val)):
        item = val[i]
        target = item["labels"]
        baselines["all_positive"].append(metrics(np.ones_like(target), target, item["video_id"]))
        prediction = np.interp(np.linspace(0, 1, len(target)), grid, prior)
        baselines["train_mean_normalized_timeline"].append(metrics(prediction, target, item["video_id"]))
    comparison = {name: {"f1": float(np.mean([r["f1"] for r in per])),
                         "spearman": float(np.mean([r["spearman"] for r in per])),
                         "per_video": per} for name, per in baselines.items()}
    all_rows = {name: variant["best"]["per_video"] for name, variant in result["variants"].items()}
    all_rows.update(baselines)
    paired = {}
    ref = {r["video_id"]: r for r in all_rows["deits_clip_mean"]}
    for name, per in all_rows.items():
        if name == "deits_clip_mean":
            continue
        observed = {r["video_id"]: r for r in per}
        assert set(observed) == set(ref)
        paired[name + "_minus_deits"] = {
            key: paired_bootstrap([observed[v][key] - ref[v][key] for v in sorted(ref)])
            for key in ["f1", "spearman"]}
    out = {"run_id": result["run_id"], "official_f_video": None,
           "competition_score": None, "baseline_fit_source": "train only",
           "baselines": comparison, "paired_development_comparison": paired}
    (args.run / "analysis.json").write_text(json.dumps(out, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"baselines": {k: {m: v[m] for m in ["f1", "spearman"]} for k, v in comparison.items()},
                      "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
