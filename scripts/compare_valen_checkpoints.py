#!/usr/bin/env python3
"""Paired source-level comparison of two Valen crop-choice evaluations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def bootstrap(values, seed=20261020, n=20000):
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    b = x[rng.integers(0, len(x), size=(n, len(x)))].mean(axis=1)
    return {"mean": float(x.mean()),
            "ci95": [float(np.quantile(b, .025)), float(np.quantile(b, .975))],
            "sources": int(len(x)),
            "positive": int(np.sum(x > 0)),
            "le_minus_005": int(np.sum(x <= -.05)),
            "le_minus_010": int(np.sum(x <= -.10))}


def report_pair(old_path: Path, new_path: Path):
    old = json.loads(old_path.read_text())
    new = json.loads(new_path.read_text())
    old_src, new_src = old["source_metrics"], new["source_metrics"]
    keys = sorted(set(old_src) & set(new_src))
    result = {"sources": len(keys)}
    result["all"] = bootstrap([new_src[k]["valen"] - old_src[k]["valen"] for k in keys])
    result["macro_old"] = float(np.mean([old_src[k]["valen"] for k in keys]))
    result["macro_new"] = float(np.mean([new_src[k]["valen"] for k in keys]))
    result["domains"] = {}
    for domain in sorted({old_src[k]["domain"] for k in keys}):
        subset = [k for k in keys if old_src[k]["domain"] == domain]
        result["domains"][domain] = {
            "macro_old": float(np.mean([old_src[k]["valen"] for k in subset])),
            "macro_new": float(np.mean([new_src[k]["valen"] for k in subset])),
            "delta": bootstrap([new_src[k]["valen"] - old_src[k]["valen"] for k in subset]),
        }
    return result


def prediction_sources(records_path: Path, predictions_path: Path, ratio=None):
    records = [json.loads(x) for x in records_path.read_text().splitlines() if x.strip()]
    preds = {int(x["row_index"]): x for x in
             (json.loads(line) for line in predictions_path.read_text().splitlines())
             if not x.get("shift", 0)}
    values = {}
    for i, row in enumerate(records):
        if i not in preds or (ratio is not None and tuple(row["ratio"]) != tuple(ratio)):
            continue
        values.setdefault(row["source"], []).append(float(row["iou"][int(preds[i]["chosen_index"])]))
    return {source: float(np.mean(rows)) for source, rows in values.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-dev", type=Path)
    ap.add_argument("--new-dev", type=Path)
    ap.add_argument("--records", type=Path)
    ap.add_argument("--old-confirm", type=Path)
    ap.add_argument("--new-confirm", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    result = {}
    if args.old_dev and args.new_dev:
        result["dev"] = report_pair(args.old_dev, args.new_dev)
    if args.records and args.old_confirm and args.new_confirm:
        old_all = prediction_sources(args.records, args.old_confirm)
        new_all = prediction_sources(args.records, args.new_confirm)
        keys = sorted(set(old_all) & set(new_all))
        result["confirm"] = {
            "macro_old": float(np.mean([old_all[k] for k in keys])),
            "macro_new": float(np.mean([new_all[k] for k in keys])),
            "delta": bootstrap([new_all[k] - old_all[k] for k in keys]),
            "ratios": {},
        }
        for name, ratio in (("1:3", [1.0, 3.0]), ("3:1", [3.0, 1.0])):
            old_r = prediction_sources(args.records, args.old_confirm, ratio)
            new_r = prediction_sources(args.records, args.new_confirm, ratio)
            kk = sorted(set(old_r) & set(new_r))
            result["confirm"]["ratios"][name] = {
                "macro_old": float(np.mean([old_r[k] for k in kk])),
                "macro_new": float(np.mean([new_r[k] for k in kk])),
                "delta": bootstrap([new_r[k] - old_r[k] for k in kk]),
            }
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
