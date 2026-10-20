#!/usr/bin/env python3
"""Source-level analysis for the frozen Valen 601--700 confirmation run."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def bootstrap(values, seed=20261020, n=20000):
    values = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(values) == 0:
        return {"sources": 0, "mean": None, "ci95": [None, None]}
    sample = values[rng.integers(0, len(values), size=(n, len(values)))].mean(1)
    return {"sources": int(len(values)), "mean": float(values.mean()),
            "ci95": [float(np.quantile(sample, .025)), float(np.quantile(sample, .975))]}


def summarize(rows):
    out = {}
    for key, vals in rows.items():
        source_means = {}
        for source, items in vals.items():
            source_means[source] = {
                name: float(np.mean([x[name] for x in items]))
                for name in ("valen", "head", "center", "oracle4", "oracle129")
            }
        d = np.asarray([v["valen"] - v["head"] for v in source_means.values()])
        out[key] = {
            "sources": len(source_means),
            "rows": sum(len(v) for v in vals.values()),
            "source_macro": {name: float(np.mean([v[name] for v in source_means.values()]))
                             for name in ("valen", "head", "center", "oracle4", "oracle129")},
            "delta_valen_minus_head": bootstrap(d),
            "candidate_recall_loss": float(np.mean([v["oracle129"] - v["oracle4"] for v in source_means.values()])),
            "candidate_selection_loss": float(np.mean([v["oracle4"] - v["valen"] for v in source_means.values()])),
            "severe_degradation": {"le_minus_005": int(np.sum(d <= -.05)),
                                    "le_minus_010": int(np.sum(d <= -.10)),
                                    "positive": int(np.sum(d > 0))},
            "source_deltas": {s: float(v["valen"] - v["head"]) for s, v in source_means.items()},
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    records = [json.loads(x) for x in Path(args.records).read_text().splitlines() if x.strip()]
    preds = [json.loads(x) for x in Path(args.predictions).read_text().splitlines() if x.strip()]
    by_index = {int(p["row_index"]): p for p in preds if not p.get("shift", 0)}
    grouped = defaultdict(lambda: defaultdict(list))
    missing = []
    for i, r in enumerate(records):
        p = by_index.get(i)
        if p is None:
            missing.append(i)
            continue
        j = int(p["chosen_index"])
        item = {"valen": float(r["iou"][j]), "head": float(r["iou"][int(r["head_index"])]),
                "center": float(r["iou"][64]), "oracle4": float(max(r["iou"][k] for k in r["candidates4"])),
                "oracle129": float(max(r["iou"]))}
        ratio = "1:3" if tuple(r["ratio"]) == (1.0, 3.0) else "3:1"
        grouped["all"][r["source"]].append(item)
        grouped[ratio][r["source"]].append(item)
    result = {"records": len(records), "predictions": len(preds), "covered": len(by_index),
              "missing": missing, "official_test_used": False, "groups": summarize(grouped)}
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "groups"}, indent=2))
    print(json.dumps(result["groups"], indent=2))


if __name__ == "__main__":
    main()
