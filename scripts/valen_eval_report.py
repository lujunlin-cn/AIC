#!/usr/bin/env python3
"""Join Valen evaluation predictions to native-crop records.

Metrics are source-macro.  The script reports candidate recall loss
(129-window oracle minus 4-candidate oracle) separately from candidate-choice
loss (4-candidate oracle minus Valen choice), with source bootstrap intervals.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def domain(source):
    return "RetargetVid" if str(source).startswith("rv:") else "LIVE" if str(source).startswith("live:") else str(source).split(":", 1)[0]


def mean(x):
    return float(np.mean(x)) if x else float("nan")


def bootstrap(source_rows, key_a, key_b, seed=20261020, n=10000):
    keys = sorted(source_rows)
    values = np.asarray([source_rows[k][key_a] - source_rows[k][key_b] for k in keys], dtype=np.float64)
    if not len(values):
        return {"mean": float("nan"), "ci95": [float("nan"), float("nan")], "sources": 0}
    rng = np.random.default_rng(seed)
    boots = values[rng.integers(0, len(values), size=(n, len(values)))].mean(axis=1)
    return {"mean": float(values.mean()), "ci95": [float(np.quantile(boots, .025)), float(np.quantile(boots, .975))], "sources": len(values)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="VALEN_CROP_SFT_V1/dev.jsonl")
    ap.add_argument("--native-records", required=True, help="VALEN_CROP_DATA_V1/records.jsonl")
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    records = [json.loads(x) for x in Path(args.records).read_text().splitlines() if x.strip()]
    native = [json.loads(x) for x in Path(args.native_records).read_text().splitlines() if x.strip()]
    native_by_identity = {(r["key"], tuple(float(x) for x in r["ratio"])): r for r in native}
    preds = [json.loads(x) for x in Path(args.predictions).read_text().splitlines() if x.strip()]
    by_index = {int(p["record_index"]): p for p in preds}
    rows = []
    missing = []
    for i, record in enumerate(records):
        pred = by_index.get(i)
        if pred is None:
            missing.append(i)
            continue
        answer = pred.get("answer", {})
        choice = answer.get("choice")
        keys = list(record["targets"]["crop"]["probabilities"])
        if choice not in keys:
            # Fall back to probability argmax only for diagnostics; this is
            # separately counted so malformed outputs cannot look valid.
            probabilities = answer.get("probabilities", {})
            choice = max(keys, key=lambda k: float(probabilities.get(k, float("-inf")))) if probabilities else None
        meta = record["meta"]
        rows.append((i, meta["source"], domain(meta["source"]), record, pred, choice))

    source_values = defaultdict(list)
    malformed = 0
    for i, source, dom, record, pred, choice in rows:
        candidates = list(record["targets"]["crop"]["probabilities"])
        # Reconstruct candidate indices from the media filenames in A-D order.
        content = record["request"]["state"]["messages"][0]["content"]
        candidate_indices = []
        for j in range(4):
            url = content[3 + 2 * j]["image_url"]["url"]
            try:
                candidate_indices.append(int(Path(url).stem))
            except ValueError:
                malformed += 1
                candidate_indices.append(-1)
        native_row = native_by_identity.get((record["meta"]["row_key"], tuple(float(x) for x in record["meta"]["ratio"])))
        if native_row is None:
            malformed += 1
            continue
        iou = native_row["iou"]
        head_index = native_row["head_index"]
        if choice not in candidates:
            malformed += 1
            continue
        source_values[source].append({
            "domain": dom,
            "valen": float(iou[candidate_indices[candidates.index(choice)]]),
            "head": float(iou[int(head_index)]),
            "center": float(iou[64]),
            "oracle4": float(max(iou[j] for j in candidate_indices)),
            "oracle129": float(max(iou)),
        })

    source_means = {}
    for source, vals in source_values.items():
        source_means[source] = {"domain": vals[0]["domain"], "rows": len(vals), **{k: mean([x[k] for x in vals]) for k in ("valen", "head", "center", "oracle4", "oracle129")}}

    report = {"records": len(records), "predictions": len(preds), "covered_records": len(rows), "missing_record_indices": missing, "malformed": malformed, "source_count": len(source_means), "source_metrics": source_means, "domains": {}}
    for dom in sorted(set(v["domain"] for v in source_means.values())):
        selected = {s: v for s, v in source_means.items() if v["domain"] == dom}
        report["domains"][dom] = {
            "sources": len(selected),
            "coverage_rows": sum(v["rows"] for v in selected.values()),
            "source_macro": {k: mean([v[k] for v in selected.values()]) for k in ("valen", "head", "center", "oracle4", "oracle129")},
            "deltas": {
                "valen_minus_head": bootstrap(selected, "valen", "head"),
                "valen_minus_center": bootstrap(selected, "valen", "center"),
            },
            "candidate_losses": {
                "recall_loss_oracle129_minus_oracle4": mean([v["oracle129"] - v["oracle4"] for v in selected.values()]),
                "selection_loss_oracle4_minus_valen": mean([v["oracle4"] - v["valen"] for v in selected.values()]),
            },
            "severe_degradation_sources": {
                "delta_le_minus_0_05": sum(v["valen"] - v["head"] <= -0.05 for v in selected.values()),
                "delta_le_minus_0_10": sum(v["valen"] - v["head"] <= -0.10 for v in selected.values()),
            },
        }
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
