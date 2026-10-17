#!/usr/bin/env python3
"""Null simulation for R44's tiny temporal perturbation.

This is a public-label diagnostic only. It replaces a Poisson(mean=.032*K)
number of selected center frames by the same number of randomly chosen
unselected frames, matching R44's 6,352/199,037 replacement fraction.
It never reads official labels or test predictions.
"""
from pathlib import Path
import importlib.util, json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("r44", ROOT / "scripts/r44_semantic_calibrated_residual.py")
r44 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r44)
labels = [
    ROOT / "reports/r42_annotation_v13_full_20261017/project1/frame_labels.csv",
    ROOT / "reports/r42_annotation_v13_full_20261017/project2/frame_labels.csv",
]
rows = r44.read_rows(labels)
by_view = {}
for row in rows:
    by_view.setdefault((row["video_id"], int(row["_stream"])), []).append(row)

SEED = 1
TRIALS = 20000
REPLACEMENT_FRACTION = 6352 / 199037
rng = np.random.default_rng(SEED)

# Preserve the same center control and proxy used by the R44 probe.
def view_delta(view_rows):
    view_rows = sorted(view_rows, key=lambda row: int(float(row["time_sec"])))
    times = np.asarray([int(float(row["time_sec"])) for row in view_rows])
    y = np.asarray([r44.GRADE[row["generic_highlight"]] for row in view_rows], dtype=float)
    k = int(round(.85 * len(times)))
    center_score = 1.0 - np.abs(times - 14.5) / 14.5
    center = set(np.argsort(-center_score, kind="mergesort")[:k])
    q = min(k, len(times) - k, int(rng.poisson(REPLACEMENT_FRACTION * k)))
    removed = set(rng.choice(list(center), q, replace=False)) if q else set()
    outside = set(range(len(times))) - center
    added = set(rng.choice(list(outside), q, replace=False)) if q else set()
    selected = np.asarray(sorted((center - removed) | added), dtype=int)
    center = np.asarray(sorted(center), dtype=int)
    return r44.proxy_f(selected, y, True) - r44.proxy_f(center, y, True)

deltas = np.asarray([
    np.mean([view_delta(view) for view in by_view.values()])
    for _ in range(TRIALS)
], dtype=float)
result = {
    "experiment_id": "R44_SCORE_NULL_20261017",
    "status": "PUBLIC_LABEL_NULL_ONLY",
    "seed": SEED,
    "trials": TRIALS,
    "views": len(by_view),
    "center_keep": 0.85,
    "r44_replacements": 6352,
    "r44_parent_predictions": 199037,
    "replacement_fraction": REPLACEMENT_FRACTION,
    "replacement_model": "per view q=min(K,N-K,Poisson(replacement_fraction*K)); random selected/unselected swaps",
    "metric": "R44 proxy_f soft grade (grade/3), macro over 18 views",
    "mean_delta_F": float(deltas.mean()),
    "sd_delta_F": float(deltas.std(ddof=1)),
    "quantiles_score_points": {str(q): float(v * 100) for q, v in zip([.025,.05,.5,.95,.975], np.quantile(deltas,[.025,.05,.5,.95,.975]))},
    "p_delta_score_ge_0.05": float(np.mean(deltas >= .0005)),
    "threshold_score_points": 0.05,
    "caveat": "Public annotations are same-account diagnostic labels, not official GT; this null is illustrative and cannot estimate official-score p-values. R44 official evaluator is deterministic; randomness here represents unknown-test utility of near-tie swaps.",
}
out = ROOT / "reports/r44_score_analysis_20261017/random_swap_null.json"
out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(result, ensure_ascii=False, indent=2))
