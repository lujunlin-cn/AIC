"""Post-result descriptive appendix, requested by parent; no fitting or selection.

Reads only the exact 40 exposed sources already bound by the preregistered run.
Native AHA ix is checked against the frozen R26 1fps actual-PTS contract.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


tick = time.monotonic()
config = json.loads(Path("reports/r29/temporal_probe_config.json").read_text())
result = json.loads(Path(config["output"], "metrics.json").read_text())
report = json.loads(Path(config["axis_report"]).read_text())
axes = {r["video_id"]: r for r in report["axis"]}
rows = []
for binding in json.loads(Path(config["output"], "bindings.json").read_text()):
    vid = binding["video_id"]
    p = Path(config["data_root"], axes[vid]["axis_array_relpath"])
    assert sha(p) == binding["axis_sha256"]
    z = np.load(p)
    pts, mask = z["timestamps"], z["raw_summary_membership"].astype(bool)
    s = Path(config["scores_root"], vid + ".json")
    assert sha(s) == binding["score_sha256"]
    ix = np.array(json.loads(s.read_text())["frame_indices"])
    targets = np.arange(0, float(pts[-1]) + 1e-9, 1.)
    expected = np.unique(np.abs(pts[:, None] - targets).argmin(0))
    assert np.array_equal(expected, ix), vid
    hit = np.zeros(len(pts), bool)
    hit[ix] = True
    missed_runs, all_runs, missed_mass = [], [], []
    for ann in mask:
        starts = np.flatnonzero(ann & ~np.r_[False, ann[:-1]])
        ends = np.flatnonzero(ann & ~np.r_[ann[1:], False]) + 1
        missed = [(a, b) for a, b in zip(starts, ends) if not hit[a:b].any()]
        all_runs.append(len(starts))
        missed_runs.append(len(missed))
        missed_mass.append(sum(b-a for a, b in missed) / max(int(ann.sum()), 1))
    rows.append(dict(video_id=vid, split=binding["split"], native_aha_requested=len(targets),
                     native_aha_unique_frames=len(ix), max_sampling_gap_s=float(np.diff(pts[ix]).max()),
                     mean_rater_runs=float(np.mean(all_runs)),
                     mean_rater_unobserved_runs=float(np.mean(missed_runs)),
                     mean_rater_positive_mass_in_unobserved_runs=float(np.mean(missed_mass))))
summary = {}
for split in ("train", "dev"):
    rr = [r for r in rows if r["split"] == split]
    summary[split] = {k: dict(mean=float(np.mean([r[k] for r in rr])),
                            min=float(np.min([r[k] for r in rr])),
                            max=float(np.max([r[k] for r in rr]))) for k in rr[0] if k not in ("video_id", "split")}
out = dict(scope="Descriptive appendix to preregistered probe; no fitting, no DEV selection", protected_opened=False,
           parent_metrics_sha256=sha(Path(config["output"], "metrics.json")), script_sha256=sha(__file__),
           command="OPENBLAS_NUM_THREADS=1 python reports/r29/public_decision_probe/native_observation_recompute.py",
           seconds=time.monotonic()-tick, summary=summary, sources=rows,
           limitations=["1fps positive-run miss is annotation coverage, not event observability or a bound on learnable utility",
                        "Actual PTS gaps can reflect unavailable/uneven media; no interpolation of GT",
                        "12/24/48 results concern the coarse sparse policy, not the denser frozen AHA input"])
Path(config["output"], "native_observation_appendix.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(summary, indent=2))
