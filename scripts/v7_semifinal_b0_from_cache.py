"""Semifinal B0 package straight from the CPU obs cache (no re-decode).

The cached b0 column IS the B0 path (same ObservedFacePath state machine that ran
during cache building), so the predictions are consistent with the cache by
construction - which is exactly what max_window_release.py's cache/B0 equality
check requires.  Writes predictions.jsonl + the frozen-B0 config fields.
"""
import argparse, json
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--index', required=True)
ap.add_argument('--cache', required=True)
ap.add_argument('--output', type=Path, required=True)
ap.add_argument('--submission-id', default='B0_SEMIFINAL')
a = ap.parse_args()

import sys  # noqa: E402
sys.path.insert(0, '/root/AIC')
from aic.contract import load_index, write_submission  # noqa: E402
from scripts.independent_submission_check import check  # noqa: E402

index = load_index(a.index)
records = [json.loads(l) for l in open(a.index)]
a.output.mkdir(parents=True, exist_ok=True)
rows = []
for r in records:
    vid = r['video_id']
    z = np.load(Path(a.cache) / f'{vid}_t.npz')
    crops = np.asarray(z['b0'])
    preds = [{'frame': i, 'bboxes': crops[i].tolist()} for i in range(len(crops))]
    rows.append({'video_id': vid, 'targetRatioWH': r['targetRatioWH'], 'predictions': preds})
dest = a.output / 'predictions.jsonl'
validation = write_submission(dest, rows, index, stage='preliminary', actual_model_size_mb=None).to_dict()
independent = check(a.index, dest, None, require_size=False)
print('VALID', validation.get('valid'), independent.get('valid'), 'videos', len(rows))
print('B0_DONE', flush=True)
