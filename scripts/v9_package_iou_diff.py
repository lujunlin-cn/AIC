"""Box-level IoU between the vision-only package and the full-model package.

bboxes are [x, y, w] with x,y the TOP-LEFT corner (aic.max_window_path.to_crops:
"free_offsets: top-left along the free axis (pixels); returns [x,y,w]"), and the
height follows from the declared target ratio, h = w * rh / rw.  Getting that
wrong makes the comparison read as 0.72 IoU when the boxes are in fact nearly
identical, so the corners are rebuilt explicitly here.

This is the check that matters for scoring: it compares the submitted bbox sets,
so "identical predictions, smaller declared size" rests on the number the
platform scores rather than on intermediate points.
"""
import json
from pathlib import Path

import numpy as np

A = Path('/data/aic/semifinal_20261001/submissions/LFM_V9_A1RAW_SEMIFINAL/predictions.jsonl')
B = Path('/data/aic/semifinal_20261001/submissions/LFM_V9_VISIONONLY_SEMIFINAL/predictions.jsonl')


def load(p):
    return {json.loads(l)['video_id']: json.loads(l)
            for l in p.read_text().splitlines() if l.strip()}


def corners(b, rw, rh):
    x, y, w = float(b[0]), float(b[1]), float(b[2])
    h = w * rh / rw
    return x, y, x + w, y + h


a, b = load(A), load(B)
assert set(a) == set(b), 'video set changed'
ious, n_missing, ident = [], 0, 0
worst = []
for vid in a:
    rw, rh = a[vid]['targetRatioWH']
    pa = {p['frame']: p['bboxes'] for p in a[vid]['predictions']}
    pb = {p['frame']: p['bboxes'] for p in b[vid]['predictions']}
    for f, ba in pa.items():
        if f not in pb:
            n_missing += 1
            continue
        ax1, ay1, ax2, ay2 = corners(ba, rw, rh)
        bx1, by1, bx2, by2 = corners(pb[f], rw, rh)
        ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
        iy = max(0.0, min(ay2, by2) - max(ay1, by1))
        inter = ix * iy
        u = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
        iou = inter / u if u > 0 else 1.0
        ious.append(iou)
        if iou > 1 - 1e-12:
            ident += 1
        worst.append((iou, vid, f))

ious = np.array(ious)
worst.sort()
print(json.dumps({
    'videos': len(a),
    'frames_compared': int(len(ious)),
    'frames_missing_in_visiononly': n_missing,
    'mean_box_iou': round(float(ious.mean()), 6),
    'p01_box_iou': round(float(np.percentile(ious, 1)), 6),
    'min_box_iou': round(float(ious.min()), 6),
    'boxes_identical': ident,
    'share_identical': round(float((ious > 1 - 1e-12).mean()), 6),
    'share_iou_below_0.99': round(float((ious < 0.99).mean()), 6),
    'worst_five': [{'iou': round(i, 4), 'video': v, 'frame': f} for i, v, f in worst[:5]],
}, indent=1))