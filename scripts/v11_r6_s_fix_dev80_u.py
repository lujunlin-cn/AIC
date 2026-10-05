"""R6 S-REREAD hotfix: recompute dev80 u labels from the RAW_LTRB boxes.

Bug found by the S0 sanity read (B3 zero-shot IoU 0.038 vs the ~0.5
expected on same-dataset untrained sources): the freeze script read the
sparse annotations' boxes_xywh and fed the numbers to the IoU as if
they were ltrb.  The sparse json carries the published side as
`raw_ltrb` dicts {left, right, top, bottom}; boxes_xywh is x,y,w,h.
This script recomputes u for every dev80 row from raw_ltrb and updates
the u arrays IN PLACE inside the 12 encode shard .pt files (crop
embeddings, B3 scores, shortlists are unaffected and stay as-is).

CPU only.  Prints the per-shard before/after so the fix is auditable.
"""
import csv, glob, json, os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np

sys.path.insert(0, '/root/AIC')
from aic.max_window_path import geometry  # noqa: E402
from scripts.benchmark_spatial import iou  # noqa: E402

RATIOS = {'1-3': [1, 3], '3-1': [3, 1], '9-16': [9, 16], '16-9': [16, 9]}

FREEZE = Path('/data/aic/experiments_910a/LFM_V11/r6_s_pool_freeze_rows.csv')
T5_IDX = Path('/data/aic/experiments/T5_CROPHEAD_V3/live_train_index.jsonl')
SPARSE = Path('/data/aic/external_datasets/LIVE_YT_VC/annotations/sparse')
SHARDS = sorted(glob.glob('/data/aic/experiments_910a/LFM_V11/r6_s_encode_shard*.pt'))


def candidate_boxes(W, H, ratio_wh, nc=129):
    w, h, axis = geometry(W, H, ratio_wh)
    span = (W - w) if axis == 0 else ((H - h) if axis == 1 else 0.0)
    offs = np.linspace(0, span, nc) if span > 0 else np.zeros(1)
    boxes = np.zeros((len(offs), 4), np.float32)
    for j, o in enumerate(offs):
        if axis == 0:
            boxes[j] = [o, 0, o + w, h]
        elif axis == 1:
            boxes[j] = [0, o, w, o + h]
        else:
            boxes[j] = [0, 0, W, H]
    return boxes


def main():
    t5 = {}
    for l in T5_IDX.read_text().splitlines():
        r = json.loads(l)
        t5[r['video_id']] = r
    new_u = {}
    for r in csv.DictReader(FREEZE.open()):
        if r['pool'] != 'dev80':
            continue
        key = f"{r['vid']}|{r['frame']}|{r['ratio']}"
        t5r = t5[r['vid']]
        W, H = float(t5r['width']), float(t5r['height'])
        ratio = t5r['targetRatioWH']
        ann = json.loads(Path(t5r['annotation']).read_text())
        assert ann.get('raw_ltrb'), f'{r["vid"]}: no raw_ltrb'
        f = int(r['frame'])
        fi = list(ann['frames']).index(f)
        b = ann['raw_ltrb'][fi]
        g = np.array([b['left'], b['top'], b['right'], b['bottom']], np.float32)
        boxes = candidate_boxes(W, H, ratio)
        u = iou(boxes[:, None, :], g[None, None, :]).reshape(-1)
        new_u[key] = u.astype(np.float32)
    print(f'dev80 rows recomputed: {len(new_u)}', flush=True)
    assert new_u, 'no dev80 rows matched'
    for p in SHARDS:
        d = torch_load(p)
        fixed = 0
        deltas = []
        for key, u in new_u.items():
            if key in d['u']:
                old = float(np.max(d['u'][key]))
                new = float(np.max(u))
                deltas.append(new - old)
                d['u'][key] = u
                fixed += 1
        if fixed:
            torch_save(p, d)
        print(f'{p.split("/")[-1]}: fixed {fixed} rows, '
              f'oracle_u_max mean before/after delta '
              f'{np.mean(deltas) if deltas else 0:+.4f}', flush=True)


def torch_load(p):
    import torch
    return torch.load(p, map_location='cpu', weights_only=False)


def torch_save(p, d):
    import torch
    torch.save(d, p)


if __name__ == '__main__':
    main()
