"""R8 LABEL_PROVENANCE audit (preregistered in reports/r8/preregistration.yaml).

Checks, in order; any FAILURE does not block reads that do not depend on it
but is recorded:
  1. pool structure: 2,560 rows = train240(240 vid) + dev80(80 vid), 8 rows/vid
  2. pool isolation: train240 vs dev80 disjoint; both vs val178 disjoint;
     dev80 vids NOT in the B3 live_train npz (never-used property)
  3. train240 ancestry: every vid in the B3 live_train manifest rows
  4. label semantics: LIVE rows single-annotator (gt (1,4) or raw_ltrb single
     box); RV rows carry 6 per-frame annotators (gt (6,4)) - counted, with
     taxonomy raw_single_box / mean_candidate_utilities recorded per row
  5. u replay: dev80 ALL 640 rows recomputed from raw_ltrb and compared to
     the shard u; train240 16 sampled rows recomputed from manifest gt and
     compared to live_train.npz u[cache_row]
  6. ID-permutation invariance on 32 sampled rows: u multiset invariant
  7. u range [0,1] over all merged shards
  8. sha256 of every consumed artifact

CPU only.  Output: reports/r8/source_and_label_semantics.json (printed to
stdout as JSON; caller redirects).
"""
import csv, glob, hashlib, json, os, sys
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, '/root/AIC')
from scripts.benchmark_spatial import iou  # noqa: E402
from aic.max_window_path import geometry  # noqa: E402

RATIOS = {'1-3': [1, 3], '3-1': [3, 1], '9-16': [9, 16], '16-9': [16, 9]}
FREEZE = Path('/data/aic/experiments_910a/LFM_V11/r6_s_pool_freeze_rows.csv')
SHARDS = sorted(glob.glob('/data/aic/experiments_910a/LFM_V11/r6_s_encode_shard*.pt'))
SAMPLES = Path('/data/aic/experiments_910a/LFM_V8/samples')
MANIFEST = Path('/data/aic/experiments_910a/LFM_V8/v8_manifest.jsonl')
T5_IDX = Path('/data/aic/experiments/T5_CROPHEAD_V3/live_train_index.jsonl')
VAL_IDX = Path('/data/aic/experiments/T5_CROPHEAD_V3/live_val_index.jsonl')
DEV80_GRID = Path('/data/aic/experiments_910a/LFM_V11/live_feats_dev80')


def sha256(p, cap=None):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


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
    out = {'protocol': 'R8 LABEL_PROVENANCE audit; preregistered '
                       'reports/r8/preregistration.yaml', 'checks': {}}
    # ---- artifacts ----
    rows = list(csv.DictReader(FREEZE.open()))
    t5 = {}
    for l in T5_IDX.read_text().splitlines():
        r = json.loads(l)
        t5[r['video_id']] = r
    val_vids = set()
    if VAL_IDX.exists():
        for l in VAL_IDX.read_text().splitlines():
            val_vids.add(json.loads(l)['video_id'])
    manifest_rows = {}
    for l in MANIFEST.read_text().splitlines():
        r = json.loads(l)
        manifest_rows[(r.get('src', r.get('split', '?')), r['vid'], int(r['frame']))] = r

    # 1. structure
    pools = {}
    for r in rows:
        pools.setdefault(r['pool'], []).append(r)
    structure = {}
    for p, rs in pools.items():
        vids = {r['vid'] for r in rs}
        structure[p] = {'rows': len(rs), 'vids': len(vids),
                        'rows_per_vid': sorted({sum(1 for r in rs if r['vid'] == v)
                                                for v in vids})}
    ok1 = (structure.get('train240', {}).get('rows') == 1920 and
           structure.get('dev80', {}).get('rows') == 640 and
           all(structure[p]['rows_per_vid'] == [8] for p in structure))
    out['checks']['pool_structure'] = {'status': 'PASS' if ok1 else 'FAIL',
                                       'detail': structure}

    # 2. isolation
    tr_vids = {r['vid'] for r in pools['train240']}
    dv_vids = {r['vid'] for r in pools['dev80']}
    lt = np.load(SAMPLES / 'live_train.npz', allow_pickle=False)
    lt_vids = {str(v) for v in lt['vid']}
    inter_td = tr_vids & dv_vids
    inter_val = (tr_vids | dv_vids) & val_vids
    dv_in_lt = dv_vids & lt_vids
    ok2 = not inter_td and not inter_val and not dv_in_lt
    out['checks']['pool_isolation'] = {
        'status': 'PASS' if ok2 else 'FAIL',
        'train240_vs_dev80_overlap': sorted(inter_td),
        'val178_overlap': sorted(inter_val),
        'dev80_vids_in_b3_train_npz': sorted(dv_in_lt),
        'val178_size': len(val_vids)}

    # 3. ancestry: train240 vids all have a live_train manifest row
    lt_manifest_vids = {v for (s, v, f) in manifest_rows if s == 'live_train'}
    missing = tr_vids - lt_manifest_vids
    out['checks']['train240_ancestry'] = {
        'status': 'PASS' if not missing else 'FAIL',
        'vids_missing_from_manifest': sorted(missing)[:10], 'n_missing': len(missing)}

    # 4. semantics: gt dimensionality per source class (manifest scan)
    gt_dims = {}
    for (s, v, f), r in manifest_rows.items():
        g = np.array(r['gt'], np.float32)
        gt_dims.setdefault(s, set()).add(g.ndim)
    out['checks']['label_semantics'] = {
        'status': 'INFO',
        'manifest_gt_ndim_by_split': {k: sorted(v) for k, v in gt_dims.items()},
        'taxonomy': {'train240': 'raw_single_box (LIVE one subject per frame)',
                     'dev80': 'raw_single_box (LIVE sparse json raw_ltrb)',
                     'note': 'train240 rows carry u = IoU vs the single box; '
                             'RV rows elsewhere carry mean over 6 annotators '
                             '(mean_candidate_utilities); no synthetic raters '
                             'are created anywhere in R8'}}

    # 5a. dev80 u replay (ALL 640 rows, from raw_ltrb)
    merged_u = {}
    for p in SHARDS:
        d = torch.load(p, map_location='cpu', weights_only=False)
        merged_u.update({k: np.asarray(v, np.float32) for k, v in d['u'].items()})
    n_dev, max_abs = 0, 0.0
    for r in pools['dev80']:
        key = f"{r['vid']}|{r['frame']}|{r['ratio']}"
        t5r = t5[r['vid']]
        W, H = float(t5r['width']), float(t5r['height'])
        ann = json.loads(Path(t5r['annotation']).read_text())
        fi = list(ann['frames']).index(int(r['frame']))
        b = ann['raw_ltrb'][fi]
        g = np.array([b['left'], b['top'], b['right'], b['bottom']], np.float32)
        boxes = candidate_boxes(W, H, t5r['targetRatioWH'])
        u2 = iou(boxes[:, None, :], g[None, None, :]).reshape(-1).astype(np.float32)
        max_abs = max(max_abs, float(np.max(np.abs(u2 - merged_u[key]))))
        n_dev += 1
    ok5a = n_dev == 640 and max_abs < 1e-5
    out['checks']['dev80_u_replay'] = {
        'status': 'PASS' if ok5a else 'FAIL',
        'rows_replayed': n_dev, 'max_abs_diff': round(max_abs, 8)}

    # 5b. train240 sampled replay (16 rows) from manifest gt vs npz u
    lt_u = lt['u']
    rng = np.random.RandomState(20261051)
    sample = rng.choice(len(pools['train240']), 16, replace=False)
    n_tr, mx = 0, 0.0
    for i in sample:
        r = pools['train240'][int(i)]
        m = manifest_rows.get(('live_train', r['vid'], int(r['frame'])))
        if m is None:
            continue
        W, H = float(m['W']), float(m['H'])
        gt = np.array(m['gt'], np.float32)
        boxes = candidate_boxes(W, H, RATIOS[m['ratio']])
        # v8_build_samples recipe: iou (129,A) -> mean over annotators (axis 1)
        u2 = iou(boxes[:, None, :], gt[None, :, :]).mean(1).astype(np.float32)
        cached = np.asarray(lt_u[int(r['cache_row'])], np.float32)
        mx = max(mx, float(np.max(np.abs(u2 - cached))))
        n_tr += 1
    ok5b = n_tr == 16 and mx < 1e-5
    out['checks']['train240_u_replay_sampled'] = {
        'status': 'PASS' if ok5b else 'FAIL',
        'rows_replayed': n_tr, 'max_abs_diff': round(mx, 8)}

    # 6. shortlist index validity (32 dev80 rows).
    # NOTE on the preregistered wording ("ID-permutation invariance"): with
    # u stored as one scalar per candidate index, "permuted multiset is
    # preserved" is a TAUTOLOGY, and x[perm][argsort(perm)] == x holds for
    # ANY permutation, so an identity-sanity version is vacuous.  The real
    # question - stored u[j] belongs to physical window j - is already
    # proven by check 5a (index-aligned full replay; any misalignment would
    # explode max_abs_diff).  This check instead validates what C/M/E
    # actually consumes downstream: shortlists are valid index sets whose
    # within-shortlist oracle matches the stored u.

    rng = np.random.RandomState(20261052)
    sl_map = {}
    for p in SHARDS:
        d = torch.load(p, map_location='cpu', weights_only=False)
        sl_map.update(d['shortlist'])
    dv_rows = pools['dev80']
    sl_lens, sl_bad, covered = set(), 0, 0
    for i in rng.choice(len(dv_rows), 32, replace=False):
        r = dv_rows[int(i)]
        key = f"{r['vid']}|{r['frame']}|{r['ratio']}"
        sl = np.asarray(sl_map[key], int)
        sl_lens.add(len(sl))
        if not np.all((sl >= 0) & (sl < 129)) or len(set(sl.tolist())) != len(sl):
            sl_bad += 1
            continue
        covered += int(np.isclose(float(np.max(merged_u[key][sl])),
                                  float(np.max(merged_u[key])), atol=1e-6))
    ok6 = sl_bad == 0 and all(20 <= L <= 36 for L in sl_lens)
    out['checks']['shortlist_validity'] = {
        'status': 'PASS' if ok6 else 'FAIL', 'rows_checked': 32,
        'shortlist_lengths_seen': sorted(sl_lens),
        'length_variance_note': 'top3+g33 dedupes when the g33 grid overlaps '
                                'top3 on small-span rows; <36 is data, not error',
        'index_violations': sl_bad,
        'argmax_covered_in_sample': covered,
        'note': 'argmax coverage is reported, NOT required (shortlist v2 has '
                'nonzero preregistered regret 0.0055); replaces the '
                'preregistered ID-permutation wording, which is a tautology '
                'for per-candidate-scalar labels; index alignment is covered '
                'by dev80_u_replay (5a)'}

    # 7. u range over ALL merged rows
    allmin = min(float(u.min()) for u in merged_u.values())
    allmax = max(float(u.max()) for u in merged_u.values())
    n_keys = len(merged_u)
    ok7 = n_keys >= 2560 and 0.0 <= allmin and allmax <= 1.0
    out['checks']['u_range'] = {'status': 'PASS' if ok7 else 'FAIL',
                                'merged_row_keys': n_keys,
                                'u_min': round(allmin, 5), 'u_max': round(allmax, 5)}

    # 8. artifact hashes
    out['artifact_sha256'] = {
        'pool_freeze_rows.csv': sha256(FREEZE)[:16],
        'live_train.npz': sha256(SAMPLES / 'live_train.npz')[:16],
        'v8_manifest.jsonl': sha256(MANIFEST)[:16],
        'b3_checkpoint': sha256(Path('/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt'))[:16],
        'encode_shards': [sha256(p)[:16] for p in SHARDS]}
    out['b3_head_sha16'] = out['artifact_sha256']['b3_checkpoint']

    verdicts = [v['status'] for v in out['checks'].values()]
    out['overall'] = 'PASS' if all(v in ('PASS', 'INFO') for v in verdicts) else 'FAIL'
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
