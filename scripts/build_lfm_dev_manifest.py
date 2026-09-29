"""Build the frozen LFM450M spatial dev manifest from RetargetVid dev2 (031-100).

Stratified, fixed-seed selection of 50 source videos covering: non-person
subjects (low face rate), fast motion, frequent shot changes, and regular
fill.  Each video contributes BOTH ratio tasks (1-3 portrait target, 3-1
landscape target); per-video pairing is preserved in every analysis.
Keyframes come from the same 1 s pool the teacher used (Qwen_RV200_POINT_D1),
so LFM, teacher and B0 are compared on identical frames.

Metadata comes only from the observation caches (YuNet face choice, B0 crop
motion, shot resets) -- never from teacher predictions or sealed sets.
"""
import argparse, hashlib, json, random
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--cache', type=Path, default=Path('/data/aic/experiments/OBS_CACHE_RETARGET_ALL200'))
ap.add_argument('--points-dir', type=Path, default=Path('/data/aic/experiments/QWEN_RV200_POINT_D1/points'))
ap.add_argument('--first', default='031')
ap.add_argument('--last', default='100')
ap.add_argument('--n', type=int, default=50)
ap.add_argument('--seed', type=int, default=0)
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()

vids = [f'{i:03d}' for i in range(int(args.first), int(args.last) + 1)]
RATIOS = {'1-3': [1, 3], '3-1': [3, 1]}
meta = {}
for vid in vids:
    entry = {}
    face_hits = face_tot = 0
    motion = []
    resets = frames = 0
    ok = True
    for r in RATIOS:
        z = args.cache / f'{vid}_{r}.npz'
        pj = args.points_dir / f'{vid}.json'
        if not z.exists() or not pj.exists():
            ok = False
            break
        npz = np.load(z)
        chosen = npz['chosen']
        b0 = np.asarray(npz['b0'])
        reset = npz['reset'].astype(bool)
        face_hits += int((np.asarray(chosen) >= 0).sum())
        face_tot += len(chosen)
        ctr = b0[:, 0] + b0[:, 1]
        if len(ctr) > 1:
            motion.append(float(np.abs(np.diff(ctr))[~reset[1:]].mean()))
        resets += int(reset.sum())
        frames += len(reset)
        if 'W' not in entry:
            m = json.loads(pj.read_text())
            entry.update({'W': m['W'], 'H': m['H'],
                          'n_keyframes': len(m['keyframes']),
                          'keyframes': m['keyframes']})
    if not ok or frames == 0:
        continue
    entry.update({'face_rate': face_hits / max(face_tot, 1),
                  'motion_px': float(np.mean(motion)) if motion else 0.0,
                  'shot_rate': resets / frames})
    meta[vid] = entry

mv = np.array([m['motion_px'] for m in meta.values()])
sr = np.array([m['shot_rate'] for m in meta.values()])
motion_hi = np.quantile(mv, 2 / 3)
shot_hi = np.quantile(sr, 2 / 3)

rng = random.Random(args.seed)
pool = sorted(meta)
layer = {}
a = [v for v in pool if meta[v]['face_rate'] < 0.10]
rng.shuffle(a)
layer['nonperson'] = sorted(a[:10])
rest = [v for v in pool if v not in layer['nonperson']]
b = [v for v in rest if meta[v]['motion_px'] >= motion_hi]
rng.shuffle(b)
layer['fast_motion'] = sorted(b[:12])
rest = [v for v in rest if v not in layer['fast_motion']]
c = [v for v in rest if meta[v]['shot_rate'] >= shot_hi]
rng.shuffle(c)
layer['shot_cuts'] = sorted(c[:12])
rest = [v for v in rest if v not in layer['shot_cuts']]
d = [v for v in rest if v not in layer['shot_cuts']]
rng.shuffle(d)
layer['regular'] = sorted(d[:args.n - sum(len(x) for x in layer.values())])

sel = {v: l for l, vs in layer.items() for v in vs}
out = {
    'protocol_id': 'lfm450_spatial_dev_v1',
    'created': '2026-09-29',
    'source_pool': f'{args.first}-{args.last} (RetargetVid dev2, never confirm2/official)',
    'seed': args.seed,
    'selection_rule': 'nonperson=face_rate<0.10 (10); fast_motion=motion>=p67 (12); shot_cuts=shot_rate>=p67 (12); regular=fill (16); layers disjoint, rng(seed=0)',
    'quantiles': {'motion_hi': float(motion_hi), 'shot_hi': float(shot_hi)},
    'ratios': RATIOS,
    'keyframes_source': str(args.points_dir),
    'frames_dir': '/data/aic/experiments/QWEN_RV200_KEYFRAMES/d1',
    'obs_cache': str(args.cache),
    'annotations': '/data/aic/experiments_910a/LFM450_EVAL_V1/annotations',
    'videos': [{'vid': v, 'layer': sel[v], **meta[v]} for v in sorted(sel)],
}
blob = json.dumps(out, ensure_ascii=False, sort_keys=True, indent=1)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(blob + '\n')
print(json.dumps({'n_videos': len(out['videos']),
                  'layers': {k: len(v) for k, v in layer.items()},
                  'sha256': hashlib.sha256(blob.encode()).hexdigest()[:16]}))
