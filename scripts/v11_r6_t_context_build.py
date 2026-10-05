"""R6 T-CONTEXT build: multi-position slicing plan + variant frame extraction.

R6 Q2/Q6 protocol (preregistered seed 20261007):
  * source selection: 300 sources from the census pool with duration info,
    each contributing 1-2 overlap-merged events; train/dev split at the
    SOURCE level (210 train / 90 dev).
  * arms:
      T-anchor (historical protocol): fragment [a-1.0, a-1.0+L) - event
        start anchored at t=1.0 s inside the fragment (the anchor-slicing
        family shape; L = 12 s, within the 8-14 s legal band).
      T-front/mid/back (multi-position protocol): the SAME event placed
        at the fragment FRONT [a, a+L), MIDDLE [c-L/2, c+L/2) and BACK
        [b-L, b); all continuous reads from the ORIGINAL video - the
        event pixels and their order are never altered, and other real
        annotations inside the window are kept as labels (R6 Q6 rule 2).
  * eligibility: all four windows inside [0, D]; ineligible events are
    counted, never deleted.
  * frames: 8 x 1 s keyframes from the fragment start (the pooled-line
    keyframe contract), decoded from the original video and saved as jpg
    for the (CPU/NPU) feature extractor reused from round-4.

  Labels: for each keyframe at absolute time t, y=1 iff t falls inside
  any merged GT interval of the source.

Step 1 (this script): plan + decode jpgs (CPU, multiprocessing).
Output: r6_t_context_plan.json + jpgs under t_context_frames/<variant>/
"""
import argparse, csv, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--seed', type=int, default=20261007)
ap.add_argument('--n-sources', type=int, default=300)
ap.add_argument('--frag-len', type=float, default=12.0)
ap.add_argument('--n-frames', type=int, default=8)
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--info-root', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/raw/youtube'))
ap.add_argument('--frames-root', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/t_context_frames'))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_t_context_plan.json'))
args = ap.parse_args()


def merge(ivs):
    ivs = sorted(ivs)
    out = []
    for a, b in ivs:
        if out and a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def main_plan():
    import av
    sel = json.loads(args.selections.read_text())
    rng = np.random.RandomState(args.seed)
    L, K = args.frag_len, args.n_frames
    cands = []
    for src in sorted(sel):
        p = args.info_root / f'{src}.info.json'
        try:
            dur = float(json.loads(p.read_text())['duration'])
        except Exception:
            continue
        ivs = []
        for recs in sel[src].values():
            for r in recs:
                if float(r['t1']) > float(r['t0']):
                    ivs.append([float(r['t0']), float(r['t1'])])
        evs = [e for e in merge(ivs) if e[1] - e[0] <= L]
        if not evs:
            continue
        # keep sources where at least one event supports all 4 windows
        ok = []
        for a, b in evs:
            c = (a + b) / 2
            wins = {'anchor': (a - 1.0, a - 1.0 + L), 'front': (a, a + L),
                    'mid': (c - L / 2, c + L / 2), 'back': (b - L, b)}
            if all(w0 >= 0 and w1 <= dur for w0, w1 in wins.values()):
                ok.append({'a': a, 'b': b, 'windows': wins})
        if ok:
            cands.append({'src': src, 'dur': dur, 'events': ok})
    print(f'eligible sources: {len(cands)}', flush=True)
    rng.shuffle(cands)
    picked = cands[:args.n_sources]
    dev_n = max(1, int(round(0.3 * len(picked))))
    plan = {'protocol': 'R6 T-CONTEXT; anchor=start-1s; front/mid/back per '
                        f'event; L={L}s continuous reads; 8x1s keyframes; '
                        'ineligible counted not deleted; seed 20261007',
            'frag_len': L, 'seed': args.seed,
            'n_eligible_sources': len(cands), 'n_picked': len(picked),
            'n_dev_sources': dev_n, 'sources': []}
    n_var = 0
    for si, rec in enumerate(picked):
        rec['split'] = 'dev' if si < dev_n else 'train'
        for e in rec['events']:
            for wname, (w0, w1) in e['windows'].items():
                e.setdefault('variants', {})[wname] = {
                    'start': round(w0, 3), 'end': round(w1, 3),
                    'frames': [round(w0 + (w1 - w0) * (k + 0.5) / K, 3)
                               for k in range(K)]}
                n_var += 1
        plan['sources'].append(rec)
    plan['n_variants'] = n_var
    args.out.write_text(json.dumps(plan, indent=1) + '\n')
    print(f'plan: {len(picked)} sources, {n_var} variants, '
          f'{dev_n} dev sources', flush=True)
    return plan


def decode_one(job):
    src, vname, times, media = job
    import av
    od = args.frames_root / src / vname
    if all((od / f'{k}.jpg').exists() for k in range(len(times))):
        return 'cached'
    want = list(enumerate(times))
    od.mkdir(parents=True, exist_ok=True)
    with av.open(media) as con:
        st = con.streams.video[0]
        tb = float(st.time_base)
        con.seek(int(max(times[0] - 2.0, 0) / tb) + int(st.start_time or 0),
                 stream=st)
        ki = 0
        for fr in con.decode(st):
            t = (float(fr.pts) - (st.start_time or 0)) * tb
            if t >= times[ki] - 0.5:
                fr.to_image().save(od / f'{ki}.jpg', quality=92)
                ki += 1
                if ki >= len(times):
                    break
    return 'ok' if ki >= len(times) else f'short {ki}/{len(times)}'


def main_decode(plan):
    jobs = []
    for rec in plan['sources']:
        media = args.info_root / f'{rec["src"]}.mp4'
        if not media.exists():
            continue
        for e in rec['events']:
            for wname, v in e['variants'].items():
                jobs.append((rec['src'], f'{e["a"]:.2f}_{e["b"]:.2f}_{wname}',
                             v['frames'], str(media)))
    print(f'decode jobs: {len(jobs)}', flush=True)
    n_ok = n_short = 0
    with ProcessPoolExecutor(max_workers=24) as ex:
        for i, res in enumerate(ex.map(decode_one, jobs, chunksize=4)):
            n_ok += res in ('ok', 'cached')
            n_short += res not in ('ok', 'cached')
            if (i + 1) % 200 == 0:
                print(f'{i+1}/{len(jobs)} ({n_short} short)', flush=True)
    print(f'decode done: {n_ok} ok, {n_short} short', flush=True)
    plan['decode'] = {'jobs': len(jobs), 'ok': n_ok, 'short': n_short}
    args.out.write_text(json.dumps(plan, indent=1) + '\n')


if __name__ == '__main__':
    p = main_plan()
    main_decode(p)
