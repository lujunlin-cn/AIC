"""PHD2 fragment pool: cut long tier-1 videos into official-shaped short clips.

Why fragments: the official semifinal drop is short (duration p50 14.1 s, p25
9.6, p75 22.6) while every downloaded PHD2 tier-1 video is 90 s - 2 h (p50
166 s).  Training on whole videos would train a temporal model on a length
distribution the evaluation never shows, and would waste teacher queries on
long non-highlight stretches.  So every training unit here is an 8-14 s clip
with an 8-frame 1 s keyframe grid - the same shape the teacher path and the
student head already consume.

Leakage: tier0 (831 official-test is_last GIF hosts) and tier2 (testing.csv
spares) are excluded outright; only tier1 intersect downloaded media is used.
Time-axis: validated 2026-10-01 - max(t1)/duration <= 1.0 for 99.3% of the
pool (0.7% overshoot), so gifs.com second offsets are used as-is.

Geometry: source orientation and target ratio are assigned jointly to match the
measured semifinal distribution (source 66% landscape / 29% portrait / 5%
square; target 71% 16:9 / 29% 9:16).  Each fragment gets a virtual video_id
`<src>_f<NN>` so the existing teacher/feature scripts consume it unchanged.

Outputs (under --out):
  index.jsonl        one row per fragment: video_id, targetRatioWH, src, t0, L, fps
  skel/<fid>.json    {keyframes, W, H, fps}  (teacher script's skeleton)
  frames/<fid>/*.jpg 8 keyframes per fragment
  pool_meta.json     counts, distribution audit, high/low label coverage
"""
import argparse, json, os, random, sys, time
from collections import Counter
from pathlib import Path

import numpy as np

D = Path('/data/aic/external_datasets/PHD2')

# official semifinal composition, measured on intake/index.enriched.jsonl
TARGET_MIX = [(16, 9), (9, 16)]
TARGET_W = [0.71, 0.29]
SOURCE_MIX = [('landscape', 0.66), ('portrait', 0.29), ('square', 0.05)]


def orient_of(w, h):
    ar = w / h
    if ar > 1.5:
        return 'landscape'
    if ar < 0.75:
        return 'portrait'
    return 'square'


def load_annotations():
    """train.json ONLY.

    PHD2 ships a second file, selections/test.json (10,222 videos), which is the
    upstream PHD2 test split - the same GIF-highlight objective the AIC official
    drop is built from (tier0 = the 831 is_last GIF hosts of that split).  667
    tier-1 videos also appear in test.json, so anchoring fragments on test.json
    intervals would pull official-test supervision into training.  v9 hard-blocks
    that by never opening test.json.
    """
    out = {}
    p = D / 'annotations' / 'selections' / 'train.json'
    if not p.exists():
        return out
    d = json.loads(p.read_text())
    for vid, users in d.items():
        ivs = []
        for u, lst in users.items():
            for s in lst:
                t0, t1 = float(s['t0']), float(s['t1'])
                if t1 > t0:
                    ivs.append((t0, t1))
        if ivs:
            out[vid] = ivs
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--n-fragments', type=int, default=6000)
    ap.add_argument('--n-per-video', type=int, default=2)
    ap.add_argument('--clip-min', type=float, default=8.0)
    ap.add_argument('--clip-max', type=float, default=14.0)
    ap.add_argument('--frames', type=int, default=8)
    ap.add_argument('--rotate-frac', type=float, default=0.30,
                    help='fraction of fragments additionally written as 90deg-CW '
                         'pseudo-portrait copies; PHD2 media is 86%% landscape / '
                         '0.7%% portrait, so rotation is the only source of the '
                         'portrait-source geometry that is 29%% of the official set')
    ap.add_argument('--max-fps', type=float, default=30.0)
    ap.add_argument('--seed', type=int, default=20261001)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--extract', action='store_true', help='also decode frames to jpg')
    ap.add_argument('--jpg-quality', type=int, default=90)
    ap.add_argument('--write-index-only', action='store_true')
    ap.add_argument('--sources-file', type=Path, default=None,
                    help='restrict the pool to exactly these source ids (JSON '
                         'list or one-per-line); still intersected with media '
                         'on disk.  Used by the round-4 confirm set, whose '
                         'exclusions are stronger than tier0/tier2.')
    a = ap.parse_args()
    rng = random.Random(a.seed)

    sub = json.loads((D / 'splits' / 'subset_download_v1.json').read_text())
    tier1 = set(sub['tiers']['tier1'])
    banned = set(sub['tiers']['tier0']) | set(sub['tiers']['tier2'])
    have = {p.stem for p in (D / 'raw' / 'youtube').glob('*.mp4')}
    if a.sources_file:
        txt = a.sources_file.read_text().strip()
        parsed = json.loads(txt)
        if isinstance(parsed, dict) and 'sources' in parsed:
            parsed = parsed['sources']
        want = set(parsed) if isinstance(parsed, list) else \
            {l.strip() for l in txt.splitlines() if l.strip()}
        missing = want - have
        if missing:
            print(f'WARN {len(missing)} sources have no media on disk', flush=True)
        pool = sorted(want & have)
    else:
        pool = sorted((tier1 & have) - banned)
    ann = load_annotations()

    # metadata pass (single glob, cache to json so reruns are instant)
    meta_cache = a.out / 'src_meta.json'
    if meta_cache.exists():
        meta = json.loads(meta_cache.read_text())
    else:
        meta = {}
        for vid in pool:
            p = D / 'raw' / 'youtube' / f'{vid}.info.json'
            try:
                d = json.loads(p.read_text())
            except Exception:
                continue
            meta[vid] = {'duration': d.get('duration'), 'width': d.get('width'),
                         'height': d.get('height'), 'fps': d.get('fps')}
        meta_cache.parent.mkdir(parents=True, exist_ok=True)
        meta_cache.write_text(json.dumps(meta))
    print(f'POOL tier1&have&not_banned={len(pool)} meta={len(meta)} '
          f'with_ann={len(set(pool) & set(ann))}', flush=True)

    usable = [v for v in pool
              if v in ann and meta.get(v, {}).get('duration')
              and 90 <= meta[v]['duration'] <= 7200
              and meta[v].get('width') and meta[v].get('height')
              and (meta[v].get('fps') or 30) <= a.max_fps]
    print(f'USABLE {len(usable)}', flush=True)

    # orientation quotas -> per-video fragment budget (keep the source mix)
    cnt = Counter(orient_of(meta[v]['width'], meta[v]['height']) for v in usable)
    print('SOURCE_ORIENT', dict(cnt), flush=True)
    quota = {k: max(1, int(round(len(usable) * w))) for k, w in SOURCE_MIX}
    buckets = {k: [v for v in usable if orient_of(meta[v]['width'], meta[v]['height']) == k]
               for k in quota}
    for k in buckets:
        rng.shuffle(buckets[k])

    order, per_src = [], Counter()
    cap = {k: int(quota[k] * a.n_fragments / max(len(usable), 1)) for k in quota}
    for k, vs in buckets.items():
        for v in vs:
            if per_src[v] >= a.n_per_video or sum(per_src.values()) >= a.n_fragments:
                break
            per_src[v] += 1
            order.append(v)
    rng.shuffle(order)
    order = order[:a.n_fragments]
    print(f'SELECTED {len(order)} videos -> fragments', flush=True)

    rows, skels = [], []
    hi = lo = 0
    for v in order:
        m = meta[v]
        dur, W, H = float(m['duration']), int(m['width']), int(m['height'])
        ivs = ann[v]
        orient = orient_of(W, H)
        for k in range(a.n_per_video):
            if len(rows) >= a.n_fragments:
                break
            L = rng.uniform(a.clip_min, a.clip_max)
            # 65% anchored on a GIF selection (temporal positive), else background
            if rng.random() < 0.65 and ivs:
                s0, s1 = rng.choice(ivs)
                c = rng.uniform(s0, s1)
                t0 = min(max(c - rng.uniform(0.0, 0.6) * L, 0.0), max(dur - L, 0.0))
            else:
                t0 = rng.uniform(0.0, max(dur - L, 0.0))
            fps = min(m.get('fps') or 30.0, a.max_fps)
            fid = f'{v}_f{k:02d}'
            if orient == 'landscape':
                rw, rh = TARGET_MIX[0] if rng.random() < TARGET_W[0] else TARGET_MIX[1]
            elif orient == 'portrait':
                # official: portrait sources are 96% 16:9 targets
                rw, rh = TARGET_MIX[0] if rng.random() < 0.96 else TARGET_MIX[1]
            else:
                rw, rh = (TARGET_MIX[0] if rng.random() < 0.52 else TARGET_MIX[1])
            ov = sum(1 for s0, s1 in ivs if s1 > t0 and s0 < t0 + L)
            hi += (ov > 0)
            lo += (ov == 0)
            # keyframe names must be byte-identical to what the extractor writes:
            # both sides format the *rounded* t0/L from index.jsonl.
            t0r, Lr = round(t0, 2), round(L, 2)
            kfs = [f'{t0r + (Lr * i) / a.frames:.3f}' for i in range(a.frames)]
            rows.append({'video_id': fid, 'targetRatioWH': [rw, rh], 'src': v,
                         't0': t0r, 'L': Lr, 'fps': round(fps, 3),
                         'W': W, 'H': H, 'n_highlight_iv': ov})
            skels.append((fid, {'keyframes': kfs, 'W': W, 'H': H,
                                'fps': round(fps, 3), 't0': t0r}))
            if rng.random() < a.rotate_frac:
                rows.append({'video_id': fid + '_r', 'targetRatioWH': [rw, rh], 'src': v,
                             't0': t0r, 'L': Lr, 'fps': round(fps, 3),
                             'W': H, 'H': W, 'n_highlight_iv': ov, 'rotated': True})
                skels.append((fid + '_r', {'keyframes': kfs,
                                           'W': H, 'H': W, 'fps': round(fps, 3),
                                           't0': t0r, 'rotated': True}))
        if len(rows) >= a.n_fragments:
            break

    rows = rows[a.shard::a.nshards]
    keep = {r['video_id'] for r in rows}
    skels = [s for s in skels if s[0] in keep]

    a.out.mkdir(parents=True, exist_ok=True)
    if a.nshards == 1:
        (a.out / 'index.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        (a.out / 'skel').mkdir(exist_ok=True)
        for fid, s in skels:
            (a.out / 'skel' / f'{fid}.json').write_text(json.dumps(s))
        tmix = Counter(tuple(r['targetRatioWH']) for r in rows)
        smix = Counter(orient_of(r['W'], r['H']) for r in rows)
        (a.out / 'pool_meta.json').write_text(json.dumps({
            'fragments': len(rows), 'source_videos': len({r['src'] for r in rows}),
            'target_mix': {f'{k[0]}:{k[1]}': v for k, v in tmix.items()},
            'source_mix': dict(smix), 'clip_len_mean': round(
                float(np.mean([r['L'] for r in rows])), 2),
            'with_highlight': hi, 'background_only': lo,
            'leakage': 'tier0+tier2 excluded; tier1 only',
            'time_axis_check': 'max(t1)/duration<=1.0 for 99.3% of pool (2026-10-01)',
        }, indent=1, ensure_ascii=False))
        print('WROTE', a.out / 'index.jsonl', len(rows), 'target_mix', dict(tmix),
              'source_mix', dict(smix), 'hi', hi, 'lo', lo, flush=True)
    else:
        (a.out / f'index.shard{a.shard}.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        print('WROTE shard', a.shard, len(rows), flush=True)

    if not a.extract:
        return

    import cv2
    cv2.setNumThreads(1)
    t_start = time.time()
    for i, r in enumerate(rows):
        fd = a.out / 'frames' / r['video_id']
        if len(list(fd.glob('*.jpg'))) >= a.frames:
            continue
        fd.mkdir(parents=True, exist_ok=True)
        rot = bool(r.get('rotated'))
        cap = cv2.VideoCapture(str(D / 'raw' / 'youtube' / f"{r['src']}.mp4"))
        want = [f"{f'{s:.3f}'}" for s in
                [r['t0'] + (r['L'] * j) / a.frames for j in range(a.frames)]]
        got = {}
        for s in want:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(s) * 1000.0)
            ok, fr = cap.read()
            if not ok:
                continue
            if rot:
                fr = cv2.rotate(fr, cv2.ROTATE_90_CLOCKWISE)
            got[s] = fr
        cap.release()
        for s, fr in got.items():
            cv2.imwrite(str(fd / f'{s}.jpg'), fr,
                        [int(cv2.IMWRITE_JPEG_QUALITY), a.jpg_quality])
        if i % 200 == 0:
            el = time.time() - t_start
            print(f'EXTRACT {i}/{len(rows)} {el:.0f}s eta={el / max(i, 1) * (len(rows) - i):.0f}s',
                  flush=True)
    n_jpg = len(list((a.out / 'frames').rglob('*.jpg')))
    print(f'EXTRACT_DONE jpg={n_jpg} wall={time.time() - t_start:.0f}s', flush=True)


if __name__ == '__main__':
    main()