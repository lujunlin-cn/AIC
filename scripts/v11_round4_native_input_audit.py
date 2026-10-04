"""Round-4 native input audit (GPT-6-PRO R3/R4/Q6.2).

Audits the decode path that produced the now-INVALID native VideoMAE caches
(phd2_vmae_native, phd2_vmae_native8).  Decode-only: no model forward, no NPU.

For N stratified source videos this script measures, against a
sequential-decode reference table:
  seek_units    decode_window seeks with int(t * av.time_base) while passing
                stream=st, so PyAV interprets the offset in st.time_base
                units.  Record st.time_base, the implied offset scale factor
                (1e6 * time_base), and the ACTUAL selected PTS per request.
  accuracy      |selected PTS - requested t| and the frame-index gap to the
                nearest frame the reference table would allow.
  duplicates    how often the t-0.25 acceptance rule repeats one frame inside
                a 16-frame window (encoding may pad, scoring frames must not).
  outside       fraction of selected frames that fall outside the sub-window.
  normalize     numeric demo: encode() feeds raw uint8->float (0..255) into
                patch_embed; the VideoMAEv2 contract is (x/255 - MEAN)/STD.

Output: experiments/20261004_v11/round4_native_input_audit.json
"""
import argparse, hashlib, json
from fractions import Fraction
from pathlib import Path
import numpy as np
import av

ap = argparse.ArgumentParser()
ap.add_argument('--index', default='/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl')
ap.add_argument('--media-dir', default='/data/aic/external_datasets/PHD2/raw/youtube')
ap.add_argument('--cache', default='/data/aic/experiments_910a/LFM_V11/phd2_vmae_native')
ap.add_argument('--n-sources', type=int, default=8)
ap.add_argument('--max-prefix-s', type=float, default=900.0,
                help='audit only fragments whose decode prefix stays below this')
ap.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/round4_native_input_audit.json')
args = ap.parse_args()


def sha16(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def load_frag_index():
    """src -> the earliest cached fragment whose decode prefix is affordable."""
    rows = [json.loads(l) for l in Path(args.index).read_text().splitlines() if l.strip()]
    cache = Path(args.cache)
    best = {}
    n_cached = 0
    for r in rows:
        if not (cache / f"{r['video_id']}.npz").exists():
            continue
        n_cached += 1
        t0, L = float(r['t0']), float(r['L'])
        if t0 + L + 2.0 > args.max_prefix_s:
            continue
        cur = best.get(r['src'])
        if cur is None or t0 < cur[0]:
            best[r['src']] = (t0, L)
    return best, n_cached


def durations_of(srcs):
    out = {}
    for s in srcs:
        p = Path(args.media_dir) / f'{s}.mp4'
        try:
            with av.open(str(p)) as c:
                st = c.streams.video[0]
                dur = float(st.duration * st.time_base) if st.duration is not None else None
                out[s] = dur
        except Exception:                                    # noqa: BLE001
            out[s] = None
    return out


def sequential_table(path, t_end):
    """Frame identity table up to t_end seconds: {pts_int: pts_sec} in order."""
    tab = {}
    order = []
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        tb = float(st.time_base)
        start = st.start_time or 0
        for fr in c.decode(st):
            if fr.pts is None:
                continue
            t = (fr.pts - start) * tb
            if fr.pts not in tab:
                tab[fr.pts] = t
                order.append(fr.pts)
            if t > t_end:
                break
    return tab, order, tb, start


def seek_path(path, sub_t0, subL, n=16):
    """Exact copy of decode_window's seek logic in v11_vmae_native_feats.py,
    instrumented.  Returns list of (requested_t, got_pts_int or None)."""
    got_list = []
    with av.open(str(path)) as c:
        st = c.streams.video[0]
        tb_frac = st.time_base
        try:
            tb_frac = Fraction(tb_frac.numerator, tb_frac.denominator)
        except AttributeError:
            tb_frac = Fraction(tb_frac).limit_denominator(10 ** 9)
        tb = float(tb_frac)
        start = st.start_time or 0
        times = sub_t0 + subL * (np.arange(n) + 0.5) / n
        for t in times:
            c.seek(int(t * av.time_base), stream=st, backward=True)
            got = None
            for fr in c.decode(st):
                if fr.time is None or fr.time >= t - 0.25:
                    got = fr
                    break
            if got is None:
                got_list.append((float(t), None))
            else:
                got_list.append((float(t), int(got.pts)))
    return got_list, tb_frac, start


def main():
    best, n_cached = load_frag_index()
    if len(best) < args.n_sources:
        print(f'WARN only {len(best)} affordable sources cached')
    durs = durations_of(sorted(best))
    afford = [s for s in sorted(best, key=lambda s: (durs.get(s) is None, durs.get(s) or 0))]
    # stratify: evenly spaced over the duration-sorted list
    picks = [afford[int(round(i * (len(afford) - 1) / (args.n_sources - 1)))]
             for i in range(args.n_sources)] if len(afford) >= 2 else afford
    picks = sorted(set(picks))
    print(f'cached frags {n_cached}; auditing {len(picks)} sources', flush=True)

    results = {}
    for s in picks:
        p = Path(args.media_dir) / f'{s}.mp4'
        t0, L = best[s]
        subL = L / 4
        try:
            tab, order, tb, start = sequential_table(p, t0 + L + 2.0)
            got_list, stb, _ = seek_path(p, t0, L)
        except Exception as e:                               # noqa: BLE001
            results[s] = {'error': f'{type(e).__name__}: {e}'}
            continue
        # seek_offset_scale: code passes offset_number = t * av.time_base
        # (microsecond count); PyAV reads it in st.time_base units, so the
        # effective seek position is t * av.time_base * st.time_base seconds.
        scale = float(av.time_base) * float(stb)          # 1e6 * time_base
        errs, dups, outside = [], 0, 0
        seen = set()
        per_req = []
        for t, pts in got_list:
            if pts is None:
                per_req.append({'t': round(t, 3), 'got': None})
                continue
            got_t = (pts - start) * tb
            near = min(tab.values(), key=lambda v: abs(v - t)) if tab else None
            errs.append(abs(got_t - t))
            if pts in seen:
                dups += 1
            seen.add(pts)
            if not (t0 <= got_t < t0 + L):
                outside += 1
            per_req.append({'t': round(t, 3), 'got_t': round(got_t, 3),
                            'nearest_t': None if near is None else round(near, 3),
                            'abs_err': round(abs(got_t - t), 3)})
        n_ok = sum(1 for _, pts in got_list if pts is not None)
        results[s] = {
            'frag_t0': round(t0, 3), 'frag_L': round(L, 3),
            'stream_time_base': f'{stb.numerator}/{stb.denominator}',
            'seek_offset_scale_1e6_tb': round(scale, 4),
            'units_compatible': bool(abs(scale - 1.0) < 1e-6),
            'sequential_frames_scanned': len(tab),
            'requests': len(got_list), 'frames_obtained': n_ok,
            'mean_abs_err_s': round(float(np.mean(errs)), 4) if errs else None,
            'max_abs_err_s': round(float(np.max(errs)), 4) if errs else None,
            'duplicate_frames_in_window': dups,
            'frames_outside_fragment': outside,
            'per_request': per_req,
        }
        print(s, json.dumps({k: v for k, v in results[s].items() if k != 'per_request'}),
              flush=True)

    # normalization demo: what encode() actually feeds vs the official contract
    raw = np.random.RandomState(0).randint(0, 256, (2, 4, 4, 3), np.uint8).astype(np.float64)
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    norm = (raw / 255.0 - MEAN) / STD
    norm_demo = {'raw_min': float(raw.min()), 'raw_max': float(raw.max()),
                 'official_min': round(float(norm.min()), 4),
                 'official_max': round(float(norm.max()), 4),
                 'finding': 'encode() sends raw 0..255 floats; VideoMAEv2 '
                            'expects (x/255 - MEAN)/STD in about [-2.1, 2.6]. '
                            'The 0..255 contract violation is CONFIRMED by code '
                            'reading; the numbers here quantify the gap.'}

    n_bad_units = sum(1 for r in results.values()
                      if isinstance(r, dict) and not r.get('units_compatible', True))
    out = {
        'meta': {
            'audited_script_sha16': sha16('/data/aic/experiments_910a/LFM_V11/v11_vmae_native_feats.py')
                if Path('/data/aic/experiments_910a/LFM_V11/v11_vmae_native_feats.py').exists() else None,
            'cache_dir': args.cache, 'media_dir': args.media_dir,
            'n_cached_frags': n_cached, 'n_sources_audited': len(results),
            'decode_only': True,
        },
        'sources': results,
        'summary': {
            'sources_with_incompatible_seek_units': n_bad_units,
            'normalization': norm_demo,
            'verdict': ('Every audited source whose seek_offset_scale != 1.0 '
                        'received seeks at the WRONG position under stream-unit '
                        'interpretation; combined with the 0..255 normalization '
                        'violation the native caches are INVALID_IMPLEMENTATION '
                        'and must be rebuilt under a corrected, hash-versioned '
                        'extraction contract.'),
        },
    }
    Path(args.out).write_text(json.dumps(out, indent=1, default=str) + '\n')
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
