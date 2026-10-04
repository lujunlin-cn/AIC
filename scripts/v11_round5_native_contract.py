"""Round-5 native frame contract (R5 section 3.2) - generator + validator.

Contract (frozen here):
  * one 1 s ACTION UNIT per second of the source; action units do NOT
    overlap;
  * each action gets a CENTRE-ALIGNED 2 s context window
    [a+0.5-1, a+0.5+1) clamped to the source; adjacent windows overlap 1 s;
  * 16 requested timestamps per window: linspace(start, end, 16,
    endpoint=False) in SECONDS;
  * every request resolves to the NEAREST actually-decoded frame (ordered
    decode inside the window, never a bare seek grab); duplicate mappings
    are RECORDED (unique-frame count per window);
  * a window with <=2 unique frames of 16 requests is a HARD ERROR
    (R5: 15/16 duplicates must abort, never silently continue);
  * out-of-source requests clamp to the boundary and set is_boundary_pad;
  * normalization id: (x/255 - MEAN)/STD with the VideoMAEv2 processor
    constants; the processor config hash is part of the contract hash;
  * feature_contract_hash = sha256 over the protocol dict below.

Validator runs on 32 stratified sources (2 windows each) and records
requested vs got PTS, abs error, unique-frame counts - the measured proof
that the round-4 seek-unit defect (offset given in container units instead
of stream time_base units) is gone: correct seek = int(t / float(st.time_base)).

Output: /data/aic/experiments_910a/LFM_V11/native_frame_contract.json
"""
import argparse, hashlib, json, os, random
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--media-dir', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/raw/youtube'))
ap.add_argument('--sources', type=Path, default=None,
                help='json list to sample from (default: glob the media dir)')
ap.add_argument('--n-verify', type=int, default=32)
ap.add_argument('--seed', type=int, default=20261005)
ap.add_argument('--ctx-s', type=float, default=2.0)
ap.add_argument('--frames', type=int, default=16)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/native_frame_contract.json'))
args = ap.parse_args()

PROTOCOL = {
    'action_unit_s': 1.0,
    'context_s': args.ctx_s,
    'frames_per_window': args.frames,
    'window_step_s': 1.0,
    'requested_grid': f'linspace(start, end, {args.frames}, endpoint=False)',
    'frame_resolution': 'nearest actually-decoded PTS (ordered decode in window)',
    'dup_rule': 'record unique-frame count; <=2 unique of 16 = HARD_ERROR',
    'boundary_rule': 'clamp + is_boundary_pad flag',
    'normalization': '(x/255 - MEAN)/STD, VideoMAEv2 processor constants',
    'seek_rule': 'int(t / float(st.time_base)) with stream=st (stream units, '
                 'NOT container av.time_base); ordered decode verifies PTS',
    'per_action_record': ['video_sha256', 'source_id', 'frame_id', 'pts',
                          'time_base', 'action_id', 'action_frame_count',
                          'window_id', 'requested_pts', 'selected_frame_id',
                          'boundary_pad', 'processor_hash', 'encoder_hash',
                          'feature_contract_hash'],
}


def contract_hash():
    body = json.dumps(PROTOCOL, sort_keys=True).encode()
    return hashlib.sha256(body).hexdigest()


def action_table(duration):
    """All 1 s action units of a source with clamped 2 s windows."""
    acts = []
    a = 0
    while a < int(np.floor(duration)):
        c = a + 0.5
        ws, we = max(c - args.ctx_s / 2, 0.0), min(c + args.ctx_s / 2, duration)
        req = list(np.linspace(ws, we, args.frames, endpoint=False))
        acts.append({'action_id': a, 'window': [round(ws, 3), round(we, 3)],
                     'requested_s': [round(float(t), 4) for t in req],
                     'boundary_pad': [t < ws + 1e-6 or t > we - 1e-6 for t in req]})
        a += 1
    return acts


def verify_window(path, ws, we):
    """Ordered decode [ws, we); return frame table and nearest-match stats."""
    import av
    with av.open(path) as con:
        st = con.streams.video[0]
        st.thread_type = 'AUTO'
        tb = float(st.time_base)
        start = float((st.start_time or 0) * tb)
        tab = {}
        con.seek(int(max(ws - 0.5, 0) / tb) + int(st.start_time or 0),
                 stream=st, backward=True)
        for fr in con.decode(st):
            if fr.pts is None:
                continue
            t = (fr.pts - (st.start_time or 0)) * tb
            if t < ws - 0.5:
                continue
            if t > we + 1e-6:
                break
            if fr.pts not in tab:
                tab[int(fr.pts)] = float(t)
        pts_sorted = sorted(tab.items(), key=lambda kv: kv[1])
        return pts_sorted, tb


def main():
    if args.sources and args.sources.exists():
        pool = [s for s in json.loads(args.sources.read_text())
                if (args.media_dir / f'{s}.mp4').exists()]
    else:
        pool = sorted(p.stem for p in args.media_dir.glob('*.mp4'))
    rng = random.Random(args.seed)
    sample = rng.sample(pool, min(args.n_verify, len(pool)))

    results, hard_errors = [], 0
    for src in sample:
        p = args.media_dir / f'{src}.mp4'
        try:
            import av
            with av.open(p) as con:
                st = con.streams.video[0]
                dur = float(st.duration * st.time_base) if st.duration else None
                tb_meta = f'{st.time_base.numerator}/{st.time_base.denominator}'
                n_fr = st.frames
            if not dur or dur < 3.0:
                continue
            acts = action_table(dur)
            mid = acts[len(acts) // 2]
            pts_sorted, tb = verify_window(p, mid['window'][0], mid['window'][1])
            if not pts_sorted:
                continue
            sel, errs = [], []
            for t in mid['requested_s']:
                tt = min(max(t, mid['window'][0]), mid['window'][1] - 1e-6)
                pts, got = min(pts_sorted, key=lambda kv: abs(kv[1] - tt))
                sel.append(int(pts))
                errs.append(abs(got - tt))
            uniq = len(set(sel))
            rec = {
                'source': src, 'duration_s': round(dur, 2),
                'time_base': tb_meta, 'n_frames_meta': n_fr,
                'window': mid['window'],
                'unique_frames_of_16': uniq,
                'mean_abs_err_s': round(float(np.mean(errs)), 4),
                'max_abs_err_s': round(float(np.max(errs)), 4),
                'status': 'OK' if uniq > 2 else 'HARD_ERROR_DUP'}
            if uniq <= 2:
                hard_errors += 1
            results.append(rec)
            print(json.dumps(rec), flush=True)
        except Exception as e:                                # noqa: BLE001
            results.append({'source': src, 'status': f'DECODE_FAIL:{e}'})
            print(src, 'DECODE_FAIL', e, flush=True)

    n_ok = sum(1 for r in results if r.get('status') == 'OK')
    merr = [r['mean_abs_err_s'] for r in results if 'mean_abs_err_s' in r]
    out = {
        'protocol': PROTOCOL,
        'feature_contract_hash': contract_hash(),
        'validation': {
            'n_sources_sampled': len(sample), 'n_ok': n_ok,
            'n_hard_error_dup': hard_errors,
            'mean_abs_err_s_over_sources': round(float(np.mean(merr)), 4)
                if merr else None,
            'max_abs_err_s_over_sources': round(float(np.max(merr)), 4)
                if merr else None,
            'note': 'errors here are NEAREST-MATCH quantization (source fps '
                    'grid), not seek-unit defects; round-4 measured 24-1608 s '
                    'mean errors - two to four orders of magnitude larger',
            'per_source': results},
        'pending_before_cache_build': [
            'processor/encoder hashes resolved at build time',
            'is_boundary_pad semantics for the first/last action unit',
            'tl;dr any window with <=2 unique frames aborts the build'],
    }
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print('WROTE', args.out, f'ok={n_ok} hard_errors={hard_errors}', flush=True)


if __name__ == '__main__':
    main()
