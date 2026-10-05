"""R6 T-CONTEXT step 1: eligible-event census for multi-position slicing.

R6 Q2 protocol: for one labeled event, read CONTINUOUS context from the
original video so the event appears at the FRONT / MIDDLE / BACK of a
legal-length fragment.  Eligibility of an event of length Le in a source
video of duration D: three disjoint placements of length L (the fragment
length) must fit, so the video needs D >= 3*L with the event held inside
the fragment.  We census, per source:

  * merged events (overlapping intervals of the same annotation protocol
    merged; kept per-source, selection unit recorded)
  * event lengths (strata <=4 / <=6 / <=8 / >8 s)
  * source durations (yt-dlp info.json)
  * eligible fraction per stratum at fragment lengths L in {8, 10, 12, 14}
  * sources usable for the paired dev stress test (>= 2 eligible events)

Ineligible events are counted and reported - NEVER deleted from test
records (R6 Q6).  CPU only; output: r6_t_context_census.json.
"""
import argparse, glob, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--info-root', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/raw/youtube'))
ap.add_argument('--frag-lens', nargs='*', type=float, default=[8, 10, 12, 14])
ap.add_argument('--max-src', type=int, default=0, help='0 = all')
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_t_context_census.json'))
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


def main():
    sel = json.loads(args.selections.read_text())
    srcs = sorted(sel) if not args.max_src else sorted(sel)[:args.max_src]
    n_ev, n_ev_ok_cache = [], []
    strata = {f'<= {x}s' if False else x: {'ev': 0, 'ok': {L: 0 for L in args.frag_lens}}
              for x in ('<=4s', '<=6s', '<=8s', '>8s')}
    strata = {'<=4s': {'ev': 0, 'ok': {L: 0 for L in args.frag_lens}},
              '<=6s': {'ev': 0, 'ok': {L: 0 for L in args.frag_lens}},
              '<=8s': {'ev': 0, 'ok': {L: 0 for L in args.frag_lens}},
              '>8s': {'ev': 0, 'ok': {L: 0 for L in args.frag_lens}}}
    n_src_dur, n_src_noinfo, ev_lens = 0, 0, []
    src_events = {}
    for si, src in enumerate(srcs):
        p = args.info_root / f'{src}.info.json'
        try:
            dur = float(json.loads(p.read_text())['duration'])
        except Exception:
            n_src_noinfo += 1
            continue
        n_src_dur += 1
        ivs = []
        for recs in sel[src].values():
            for r in recs:
                if float(r['t1']) > float(r['t0']):
                    ivs.append([float(r['t0']), float(r['t1'])])
        evs = merge(ivs)
        src_events[src] = {'dur': dur, 'events': evs}
        for a, b in evs:
            Le = b - a
            ev_lens.append(Le)
            st = '<=4s' if Le <= 4 else '<=6s' if Le <= 6 else '<=8s' if Le <= 8 else '>8s'
            strata[st]['ev'] += 1
            for L in args.frag_lens:
                # event must fit inside ANY placement of fragment length L:
                # placements can start anywhere in [0, D-L]; event [a,b] fits
                # iff D >= L and there is room for 3 disjoint fragments each
                # containing the event at front/middle/back: D >= 3*L is the
                # sufficient rule (front: [a, a+L], back: [b-L, b], middle in
                # between when D >= 3L).  Ineligible events stay in the count.
                if dur >= 3 * L and Le <= L:
                    strata[st]['ok'][L] += 1
        if (si + 1) % 20000 == 0:
            print(f'{si+1}/{len(srcs)} sources scanned', flush=True)
    ev_lens = np.array(ev_lens)
    n_ev_multi = sum(1 for s in src_events.values()
                     if len(s['events']) >= 2 and s['dur'] >= 3 * min(args.frag_lens))
    out = {
        'protocol': 'R6 Q2 eligible-event census; overlap-merged events per '
                    'source; eligibility = duration >= 3*fragment_length AND '
                    'event length <= fragment length (sufficient rule); '
                    'ineligible events are counted, never deleted',
        'n_sources_total': len(srcs),
        'n_sources_with_duration': n_src_dur,
        'n_sources_no_info': n_src_noinfo,
        'n_events': int(len(ev_lens)),
        'event_len_s': {'p10': round(float(np.percentile(ev_lens, 10)), 2),
                        'p50': round(float(np.percentile(ev_lens, 50)), 2),
                        'p90': round(float(np.percentile(ev_lens, 90)), 2)} if len(ev_lens) else None,
        'strata': {k: {'events': v['ev'],
                       'eligible': {str(L): v['ok'][L] for L in args.frag_lens},
                       'eligible_rate': {str(L): round(v['ok'][L] / max(v['ev'], 1), 3)
                                         for L in args.frag_lens}}
                   for k, v in strata.items()},
        'sources_with_2plus_eligible_events': n_ev_multi,
        'note': 'Larger-than-needed D makes eligibility EASIER; the census is '
                'an upper bound on build cost and a lower bound on eligibility.',
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
