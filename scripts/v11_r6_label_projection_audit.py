"""R6 Q7-B: quantify the 1 s block label-projection error on original frames.

R6 section Q7-B: any block-constant label loses information vs original-frame
labels.  This script measures the loss on the native-contract dev pool with
REAL per-frame data:

  * per-frame PTS: PyAV packet-level demux (no pixel decode), sorted and
    deduplicated.  Packet PTS equals decoded-frame PTS for these streams;
    verified on the 12 contract-validation sources by recomputing the
    nearest-decoded-PTS match error and comparing with the stored
    contract records.
  * action block t: the 1 s half-open interval centred on the action,
    [c - 0.5, c + 0.5), c = midpoint of the stored contract window.
  * d_t = # original frames with PTS in block t
  * h_t = # original frames with PTS in block t AND inside GT
  * y_t = 1[h_t > 0]  (frame-truth projection; compared with the
    tubelet-centre labels the posbl line trained on - agreement reported)
  * a_t = O-arm (3 seeds, mean score) keep-0.80 mask per source

Reported (R6 Q7-B formulas, exact):
  e_proj = sum_t [y_t (d_t - h_t) + (1 - y_t) h_t] / sum_t d_t
  e_min  = sum_t min(h_t, d_t - h_t) / sum_t d_t
  partial_block_rate = #(0 < h_t < d_t) / #blocks
  F_T(a) = 2 sum_t a_t h_t / (sum_t a_t d_t + G),  G = sum_t h_t
           (empty-GT sources are excluded from the macro, matching the
           posbl pool construction; single-zero side -> F = 0)
  block_F1  = standard binary F1 on (a_t, y_t)
  frame_oracle_F = F_T(a = y_t) - what a perfect block predictor keeps
  frame_perfect_F = 1.0 by construction (selecting exactly the GT frames)
  gap = 1 - frame_oracle_F  = irreducible loss of BLOCK-CONSTANT output

CPU only.  Output: r6_label_projection_audit.json (+ per-source csv)
"""
import argparse, csv, glob, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--feat-dir', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/R5_NATIVE_FEATS_V1'))
ap.add_argument('--sources-file', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/round5_native_dev_sources.json'))
ap.add_argument('--media-root', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/raw/youtube'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--posbl-glob', default='/data/aic/experiments_910a/LFM_V11/round5_posbl_matrix_task_O_s*_lr0.0001.pt')
ap.add_argument('--contract-audit', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/native_frame_contract.json'))
ap.add_argument('--keep', type=float, default=0.80)
ap.add_argument('--pts-cache', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_pts_cache'))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_label_projection_audit.json'))
args = ap.parse_args()


def load_labels(sel):
    out = {}
    for src, users in sel.items():
        ivs = []
        for recs in users.values():
            for r in recs:
                if float(r['t1']) > float(r['t0']):
                    ivs.append((float(r['t0']), float(r['t1'])))
        if ivs:
            out[src] = ivs
    return out


def pts_of_source(src, media_root, cache_dir):
    """Packet-level PTS seconds, sorted deduped. Cached as npz."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    c = cache_dir / f'{src}.pts.npz'
    if c.exists():
        d = np.load(c)
        return d['pts']
    import av
    # media filename: PHD2 videos are stored as <src>.<ext> under the root
    p = None
    for ext in ('.mp4', '.webm', '.mkv', '.avi', '.mov'):
        q = media_root / f'{src}{ext}'
        if q.exists():
            p = q
            break
    if p is None:
        raise FileNotFoundError(f'no media for {src} under {media_root}')
    con = av.open(str(p))
    st = con.streams.video[0]
    tb = float(st.time_base)
    start = float(st.start_time or 0)
    seen = set()
    pts = []
    for pkt in con.demux(st):
        if pkt.pts is None:
            continue
        v = (float(pkt.pts) - start) * tb
        key = int(pkt.pts)
        if key in seen:
            continue
        seen.add(key)
        pts.append(v)
    con.close()
    pts = np.array(sorted(pts), np.float64)
    np.savez_compressed(c, pts=pts)
    return pts


def block_stats(pts, lo, hi, ivs):
    """d_t, h_t for one 1 s half-open block."""
    m = (pts >= lo) & (pts < hi)
    d = int(m.sum())
    h = 0
    for a, b in ivs:
        h += int(((pts >= max(lo, a)) & (pts < min(hi, b))).sum())
    return d, min(h, d)


def f_frame(a, h, d, G):
    """R6 Q7-B F_T: original-frame temporal F of a block mask."""
    num = 2.0 * float(np.sum(a * h))
    den = float(np.sum(a * d)) + G
    if den == 0:
        return 0.0
    return num / den


def main():
    sel = json.loads(args.selections.read_text())
    labels = load_labels(sel)
    src_sel = json.loads(args.sources_file.read_text())
    eval_srcs = sorted(set(src_sel['eval_sources']))
    # posbl O-arm dev scores
    sc = {}
    keep_src_order = None
    for p in sorted(glob.glob(args.posbl_glob)):
        import torch
        dd = torch.load(p, map_location='cpu', weights_only=False)
        for i, s in enumerate(dd['keep']):
            if s in eval_srcs:
                sc.setdefault(s, []).append(dd['scores'][i])
        keep_src_order = dd['keep']
    print(f'O-arm score coverage: {len(sc)} eval sources '
          f'x {len(sc[list(sc)[0]]) if sc else 0} seeds', flush=True)

    contract = json.loads(args.contract_audit.read_text()) if args.contract_audit.exists() else {}
    contract_srcs = set()
    if isinstance(contract.get('sources'), list):
        contract_srcs = {r.get('src') for r in contract['sources'] if isinstance(r, dict)}

    rows, tot = [], dict(d=0, h=0, blocks=0, partial=0, e_proj_num=0.0, e_min_num=0.0)
    f_frame_list, f_block_list, agree_list = [], [], []
    per_src = []
    for si, src in enumerate(eval_srcs):
        if src not in sc:
            continue
        p = args.feat_dir / f'{src}.npz'
        if not p.exists():
            continue
        d_np = np.load(p)
        meta = json.loads(str(d_np['meta']))
        try:
            pts = pts_of_source(src, args.media_root, args.pts_cache)
        except FileNotFoundError as e:
            print('SKIP', src, e, flush=True)
            continue
        ivs = labels.get(src, [])
        if not ivs:
            continue
        s_mean = np.mean(sc[src], axis=0)                # [n_act] mean over seeds
        n_act = len(meta)
        assert len(s_mean) == n_act, f'{src}: {len(s_mean)} vs {n_act}'
        d_t = np.zeros(n_act, np.float64)
        h_t = np.zeros(n_act, np.float64)
        y_tube = np.zeros(n_act, np.float64)             # posbl labels (tubelet proj)
        for ai, m in enumerate(meta):
            w0, w1 = m['window']
            c = 0.5 * (w0 + w1)
            dd, hh = block_stats(pts, c - 0.5, c + 0.5, ivs)
            d_t[ai], h_t[ai] = dd, hh
            y_tube[ai] = float(any(
                any(lo <= tc <= hi for lo, hi in ivs)
                for tc in m['tubelet_centers_s']))
        y_t = (h_t > 0).astype(np.float64)
        if h_t.sum() == 0 or d_t.sum() == 0:
            continue
        k = max(1, int(round(args.keep * n_act)))
        a_t = np.zeros(n_act, np.float64)
        a_t[np.argsort(-s_mean)[:k]] = 1.0
        G = float(h_t.sum())
        # per-source aggregation of the exact formulas
        e_proj_num = float(np.sum(y_t * (d_t - h_t) + (1 - y_t) * h_t))
        e_min_num = float(np.sum(np.minimum(h_t, d_t - h_t)))
        nb = float(d_t.sum())
        partial = int(np.sum((h_t > 0) & (h_t < d_t)))
        Ffr = f_frame(a_t, h_t, d_t, G)
        # block-level F1 with the same mask
        tp = float(np.sum(a_t * y_t))
        prec = tp / max(float(a_t.sum()), 1e-9)
        rec = tp / max(G, 1e-9)
        Fb = 2 * prec * rec / max(prec + rec, 1e-9) if (prec + rec) > 0 else 0.0
        Fora = f_frame(y_t, h_t, d_t, G)
        agree = float(np.mean(y_t == y_tube))
        rows.append([src, n_act, nb, G, e_proj_num / nb, e_min_num / nb,
                     partial / n_act, Ffr, Fb, Fora, agree])
        per_src.append(dict(src=src, n_act=n_act, frames=int(nb), gt_frames=int(G),
                            e_proj=e_proj_num / nb, e_min=e_min_num / nb,
                            partial_rate=partial / n_act, F_frame=Ffr,
                            F_block=Fb, F_frame_oracle=Fora, label_agree=agree))
        tot['d'] += nb; tot['h'] += G; tot['blocks'] += n_act
        tot['partial'] += partial; tot['e_proj_num'] += e_proj_num
        tot['e_min_num'] += e_min_num
        f_frame_list.append(Ffr); f_block_list.append(Fb); agree_list.append(agree)
        if (si + 1) % 32 == 0:
            print(f'  {si+1}/{len(eval_srcs)} sources done', flush=True)

    e_proj = tot['e_proj_num'] / max(tot['d'], 1)
    e_min = tot['e_min_num'] / max(tot['d'], 1)
    # pooled frame-oracle macro
    Fora_macro = float(np.mean([r[9] for r in rows])) if rows else 0.0
    out = {
        'protocol': 'R6 Q7-B label projection audit; packet-level PTS (deduped), '
                    '1 s half-open blocks at action centres; formulas exact per R6; '
                    'O-arm keep-0.80 mask (3-seed mean); frame_oracle = perfect '
                    'block predictor (a=y); frame_perfect = 1.0 by construction',
        'n_sources': len(rows),
        'totals': {'frames': tot['d'], 'gt_frames': tot['h'], 'blocks': tot['blocks'],
                   'partial_blocks': tot['partial']},
        'e_proj': round(e_proj, 5),
        'e_min': round(e_min, 5),
        'partial_block_rate': round(tot['partial'] / max(tot['blocks'], 1), 5),
        'F_frame_macro': round(float(np.mean(f_frame_list)), 4) if f_frame_list else None,
        'F_block_macro': round(float(np.mean(f_block_list)), 4) if f_block_list else None,
        'F_frame_minus_block_macro': round(float(np.mean(f_frame_list) - np.mean(f_block_list)), 4) if f_frame_list else None,
        'frame_oracle_macro': round(Fora_macro, 4),
        'block_constant_loss_oracle': round(1.0 - Fora_macro, 4),
        'label_agree_frame_vs_tubelet_macro': round(float(np.mean(agree_list)), 4) if agree_list else None,
        'contract_srcs_overlap': len(set(r['src'] for r in per_src) & contract_srcs),
        'per_source_csv': args.out.with_name(args.out.stem + '_per_source.csv').as_posix(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    with open(args.out.with_name(args.out.stem + '_per_source.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['src', 'n_act', 'frames', 'gt_frames', 'e_proj', 'e_min',
                    'partial_rate', 'F_frame', 'F_block', 'F_frame_oracle', 'label_agree'])
        w.writerows(rows)
    print(json.dumps({k: v for k, v in out.items() if k != 'per_source_csv'},
                     indent=1), flush=True)
    print('WROTE', args.out, flush=True)


if __name__ == '__main__':
    main()
