"""R8 diagnostic packages D_N / D_X / D_Y (r7 slot_strategy default; user
approved 2026-10-06, quota is DAILY-5 so diagnostics do not crowd out real
candidates).

Design frozen before construction (baseline = the scored champion
LFM_V11_VTREPLAY, official 34.95; each package applies ONE known synthetic
perturbation to the champion's predictions, so the official score drop
buys the transfer slope of one domain):

  D_N  temporal localisation : every video's kept-frame indices shift by
       delta = round(0.1 * n_frames) (out-of-range frames dropped; frame
       COVERAGE unchanged, only timing moves).  Reads the marginal
       official value of temporal localisation (orthogonal to the
       historical keep-rate/k_size slope).
  D_X  horizontal localisation : x' = clamp(x + 0.05*W, 0, W-w).
  D_Y  vertical localisation   : y' = clamp(y + 0.05*H, 0, H-h)
       (h derived from w and targetRatioWH exactly as aic.contract does).

Identifiability (synthetic scenarios, no media IO): on a synthetic
per-video GT (known highlight frames + known centered crop), the SAME
perturbation operators must move the local metrics monotonically
(binary temporal F1 down for D_N, IoU down for D_X/D_Y) by clearly more
than numeric noise - otherwise the official delta could not be attributed.

Outputs: three submission dirs + zips next to the champion, plus a JSON
design/verification record.  The official readbacks land in
transfer_pairs with this file as lineage.
"""
import argparse, json, math, shutil, sys, time, zipfile
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--champion-dir', type=Path,
                default=Path('/data/aic/semifinal_20261001/submissions/LFM_V11_VTREPLAY_SEMIFINAL'))
ap.add_argument('--index', type=Path,
                default=Path('/data/aic/semifinal_20261001/intake/index.enriched.jsonl'))
ap.add_argument('--out-root', type=Path,
                default=Path('/data/aic/semifinal_20261001/submissions'))
ap.add_argument('--record', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r8_npu/diagnostic_pkgs.json'))
args = ap.parse_args()

sys.path.insert(0, '/root/AIC')
from aic.contract import load_index, load_jsonl, write_submission  # noqa: E402


def shift_frames(row, delta_frac=0.1):
    delta = round(delta_frac * len(row['predictions']))
    preds = [dict(p) for p in row['predictions']]
    shifted = []
    for p in preds:
        nf = p['frame'] + delta
        if nf < row['_n_frames']:
            shifted.append({'frame': nf, 'bboxes': list(p['bboxes'])})
    row['predictions'] = shifted
    return delta


def shift_box(row, axis, frac, meta):
    W = meta.width if hasattr(meta, 'width') else meta['width']
    H = meta.height if hasattr(meta, 'height') else meta['height']
    out = []
    for p in row['predictions']:
        x, y, w = p['bboxes']
        rw, rh = row['targetRatioWH']
        h = w * rh / rw
        if axis == 'x':
            x2 = min(max(x + frac * W, 0.0), W - w)
            out.append({'frame': p['frame'], 'bboxes': [round(x2, 5), y, w]})
        else:
            y2 = min(max(y + frac * H, 0.0), H - h)
            out.append({'frame': p['frame'], 'bboxes': [x, round(y2, 5), w]})
    row['predictions'] = out


# ---------- synthetic identifiability ----------
def synth_identifiability():
    """Pure-synthetic GT: 40 videos x 300 frames, highlight = frames
    [60,140), crop = centered legal box; perturb; local metrics must fall."""
    rng_time, ious_t = [], []
    for seed in range(8):
        n = 300
        gt_frames = set(range(60, 140))
        W, H = 1280, 720
        gx, gy, gw = 280.0, 180.0, 720.0
        gh = gw * 9 / 16
        # oracle submission rows
        pred = [{'frame': f, 'bboxes': [gx, gy, gw]} for f in sorted(gt_frames)]
        row = {'video_id': 's', 'targetRatioWH': [16, 9], 'predictions': pred,
               '_n_frames': n}
        meta = {'width': W, 'height': H}

        def t_f1(prs):
            pr = set(p['frame'] for p in prs)
            hit = len(pr & gt_frames)
            return 2 * hit / max(len(pr) + len(gt_frames), 1)

        def iou(prs):
            x, y, w = prs[0]['bboxes']
            h = w * 9 / 16
            ax1, ay1, ax2, ay2 = x, y, x + w, y + h
            bx1, by1, bx2, by2 = gx, gy, gx + gw, gy + gh
            ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
            iy = max(0.0, min(ay2, by2) - max(ay1, by1))
            inter = ix * iy
            a1, a2 = w * h, gw * gh
            return inter / (a1 + a2 - inter)

        base_t, base_i = t_f1(pred), iou(pred)
        r1 = json.loads(json.dumps(row)); d = shift_frames(r1, 0.1)
        t_n = t_f1(r1['predictions'])
        r2 = json.loads(json.dumps(row)); shift_box(r2, 'x', 0.05, meta)
        i_x = iou(r2['predictions'])
        r3 = json.loads(json.dumps(row)); shift_box(r3, 'y', 0.05, meta)
        i_y = iou(r3['predictions'])
        rng_time.append((base_t, t_n, d))
        ious_t.append((base_i, i_x, i_y))
    t_drop = float(np_mean([b - a for b, a, _ in rng_time]))
    i_drop_x = float(np_mean([b - c for b, c, _ in ious_t]))
    i_drop_y = float(np_mean([b - c for b, _, c in ious_t]))
    return {'time_f1_base': round(rng_time[0][0], 4),
            'time_f1_after_shift': round(float(np_mean([a for _, a, _ in rng_time])), 4),
            'time_f1_drop': round(t_drop, 4),
            'delta_frames_applied': rng_time[0][2],
            'iou_base': round(ious_t[0][0], 4),
            'iou_after_x': round(float(np_mean([c for _, c, _ in ious_t])), 4),
            'iou_after_y': round(float(np_mean([c for _, _, c in ious_t])), 4),
            'iou_drop_x': round(i_drop_x, 5), 'iou_drop_y': round(i_drop_y, 5),
            'identifiable': bool(t_drop > 0.05 and i_drop_x > 0.02 and i_drop_y > 0.02)}


def np_mean(xs):
    return sum(xs) / max(len(xs), 1)


def main():
    t0 = time.time()
    ident = synth_identifiability()
    print('identifiability:', json.dumps(ident), flush=True)
    assert ident['identifiable'], 'perturbation not identifiable on synthetic GT'

    index = load_index(args.index)
    rows = load_jsonl(args.champion_dir / 'predictions.jsonl')
    print(f'champion rows {len(rows)} ({time.time()-t0:.0f}s)', flush=True)

    record = {'design': 'D_N shift+10% frames / D_X x+5%W / D_Y y+5%H on '
                        'frozen champion LFM_V11_VTREPLAY (34.95); one '
                        'perturbation per package; official drops buy '
                        'per-domain transfer slopes',
              'baseline': str(args.champion_dir),
              'synthetic_identifiability': ident, 'packages': {}}

    for name in ('DN', 'DX', 'DY'):
        new_rows = [json.loads(json.dumps(
            {k: v for k, v in r.items() if k != '_n_frames'})) for r in rows]
        # recompute n_frames from the index (predictions len != n_frames)
        infos = []
        for r, orig in zip(new_rows, rows):
            meta = index[r['video_id']]
            orig_n = len(orig['predictions'])
            r['_n_frames'] = meta.frame_count
            infos.append((r, meta, orig_n))
        for r, meta, orig_n in infos:
            if name == 'DN':
                d = shift_frames(r, 0.1)
            elif name == 'DX':
                shift_box(r, 'x', 0.05, meta)
                d = 0.05
            else:
                shift_box(r, 'y', 0.05, meta)
                d = 0.05
        out_dir = args.out_root / f'LFM_D{name}_DIAG'
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        for r in new_rows:
            r.pop('_n_frames', None)   # private bookkeeping never ships
        write_submission(out_dir / 'predictions.jsonl', new_rows, index,
                         stage='preliminary')
        zp = out_dir.parent / f'LFM_D{name}_DIAG.zip'
        with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as z:
            z.write(out_dir / 'predictions.jsonl', 'predictions.jsonl')
        kept = [len(r['predictions']) for r in new_rows]
        record['packages'][name] = {
            'dir': str(out_dir), 'zip': str(zp),
            'perturbation': ('shift frames +round(0.1*n)' if name == 'DN' else
                             ('x + 5%W' if name == 'DX' else 'y + 5%H')),
            'preds_mean': round(float(np_mean(kept)), 1),
            'preds_min': min(kept), 'preds_max': max(kept)}
        print(name, 'written', zp, flush=True)

    record['elapsed_min'] = round((time.time() - t0) / 60, 1)
    args.record.parent.mkdir(parents=True, exist_ok=True)
    args.record.write_text(json.dumps(record, indent=1) + '\n')
    print('WROTE', args.record, flush=True)


if __name__ == '__main__':
    main()
