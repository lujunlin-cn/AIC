"""V6-NEXT E3/E4: target-domain temporal transfer on YouTube-Highlights (YTH).

YouTube-Highlights mirror (raw/mirror/youtube_highlights_full/<domain>/videos):
raw source footage; annotations/segments/<domain>__<vid>.json hold the author
clip windows with match_label (1 = survived the user's highlight edit, 0 =
borderline, -1 unused) on original-upload frame numbers, seconds = frame/29.97
(mirror fps).  No human-in-the-loop here: match_label IS the public label.

Evidence class: TRANSFER_INDICATIVE (n videos small, exposure-diagnostic
only -- never used for selection, training, or thresholds).

Stages:
  list   --n 8  pick videos that have both mirror mp4 + segment json (stride)
  extract              E3 official chain (delegates to v6n_e3_yt8m_extract)
  eval   --weights W   head scores -> per-second GT indicator -> Spearman/AP
"""
import argparse, csv, glob, json, subprocess, sys
from pathlib import Path

YTH = Path('/data/aic/external_datasets/YouTubeHighlights')
WORK = Path('/data/aic/experiments/V6N_E3_YTH')
MIRROR = YTH / 'raw/mirror/youtube_highlights_full'
FPS_MIRROR = 29.970030


def stage_list(n):
    have = {}
    for dom_dir in sorted(MIRROR.iterdir()):
        for mp4 in dom_dir.glob('videos/*.mp4'):
            have[mp4.stem] = mp4
    rows = []
    for seg in sorted(glob.glob(str(YTH / 'annotations/segments/*.json'))):
        d = json.loads(Path(seg).read_text())
        vid = d['video_id']
        if vid not in have:
            continue
        labs = [s['match_label'] for s in d['segments_frames']]
        if labs.count(1) < 3:
            continue
        rows.append({'vid': vid, 'domain': d['domain'], 'mp4': str(have[vid]),
                     'segments': seg, 'official_split': d.get('official_split'),
                     'n_seg': len(labs), 'n_pos': labs.count(1)})
    rows.sort(key=lambda r: (r['domain'], r['vid']))
    picked, stride = [], max(1, len(rows) // n)
    for i in range(0, len(rows), stride):
        picked.append(rows[i])
        if len(picked) >= n:
            break
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / 'yth_transfer_list.json').write_text(json.dumps(picked, indent=1) + '\n')
    print(json.dumps({'picked': len(picked), 'pool': len(rows),
                      'domains': sorted({r['domain'] for r in picked})}), flush=True)


def stage_extract():
    sys.path.insert(0, '/home/supie/AIC')
    from scripts.v6n_e3_yt8m_extract import stage_extract
    rows = json.loads((WORK / 'yth_transfer_list.json').read_text())
    stage_extract([r['mp4'] for r in rows], WORK / 'feats', device='cpu')


def per_second_gt(segments_json, steps):
    d = json.loads(Path(segments_json).read_text())
    y = [0.0] * steps
    cov = [[] for _ in range(steps)]
    for s in d['segments_frames']:
        a = int(s['start_frame'] / FPS_MIRROR)
        b = min(steps - 1, int(round(s['end_frame'] / FPS_MIRROR)) - 1)
        for t in range(max(0, a), b + 1):
            cov[t].append(s['match_label'])
    for t in range(steps):
        c = cov[t]
        if 1 in c:
            y[t] = 1.0
        elif 0 in c:
            y[t] = 0.5
    return y


def stage_eval(weights):
    sys.path.insert(0, '/home/supie/AIC')
    import numpy as np
    import torch
    from scipy.stats import spearmanr
    from sklearn.metrics import average_precision_score
    from scripts.v6n_e3_yt8m_extract import dequant_v1
    from scripts.v6n_e4_temporal import build_model
    ck = torch.load(weights, map_location='cpu', weights_only=False)
    model = build_model(ck['config']).to('cuda:0').eval()
    model.load_state_dict(ck['state_dict'])
    rows = json.loads((WORK / 'yth_transfer_list.json').read_text())
    out = []
    for r in rows:
        f = WORK / 'feats' / f"{r['vid']}.npz"
        if not f.exists():
            continue
        d = np.load(f)
        T = int(d['steps'])
        feats = np.concatenate([dequant_v1(d['rgb']), dequant_v1(d['audio'])], 1)
        x = torch.from_numpy(feats).unsqueeze(0).to('cuda:0')
        with torch.inference_mode():
            s = model(x).squeeze(0).cpu().numpy()
        y = np.asarray(per_second_gt(r['segments'], T))
        rs = float(spearmanr(s, y).statistic) if y.std() > 0 else float('nan')
        ap = float(average_precision_score(y == 1, s)) if (y == 1).any() else float('nan')
        k = max(1, int(round(0.2 * T)))
        top = np.argsort(-s)[:k]
        base = float((y == 1).mean())
        out.append({'vid': r['vid'], 'domain': r['domain'], 'steps': T,
                    'pos_rate': round(base, 4),
                    'spearman_vs_highlight': round(rs, 4),
                    'ap_highlight': round(ap, 4),
                    'prec_at_top20': round(float((y[top] == 1).mean()), 4),
                    'base_rate': round(base, 4)})
        print(json.dumps(out[-1]), flush=True)
    (WORK / 'yth_transfer.json').write_text(json.dumps(out, indent=1) + '\n')
    if out:
        sp = [o['spearman_vs_highlight'] for o in out if o['spearman_vs_highlight'] == o['spearman_vs_highlight']]
        aps = [o['ap_highlight'] for o in out if o['ap_highlight'] == o['ap_highlight']]
        pr = [o['prec_at_top20'] for o in out]
        br = [o['base_rate'] for o in out]
        summ = {'videos': len(out), 'macro_spearman': round(float(np.mean(sp)), 4) if sp else None,
                'macro_ap_highlight': round(float(np.mean(aps)), 4) if aps else None,
                'macro_prec_at_top20': round(float(np.mean(pr)), 4),
                'macro_base_rate': round(float(np.mean(br)), 4),
                'evidence_class': 'TRANSFER_INDICATIVE'}
        (WORK / 'yth_transfer_summary.json').write_text(json.dumps(summ, indent=1) + '\n')
        print(json.dumps(summ), flush=True)


def stage_extract_pilot():
    """E3 chain on the 32 PM400 E1-AUTO pilot clips (same sample_ids on V100)."""
    sys.path.insert(0, '/home/supie/AIC')
    from scripts.v6n_e3_yt8m_extract import stage_extract
    rows = json.loads((WORK / 'pilot_list.json').read_text())
    base = Path('/data/aic/external_datasets/PM400/raw/clips_partial_v1/videos')
    stage_extract([str(base / (r['pm_video_id'] + '.mp4')) for r in rows],
                  WORK.parent / 'V6N_E1A' / 'feats_pilot', device='cpu')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=('list', 'extract', 'extract-pilot', 'eval'))
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--weights', type=Path,
                    default=Path('/data/aic/experiments/V6N_E4/weights_t1_s3.pt'))
    a = ap.parse_args()
    if a.stage == 'list':
        stage_list(a.n)
    elif a.stage == 'extract':
        stage_extract()
    elif a.stage == 'extract-pilot':
        stage_extract_pilot()
    else:
        stage_eval(a.weights)


if __name__ == '__main__':
    main()
