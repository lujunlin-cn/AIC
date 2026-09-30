"""V6-NEXT E1-AUTO: pilot-scale training contrast on PM400 clips (V100).

User directive: compare 'public ready-made labels only' (Arm A) vs
'public labels + filtered pseudo-labels' (Arm B) using the public labels
themselves for evaluation; teacher agreement reported separately.

  Arm A  target = PM400 clip_context event window (public, weak temporal):
         y_t = 1 iff event_start_s <= t+0.5 < event_end_s in clip time
         (clips were cut at context start_s, so clip t == context t - start_s)
  Arm B  identical, but each step is weighted by the E1-AUTO teacher QC
         keep_weight (nearest annotated frame; 1 clean / 0.5 uncertain or
         jittery / 0 unparsed) -- consistency filtering only, per the user
         override this is a screening signal, NOT ground truth.

Evidence class: PILOT_SCALE feasibility (24/8 hash split, tiny), never a
selection or packaging signal.  Model: the same TemporalTCN(T1) skeleton,
trained from scratch, 3 seeds per arm.
"""
import argparse, hashlib, json
from pathlib import Path

import numpy as np
import torch

WORK = Path('/data/aic/experiments/V6N_E1A')
FEATS = 1024 + 128


def dequant(q):
    return (q.astype('float32') + 0.5) * (2.0 / 255.0) - 1.0


def load_samples():
    pilot = {r['pm_video_id']: r for r in json.loads((WORK / 'pilot_list.json').read_text())}
    plabels = {}
    pl_path = WORK / 'pseudo_labels.jsonl'
    if pl_path.exists():
        for l in pl_path.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                plabels[r['pm_video_id']] = r
    out = []
    for f in sorted((WORK / 'feats_pilot').glob('*.npz')):
        vid = f.stem
        ctx = pilot[vid]['clip_context']
        d = np.load(f)
        T = int(d['steps'])
        feats = np.concatenate([dequant(d['rgb']), dequant(d['audio'])], 1)
        ts = np.arange(T) + 0.5
        a = max(0.0, ctx['event_start_s'] - ctx['start_s'])
        b = max(a + 1.0, ctx['event_end_s'] - ctx['start_s'])
        y = ((ts >= a) & (ts < b)).astype(np.float32)
        w = np.ones(T, np.float32)
        if vid in plabels:
            fr = plabels[vid]['frames']
            at = np.asarray([fr_['t'] for fr_ in fr])
            aw = np.asarray([fr_['keep_weight'] for fr_ in fr], np.float32)
            idx = np.abs(ts[:, None] - at[None, :]).argmin(1)
            w = aw[idx]
        heldout = int(hashlib.sha1(vid.encode()).hexdigest(), 16) % 4 == 0
        out.append({'vid': vid, 'feats': feats.astype(np.float32), 'y': y, 'w': w,
                    'heldout': heldout, 'pos_rate': float(y.mean())})
    return out


def train_arm(samples, seed, use_w, epochs=30, device='cuda:0'):
    import sys
    sys.path.insert(0, '/home/supie/AIC')
    from scripts.v6n_e4_temporal import build_model
    from aic.train import set_seed
    set_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device(device)
    tr = [s for s in samples if not s['heldout']]
    ho = [s for s in samples if s['heldout']]
    model = build_model('t1').to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for ep in range(epochs):
        model.train()
        for s in tr:
            x = torch.from_numpy(s['feats']).unsqueeze(0).to(dev)
            y = torch.from_numpy(s['y']).unsqueeze(0).to(dev)
            w = torch.from_numpy(s['w'] if use_w else np.ones_like(s['w'])).unsqueeze(0).to(dev)
            opt.zero_grad()
            z = model(x)
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(
                z, y, reduction='none') * w).sum() / w.sum().clamp_min(1.0)
            loss.backward()
            opt.step()
    model.eval()
    rows = []
    with torch.inference_mode():
        for s in ho:
            x = torch.from_numpy(s['feats']).unsqueeze(0).to(dev)
            z = model(x).squeeze(0).cpu().numpy()
            from sklearn.metrics import average_precision_score
            ap = float(average_precision_score(s['y'], z)) if 0 < s['y'].sum() < len(s['y']) else float('nan')
            rows.append({'vid': s['vid'], 'ap': ap, 'pos_rate': round(s['pos_rate'], 3)})
    aps = [r['ap'] for r in rows if r['ap'] == r['ap']]
    return {'seed': seed, 'heldout_macro_ap': float(np.mean(aps)) if aps else None, 'per_video': rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seeds', nargs='*', type=int, default=[20260929, 20260930, 20260931])
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    samples = load_samples()
    print(json.dumps({'videos': len(samples), 'heldout': sum(s['heldout'] for s in samples),
                      'mean_pos_rate': round(float(np.mean([s['pos_rate'] for s in samples])), 3),
                      'with_pseudo': WORK.joinpath('pseudo_labels.jsonl').exists()}), flush=True)
    out = {'arm_A_public_only': [], 'arm_B_public_plus_pseudo': []}
    for seed in a.seeds:
        ra = train_arm(samples, seed, use_w=False, device=a.device)
        out['arm_A_public_only'].append(ra)
        print(json.dumps({'arm': 'A', **{k: ra[k] for k in ('seed', 'heldout_macro_ap')}}), flush=True)
        rb = train_arm(samples, seed, use_w=True, device=a.device)
        out['arm_B_public_plus_pseudo'].append(rb)
        print(json.dumps({'arm': 'B', **{k: rb[k] for k in ('seed', 'heldout_macro_ap')}}), flush=True)
    for key in list(out.keys()):
        vals = [r['heldout_macro_ap'] for r in out[key]]
        out[key + '_mean'] = float(np.mean(vals))
    a_m, b_m = out['arm_A_public_only_mean'], out['arm_B_public_plus_pseudo_mean']
    out['delta_B_minus_A'] = round(b_m - a_m, 4)
    out['evidence_class'] = 'PILOT_SCALE_FEASIBILITY'
    (WORK / 'pilot_contrast.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'A_mean': round(a_m, 4), 'B_mean': round(b_m, 4),
                      'delta': out['delta_B_minus_A']}), flush=True)


if __name__ == '__main__':
    main()
