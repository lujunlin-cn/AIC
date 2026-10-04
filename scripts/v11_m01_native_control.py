"""M01 native-framerate control: ord vs rep at 4x the keyframe-grid time
resolution.

Per fragment: 4 sub-windows x 16 native frames each, VideoMAE features
[4,8,768] flattened to a 32-step TCN input.  rep repeats each sub-window's
middle frame (same decode, zero motion).  Labels: the sub-window center
time's binary GT, replicated over the 8 slots of that sub-window.

Paired arms (same TCN, init seed, sampler stream, MSE objective).
PHD2 in-domain diagnosis only.
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--feat-root', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/phd2_vmae_native'))
ap.add_argument('--index', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--eval-sources', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/m01_native_control.json'))
ap.add_argument('--steps', type=int, default=900)
ap.add_argument('--batch', type=int, default=8)
args = ap.parse_args()


def load(feat_root, index, sel, ev_src):
    rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
    X, Y, SRC = [], [], []
    for r in rows:
        f = feat_root / f"{r['video_id']}.npz"
        if not f.exists():
            continue
        m = np.load(f)
        t0, L = float(r['t0']), float(r['L'])
        ivs = []
        for recs in sel.get(r['src'], {}).values():
            for rec in recs:
                a_, b_ = float(rec['t0']) - t0, float(rec['t1']) - t0
                if b_ > a_:
                    ivs.append((a_, b_))
        if not ivs:
            continue
        y4 = np.zeros(4, np.float32)
        for i in range(4):
            c = (i + 0.5) * L / 4
            for a_, b_ in ivs:
                if a_ <= c < b_:
                    y4[i] = 1.0
                    break
        if not (0 < y4.sum() < 4):
            continue
        y32 = np.repeat(y4, 8)
        X.append({'ord': m['ord4'].astype(np.float32).reshape(32, 768),
                  'rep': m['rep4'].astype(np.float32).reshape(32, 768)})
        Y.append(y32)
        SRC.append(r['src'])
    return X, Y, SRC


class TCN(torch.nn.Module):
    def __init__(self, d_in=768, ch=128, dils=(1, 2, 4)):
        super().__init__()
        self.proj = torch.nn.Conv1d(d_in, ch, 1)
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in dils])
        self.out = torch.nn.Conv1d(ch, 1, 1)

    def forward(self, x):
        h = self.proj(x.transpose(1, 2))
        for c in self.convs:
            h = torch.nn.functional.gelu(c(h) + h)
        return self.out(h).squeeze(1)


def ap_of(s, y):
    o = np.argsort(-s)
    ys = y[o]
    return float((np.cumsum(ys) / np.arange(1, len(ys) + 1) * ys).sum() / ys.sum())


def f1_keep(s, y, keep=0.8):
    k = max(1, int(round(keep * len(y))))
    idx = set(np.argsort(-s)[:k].tolist())
    return float(2 * sum(y[i] for i in idx) / (k + y.sum()))


def run_arm(name, key, Xtr, Ytr, Xev, Yev, seed=20261009):
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)
    model = TCN()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for step in range(args.steps):
        idx = rng.choice(len(Xtr), args.batch)
        opt.zero_grad()
        losses = []
        for i in idx:
            logits = model(torch.from_numpy(Xtr[i][key])[None])[0]
            y = torch.from_numpy(Ytr[i])
            losses.append(((torch.sigmoid(logits) - y) ** 2).mean())
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    model.eval()
    aps, f1s = [], []
    with torch.no_grad():
        for x, y in zip(Xev, Yev):
            s = model(torch.from_numpy(x[key])[None])[0].numpy()
            aps.append(ap_of(s, y))
            f1s.append(f1_keep(s, y))
    return {'arm': name, 'ap': round(float(np.mean(aps)), 4),
            'f1_keep080': round(float(np.mean(f1s)), 4),
            'state': {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}}


def main():
    sel = json.loads(args.selections.read_text())
    ev_src = set(json.loads(args.eval_sources.read_text()))
    X, Y, SRC = load(args.feat_root, args.index, sel, ev_src)
    tr = [i for i in range(len(X)) if SRC[i] not in ev_src]
    ev = [i for i in range(len(X)) if SRC[i] in ev_src]
    Xtr = [X[i] for i in tr]; Ytr = [Y[i] for i in tr]
    Xev = [X[i] for i in ev]; Yev = [Y[i] for i in ev]
    print(f'native frags train {len(Xtr)} eval {len(Xev)}', flush=True)
    out = {'pool': {'train': len(Xtr), 'eval': len(Xev),
                    'note': 'native-framerate 4x16-frame windows, 32-step TCN input, ord vs rep, MSE'},
           'arms': {}}
    for name, key in (('ord', 'ord'), ('rep', 'rep')):
        r = run_arm(name, key, Xtr, Ytr, Xev, Yev)
        st = r.pop('state')
        torch.save({'state_dict': {k: torch.from_numpy(v) for k, v in st.items()},
                    'ch': 128, 'dils': [1, 2, 4], 'arm': name},
                   args.out.parent / f'm01_native_arm_{name}.pt')
        out['arms'][name] = r
        print(name, json.dumps(r), flush=True)
    args.out.write_text(json.dumps(out, indent=1))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
