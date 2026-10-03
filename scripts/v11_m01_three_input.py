"""M01 three-input control: does video-native motion evidence help ranking?

Arms, same TCN, same init, same sampler stream:
  ord  VideoMAE features from the time-ordered 16-frame window
  rep  features from the middle frame repeated 16x (same static content,
       zero motion evidence)

If ord > rep passes the in-domain gate, motion evidence exists in the
VideoMAE representation.  If both tie, the gain (if any) is static
capacity.  PHD2 in-domain diagnosis only (V10 report rule 4).  CPU.
"""
import argparse, json, os
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--feat-root', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/phd2_vmae_feats'))
ap.add_argument('--frame-root', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/frames'))
ap.add_argument('--index', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--eval-sources', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/m01_three_input.json'))
ap.add_argument('--steps', type=int, default=900)
ap.add_argument('--batch', type=int, default=8)
args = ap.parse_args()


def load(feat_root, frame_root, index, sel, ev_src):
    rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
    X, Y, SRC = [], [], []
    for r in rows:
        f = feat_root / f"{r['video_id']}.npz"
        if not f.exists():
            continue
        m = np.load(f)
        t0 = float(r['t0'])
        ivs = []
        for recs in sel.get(r['src'], {}).values():
            for rec in recs:
                a_, b_ = float(rec['t0']), float(rec['t1'])
                if b_ > a_:
                    ivs.append((a_ - t0, b_ - t0))
        jpgs = sorted((frame_root / r['video_id']).glob('*.jpg'), key=lambda p: float(p.stem))
        if len(jpgs) < 4 or not ivs:
            continue
        stems = np.array([float(p.stem) for p in jpgs], np.float32)
        T = len(stems)
        i_ord = np.round(np.linspace(0, T - 1, 8)).astype(int).clip(0, T - 1)
        times = stems[i_ord] - t0
        half = ((stems[-1] - stems[0]) / max(T - 1, 1)) * 0.5
        y = np.zeros(8, np.float32)
        for a_, b_ in ivs:
            y[(times + half >= a_) & (times - half < b_)] = 1.0
        if not (0 < y.sum() < 8):
            continue
        X.append({'ord': m['ord'].astype(np.float32), 'rep': m['rep'].astype(np.float32)})
        Y.append(y)
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


def run_arm(name, key, Xtr, Ytr, Xev, Yev, seed=20261007):
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
    X, Y, SRC = load(args.feat_root, args.frame_root, args.index, sel, ev_src)
    tr = [i for i in range(len(X)) if SRC[i] not in ev_src]
    ev = [i for i in range(len(X)) if SRC[i] in ev_src]
    Xtr = [X[i] for i in tr]; Ytr = [Y[i] for i in tr]
    Xev = [X[i] for i in ev]; Yev = [Y[i] for i in ev]
    print(f'frags train {len(Xtr)} eval {len(Xev)}', flush=True)
    out = {'pool': {'train': len(Xtr), 'eval': len(Xev),
                    'note': 'VideoMAE 8-slot features; ord vs rep paired arms; MSE objective'},
           'arms': {}}
    for name, key in (('ord', 'ord'), ('rep', 'rep')):
        r = run_arm(name, key, Xtr, Ytr, Xev, Yev)
        st = r.pop('state')
        torch.save({'state_dict': {k: torch.from_numpy(v) for k, v in st.items()},
                    'ch': 128, 'dils': [1, 2, 4], 'arm': name},
                   args.out.parent / f'm01_arm_{name}.pt')
        out['arms'][name] = r
        print(name, json.dumps(r), flush=True)
    args.out.write_text(json.dumps(out, indent=1))
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
