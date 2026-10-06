"""R9 A0 metric-space audit export (GPT6PRO R9 report Q1.4/Q7.5; user approved
the R9 plan 2026-10-06).  CPU only; runs on the 910A box where the pool lives.

Exports the FROZEN common pool into the R9 audit-bundle JSONL schema
  {source_id, fragment_id, labels[8], scores[8]}
with scores = champion head (probe_deploy_head.pt) logits computed here
(CPU, deterministic, same weights the gate used).  Train/eval follow the
frozen eval_sources_50 source split; the audit tool rejects any overlap.

Also writes the pool counting decomposition (Q1.4): index -> features ->
mixed-only -> G histogram, train and eval separately, so downstream
comparisons share one manifest.

Consumed by reports/r9/gpt6pro/metric_sensitivity_audit.py with EXPLICIT
integer K.  Actual K: f1_at_keep(0.80) on n=8 rounds to k = round(6.4) = 6
(real keep 0.75) -- verified in scripts/r8_temporal_gate.py:114 and
scripts/v10_keep_curve_heads.py:151.  The runner also sweeps K=1..8 on the
same frozen list.
"""
import argparse, json, sys, time
from pathlib import Path
from collections import Counter
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('--feat-root', type=Path, default=Path('/data/aic/experiments_910a/LFM_V10/pool_feats'))
ap.add_argument('--index', type=Path, default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/index_clean.jsonl'))
ap.add_argument('--selections', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/annotations/selections/train.json'))
ap.add_argument('--eval-sources', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/eval_sources_50.json'))
ap.add_argument('--champion', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V10/probe_deploy_head.pt'))
ap.add_argument('--out-dir', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit'))
args = ap.parse_args()


class TCN(torch.nn.Module):
    """Verbatim from r8_temporal_gate.py (= champion family)."""

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


def load_fragments(feat_root, index, sel):
    """Verbatim loader (mixed-only) from r8_temporal_gate.py/v11_expected_f_train.py."""
    rows = [json.loads(l) for l in index.read_text().splitlines() if l.strip()]
    X, Y, SRC, FID = [], [], [], []
    for r in rows:
        f = feat_root / f"{r['video_id']}.npz"
        if not f.exists():
            continue
        m = np.load(f)
        if 'mean' not in m or 't' not in m:
            continue
        Xf = m['mean'].astype(np.float32)
        L = len(Xf)
        t0 = float(r['t0'])
        ivs = []
        for recs in sel.get(r['src'], {}).values():
            for rec in recs:
                a_, b_ = float(rec['t0']), float(rec['t1'])
                if b_ > a_:
                    ivs.append((a_ - t0, b_ - t0))
        stamps = sorted(float(x) for x in m['t'])
        times = np.array([stamps[i] - t0 for i in range(L)], np.float32)
        y = np.zeros(L, np.float32)
        half = ((stamps[-1] - stamps[0]) / max(len(stamps) - 1, 1)) * 0.5 if L > 1 else 0.5
        for a_, b_ in ivs:
            y[(times + half >= a_) & (times - half < b_)] = 1.0
        if not (0 < y.sum() < L):
            continue
        X.append(Xf); Y.append(y); SRC.append(r['src']); FID.append(r['video_id'])
    return X, Y, SRC, FID


def main():
    t0 = time.time()
    sel = json.loads(args.selections.read_text())
    ev_src = set(json.loads(args.eval_sources.read_text()))
    X, Y, SRC, FID = load_fragments(args.feat_root, args.index, sel)
    tr = [i for i in range(len(X)) if SRC[i] not in ev_src]
    ev = [i for i in range(len(X)) if SRC[i] in ev_src]
    assert not (set(SRC[i] for i in tr) & set(SRC[i] for i in ev)), 'split overlap'
    print(f'pool: train {len(tr)} eval {len(ev)} frags ({time.time()-t0:.0f}s)', flush=True)

    ck = torch.load(args.champion, map_location='cpu', weights_only=False)
    champ = TCN(ch=ck.get('ch', 128), dils=tuple(ck.get('dils', (1, 2))))
    champ.load_state_dict(ck['state_dict'])
    champ.eval()
    with torch.no_grad():
        S = [champ(torch.from_numpy(X[i])[None])[0].numpy().astype(np.float64)
             for i in range(len(X))]
    print(f'champion forward done ({time.time()-t0:.0f}s)', flush=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    def write(path, idx):
        with path.open('w') as fh:
            for i in idx:
                fh.write(json.dumps({
                    'source_id': SRC[i], 'fragment_id': FID[i],
                    'labels': [int(v) for v in Y[i]],
                    'scores': [round(float(v), 6) for v in S[i]]}) + '\n')

    write(args.out_dir / 'train_public.jsonl', tr)
    write(args.out_dir / 'eval_public.jsonl', ev)

    idx_rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
    decomp = {
        'index_clean_rows': len(idx_rows),
        'index_clean_sources': len(set(r['src'] for r in idx_rows)),
        'feature_available': sum(1 for r in idx_rows
                                 if (args.feat_root / f"{r['video_id']}.npz").exists()),
        'mixed_only_kept': len(X),
        'mixed_only_sources': len(set(SRC)),
        'g_hist_train': dict(sorted(Counter(int(Y[i].sum()) for i in tr).items())),
        'g_hist_eval': dict(sorted(Counter(int(Y[i].sum()) for i in ev).items())),
        'slots_hist_eval': dict(sorted(Counter(len(X[i]) for i in ev).items())),
    }
    rec = {'champion_ckpt': str(args.champion),
           'split': 'eval_sources_50 (source-disjoint)',
           'actual_k_note': 'f1_at_keep(0.80) on n=8 -> k=round(6.4)=6, real keep 0.75',
           'pool_decomposition': decomp}
    (args.out_dir / 'export_record.json').write_text(json.dumps(rec, indent=1) + '\n')
    print(json.dumps(decomp, indent=1), flush=True)
    print('WROTE', args.out_dir, flush=True)


if __name__ == '__main__':
    main()
