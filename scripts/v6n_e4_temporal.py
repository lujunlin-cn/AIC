"""V6-NEXT E4: temporal candidates over the frozen mrhisum_feat_subset_v1.

Prereg: configs/V6_NEXT10H_PREREG.json (e4_temporal_candidates).

  T0  TemporalUNet(1152), the E2 config, retrained with weight saving
      (the E2 pilot never persisted weights; seed 20260929 reproduces run1
      up to the declared upsample-backward run-to-run variance).
  T1  TemporalTCN(1152): full temporal resolution, dilated residual blocks,
      no down/upsample anywhere (removes the nondeterministic
      upsample_linear1d_backward kernel and the decode blur).
  T2  T0 skeleton + intra-video pairwise margin ranking loss
      (64 uniformly sampled valid pairs per video, lambda=0.5).

Stages:
  export-confirm  npz cache for the release confirmation manifest (4,008
                  videos, fresh reserve never used for selection) -- run once
                  before any confirm eval.
  train           --config t0|t1|t2 --seed S --tag TAG; saves weights +
                  metrics JSON (params, history, wall).
  eval            --weights W --split dev|confirm --out CSV; per-video
                  Spearman rows + macro summary JSON.
"""
import argparse, json, time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

FEATS = 1024 + 128
RELEASE = Path('/data/aic/external_datasets/_releases/mrhisum_feat_subset_v1')
E2_WORK = Path('/data/aic/experiments/V6_E2')


def read_manifest(name):
    return [json.loads(l) for l in (RELEASE / 'manifests' / f'{name}.jsonl').read_text().splitlines() if l.strip()]


def dequant(q):
    return (q.astype('float32') + 0.5) * (2.0 / 255.0) - 1.0


def stage_export_confirm(out_dir):
    import h5py
    rows = read_manifest('confirmation')
    by_shard = defaultdict(list)
    for r in rows:
        by_shard[r['feature_h5']].append(r)
    out = out_dir / 'npz_confirm'; out.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / 'cache_manifest_confirm.jsonl'
    n = 0; t0 = time.time()
    with manifest.open('w') as mf:
        for shard_rel in sorted(by_shard):
            with h5py.File(RELEASE / shard_rel, 'r') as fz:
                for r in by_shard[shard_rel]:
                    vid = r['video_id']
                    g = fz[vid]
                    T = g['rgb'].shape[0]
                    if T != r['steps'] or g['audio'].shape[0] != T:
                        raise ValueError(f'{vid}: steps mismatch')
                    feats = np.concatenate([dequant(g['rgb'][...]), dequant(g['audio'][...])], 1)
                    with h5py.File(RELEASE / r['labels_h5'], 'r') as fl:
                        y = fl[vid]['gtscore'][...].astype('float32')
                    if len(y) != T:
                        raise ValueError(f'{vid}: gtscore {len(y)} != {T}')
                    np.savez(out / f'{vid}.npz', features=feats.astype('float16'), labels=y,
                             mask=np.ones(T, bool), frame_indices=np.arange(T),
                             timestamps=np.arange(T) + 0.5,
                             metadata_json=np.array(json.dumps({'group_id': r['group_id']})))
                    mf.write(json.dumps({'path': str(out / f'{vid}.npz'), 'video_id': vid}) + '\n')
                    n += 1
    print(json.dumps({'exported_confirm': n, 'shards': len(by_shard), 'seconds': round(time.time() - t0, 1)}), flush=True)


def build_model(config):
    import torch.nn as nn
    from aic.models import TemporalUNet

    class TemporalTCN(nn.Module):
        """Full-resolution dilated TCN: adapter -> residual ConvBlocks -> 1x1.

        No pooling, no interpolation: every output step sees a causal-symmetric
        dilated receptive field and gradients never touch
        upsample_linear1d_backward (the nondeterministic kernel root-caused in
        V6_E2/det_ab.log).  lengths handling mirrors TemporalUNet: padded
        batches are decomposed to true-length samples so GroupNorm never sees
        padding.
        """

        DILATIONS = (1, 2, 4, 8, 16, 32)

        def __init__(self, input_dim=FEATS, channels=128):
            super().__init__()
            self.input_dim = input_dim
            self.adapter = nn.Linear(input_dim, channels)
            self.blocks = nn.ModuleList()
            for i, d in enumerate(self.DILATIONS):
                ch_in = channels if i == 0 else channels
                self.blocks.append(nn.Sequential(
                    nn.Conv1d(ch_in, channels, 3, padding=d, dilation=d, bias=False),
                    nn.GroupNorm(32, channels),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(channels, channels, 3, padding=d, dilation=d, bias=False),
                    nn.GroupNorm(32, channels),
                ))
            self.act = nn.ReLU(inplace=True)
            self.output = nn.Conv1d(channels, 1, 1)

        def _forward_core(self, x):
            # x: [B, T, D] at true length (batch-agnostic: conv/linear only)
            h = self.adapter(x).transpose(1, 2)
            for blk in self.blocks:
                h = self.act(h + blk(h))
            return self.output(h).squeeze(1)

        def forward(self, features, aux=None, lengths=None):
            if features.ndim != 3 or features.shape[-1] != self.input_dim:
                raise ValueError(f'Expected [B,T,{self.input_dim}], got {tuple(features.shape)}')
            if lengths is not None:
                lengths = lengths.to(features.device, dtype=torch.long)
                if bool(torch.all(lengths == features.shape[1])):
                    lengths = None
            if lengths is None:
                return self._forward_core(features)[:, :features.shape[1]]
            out = features.new_zeros((features.shape[0], features.shape[1]))
            for i, L in enumerate(lengths.tolist()):
                out[i, :L] = self._forward_core(features[i:i + 1, :L])[0]
            return out

    if config == 't0':
        return TemporalUNet(FEATS)
    if config == 't1':
        return TemporalTCN(FEATS)
    if config == 't2':
        return TemporalUNet(FEATS)
    raise ValueError(config)


def rank_loss(logits, labels, mask, pairs=64, generator=None):
    """Intra-video pairwise margin ranking on uniformly sampled valid pairs."""
    import torch
    B, T = labels.shape
    losses = []
    for b in range(B):
        valid = torch.nonzero(mask[b], as_tuple=False).squeeze(1)
        n = int(valid.numel())
        if n < 2:
            continue
        ii = torch.randint(0, n, (pairs,), generator=generator).to(logits.device)
        jj = torch.randint(0, n, (pairs,), generator=generator).to(logits.device)
        keep = ii != jj
        if not bool(keep.any()):
            continue
        ii, jj = valid[ii[keep]], valid[jj[keep]]
        sign = torch.sign(labels[b, ii] - labels[b, jj])
        nz = sign != 0
        if not bool(nz.any()):
            continue
        losses.append(torch.nn.functional.softplus(-(sign[nz] * (logits[b, ii[nz]] - logits[b, jj[nz]]))).mean())
    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


def dev_spearman(model, loader, device):
    import torch
    from aic.train import _spearman
    model.eval(); rhos = []
    with torch.inference_mode():
        for b in loader:
            z = model(b['features'].to(device), lengths=b['lengths'].to(device))
            for i, vid in enumerate(b['video_ids']):
                m = b['mask'][i].bool().numpy()
                r = _spearman(z[i, m].cpu().numpy(), b['labels'][i][m].numpy())
                if r is not None:
                    rhos.append((vid, r))
    return rhos


def stage_train(config, seed, tag, epochs, device, work):
    import torch
    from torch.utils.data import DataLoader
    from aic.features import FeatureCacheDataset, collate_feature_batch
    from aic.train import masked_regression, set_seed
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device(device)
    set_seed(seed)
    recs = [json.loads(l) for l in (E2_WORK / 'cache_manifest.jsonl').read_text().splitlines() if l.strip()]
    dev_ids = {r['video_id'] for r in read_manifest('dev')}
    tr_recs = [r for r in recs if r['video_id'] not in dev_ids]
    dv_recs = [r for r in recs if r['video_id'] in dev_ids]
    train_ds = FeatureCacheDataset(tr_recs); dev_ds = FeatureCacheDataset(dv_recs)
    model = build_model(config).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(); g.manual_seed(seed)
    dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4,
                    collate_fn=collate_feature_batch, generator=g, drop_last=True)
    dvl = DataLoader(dev_ds, batch_size=8, shuffle=False, num_workers=2,
                     collate_fn=collate_feature_batch)
    pg = torch.Generator(device='cpu'); pg.manual_seed(seed + 1)
    lam = 0.5
    hist = []; t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0; tot_rank = 0.0; nb = 0
        for b in dl:
            opt.zero_grad()
            z = model(b['features'].to(dev), lengths=b['lengths'].to(dev))
            l_reg = masked_regression(z, b['labels'].to(dev), b['mask'].to(dev))
            if config == 't2':
                l_rank = rank_loss(z, b['labels'].to(dev), b['mask'].to(dev), generator=pg)
                loss = l_reg + lam * l_rank
            else:
                l_rank = z.new_zeros(()); loss = l_reg
            loss.backward(); opt.step()
            tot += float(l_reg); tot_rank += float(l_rank); nb += 1
        rhos = dev_spearman(model, dvl, dev)
        macro = float(np.mean([r for _, r in rhos]))
        hist.append({'epoch': ep, 'train_reg': tot / nb, 'train_rank': tot_rank / nb,
                     'dev_macro_spearman': macro})
        print(json.dumps({'config': config, 'seed': seed, 'tag': tag, 'epoch_done': ep, **hist[-1]}), flush=True)
    weights = work / f'weights_{config}_{tag}.pt'
    torch.save({'config': config, 'seed': seed, 'state_dict': model.state_dict(),
                'input_dim': FEATS, 'params': n_params}, weights)
    out = {'config': config, 'seed': seed, 'tag': tag, 'epochs': epochs,
           'history': hist, 'model_params': n_params,
           'model_params_b': round(n_params / 1e9, 9),
           'final_dev_macro_spearman': hist[-1]['dev_macro_spearman'],
           'wall_seconds': round(time.time() - t0, 1),
           'weights_sha256': __import__('hashlib').sha256(weights.read_bytes()).hexdigest(),
           'lambda_rank': lam if config == 't2' else None}
    (work / f'train_{config}_{tag}.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'config': config, 'seed': seed, 'tag': tag, 'final': hist[-1]['dev_macro_spearman'],
                      'params': n_params, 'weights': str(weights)}), flush=True)


def stage_eval(weights, split, out_csv, device, work):
    import torch
    from torch.utils.data import DataLoader
    from aic.features import FeatureCacheDataset, collate_feature_batch
    from aic.train import set_seed
    set_seed(0)
    dev = torch.device(device)
    ckpt = torch.load(weights, map_location='cpu', weights_only=False)
    model = build_model(ckpt['config']).to(dev)
    model.load_state_dict(ckpt['state_dict']); model.eval()
    if split == 'dev':
        recs = [json.loads(l) for l in (E2_WORK / 'cache_manifest.jsonl').read_text().splitlines() if l.strip()]
        dev_ids = {r['video_id'] for r in read_manifest('dev')}
        recs = [r for r in recs if r['video_id'] in dev_ids]
    else:
        recs = [json.loads(l) for l in (work / 'cache_manifest_confirm.jsonl').read_text().splitlines() if l.strip()]
    ds = FeatureCacheDataset(recs)
    dvl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                     collate_fn=collate_feature_batch)
    rhos = dev_spearman(model, dvl, dev)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open('w') as f:
        f.write('video_id,spearman\n')
        for vid, r in sorted(rhos):
            f.write(f'{vid},{r:.6f}\n')
    vals = [r for _, r in rhos]
    summary = {'weights': str(weights), 'config': ckpt['config'], 'seed': ckpt.get('seed'),
               'split': split, 'videos_scored': len(vals),
               'macro_spearman': float(np.mean(vals)), 'spearman_sd': float(np.std(vals)),
               'params': ckpt['params']}
    out_csv.with_suffix('.summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', required=True, choices=('export-confirm', 'train', 'eval'))
    ap.add_argument('--config', default='t0', choices=('t0', 't1', 't2'))
    ap.add_argument('--seed', type=int, default=20260929)
    ap.add_argument('--tag', default='s1')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--work', type=Path, default=Path('/data/aic/experiments/V6N_E4'))
    ap.add_argument('--weights', type=Path)
    ap.add_argument('--split', default='dev', choices=('dev', 'confirm'))
    ap.add_argument('--out', type=Path)
    a = ap.parse_args(); a.work.mkdir(parents=True, exist_ok=True)
    if a.stage == 'export-confirm':
        stage_export_confirm(a.work)
    elif a.stage == 'train':
        stage_train(a.config, a.seed, a.tag, a.epochs, a.device, a.work)
    else:
        stage_eval(a.weights, a.split, a.out, a.device, a.work)


if __name__ == '__main__':
    main()
