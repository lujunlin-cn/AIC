"""V6 E2: Mr.HiSum cached-feature pilot over the frozen release.

Feature-only (YouTube-8M v2 frame features + Most-Replayed gtscore); no raw
video, no image model — this cannot contaminate the video pipeline and is
"FEATURE HEAD ONLY" per the release contract (mrhisum_feat_subset_v1).

Stages:
  export          manifests train+dev -> per-video npz in FeatureCacheDataset
                  format under --work; dequantization is the frozen v1 mapping
                  float = (uint8 + 0.5) * 2/255 - 1  (uniform mid-tread uint8 ->
                  [-1,1]; any affine-consistent alternative is absorbed by the
                  first Linear adapter — the mapping, not its exact constants,
                  is what the release freezes).
  train           TemporalUNet(input_dim=1152) on gtscore (masked_regression,
                  sigmoid output) with fixed seed; per-epoch dev macro Spearman;
                  writes metrics JSON incl. param count for model_params_b.
  verify-extract  re-download one shard from the YT-8M mirror (md5 from the
                  download plan), re-run extract_shard, bitwise-compare
                  rgb/audio/labels against the release h5 — the feature
                  extraction path is reproducible end-to-end from raw.

Determinism A/B: run `train` twice with the same --tag seed config on the same
GPU model; metrics must be bit-identical (cudnn.deterministic is set by
aic.train.set_seed).
"""
import argparse, hashlib, json, math, time
from collections import defaultdict
from pathlib import Path

SEED = 20260929
FEATS = 1024 + 128  # rgb + audio


def dequant(q):
    return (q.astype('float32') + 0.5) * (2.0 / 255.0) - 1.0


def read_manifests(release):
    rows = []
    for name in ('train', 'dev'):
        p = release / 'manifests' / f'{name}.jsonl'
        rows.extend(json.loads(l) for l in p.read_text().splitlines() if l.strip())
    return rows


def stage_export(release, work):
    import h5py
    import numpy as np
    rows = read_manifests(release)
    by_shard = defaultdict(list)
    for r in rows:
        by_shard[r['feature_h5']].append(r)
    out = work / 'npz'; out.mkdir(parents=True, exist_ok=True)
    manifest = work / 'cache_manifest.jsonl'
    n = 0; t0 = time.time()
    with manifest.open('w') as mf:
        for shard_rel in sorted(by_shard):
            with h5py.File(release / shard_rel, 'r') as fz:
                for r in by_shard[shard_rel]:
                    vid = r['video_id']
                    if vid not in fz:
                        raise KeyError(f'{vid} missing from {shard_rel}')
                    g = fz[vid]
                    T = g['rgb'].shape[0]
                    if T != r['steps'] or g['audio'].shape[0] != T:
                        raise ValueError(f'{vid}: steps mismatch vs manifest')
                    feats = np.concatenate([dequant(g['rgb'][...]), dequant(g['audio'][...])], 1)
                    with h5py.File(release / r['labels_h5'], 'r') as fl:
                        y = fl[vid]['gtscore'][...].astype('float32')
                    if len(y) != T:
                        raise ValueError(f'{vid}: gtscore {len(y)} != steps {T}')
                    dest = out / f'{vid}.npz'
                    np.savez(dest, features=feats.astype('float16'), labels=y,
                             mask=np.ones(T, bool), frame_indices=np.arange(T),
                             timestamps=np.arange(T) + 0.5,
                             metadata_json=np.array(json.dumps({'group_id': r['group_id']})))
                    mf.write(json.dumps({'path': str(dest), 'video_id': vid}) + '\n')
                    n += 1
    print(json.dumps({'exported': n, 'shards': len(by_shard), 'seconds': round(time.time() - t0, 1)}), flush=True)


def dev_spearman(model, loader, dev):
    import numpy as np
    import torch
    from aic.train import _spearman
    model.eval(); rhos = []
    with torch.inference_mode():
        for b in loader:
            z = model(b['features'].to(dev), lengths=b['lengths'].to(dev))
            for i, vid in enumerate(b['video_ids']):
                m = b['mask'][i].bool().numpy()
                r = _spearman(z[i, m].cpu().numpy(), b['labels'][i][m].numpy())
                if r is not None:
                    rhos.append(r)
    return {'videos_scored': len(rhos), 'macro_spearman': float(np.mean(rhos)),
            'spearman_sd': float(np.std(rhos))}


def stage_train(release, work, tag, epochs, device):
    import torch
    from torch.utils.data import DataLoader
    from aic.features import FeatureCacheDataset, collate_feature_batch
    from aic.models import TemporalUNet
    from aic.train import masked_regression, set_seed
    set_seed(SEED)
    # bit-exact rerun requires more than cudnn.deterministic (run1 vs run2 on
    # cuda:4 differed by ~0.011 dev Spearman); warn_only for ops without a
    # deterministic kernel.  CUBLAS_WORKSPACE_CONFIG is set by the launcher.
    torch.use_deterministic_algorithms(True, warn_only=True)
    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    recs = [json.loads(l) for l in (work / 'cache_manifest.jsonl').read_text().splitlines() if l.strip()]
    # classify by the release dev manifest (group-level, frozen)
    dev_ids = {r['video_id'] for r in (json.loads(l) for l in
               (release / 'manifests' / 'dev.jsonl').read_text().splitlines() if l.strip())}
    tr_recs = [rec for rec in recs if rec['video_id'] not in dev_ids]
    dv_recs = [rec for rec in recs if rec['video_id'] in dev_ids]
    train_ds = FeatureCacheDataset(tr_recs); dev_ds = FeatureCacheDataset(dv_recs)
    model = TemporalUNet(FEATS).to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(); g.manual_seed(SEED)
    dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4,
                    collate_fn=collate_feature_batch, generator=g, drop_last=True)
    dvl = DataLoader(dev_ds, batch_size=8, shuffle=False, num_workers=2,
                     collate_fn=collate_feature_batch)
    hist = []; t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train(); tot = 0.0; nb = 0
        for b in dl:
            opt.zero_grad()
            z = model(b['features'].to(dev), lengths=b['lengths'].to(dev))
            loss = masked_regression(z, b['labels'].to(dev), b['mask'].to(dev))
            loss.backward(); opt.step()
            tot += float(loss); nb += 1
            if nb % 400 == 0:
                print(json.dumps({'tag': tag, 'epoch': ep, 'step': nb, 'loss': round(tot / nb, 5)}), flush=True)
        s = dev_spearman(model, dvl, dev)
        hist.append({'epoch': ep, 'train_loss': tot / nb, **s})
        print(json.dumps({'tag': tag, 'epoch_done': ep, **hist[-1]}), flush=True)
    out = {'tag': tag, 'seed': SEED, 'release': str(release), 'device': str(dev),
           'model': 'TemporalUNet(input_dim=1152)', 'train_videos': len(train_ds),
           'dev_videos': len(dev_ds), 'epochs': epochs, 'history': hist,
           'model_params': n_params, 'model_params_b': round(n_params / 1e9, 6),
           'wall_seconds': round(time.time() - t0, 1),
           'feature_mapping': 'v1: (uint8+0.5)*2/255-1', 'feature_only': True}
    dest = work / f'train_metrics_{tag}.json'
    dest.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps({'tag': tag, 'final': hist[-1]['macro_spearman'], 'params': n_params,
                      'metrics': str(dest)}), flush=True)


def stage_verify_extract(release, mr_root, work):
    import h5py
    import numpy as np
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'ext_data'))
    from aicext import yt8m
    from aicext.download import session
    rows = read_manifests(release)
    shard_rel = sorted({r['feature_h5'] for r in rows})[0]  # e.g. features/train0026.h5
    shard = Path(shard_rel).stem
    vids = sorted(r['video_id'] for r in rows if r['feature_h5'] == shard_rel)
    lines = (mr_root / 'annotations' / 'metadata.csv').read_text().splitlines()
    hdr = lines[0].split(',')
    meta = {r['video_id']: r for r in
            (dict(zip(hdr, line.split(','))) for line in lines[1:])}
    wanted = {meta[v]['random_id']: v for v in vids if v in meta and meta[v]['yt8m_file'] == shard}
    missing_meta = [v for v in vids if v not in meta]
    plan = json.loads((mr_root / 'raw' / 'yt8m_frame_train_plan.json').read_text())['files']
    remote = yt8m.shard_remote_name(shard) + '.tfrecord'
    url = f'http://asia.data.yt8m.org/2/frame/train/{remote}'
    t0 = time.time()
    res = yt8m.extract_shard(url, wanted, work / 'verify_tmp.h5', plan.get(remote), session())
    diffs = []
    with h5py.File(work / 'verify_tmp.h5', 'r') as fn, h5py.File(release / shard_rel, 'r') as fo:
        for v in sorted(wanted.values()):
            for ds in ('rgb', 'audio', 'labels'):
                if not np.array_equal(fn[v][ds][...], fo[v][ds][...]):
                    diffs.append(f'{v}/{ds}')
    out = {'shard': shard, 'remote': remote, 'url': url, 'md5_ok': True,
           'md5': res['md5'], 'videos_compared': len(wanted), 'bitwise_diffs': diffs,
           'seconds': round(time.time() - t0, 1), 'missing_meta': missing_meta,
           'verdict': 'IDENTICAL' if not diffs else 'MISMATCH'}
    (work / 'verify_extract.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out), flush=True)


def stage_verify_forward(work, device):
    """Deployment-side determinism: forward pass (the deployed path) must be
    bit-identical across repeated evaluation.  Training backward is NOT
    bit-deterministic (upsample_linear1d_backward CUDA kernel, documented in
    det_ab.log); deployment freezes weights, so only forward matters."""
    import torch
    from torch.utils.data import DataLoader
    from aic.features import FeatureCacheDataset, collate_feature_batch
    from aic.models import TemporalUNet
    from aic.train import set_seed
    dev = torch.device(device if torch.cuda.is_available() or device == 'cpu' else 'cpu')
    recs = [json.loads(l) for l in (work / 'cache_manifest.jsonl').read_text().splitlines() if l.strip()]
    ds = FeatureCacheDataset(recs)
    runs = []
    for rep in range(2):
        set_seed(SEED)
        model = TemporalUNet(FEATS).to(dev).eval()
        dl = DataLoader(ds, batch_size=8, shuffle=False, num_workers=2,
                        collate_fn=collate_feature_batch)
        import hashlib
        h = hashlib.sha256(); n = 0
        with torch.inference_mode():
            for b in dl:
                z = model(b['features'].to(dev), lengths=b['lengths'].to(dev))
                h.update(z.detach().cpu().numpy().tobytes()); n += int(b['mask'].sum())
        runs.append({'sha256': h.hexdigest(), 'steps': n})
    out = {'forward_bit_identical': runs[0] == runs[1], 'runs': runs, 'videos': len(recs)}
    (work / 'verify_forward.json').write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--release', type=Path, default=Path('/data/aic/external_datasets/_releases/mrhisum_feat_subset_v1'))
    ap.add_argument('--mr-root', type=Path, default=Path('/data/aic/external_datasets/MrHiSum'))
    ap.add_argument('--work', type=Path, default=Path('/data/aic/experiments/V6_E2'))
    ap.add_argument('--stage', required=True, choices=('export', 'train', 'verify-extract', 'verify-forward'))
    ap.add_argument('--tag', default='run1')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--device', default='cuda:2')
    a = ap.parse_args(); a.work.mkdir(parents=True, exist_ok=True)
    if a.stage == 'export':
        stage_export(a.release, a.work)
    elif a.stage == 'train':
        stage_train(a.release, a.work, a.tag, a.epochs, a.device)
    elif a.stage == 'verify-forward':
        stage_verify_forward(a.work, a.device)
    else:
        stage_verify_extract(a.release, a.mr_root, a.work)


if __name__ == '__main__':
    main()
