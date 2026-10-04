"""Round-5 B4 (multi-process): native VideoMAEv2-B window features.

Same frozen contract as v11_round5_native_feats.py (hash ac3b657f42264c9a
family - re-derived from the same args), but decode runs in a CPU worker
pool and the NPU only sees normalized fp16 batches:

  worker (per SOURCE, one av.open):  ordered decode per 2 s window,
  nearest-PTS selection, resize, normalize -> fp16 [3,16,224,224] per
  window; returns windows + meta (unique count, err, boundary, tubelet
  centers).
  main (NPU): batches of --batch-windows windows -> forward_tokens ->
  spatial mean -> [8, 768] tubelet anchors -> per-source npz.

Throughput target: the single-process probe measured 0.775 win/s; the
serial decode was the bottleneck, hence workers.  MEASURE, do not promise
(linear card scaling is explicitly forbidden in this repo).

Hard error contract unchanged: <=2 unique frames of 16 in any window
aborts the build (strict), the worker raises and the source is logged.

Output: --out-dir/<src>.npz (feats fp16 [n_actions, 8, 768], meta json,
contract_hash, encoder), --out-dir/build_log.jsonl, and a build summary
json.
"""
import argparse, hashlib, json, os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--media-dir', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/raw/youtube'))
ap.add_argument('--sources-file', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/round5_native_dev_sources.json'))
ap.add_argument('--out-dir', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/R5_NATIVE_FEATS_V1'))
ap.add_argument('--weights', default='/data/aic/pretrained/videomaev2')
ap.add_argument('--ctx-s', type=float, default=2.0)
ap.add_argument('--frames', type=int, default=16)
ap.add_argument('--size', type=int, default=224)
ap.add_argument('--batch-windows', type=int, default=32)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--limit-windows', type=int, default=0)
ap.add_argument('--max-sources', type=int, default=0)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/round5_native_feats_build.json'))
args = ap.parse_args()

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
CONTRACT_HASH = hashlib.sha256(json.dumps({
    'ctx_s': args.ctx_s, 'frames': args.frames, 'size': args.size,
    'mean_std_id': 'videomaev2-in1k', 'seek_rule': 'stream time_base units',
    'frame_rule': 'nearest decoded PTS'}, sort_keys=True).encode()).hexdigest()[:16]
ENCODER_ID = 'videomaev2-vitb-distilled'


def action_table(duration, ctx_s, frames):
    acts = []
    a = 0
    while a < int(np.floor(duration)):
        c = a + 0.5
        ws, we = max(c - ctx_s / 2, 0.0), min(c + ctx_s / 2, duration)
        req = list(np.linspace(ws, we, frames, endpoint=False))
        acts.append({'action_id': a, 'window': [round(ws, 3), round(we, 3)],
                     'requested_s': [round(float(t), 4) for t in req]})
        a += 1
    return acts


def decode_source(task):
    """Worker: decode ALL windows of one source (single av.open).
    Returns (src, status, [meta], [fp16 arrs])."""
    src, path_str = task
    import cv2
    cv2.setNumThreads(1)
    import av
    metas, arrs = [], []
    try:
        with av.open(path_str) as con:
            st = con.streams.video[0]
            tb = float(st.time_base)
            dur = float(st.duration * st.time_base) if st.duration else None
            if not dur or dur < 3.0:
                return src, 'SKIP_SHORT', [], []
            for act in action_table(dur, args.ctx_s, args.frames):
                ws, we = act['window']
                con.seek(int(max(ws - 0.5, 0) / tb) + int(st.start_time or 0),
                         stream=st, backward=True)
                tab = {}
                for fr in con.decode(st):
                    if fr.pts is None:
                        continue
                    t = (fr.pts - (st.start_time or 0)) * tb
                    if t < ws - 0.5:
                        continue
                    if t > we + 1e-6:
                        break
                    if int(fr.pts) not in tab:
                        img = fr.to_ndarray(format='rgb24')
                        img = cv2.resize(img, (args.size, args.size),
                                         interpolation=cv2.INTER_AREA)
                        tab[int(fr.pts)] = (float(t), img)
                if not tab:
                    continue
                sel, errs = [], []
                for t in act['requested_s']:
                    tt = min(max(t, ws), we - 1e-6)
                    pts, (got, _img) = min(tab.items(),
                                           key=lambda kv: abs(kv[1][0] - tt))
                    sel.append(pts); errs.append(abs(got - tt))
                uniq = len(set(sel))
                if uniq <= 2:
                    return src, f'HARD_ERROR_DUP:action={act["action_id"]}:uniq={uniq}', [], []
                imgs = [tab[p][1] for p in sel]
                arr = np.stack(imgs).astype(np.float32) / 255.0
                arr = (arr - MEAN) / STD
                arrs.append(np.ascontiguousarray(
                    arr.transpose(3, 0, 1, 2)).astype(np.float16))
                metas.append({
                    'action_id': act['action_id'], 'window': act['window'],
                    'unique_of_16': uniq,
                    'mean_err_s': round(float(np.mean(errs)), 4),
                    'tubelet_centers_s': [round(
                        (act['requested_s'][2 * j] + act['requested_s'][2 * j + 1]) / 2, 3)
                        for j in range(8)]})
        return src, 'OK', metas, arrs
    except Exception as e:                                       # noqa: BLE001
        return src, f'DECODE_FAIL:{type(e).__name__}:{e}', [], []


def forward_tokens(model, x):
    """Token-level forward; this port's patch_embed has NO CLS token
    (verified: patch_embed out is exactly 8x14x14 = 1568)."""
    B = x.size(0)
    h = model.patch_embed(x)
    if model.pos_embed is not None:
        h = h + model.pos_embed.expand(B, -1, -1).type_as(h).to(
            h.device).clone().detach()
    h = model.pos_drop(h)
    for blk in model.blocks:
        h = blk(h)
    if model.fc_norm is not None:
        h = model.fc_norm(h)
    return h


def main():
    sys.path.insert(0, '/data/aic/pretrained')
    sys.path.insert(0, '/data/aic/experiments_910a/LFM_V11')
    import torch
    import torch_npu                                        # noqa: F401
    from videomaev2.modeling_videomaev2 import VisionTransformer  # noqa: E402
    from v11_m01_conv3d_bridge import patch_videomae_conv3d  # noqa: E402
    from safetensors.torch import load_file

    DEV = 'npu'
    cfg = json.load(open(Path(args.weights) / 'config.json'))['model_config']
    model = VisionTransformer(**cfg)
    sd = load_file(Path(args.weights) / 'model.safetensors')
    sd = {k[len('model.'):] if k.startswith('model.') else k: v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    patch_videomae_conv3d(model)
    model = model.float().to(DEV).eval()
    torch.set_grad_enabled(False)
    print('encoder loaded', flush=True)

    sel = json.loads(Path(args.sources_file).read_text())
    srcs = sorted(set(sel['train_sources']) | set(sel['eval_sources'])) \
        if isinstance(sel, dict) else sel
    if args.max_sources:
        srcs = srcs[:args.max_sources]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    todo = [s for s in srcs if not (args.out_dir / f'{s}.npz').exists()
            and (args.media_dir / f'{s}.mp4').exists()]
    print(f'{len(srcs)} sources, {len(todo)} to build', flush=True)

    tasks = [(s, str(args.media_dir / f'{s}.mp4')) for s in todo]
    import multiprocessing.pool as mpool
    pool = mpool.Pool(args.workers, initializer=_init_worker)
    blog = (args.out_dir / 'build_log.jsonl').open('a')
    t0, n_win, hard, done = time.time(), 0, 0, 0
    try:
        for src, status, metas, arrs in pool.imap(decode_source, tasks):
            if status.startswith('HARD_ERROR'):
                hard += 1
                blog.write(json.dumps({'src': src, 'status': status}) + '\n')
                blog.flush()
                print('HARD_ERROR', src, status, flush=True)
                pool.terminate()
                break
            if status != 'OK' or not arrs:
                blog.write(json.dumps({'src': src, 'status': status}) + '\n')
                blog.flush()
            else:
                p = args.out_dir / f'{src}.npz'
                all_feats = []
                ts = cfg['tubelet_size']
                n_t = cfg['num_frames'] // (ts[0] if isinstance(ts, (list, tuple)) else ts)
                for i in range(0, len(arrs), args.batch_windows):
                    batch = np.stack(arrs[i:i + args.batch_windows])
                    xx = torch.from_numpy(batch).float().to(DEV)
                    with torch.autocast('npu', dtype=torch.float16):
                        y = forward_tokens(model, xx)
                    tok = y.float()
                    B, Ntok, D = tok.shape
                    n_s = int(round((Ntok / n_t) ** 0.5))
                    all_feats.append(tok.reshape(B, n_t, n_s, n_s, D)
                                     .mean((2, 3)).half().cpu().numpy())
                F = np.concatenate(all_feats).astype(np.float16)
                path = args.media_dir / f'{src}.mp4'
                np.savez_compressed(
                    p, feats=F, meta=json.dumps(metas),
                    contract_hash=CONTRACT_HASH, encoder=ENCODER_ID,
                    video_sha256=hashlib.sha256(path.read_bytes()).hexdigest()[:16])
                n_win += len(metas)
                blog.write(json.dumps({'src': src, 'status': 'OK',
                                       'n_windows': len(metas)}) + '\n')
                blog.flush()
            done += 1
            if args.limit_windows and n_win >= args.limit_windows:
                pool.terminate()
                break
            if done % 10 == 0 or done == len(todo):
                el = time.time() - t0
                print(f'[{done}/{len(todo)}] win={n_win} '
                      f'{n_win / max(el, 1e-9):.2f} win/s', flush=True)
    finally:
        pool.terminate()
    blog.close()

    el = time.time() - t0
    out = {'contract_hash': CONTRACT_HASH, 'encoder': ENCODER_ID,
           'sources_requested': len(todo), 'hard_errors': hard,
           'windows_built': n_win,
           'windows_per_sec': round(n_win / max(el, 1e-9), 3),
           'wall_s': round(el, 1),
           'limit_windows': args.limit_windows}
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


def _init_worker():
    import signal
    signal.signal(signal.SIGINT, signal.SIG_IGN)




if __name__ == '__main__':
    main()
