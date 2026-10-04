"""Round-5 B4: native VideoMAEv2-B window features on NPU (R5 section 3.2).

Implements the frozen native frame contract:
  1 s action units, centre-aligned 2 s windows, 16 requested timestamps,
  ordered decode inside the window (stream-time_base seek units!), nearest
  decoded PTS per request, duplicate counting with HARD_ERROR on <=2
  unique frames of 16, boundary pads flagged.
Per window the encoder keeps tubelet anchors: [8, 768] (16 frames -> 8
tubelets of 2, spatial token mean over the real 14x14 grid), fp16 on disk,
plus the per-window record (requested/got pts, unique count, boundary).

Cache identity: every npz carries feature_contract_hash + encoder id +
processor id.  Throughput is measured first (--limit windows) before any
full build (R5: never promise linear scaling in cards).

CPU decode + NPU forward pipeline.  Single GPU (logical 0).
Output: --out-dir/<src>.npz + --out-dir/build_log.jsonl (per-source stats)
"""
import argparse, hashlib, json, os, sys, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '8')
from pathlib import Path
import numpy as np
import cv2
cv2.setNumThreads(1)

ap = argparse.ArgumentParser()
ap.add_argument('--media-dir', type=Path,
                default=Path('/data/aic/external_datasets/PHD2/raw/youtube'))
ap.add_argument('--sources-file', type=Path, default=None)
ap.add_argument('--out-dir', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/R5_NATIVE_FEATS_V1'))
ap.add_argument('--weights', default='/data/aic/pretrained/videomaev2')
ap.add_argument('--ctx-s', type=float, default=2.0)
ap.add_argument('--frames', type=int, default=16)
ap.add_argument('--size', type=int, default=224)
ap.add_argument('--batch-windows', type=int, default=8)
ap.add_argument('--limit-windows', type=int, default=0,
                help='throughput probe: stop after N windows total')
ap.add_argument('--max-sources', type=int, default=0)
ap.add_argument('--strict-dup', action='store_true', default=True)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/round5_native_feats_build.json'))
args = ap.parse_args()

sys.path.insert(0, '/data/aic/pretrained')
sys.path.insert(0, str(Path(args.weights).parent))
sys.path.insert(0, '/data/aic/experiments_910a/LFM_V11')

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


def decode_window(con_st, con, ws, we):
    """Ordered decode [ws, we); returns sorted [(pts_int, t_sec, img224)].
    Pixels are kept so the window is decoded exactly ONCE (the previous
    window_pixels re-opened and re-decoded the file - 2x cost)."""
    st = con_st
    tb = float(st.time_base)
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
    return sorted(tab.items(), key=lambda kv: kv[1][0])


def forward_tokens(model, x):
    """Token-level forward: the HF forward_features pools internally
    (fc_norm(x.mean(1)) -> [B, D]); we need the [B, 1+Ntok, D] sequence.
    Same ops as forward_features, then the TRAINED fc_norm on tokens."""
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
    return h                                                  # [B, 1+Ntok, D]


def main():
    import cv2
    cv2.setNumThreads(1)
    import torch
    import torch_npu                                        # noqa: F401
    from videomaev2.modeling_videomaev2 import VisionTransformer  # noqa: E402
    from v11_m01_conv3d_bridge import patch_videomae_conv3d  # noqa: E402

    DEV = 'npu'
    cfg = json.load(open(Path(args.weights) / 'config.json'))['model_config']
    model = VisionTransformer(**cfg)
    from safetensors.torch import load_file
    sd = load_file(Path(args.weights) / 'model.safetensors')
    sd = {k[len('model.'):] if k.startswith('model.') else k: v
          for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    patch_videomae_conv3d(model)
    model = model.float().to(DEV).eval()
    torch.set_grad_enabled(False)
    print('encoder loaded', sum(p.numel() for p in model.parameters()) / 1e6,
          'M params', flush=True)

    if args.sources_file and Path(args.sources_file).exists():
        parsed = json.loads(Path(args.sources_file).read_text())
        srcs = parsed['sources'] if isinstance(parsed, dict) else parsed
    else:
        srcs = sorted(p.stem for p in args.media_dir.glob('*.mp4'))
    if args.max_sources:
        srcs = srcs[:args.max_sources]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    blog = (args.out_dir / 'build_log.jsonl').open('a')

    n_win_total, t_start, hard_errors = 0, time.time(), 0
    stats = {'sources_done': 0, 'windows_done': 0, 'windows_skipped_dup': 0,
             'mean_unique_frac': [], 'mean_err_s': []}
    import av
    for si, src in enumerate(srcs):
        op = args.out_dir / f'{src}.npz'
        if op.exists():
            continue
        p = args.media_dir / f'{src}.mp4'
        if not p.exists():
            continue
        try:
            con = av.open(p)
            st = con.streams.video[0]
            dur = float(st.duration * st.time_base) if st.duration else None
            if not dur or dur < 3.0:
                con.close(); continue
            acts = action_table(dur, args.ctx_s, args.frames)
            F, meta = [], []
            buf_windows, buf_meta = [], []
            for act in acts:
                tab = decode_window(st, con, act['window'][0], act['window'][1])
                if not tab:
                    continue
                sel, errs = [], []
                for t in act['requested_s']:
                    tt = min(max(t, act['window'][0]), act['window'][1] - 1e-6)
                    pts, (got, _img) = min(tab, key=lambda kv: abs(kv[1][0] - tt))
                    sel.append(pts); errs.append(abs(got - tt))
                uniq = len(set(sel))
                if uniq <= 2:
                    hard_errors += 1
                    stats['windows_skipped_dup'] += 1
                    if args.strict_dup:
                        raise RuntimeError(
                            f'HARD_ERROR dup window src={src} '
                            f'action={act["action_id"]} uniq={uniq}')
                    continue
                pts2img = dict(tab)
                buf_windows.append([pts2img[p_][1] for p_ in sel])
                buf_meta.append((act, uniq, errs))
                if len(buf_windows) == args.batch_windows or act is acts[-1]:
                    batch = np.stack(buf_windows)
                    arr = batch.astype(np.float32) / 255.0
                    arr = (arr - MEAN) / STD
                    x = torch.from_numpy(np.ascontiguousarray(
                        arr.transpose(0, 4, 1, 2, 3))).float().to(DEV)
                    with torch.autocast('npu', dtype=torch.float16):
                        y = forward_tokens(model, x)               # [B, Ntok, D]
                    tok = y.float()      # this port's patch_embed has NO CLS
                                                 # (verified: patch_embed out is
                                                 # exactly 8x14x14 = 1568)
                    B, Ntok, D = tok.shape
                    ts = cfg['tubelet_size']
                    n_t = cfg['num_frames'] // (ts[0] if isinstance(ts, (list, tuple)) else ts)
                    n_s = int(round((Ntok / n_t) ** 0.5))        # 14 for ViT-B/16
                    tok = tok.reshape(B, n_t, n_s, n_s, D).mean((2, 3))
                    F.append(tok.half().cpu().numpy())           # [B, 8, D]
                    for (act_, uniq_, errs_) in buf_meta:
                        meta.append({
                            'action_id': act_['action_id'],
                            'window': act_['window'],
                            'unique_of_16': uniq_,
                            'mean_err_s': round(float(np.mean(errs_)), 4),
                            'tubelet_centers_s': [round(
                                act_['requested_s'][2 * j + 0] * 0.5 +
                                act_['requested_s'][2 * j + 1] * 0.5, 3)
                                for j in range(8)]})
                        stats['mean_unique_frac'].append(uniq_ / 16)
                        stats['mean_err_s'].append(float(np.mean(errs_)))
                    buf_windows, buf_meta = [], []
                n_win_total += 1
                stats['windows_done'] += 1
                if args.limit_windows and n_win_total >= args.limit_windows:
                    break
            if F:
                np.savez_compressed(
                    op, feats=np.concatenate(F).astype(np.float16),
                    meta=json.dumps(meta), contract_hash=CONTRACT_HASH,
                    encoder=ENCODER_ID, video_sha256=hashlib.sha256(
                        p.read_bytes()).hexdigest()[:16])
            con.close()
            stats['sources_done'] += 1
            blog.write(json.dumps({'src': src, 'n_windows': len(meta),
                                   'ts': time.time()}) + '\n')
            blog.flush()
            if si % 5 == 0:
                el = time.time() - t_start
                wps = stats['windows_done'] / max(el, 1e-9)
                print(f'[{si}/{len(srcs)}] {src} win={stats["windows_done"]} '
                      f'{wps:.2f} win/s eta_h={((len(srcs)-si)*15)/max(wps,1e-9)/3600:.2f}',
                      flush=True)
            if args.limit_windows and n_win_total >= args.limit_windows:
                break
        except RuntimeError as e:
            if 'HARD_ERROR' in str(e):
                print(str(e), flush=True)
                blog.write(json.dumps({'src': src, 'status': 'HARD_ERROR_DUP'}) + '\n')
                blog.flush()
                if args.strict_dup:
                    break
            else:
                raise
    blog.close()
    el = time.time() - t_start
    out = {
        'contract_hash': CONTRACT_HASH, 'encoder': ENCODER_ID,
        'n_sources_done': stats['sources_done'],
        'n_windows': stats['windows_done'],
        'windows_per_sec': round(stats['windows_done'] / max(el, 1e-9), 3),
        'hard_errors': hard_errors,
        'mean_unique_frac': round(float(np.mean(stats['mean_unique_frac'])), 4)
            if stats['mean_unique_frac'] else None,
        'mean_err_s': round(float(np.mean(stats['mean_err_s'])), 4)
            if stats['mean_err_s'] else None,
        'wall_s': round(el, 1), 'limit_windows': args.limit_windows}
    args.out.write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
