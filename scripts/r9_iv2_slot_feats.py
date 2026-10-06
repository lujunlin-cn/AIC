"""R9 main line: InternVideo2-1B Stage1 multi-layer features for the PHD2
slot pool (R9 report Q3 first batch, user-approved 2026-10-06).

Pool     = the FROZEN R9 audit manifest (mixed-only, 3,917 frags / 1,963
           sources): r9_audit/{train,eval}_public.jsonl (fragment_id +
           source_id only; labels are NOT read by this script).
Slots    = the 8 frozen sampling times copied verbatim from
           LFM_V10/pool_feats/<fid>.npz 't' (absolute seconds) - the time
           axis is NEVER re-derived from media.
Input    = per slot, the 8 integer-second context frames [t-3 .. t+4]
           (identical context protocol to the QVH IV2 extraction,
           r8_iv2_frag_feats.py), read from
           /data/aic/external_datasets/PHD2/raw/youtube/<src>.mp4.

Decoder  = PyAV 18 / libdav1d.  REASON: 685/1,963 manifest sources (34.9%)
           are AV1 (cv2 probe 2026-10-06) and the container cv2/ffmpeg
           cannot decode AV1 (silent grab failures + zero/padded frames);
           PyAV wheels bundle libdav1d and decode all three codecs present
           (h264 1254 / AV01 685 / VP90 24).  Frame index = round(t*fps)
           with fps MEASURED from the stream (index fps can be wrong:
           -3ixfTKGG8A is 23.976 vs index 25).  Missing frames are NEVER
           padded: they stay zero and are counted in the 'nmiss' key
           (quality gate: nmiss must be 0 for a clean run).
Readouts (one forward, hook on blocks[34] = layer 35/40 = "minus-5"):
  mean768   fc_norm(clip_projector(x_vis_last))   - the exact QVH-validated
            path (parity PASS, cosine 0.99999)
  m1408_l   mean(non-cls tokens, last layer)      - norm-free readout
  m1408_m5  mean(non-cls tokens, layer-35 output) - norm-free readout
  Keeping both readouts defers the layer-protocol decision to the (cheap)
  head-training stage; R9 Q3 evidence (InternVideo2 Table23: THUMOS14
  67.9 last -> 69.9 minus5) motivates the minus-5 arm.
t' = 8 (tubelet_size=1): NO temporal downsample - the R9 report's
  temporal-patch trap does not apply to this config.
Sharded by frag index (i % nshards == shard), one process per card;
incremental npz per frag (resume-safe).  No official media.
"""
import argparse, json, os, sys, time, types
from pathlib import Path
import numpy as np
import torch

for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(_v, '4')

p = argparse.ArgumentParser()
p.add_argument('--source-root', default='/data/aic/tmp/InternVideo2')
p.add_argument('--weights', default='/data/aic/pretrained/internvideo2_stage1_1b/1B_ft_k710_ft_k700_f8.pth')
p.add_argument('--media-root', default='/data/aic/external_datasets/PHD2/raw/youtube')
p.add_argument('--pool-feats', default='/data/aic/experiments_910a/LFM_V10/pool_feats')
p.add_argument('--manifest-dir', default='/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit')
p.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r9_iv2_slot')
p.add_argument('--shard', type=int, default=0)
p.add_argument('--nshards', type=int, default=1)
p.add_argument('--limit', type=int, default=0, help='debug: first N frags only')
p.add_argument('--dump-frames', type=int, default=0, help='debug: save slot context frames of first N frags')
a = p.parse_args()

src = Path(a.source_root) / 'llava-train_videochat/llava/model/multimodal_encoder/internvideo2'
pkg = types.ModuleType('aic_iv2_enc'); pkg.__path__ = [str(src)]; sys.modules[pkg.__name__] = pkg
stub = types.ModuleType('aic_iv2_enc.flash_attention_class')
class DF(torch.nn.Module):
    def __init__(self, *args, **k): raise RuntimeError('flash disabled')
stub.FlashAttention = DF; sys.modules[stub.__name__] = stub
from aic_iv2_enc.vit_scale_clean import PretrainVisionTransformer_clean, interpolate_pos_embed_internvideo2  # noqa: E402
import cv2  # noqa: E402  (resize/imwrite only - never used for decoding)
import av  # noqa: E402

os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', str(5))
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
assert torch.npu.device_count() > 0, 'no NPU visible - refusing CPU fallback'

mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1, 1)
std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1, 1)

s = torch.load(a.weights, map_location='cpu')['module']
s = {k.replace('.ls1.gamma', '.ls1.weight').replace('.ls2.gamma', '.ls2.weight'): v
     for k, v in s.items()}
kw = dict(in_chans=3, img_size=224, patch_size=14, embed_dim=1408, depth=40,
          num_heads=16, mlp_ratio=48/11, qkv_bias=False, drop_path_rate=0.0,
          init_values=1e-5, qk_normalization=True, use_flash_attn=False,
          use_fused_rmsnorm=False, use_fused_mlp=False, attn_pool_num_heads=16,
          layerscale_no_force_fp32=False, num_frames=8, tubelet_size=1,
          sep_pos_embed=False, sep_image_video_pos_embed=False,
          use_checkpoint=False, checkpoint_num=0, x_vis_return_idx=-1, x_vis_only=False)
m = PretrainVisionTransformer_clean(**kw)
interpolate_pos_embed_internvideo2(s, m, orig_t_size=8)
missing, extra = m.load_state_dict(s, strict=False)
assert missing == [], missing
fc = torch.nn.LayerNorm(768, eps=1e-6)
fc.load_state_dict({k.removeprefix('fc_norm.'): v for k, v in s.items()
                    if k.startswith('fc_norm.')})

# minus-5 hook: capture the post-block residual stream of layer 35/40.
# Block forwards may return (x, residual) (fused ops); the main loop would
# add them before the next block, so the hook does the same.
M5 = {}
def _hook(_mod, _inp, out):
    x, res = out if isinstance(out, tuple) and len(out) == 2 else (out, None)
    if res is not None:
        x = x + res
    M5['tok'] = x
m.blocks[34].register_forward_hook(_hook)

m = m.eval().half().npu(); fc = fc.eval().half().npu()
print('IV2 Stage1 loaded (depth 40, tubelet 1, hook blocks[34])', flush=True)

# ---- frozen manifest ----
man = []
for split in ('train_public', 'eval_public'):
    for l in (Path(a.manifest_dir) / f'{split}.jsonl').read_text().splitlines():
        if l.strip():
            man.append(json.loads(l))
fids = [r['fragment_id'] for r in man]
srcs = {r['fragment_id']: r['source_id'] for r in man}
mine = [f for i, f in enumerate(fids) if i % a.nshards == a.shard]
if a.limit:
    mine = mine[:a.limit]
OUTP = Path(a.out) / f'p{a.shard}'
OUTP.mkdir(parents=True, exist_ok=True)
print(f'shard {a.shard}/{a.nshards}: {len(mine)} frags', flush=True)

MEDIA = Path(a.media_root)
POOL = Path(a.pool_feats)


def read_ctx(vid_path, t8, dump_dir=None):
    """(8 slots, 3, 8, 224, 224) float32 in 0..1, plus missing-frame count.

    PyAV path: seek to 2 s before the first needed frame, sequential decode
    to the last one.  Frame index = round(pts * time_base * measured fps)
    (CFR assumption; PHD2 yt rips are CFR).  Missing frames stay zero and
    are counted - never padded with a neighbour frame.
    """
    need = {}
    c = av.open(str(vid_path))
    st = c.streams.video[0]
    fps = float(st.average_rate)
    for si, t in enumerate(t8.tolist()):
        for oi in range(-3, 5):
            fr = max(int(round((t + oi) * fps)), 0)
            need.setdefault(fr, []).append((si, oi))
    minfr, maxfr = min(need), max(need)
    c.seek(max(int((minfr - 2) / fps * av.time_base), 0), backward=True)
    imgs = {}
    for fr in c.decode(video=0):
        idx = int(round(fr.pts * fr.time_base * fps))
        if idx in need:
            imgs[idx] = cv2.resize(fr.to_ndarray(format='rgb24'), (224, 224),
                                   interpolation=cv2.INTER_CUBIC)
        if idx >= maxfr:
            break
    c.close()
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)
        for si in (0, 4):
            t = t8[si]
            fr = max(int(round((t - 3) * fps)), 0)
            if fr in imgs:
                cv2.imwrite(str(dump_dir / f'{vid_path.stem}_s{si}_t{t:.2f}_f{fr}.jpg'),
                            cv2.cvtColor(imgs[fr], cv2.COLOR_RGB2BGR))
    nmiss = len(need) - len(imgs)
    slots = np.zeros((8, 3, 8, 224, 224), np.float32)
    for si, t in enumerate(t8.tolist()):
        for k, oi in enumerate(range(-3, 5)):
            fr = max(int(round((t + oi) * fps)), 0)
            if fr in imgs:
                slots[si, :, k] = torch.from_numpy(imgs[fr]).permute(2, 0, 1).numpy() / 255
    x = torch.from_numpy(slots)          # (8,3,8,224,224) in 0..1
    x = (x.unsqueeze(1) - mean) / std    # 5-D mean/std like QVH read_win
    return x.squeeze(1), fps, nmiss


t0 = time.time(); done = 0; errs = 0; miss_tot = 0
for w in mine:
    outp = OUTP / f'{w}.npz'
    if outp.exists():
        done += 1
        continue
    try:
        t8 = np.load(POOL / f'{w}.npz')['t'].astype(np.float64)
        x, mfps, nmiss = read_ctx(MEDIA / f'{srcs[w]}.mp4', t8,
                                  dump_dir=OUTP / '_frames' if a.dump_frames > done else None)
        with torch.inference_mode():
            xvis, pooled768, _, _ = m(x.half().npu())
            pool_fc = fc(pooled768).float()                      # (8,768)
            last = xvis[:, 1:].mean(1).float()                   # (8,1408)
            m5 = M5['tok'][:, 1:].mean(1).float()                # (8,1408)
        assert xvis.shape[1] == 1 + 8 * 256, f"T' trap: tokens {xvis.shape}"
        np.savez(outp,
                 mean768=pool_fc.cpu().numpy().astype(np.float16),
                 m1408_l=last.cpu().numpy().astype(np.float16),
                 m1408_m5=m5.cpu().numpy().astype(np.float16),
                 t=t8.astype(np.float32), fps=np.float32(mfps),
                 nmiss=np.int32(nmiss))
        miss_tot += nmiss
        done += 1
        if done % 200 == 0:
            el = time.time() - t0
            print(f'{done}/{len(mine)} {done/el:.2f}/s eta {(len(mine)-done)/(done/el)/60:.0f}min miss {miss_tot}',
                  flush=True)
    except Exception as e:
        errs += 1
        print(f'ERR {w}: {e}', flush=True)
print(f'DONE shard {a.shard}: {done}/{len(mine)} errs {errs} missed_frames {miss_tot} '
      f'in {(time.time()-t0)/60:.1f}min', flush=True)
