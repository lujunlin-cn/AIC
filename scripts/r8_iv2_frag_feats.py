"""R8: InternVideo2-1B features for the QVH frag pool (IV2_1B_TEMPORAL prep).

Input : <frag>/frames/<window>/<second>.jpg (8 frames per window, 1 Hz),
        as consumed by v10_qvh_frag_tcn.py; 9,000 train windows.
Output: <out>/p<pid>/<window>.npz  {pooled (8,768) fp16, t (8,) float32}
        - same schema as frag_feats_train so the temporal head pipeline is
          a drop-in swap of the feature ROOT.

Protocol (recorded, transplant decision): the pooled vector for frame i of
a window is the InternVideo2 stage-1 (k710) pooled feature of the CENTERED
8-frame window frames[clamp(i-3)..clamp(i+4)] (edge-clamped).  The 8
per-frame windows of one window directory are stacked into ONE batch-8
forward per directory.  This gives every frame 8 s of temporal context -
the property a single-frame vision tower cannot provide and the reason this
backbone swap is preregistered as a feature change.

Sharded by window index (i % nshards == shard); one process per card.
CPU/NPU parity for this encoder passed (r8_npu/iv2_parity.json, cosine
0.99999).  Public QVHighlights pool only - no official media.
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
p.add_argument('--frag', default='/data/aic/experiments_910a/QVH_V10/frag_train')
p.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r8_iv2/iv2_frag_feats_train')
p.add_argument('--shard', type=int, default=0)
p.add_argument('--nshards', type=int, default=1)
p.add_argument('--limit', type=int, default=0, help='debug: first N windows only')
p.add_argument('--frames', type=int, default=8)
a = p.parse_args()

src = Path(a.source_root) / 'llava-train_videochat/llava/model/multimodal_encoder/internvideo2'
pkg = types.ModuleType('aic_iv2_enc'); pkg.__path__ = [str(src)]; sys.modules[pkg.__name__] = pkg
stub = types.ModuleType('aic_iv2_enc.flash_attention_class')
class DF(torch.nn.Module):
    def __init__(self, *args, **k): raise RuntimeError('flash disabled')
stub.FlashAttention = DF; sys.modules[stub.__name__] = stub
from aic_iv2_enc.vit_scale_clean import PretrainVisionTransformer_clean, interpolate_pos_embed_internvideo2  # noqa: E402
import cv2  # noqa: E402

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
          layerscale_no_force_fp32=False, num_frames=a.frames, tubelet_size=1,
          sep_pos_embed=False, sep_image_video_pos_embed=False,
          use_checkpoint=False, checkpoint_num=0, x_vis_return_idx=-1, x_vis_only=False)
m = PretrainVisionTransformer_clean(**kw)
interpolate_pos_embed_internvideo2(s, m, orig_t_size=a.frames)
missing, extra = m.load_state_dict(s, strict=False)
assert missing == [], missing
fc = torch.nn.LayerNorm(768, eps=1e-6)
fc.load_state_dict({k.removeprefix('fc_norm.'): v for k, v in s.items()
                    if k.startswith('fc_norm.')})
m = m.eval().half().npu(); fc = fc.eval().half().npu()
print('IV2 loaded', flush=True)

FRAG = Path(a.frag)
wins = sorted(d.name for d in (FRAG / 'frames').iterdir() if d.is_dir())
mine = [w for i, w in enumerate(wins) if i % a.nshards == a.shard]
if a.limit:
    mine = mine[:a.limit]
OUTP = Path(a.out) / f'p{a.shard}'
OUTP.mkdir(parents=True, exist_ok=True)
print(f'shard {a.shard}/{a.nshards}: {len(mine)} windows', flush=True)


def read_win(win, idxs):
    """Return (3,T,224,224) float32 for the given frame indices (1 Hz jpgs).

    Normalisation follows the smoke protocol exactly: unsqueeze(0) first,
    then 5-D mean/std, then squeeze back - a 4-D x with a 5-D view silently
    broadcasts to dim-5 and broke the first pilot run.
    """
    fs = sorted((FRAG / 'frames' / win).glob('*.jpg'),
                key=lambda p: float(p.stem))
    n = len(fs)
    sel = [fs[min(max(i, 0), n - 1)] for i in idxs]
    imgs = [cv2.resize(cv2.cvtColor(cv2.imread(str(f)), cv2.COLOR_BGR2RGB),
                       (224, 224), interpolation=cv2.INTER_CUBIC)
            for f in sel]
    x = torch.tensor(np.stack(imgs), dtype=torch.float32).permute(3, 0, 1, 2) / 255
    x = (x.unsqueeze(0) - mean) / std          # (1,3,T,224,224)
    return x.squeeze(0)                        # (3,T,224,224)


def pooled(x):  # x (B,3,T,H,W) float32 -> (B,768) float32 cpu
    x = x.half().npu()
    with torch.inference_mode():
        y = fc(m(x)[1]).float()
    return y.cpu()


t0 = time.time(); done = 0
for w in mine:
    outp = OUTP / f'{w}.npz'
    if outp.exists():
        done += 1
        continue
    try:
        secs = sorted(float(f.stem) for f in (FRAG / 'frames' / w).glob('*.jpg'))
        n = len(secs)
        if n != a.frames:
            print(f'SKIP {w}: {n} frames', flush=True)
            continue
        batch = torch.stack([read_win(w, [i - 3 + k for k in range(a.frames)])
                             for i in range(n)])  # (8,3,8,224,224)
        with torch.inference_mode():
            y = fc(m(batch.half().npu())[1]).float().cpu()   # (8,768)
        np.savez(outp, pooled=y.numpy().astype(np.float16),
                 t=np.array(secs, np.float32))
        done += 1
        if done % 200 == 0:
            el = time.time() - t0
            print(f'{done}/{len(mine)} {done/el:.2f}/s eta {(len(mine)-done)/(done/el)/60:.0f}min',
                  flush=True)
    except Exception as e:
        print(f'ERR {w}: {e}', flush=True)
print(f'DONE shard {a.shard}: {done}/{len(mine)} in {(time.time()-t0)/60:.1f}min', flush=True)
