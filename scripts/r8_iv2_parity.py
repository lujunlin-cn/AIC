"""R8: InternVideo2-1B CPU/NPU parity gate (preregistered IV2_1B_TEMPORAL step 1).

Recipe from probe_internvideo2_smoke.py (stage1 k710 weights): isolated import
of upstream vit_scale_clean, LayerScale gamma->weight rename, flash stub,
bicubic 224 ImageNet norm.  This probe runs the SAME 8-frame sample through
CPU fp32 and NPU fp16 and compares the fc_norm'd pooled features.

Gate (preregistered r8: parity before any NPU number): cosine >= 0.999 and
relative L2 <= 0.02 on the pooled feature.  Public QVHighlights train row
only - no official media.

Writes iv2_parity.json next to itself.  One card; ASCEND_RT_VISIBLE_DEVICES
selects it.
"""
import argparse, hashlib, json, sys, time, types
from pathlib import Path
import cv2
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument('--source-root', default='/data/aic/tmp/InternVideo2')
p.add_argument('--weights', default='/data/aic/pretrained/internvideo2_stage1_1b/1B_ft_k710_ft_k700_f8.pth')
p.add_argument('--manifest', default='/data/aic/experiments/QVH_DEITS_QUICK_5PCT/train_features.jsonl')
p.add_argument('--frames', type=int, default=8)
p.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r8_npu/iv2_parity.json')
a = p.parse_args()

src = Path(a.source_root) / 'llava-train_videochat/llava/model/multimodal_encoder/internvideo2'
pkg = types.ModuleType('aic_iv2_probe'); pkg.__path__ = [str(src)]; sys.modules[pkg.__name__] = pkg
stub = types.ModuleType('aic_iv2_probe.flash_attention_class')
class DisabledFlashAttention(torch.nn.Module):
    def __init__(self, *args, **kwargs): raise RuntimeError('flash disabled')
stub.FlashAttention = DisabledFlashAttention; sys.modules[stub.__name__] = stub
from aic_iv2_probe.vit_scale_clean import PretrainVisionTransformer_clean, interpolate_pos_embed_internvideo2

row = json.loads(Path(a.manifest).read_text().splitlines()[0])
assert row['split'] == 'train'
torch.set_num_threads(8)
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

def build():
    m = PretrainVisionTransformer_clean(**kw)
    interpolate_pos_embed_internvideo2(s, m, orig_t_size=8)
    missing, extra = m.load_state_dict(s, strict=False)
    assert missing == [], missing
    fc = torch.nn.LayerNorm(768, eps=1e-6)
    fc.load_state_dict({k.removeprefix('fc_norm.'): v for k, v in s.items()
                        if k.startswith('fc_norm.')})
    return m.eval(), fc.eval()

cap = cv2.VideoCapture(row['video_path']); fps = cap.get(cv2.CAP_PROP_FPS); fr = []
for i in range(a.frames):
    cap.set(cv2.CAP_PROP_POS_FRAMES, round(i * fps / 2)); ok, f = cap.read()
    if not ok: raise ValueError('decode failed')
    fr.append(cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2RGB), (224, 224),
                         interpolation=cv2.INTER_CUBIC))
cap.release()
x = torch.tensor(np.stack(fr), dtype=torch.float32).permute(3, 0, 1, 2).unsqueeze(0) / 255
mean = torch.tensor([.485, .456, .406]).view(1, 3, 1, 1, 1)
std = torch.tensor([.229, .224, .225]).view(1, 3, 1, 1, 1)
x = (x - mean) / std

out = {'protocol': 'R8 IV2_1B parity; stage1 k710; pooled fc_norm feature; gate cos>=0.999 relL2<=0.02',
       'video_path': row['video_path'], 'frames': a.frames,
       'weight_sha256_16': hashlib.sha256(Path(a.weights).read_bytes()).hexdigest()[:16]}

t0 = time.time()
mc, fcc = build()
with torch.inference_mode():
    y_cpu = fcc(mc(x)[1]).float()
out['cpu_s'] = round(time.time() - t0, 1)
out['cpu_feat_shape'] = list(y_cpu.shape)

import torch_npu  # noqa: E402
dev = 'npu'
t0 = time.time()
mn, fn = build()
mn = mn.half().to(dev); fn = fn.half().to(dev)
with torch.inference_mode():
    y_npu = fn(mn(x.half().to(dev))[1]).float(); torch.npu.synchronize()
out['npu_s'] = round(time.time() - t0, 1)
out['npu_devices_visible'] = torch.npu.device_count()

y_npu_c = y_npu.float().cpu()
cos = float(torch.nn.functional.cosine_similarity(y_cpu.flatten(), y_npu_c.flatten(), dim=0))
rel = float((y_cpu - y_npu_c).norm() / y_cpu.norm().clamp_min(1e-9))
out['cosine'] = round(cos, 7)
out['rel_l2'] = round(rel, 6)
out['finite'] = bool(torch.isfinite(y_npu_c).all())
out['gate_pass'] = bool(cos >= 0.999 and rel <= 0.02 and out['finite'])
out['hbm_peak_GB'] = round(torch.npu.max_memory_allocated() / 2**30, 2)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
Path(a.out).write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
