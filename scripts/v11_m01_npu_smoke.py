"""M01 smoke: VideoMAEv2 ViT-B forward+backward on torch_npu (Ascend 910B).

Loads the VisionTransformer directly (config dict + safetensors), bypassing
transformers' meta-device init which breaks on the custom modeling's
linspace().item().  This is also the deployment-true path.

Pass criteria (V10 review 3.5): eager forward, non-constant scalar loss
backward, finite non-zero grads, one update, save/reload output identity.
"""
import argparse, json, os, sys, time
from pathlib import Path
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(_v, '8')

ap = argparse.ArgumentParser()
ap.add_argument('--weights', default='/data/aic/pretrained/videomaev2')
ap.add_argument('--cards', default='2')
ap.add_argument('--batch', type=int, default=2)
ap.add_argument('--frames', type=int, default=16)
ap.add_argument('--size', type=int, default=224)
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/m01_videomaev2_smoke.json'))
args = ap.parse_args()

import numpy as np  # noqa: E402
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
assert torch.npu.device_count() > 0, 'no NPU'
from safetensors.torch import load_file  # noqa: E402

DEV = 'npu'
log = {'weights': args.weights, 'torch': torch.__version__,
       'input': [args.batch, 3, args.frames, args.size, args.size],
       'load': 'direct VisionTransformer + safetensors'}

sys.path.insert(0, str(Path(args.weights).parent))
from videomaev2.modeling_videomaev2 import VisionTransformer  # noqa: E402
sys.path.insert(0, '/data/aic/experiments_910a/LFM_V11')
from v11_m01_conv3d_bridge import patch_videomae_conv3d  # noqa: E402

cfg = json.load(open(Path(args.weights) / 'config.json'))['model_config']
log['model_config'] = cfg
model = VisionTransformer(**cfg)
sd = load_file(Path(args.weights) / 'model.safetensors')
sd_clean = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd.items()}
missing, unexpected = model.load_state_dict(sd_clean, strict=False)
log['missing_keys'] = list(missing)[:8]
log['unexpected_keys'] = list(unexpected)[:8]
patch_videomae_conv3d(model)
log['conv3d_bridge'] = 'patched (aclnnConvolutionBackward unavailable for tubelet Conv3d)'
log['n_params_m'] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
model = model.float().to(DEV)
model.train()

torch.manual_seed(20261005)
x = torch.randn(args.batch, 3, args.frames, args.size, args.size, device=DEV)

t0 = time.time()
with torch.autocast('npu', dtype=torch.float16):
    y = model(x)
torch.npu.synchronize()
log['forward_s'] = round(time.time() - t0, 2)
log['output_shape'] = list(y.shape)
log['output_dtype'] = str(y.dtype)
log['output_finite'] = bool(torch.isfinite(y.float()).all().item())

proj = torch.randn(y.shape[1], device=DEV)
loss = ((y.float() * proj).sum() / y.numel()).float()
t0 = time.time()
loss.backward()
torch.npu.synchronize()
log['backward_s'] = round(time.time() - t0, 2)

gnorms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
log['n_grads'] = len(gnorms)
log['grad_finite_nonzero'] = bool(all(np.isfinite(gnorms)) and max(gnorms) > 0)
log['grad_norm_mean'] = round(float(sum(gnorms) / max(1, len(gnorms))), 8)

opt = torch.optim.SGD(model.parameters(), lr=1e-5)
opt.step()
log['update'] = 'ok'

model.eval()
with torch.no_grad():
    y1 = model(x[:1])
sd_path = str(args.out).replace('.json', '_state.pt')
torch.save(model.state_dict(), sd_path)
model2 = VisionTransformer(**cfg)
patch_videomae_conv3d(model2)
model2.load_state_dict(torch.load(sd_path, map_location='cpu'))
model2 = model2.float().to(DEV).eval()
with torch.no_grad():
    y2 = model2(x[:1])
log['reload_max_abs_diff'] = round(float((y1 - y2).abs().max().item()), 8)

log['peak_mem_gb'] = round(torch.npu.max_memory_allocated() / 2**30, 2)
log['pass'] = bool(log.get('output_finite') and log.get('grad_finite_nonzero')
                   and log.get('reload_max_abs_diff', 1.0) < 1e-4)
print(json.dumps(log, indent=1))
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(log, indent=1))
