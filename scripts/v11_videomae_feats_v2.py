"""VideoMAEv2 temporal features for the 426-video semifinal drop.

Each video = ONE 16-frame window, temporally interpolated over its 1 fps
keyframes (short clips repeat frames).  The VisionTransformer (with the
Conv2d bridge) emits patch tokens [1, 8*14*14, 768]; mean per tubelet time
slot gives [8, 768] temporal features + slot timestamps.

Output per video: <vid>.npz {feat [8,768], t [8]} under --out.
NPU.  426 forwards total - minutes per card.
"""
import argparse, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--kf-root', type=Path, default=Path('/data/aic/semifinal_20261001/keyframes/keyframes'))
ap.add_argument('--weights', default='/data/aic/pretrained/videomaev2')
ap.add_argument('--out', type=Path, default=Path('/data/aic/semifinal_20261001/videomae_feats'))
ap.add_argument('--cards', default='4')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--bridge-mod', default='/data/aic/experiments_910a/LFM_V11/v11_m01_conv3d_bridge.py')
args = ap.parse_args()

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import numpy as np
import torch
import torch_npu
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
assert torch.npu.device_count() > 0, 'no NPU'
from PIL import Image
from safetensors.torch import load_file

import sys
sys.path.insert(0, str(Path(args.weights).parent))
from videomaev2.modeling_videomaev2 import VisionTransformer
sys.path.insert(0, str(Path(args.bridge_mod).parent))
mod_name = Path(args.bridge_mod).stem
import importlib.util as _ilu
spec = _ilu.spec_from_file_location('v11_bridge', args.bridge_mod)
bridge_mod = _ilu.module_from_spec(spec)
spec.loader.exec_module(bridge_mod)
spec.loader.exec_module(bridge_mod)

cfg = json.load(open(Path(args.weights) / 'config.json'))['model_config']
model = VisionTransformer(**cfg)
sd = load_file(Path(args.weights) / 'model.safetensors')
sd = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in sd.items()}
model.load_state_dict(sd)
bridge_mod.patch_videomae_conv3d(model)
model = model.float().eval().to('npu')

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def load_video_window(vd):
    pngs = sorted(vd.glob('*.png'))
    if len(pngs) < 4:
        return None, None
    T = len(pngs)
    idx = np.round(np.linspace(0, T - 1, 16)).astype(int).clip(0, T - 1)
    imgs = []
    for i in idx:
        im = Image.open(pngs[i]).convert('RGB').resize((224, 224))
        a = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
        imgs.append(a)
    return np.stack(imgs), None


vids = sorted((d for d in args.kf_root.iterdir() if d.is_dir()), key=lambda p: int(p.name))
mine = [v for i, v in enumerate(vids) if i % args.nshards == args.shard]
print(f'VMFEATS shard {args.shard}/{args.nshards}: {len(mine)} videos', flush=True)
t0, n = time.time(), 0
args.out.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    for vi, vd in enumerate(mine):
        od = args.out / f'{vd.name}.npz'
        if od.exists():
            continue
        win, _ = load_video_window(vd)
        if win is None:
            continue
        x = torch.from_numpy(win)[None].float().permute(0, 4, 1, 2, 3).contiguous().to('npu')  # NHWC -> N C T H W
        with torch.autocast('npu', dtype=torch.float16):
            h = model.patch_embed(x)
            h = h + model.pos_embed.expand(1, -1, -1).type_as(h).to(h.device).clone().detach()
            h = model.pos_drop(h)
            for blk in model.blocks:
                h = blk(h)
        toks = h.float()  # [1, L, 768], L=8*14*14, time-major
        L = toks.shape[1]
        slots = toks.view(1, 8, L // 8, 768).mean(2)[0].cpu().numpy()  # [8,768]
        pngs = sorted(vd.glob('*.png'))
        T = len(pngs)
        slot_t = ((np.arange(8) + 0.5) * T / 8).astype(int)
        np.savez(od, feat=slots.astype(np.float32),
                 t=slot_t.astype(np.float32),
                 n_keyframes=T)
        n += 1
        if n % 40 == 0:
            print(f'VMFEATS {n} videos {time.time() - t0:.0f}s', flush=True)
print('VMFEATS_SUMMARY', json.dumps({'shard': args.shard, 'videos': n,
                                     'wall_s': round(time.time() - t0, 1)}), flush=True)
