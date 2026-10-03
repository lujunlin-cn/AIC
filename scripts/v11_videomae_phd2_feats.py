"""VideoMAEv2 features for PHD2 fragments, two input conditions per fragment.

ord: one 16-frame window, linspace-resampled over the fragment keyframes.
rep: the middle keyframe repeated 16 times (static control, zero motion).

Both windows share the same static content; the difference isolates the
motion contribution for the three-input control experiment.  Output per
fragment: {ord [8,768], rep [8,768], n_frames}.
"""
import argparse, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--frag-root', type=Path,
                default=Path('/data/aic/experiments_910a/PHD2_FRAG_V1/frames'))
ap.add_argument('--weights', default='/data/aic/pretrained/videomaev2')
ap.add_argument('--bridge-mod',
                default='/data/aic/experiments_910a/LFM_V11/v11_m01_conv3d_bridge.py')
ap.add_argument('--out', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/phd2_vmae_feats'))
ap.add_argument('--cards', default='0')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
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
import importlib.util as _ilu
spec = _ilu.spec_from_file_location('v11_bridge', args.bridge_mod)
bridge_mod = _ilu.module_from_spec(spec)
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


def encode(pngs, idx):
    imgs = []
    for i in idx:
        im = Image.open(pngs[i]).convert('RGB').resize((224, 224))
        a = (np.asarray(im, np.float32) / 255.0 - MEAN) / STD
        imgs.append(a)
    x = torch.from_numpy(np.stack(imgs))[None].float().permute(0, 4, 1, 2, 3).contiguous().to('npu')
    with torch.autocast('npu', dtype=torch.float16):
        h = model.patch_embed(x)
        h = h + model.pos_embed.expand(1, -1, -1).type_as(h).to(h.device).clone().detach()
        h = model.pos_drop(h)
        for blk in model.blocks:
            h = blk(h)
    L = h.shape[1]
    return h.float().view(1, 8, L // 8, 768).mean(2)[0].cpu().numpy().astype(np.float32)


frags = sorted((d for d in args.frag_root.iterdir() if d.is_dir()), key=lambda p: p.name)
mine = [v for i, v in enumerate(frags) if i % args.nshards == args.shard]
print(f'PHDMAE shard {args.shard}/{args.nshards}: {len(mine)} fragments', flush=True)
t0, n = time.time(), 0
args.out.mkdir(parents=True, exist_ok=True)
with torch.no_grad():
    for vi, fd in enumerate(mine):
        od = args.out / f'{fd.name}.npz'
        if od.exists():
            continue
        jpgs = sorted(fd.glob('*.jpg'), key=lambda p: float(p.stem))
        if len(jpgs) < 4:
            continue
        T = len(jpgs)
        i_ord = np.round(np.linspace(0, T - 1, 16)).astype(int).clip(0, T - 1)
        i_rep = np.full(16, T // 2, dtype=int)
        np.savez(od, ord=encode(jpgs, i_ord), rep=encode(jpgs, i_rep), n_frames=T)
        n += 1
        if n % 200 == 0:
            print(f'PHDMAE {n} frags {time.time() - t0:.0f}s', flush=True)
print('PHDMAE_SUMMARY', json.dumps({'shard': args.shard, 'frags': n,
                                    'wall_s': round(time.time() - t0, 1)}), flush=True)
