"""V8: rotated-view feature extraction for RetargetVid keyframes.

90 deg clockwise: (640x360, ratio r) -> (360x640, ratio 1/r); a 1:3 vertical
window sliding on x becomes a 3:1 horizontal window sliding on y and vice
versa.  This supplies the vertical-source geometry of the official mobile
videos using real RV annotations (training augmentation only; evaluation
stays on unrotated frames).  PNGs already exist from the V7 cache run.
"""
import argparse, json, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--cards', default='2')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--frames-root', type=Path, default=Path('/data/aic/experiments/QWEN_RV200_KEYFRAMES/d1'))
ap.add_argument('--output-root', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/rv_feats_rot'))
args = ap.parse_args()

import os  # noqa: E402
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

vids = sorted({p.name.split('_')[0] for p in args.frames_root.glob('*') if p.is_dir()})
mine = [v for i, v in enumerate(vids) if i % args.nshards == args.shard]
print(f'DEBUG shard {args.shard}/{args.nshards}: {len(mine)} vids of {len(vids)}', flush=True)

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1, max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation='eager').eval().to('npu')


def vision(im: Image.Image):
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                         {'type': 'text', 'text': 'describe'}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors='pt')
    x = {k: v.to('npu') for k, v in x.items() if isinstance(v, torch.Tensor)}
    x['pixel_values'] = x['pixel_values'].to(torch.float16)
    with torch.no_grad():
        out = model.model.vision_tower(pixel_values=x['pixel_values'], spatial_shapes=x['spatial_shapes'],
                                       pixel_attention_mask=x['pixel_attention_mask'], return_dict=True)
        valid = int(x['pixel_attention_mask'][0].sum())
        fh, fw = [int(v) for v in x['spatial_shapes'][0]]
        grid = out.last_hidden_state[0, :valid].reshape(fh, fw, -1)[:, : (fw // 2) * 2, :]
        grid = grid[: (fh // 2) * 2]
        pooled = grid.float().mean((0, 1))
    return grid.cpu().numpy().astype(np.float16), pooled.cpu().numpy().astype(np.float16)


t0, n = time.time(), 0
for vi, vid in enumerate(mine):
    pngs = sorted(int(p.stem) for p in (args.frames_root / vid).glob('*.png'))
    need = [k for k in pngs if not (args.output_root / vid / f'{k}.npz').exists()]
    if not need:
        continue
    (args.output_root / vid).mkdir(parents=True, exist_ok=True)
    for kf in need:
        im = Image.open(args.frames_root / vid / f'{kf}.png').convert('RGB')
        g, p = vision(im.transpose(Image.ROTATE_270))
        np.savez_compressed(args.output_root / vid / f'{kf}.npz',
                            grid=g, pooled=p, shape=np.array([g.shape[0], g.shape[1]], dtype=np.int32))
        n += 1
    if vi % 10 == 0:
        print(f'DEBUG vid {vi}/{len(mine)} fw={n} {time.time() - t0:.0f}s', flush=True)

summ = {'shard': args.shard, 'nshards': args.nshards, 'vids': len(mine), 'frames_written': n,
        'wall_s': round(time.time() - t0, 1), 'rot': 'ROTATE_270 (90 cw)'}
(args.output_root / f'extract_s{args.shard}_summary.json').write_text(json.dumps(summ, indent=1) + '\n')
print('SUMMARY', json.dumps(summ), flush=True)
