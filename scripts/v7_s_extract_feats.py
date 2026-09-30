"""V7 S-line: extract LFM2.5-VL vision-tower 2D token grids for all RetargetVid
keyframes (single-tile NaFlex, pre-projector grid) + pooled vector for the
temporal line.  One npz per keyframe; shard per NPU card.

The same cache serves: (a) frozen candidate-utility head training/eval,
(b) the T-line pooled features.  If the vision tower is later unfrozen this
cache is invalidated and must be re-extracted (prereg rule).
"""
import argparse, json, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--cards', default='2')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--output-root', type=Path, default=Path('/data/aic/experiments_910a/LFM_V7/feats'))
ap.add_argument('--frames-root', type=Path, default=Path('/data/aic/experiments/QWEN_RV200_KEYFRAMES/d1'))
ap.add_argument('--manifest', type=Path, default=Path('/data/aic/experiments/OBS_CACHE_RETARGET_ALL200'))
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

vids = sorted({p.name.split('_')[0] for p in args.manifest.glob('*_1-3.npz')})
mine = [v for i, v in enumerate(vids) if i % args.nshards == args.shard]
print(f'DEBUG shard {args.shard}/{args.nshards}: {len(mine)} vids of {len(vids)}', flush=True)

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1, max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation='eager').eval().to('npu')

t0, n, lat = time.time(), 0, []
for vi, vid in enumerate(mine):
    outd = args.output_root / vid
    kfs = sorted(int(p.stem) for p in (args.frames_root / vid).glob('*.png'))
    if outd.exists() and len(list(outd.glob('*.npz'))) == len(kfs):
        continue
    outd.mkdir(parents=True, exist_ok=True)
    for kf in kfs:
        outf = outd / f'{kf}.npz'
        if outf.exists():
            continue
        im = Image.open(args.frames_root / vid / f'{kf}.png').convert('RGB')
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                             {'type': 'text', 'text': 'describe'}]}]
        x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                     return_dict=True, return_tensors='pt')
        x = {k: v.to('npu') for k, v in x.items() if isinstance(v, torch.Tensor)}
        if 'pixel_values' in x:
            x['pixel_values'] = x['pixel_values'].to(torch.float16)
        torch.npu.synchronize()
        ts = time.time()
        with torch.no_grad():
            out = model.model.vision_tower(pixel_values=x['pixel_values'], spatial_shapes=x['spatial_shapes'],
                                     pixel_attention_mask=x['pixel_attention_mask'], return_dict=True)
            valid = int(x['pixel_attention_mask'][0].sum())
            fh, fw = [int(v) for v in x['spatial_shapes'][0]]
            grid = out.last_hidden_state[0, :valid].reshape(fh, fw, -1)[:, : (fw // 2) * 2, :]
            grid = grid[: (fh // 2) * 2]
            pooled = grid.float().mean((0, 1))
        torch.npu.synchronize()
        lat.append(time.time() - ts)
        np.savez_compressed(outf, grid=grid.cpu().numpy().astype(np.float16),
                            pooled=pooled.cpu().numpy().astype(np.float16),
                            shape=np.array([grid.shape[0], grid.shape[1]], dtype=np.int32))
        n += 1
    if vi % 10 == 0:
        print(f'DEBUG vid {vi}/{len(mine)} done, {n} frames, {time.time() - t0:.0f}s', flush=True)

summ = {'shard': args.shard, 'nshards': args.nshards, 'vids': len(mine), 'frames_written': n,
        'wall_s': round(time.time() - t0, 1), 'mean_s_per_frame': round(float(np.mean(lat)), 3) if lat else None,
        'grid_hw': [int(fh), int(fw)], 'feat_dim': int(grid.shape[-1]),
        'processor': 'min_tiles=1 max_tiles=1 max_image_tokens=256'}
(args.output_root / f'extract_s{args.shard}_summary.json').write_text(json.dumps(summ, indent=1) + '\n')
print('SUMMARY', json.dumps(summ), flush=True)
