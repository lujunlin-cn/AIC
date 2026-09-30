"""V8: extract LFM2.5-VL vision grids for LIVE-YT-VC annotated frames.

- Reads only the frozen release (media_path + packed frame_idx + ltrb_raw);
  writes nothing back into ext_data.
- Frames are resized to long side 640 BEFORE the vision tower (same as the
  official/semifinal keyframe pipeline), so train/deploy preprocessing match.
- One npz per (video, frame): grid (fh,fw,768) fp16 + pooled.
  (No rotation here: LIVE's 9:16 target rotated onto a vertical source has
  span 0 - no crop freedom.  Vertical-source augmentation is done on RV
  instead, see v8_rv_rot_extract.py.)
"""
import argparse, json, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--cards', default='2')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--release', default='/data/aic/external_datasets/_releases/spatial_crop_v1')
ap.add_argument('--splits', nargs='*', default=['train', 'dev', 'confirmation'])
ap.add_argument('--output-root', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/live_feats'))
ap.add_argument('--long-side', type=int, default=640)
args = ap.parse_args()

import os  # noqa: E402
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
import numpy as np  # noqa: E402
import av  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, '/root/AIC/ext_data')
from aicext.release import Release  # noqa: E402

units = []
rel = Release('spatial_crop_v1', path=Path(args.release))
for sp in args.splits:
    for m in rel.manifest(sp):
        units.append({'split': sp, 'vid': m['unit_id'].split(':', 1)[1], 'media': m['media_path'],
                      'W': float(m['width']), 'H': float(m['height']),
                      'ratio_key': m['ratio_key'], 'ratio_wh': m['target_ratio_wh']})
units.sort(key=lambda u: u['vid'])
mine = [u for i, u in enumerate(units) if i % args.nshards == args.shard]
print(f'DEBUG shard {args.shard}/{args.nshards}: {len(mine)} units of {len(units)}', flush=True)

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
for vi, u in enumerate(mine):
    od = args.output_root / u['vid']
    pk = np.load(Path(args.release) / 'packed' / 'LIVE_YT_VC' / f"{u['vid']}.npz")
    fidx = pk['frame_idx'].astype(int)
    need = [int(f) for f in fidx if not (od / f'{int(f)}.npz').exists()]
    if not need:
        continue
    od.mkdir(parents=True, exist_ok=True)
    want = set(need)
    s = args.long_side / max(u['W'], u['H'])
    with av.open(u['media']) as cont:
        stream = cont.streams.video[0]
        fi = 0
        for frame in cont.decode(stream):
            if fi in want:
                im = frame.to_image().convert('RGB')
                im640 = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
                g, p = vision(im640)
                np.savez_compressed(args.output_root / u['vid'] / f'{fi}.npz',
                                    grid=g, pooled=p, shape=np.array([g.shape[0], g.shape[1]], dtype=np.int32))
                n += 1
            fi += 1
            if fi > max(want):
                break
    if vi % 20 == 0:
        print(f'DEBUG unit {vi}/{len(mine)} fw={n} {time.time() - t0:.0f}s', flush=True)

summ = {'shard': args.shard, 'nshards': args.nshards, 'units': len(mine), 'frames_written': n,
        'wall_s': round(time.time() - t0, 1), 'long_side': args.long_side,
        'processor': 'min_tiles=1 max_tiles=1 max_image_tokens=256'}
(args.output_root / f'extract_s{args.shard}_summary.json').write_text(json.dumps(summ, indent=1) + '\n')
print('SUMMARY', json.dumps(summ), flush=True)
