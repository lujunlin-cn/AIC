"""R6 S-REREAD step 4: grid features for the dev80 pool.

The B3 candidate-utility head needs Siglip2 grid features (window/outside
pooled).  The dev80 pool (80 never-used sources, frozen in
r6_s_pool_freeze.json) has no grid cache yet - this extracts it with the
EXACT v8_live_extract protocol (spatial_crop_v1 release decode, packed
frame counters, long side 640 BILINEAR, fp16 grid with the fw/fh even
crop, deployment processor contract, Siglip2 NPU patch applied).

Run: 6 shards on physical cards 2-7 (one process per card).
Output: live_feats_dev80/<vid>/<frame>.npz (grid, pooled, shape)
"""
import argparse, json, os, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '16')
os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
from pathlib import Path
import sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--freeze', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/r6_s_pool_freeze.json'))
ap.add_argument('--t5-train-index', type=Path,
                default=Path('/data/aic/experiments/T5_CROPHEAD_V3/live_train_index.jsonl'),
                help='dev80 sources are NOT in the release manifests; media_path '
                     'and W,H come from the T5 train index (video_path field)')
ap.add_argument('--release', default='/data/aic/external_datasets/_releases/spatial_crop_v1')
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--long-side', type=int, default=640)
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=6)
ap.add_argument('--output-root', type=Path,
                default=Path('/data/aic/experiments_910a/LFM_V11/live_feats_dev80'))
args = ap.parse_args()

import torch
import torch_npu                                            # noqa: F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
import av                                                   # noqa: E402
from PIL import Image
sys.path.insert(0, '/root/AIC')
sys.path.insert(0, '/root/AIC/ext_data')
from v11_npu_siglip2_patch import patch_siglip2_npu         # noqa: E402
from aicext.release import Release                          # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration

freeze = json.loads(args.freeze.read_text())
import csv
vids = sorted({r['vid'] for r in csv.DictReader(
    args.freeze.with_name(args.freeze.stem + '_rows.csv').open())
    if r['pool'] == 'dev80'})
mine = [v for i, v in enumerate(vids) if i % args.nshards == args.shard]
print(f'shard {args.shard}/{args.nshards}: {len(mine)} vids', flush=True)

rel = Release('spatial_crop_v1', path=Path(args.release))
t5meta = {}
for l in args.t5_train_index.read_text().splitlines():
    r = json.loads(l)
    t5meta[r['video_id']] = (r['video_path'], float(r['width']), float(r['height']))

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1,
                                     max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True,
    attn_implementation='eager').eval().to('npu')
assert patch_siglip2_npu(model)


def vision(im):
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                         {'type': 'text', 'text': 'describe'}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors='pt')
    x = {k: v.to('npu') for k, v in x.items() if isinstance(v, torch.Tensor)}
    x['pixel_values'] = x['pixel_values'].to(torch.float16)
    with torch.no_grad():
        out = model.model.vision_tower(pixel_values=x['pixel_values'],
                                       spatial_shapes=x['spatial_shapes'],
                                       pixel_attention_mask=x['pixel_attention_mask'],
                                       return_dict=True)
        valid = int(x['pixel_attention_mask'][0].sum())
        fh, fw = [int(v) for v in x['spatial_shapes'][0]]
        grid = out.last_hidden_state[0, :valid].reshape(fh, fw, -1)[:, :(fw // 2) * 2, :]
        grid = grid[:(fh // 2) * 2]
        pooled = grid.float().mean((0, 1))
    return grid.cpu().numpy().astype(np.float16), pooled.cpu().numpy().astype(np.float16)


t0, n = time.time(), 0
for v in mine:
    od = args.output_root / v
    pk = np.load(Path(args.release) / 'packed' / 'LIVE_YT_VC' / f'{v}.npz')
    want = {int(f) for f in pk['frame_idx'].astype(int)
            if not (od / f'{int(f)}.npz').exists()}
    if not want:
        continue
    od.mkdir(parents=True, exist_ok=True)
    mp, W, H = t5meta[v]
    s = args.long_side / max(W, H)
    with av.open(mp) as cont:
        stream = cont.streams.video[0]
        fi = 0
        for frame in cont.decode(stream):
            if fi in want:
                im = frame.to_image().convert('RGB')
                im = im.resize((max(1, round(im.width * s)),
                                max(1, round(im.height * s))), Image.BILINEAR)
                g, p = vision(im)
                np.savez_compressed(od / f'{fi}.npz', grid=g, pooled=p,
                                    shape=np.array([g.shape[0], g.shape[1]], np.int32))
                n += 1
            fi += 1
            if fi > max(want):
                break
print(f'SUMMARY shard {args.shard}: {n} frames {time.time() - t0:.0f}s', flush=True)
