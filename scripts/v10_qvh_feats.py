"""Pooled vision-tower features for QVHighlights on the 1 Hz grid (NPU, sharded).

Same frozen inference path as the V7/V8/V9 student head - single-tile NaFlex
grid, eager attention, fp16, jit_compile=False, vision tower only - so the
pooled vectors are directly comparable with the pools the head was trained on.

QVHighlights needs pooled-only storage: ~1.06 M frames x (grid 3.4 MB) would be
3.6 TB, which is not writable.  The pooled 768-d vector is what the temporal
head consumes, so each video collapses to one npz of shape [n_frames, 768]
(~115 KB).  The spatial grid is NOT stored here - QV enters the pipeline as a
temporal-domain signal, and any heat-map head would need a separate pass.

Reads <out>/frames/<vid>/<t>.jpg produced by v10_qvh_extract.py and writes
<out>/feats/<vid>.npz {pooled [n,768] float16, t float32[n]}.
"""
import argparse, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--pool', type=Path, required=True,
                help='dir with frames/<vid>/*.jpg and index.jsonl')
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--cards', default='0,1')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--limit', type=int, default=0)
ap.add_argument('--ext', default='jpg')
ap.add_argument('--output-root', type=Path, required=True)
args = ap.parse_args()

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
assert torch.npu.device_count() > 0, 'no NPU visible - refusing to fall back to CPU'
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

vids = sorted(d.name for d in (args.pool / 'frames').iterdir() if d.is_dir())
mine = [v for i, v in enumerate(vids) if i % args.nshards == args.shard]
if args.limit:
    mine = mine[:args.limit]
print(f'QV_FEAT shard {args.shard}/{args.nshards}: {len(mine)} videos', flush=True)

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1,
                                     max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True,
    attn_implementation='eager').eval().to('npu')


def pooled_of(im):
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
        grid = out.last_hidden_state[0, :valid].reshape(fh, fw, -1)
    return grid.float().mean((0, 1)).cpu().numpy().astype(np.float16)


args.output_root.mkdir(parents=True, exist_ok=True)
t0, n_fr, n_v = time.time(), 0, 0
for vid in mine:
    op = args.output_root / f'{vid}.npz'
    if op.exists():
        continue
    jpgs = sorted((args.pool / 'frames' / vid).glob(f'*.{args.ext}'),
                  key=lambda p: float(p.stem))
    if not jpgs:
        continue
    feats = np.stack([pooled_of(Image.open(p).convert('RGB')) for p in jpgs])
    ts = np.array([float(p.stem) for p in jpgs], dtype=np.float32)
    np.savez(op, pooled=feats, t=ts)
    n_fr += len(jpgs)
    n_v += 1
    if n_v % 20 == 0:
        el = time.time() - t0
        fps = n_fr / el
        eta = (len(mine) - n_v) * (n_fr / n_v) / fps / 3600
        print(f'QV_FEAT {n_v}/{len(mine)} frames={n_fr} {fps:.1f}fps eta={eta:.1f}h', flush=True)

summ = {'shard': args.shard, 'videos': n_v, 'frames': n_fr,
        'wall_s': round(time.time() - t0, 1)}
print('SUMMARY ' + json.dumps(summ), flush=True)
