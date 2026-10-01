"""LFM2.5-VL vision-tower features for the PHD2 fragment pool (sharded, NPU).

Same frozen path as the V7/V8 student head (single-tile NaFlex grid, eager
attention, fp16, jit_compile=False, vision tower only - no text prompt, no
generate), so the PHD2 rows are directly comparable with the RetargetVid /
LIVE rows the head was built on.

Stores the full 2D token grid, not just the pooled vector: the pooled vector is
what the V7/V8 head consumes, but the grid is what a heat-map head (V9 P1-5)
needs, and re-extracting 48k frames later would cost another NPU hour.

Runs on the two aic-batch cards the teacher job does not use (the teacher takes
logical 2,3,4,5 = physical 4,5,6,7; logical 0,1 = physical 2,3 are free).
"""
import argparse, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--pool', type=Path, required=True)
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--cards', default='0,1')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
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

rows = [json.loads(l) for l in (args.pool / 'index.jsonl').read_text().splitlines() if l.strip()]
mine = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
print(f'FEAT shard {args.shard}/{args.nshards}: {len(mine)} fragments', flush=True)

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1, max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation='eager'
).eval().to('npu')


def vision(im: Image.Image):
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
        grid = out.last_hidden_state[0, :valid].reshape(fh, fw, -1)[:, : (fw // 2) * 2, :]
        grid = grid[: (fh // 2) * 2]
        pooled = grid.float().mean((0, 1))
    return grid.cpu().numpy().astype(np.float16), pooled.cpu().numpy().astype(np.float16)


t0, n = time.time(), 0
for vi, r in enumerate(mine):
    vid = r['video_id']
    skel = json.loads((args.pool / 'skel' / f'{vid}.json').read_text())
    od = args.output_root / vid
    need = [k for k in skel['keyframes'] if not (od / f'{k}.npz').exists()]
    if not need:
        continue
    od.mkdir(parents=True, exist_ok=True)
    for kf in need:
        im = Image.open(args.pool / 'frames' / vid / f'{kf}.{args.ext}').convert('RGB')
        g, p = vision(im)
        np.savez(od / f'{kf}.npz', grid=g, pooled=p,
                 shape=np.array([g.shape[0], g.shape[1]], dtype=np.int32))
        n += 1
    if vi % 100 == 0:
        el = time.time() - t0
        print(f'FEAT {vi}/{len(mine)} written={n} {el:.0f}s '
              f'eta={el / max(n, 1) * (len(mine) - vi) * 8 / 8:.0f}s', flush=True)

summ = {'shard': args.shard, 'nshards': args.nshards, 'fragments': len(mine),
        'frames_written': n, 'wall_s': round(time.time() - t0, 1)}
(args.output_root / f'extract_s{args.shard}_summary.json').write_text(json.dumps(summ, indent=1) + '\n')
print('SUMMARY', json.dumps(summ), flush=True)