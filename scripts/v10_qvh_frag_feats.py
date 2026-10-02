"""Pooled vision-tower features for QV fragments (NPU, sharded).

A QV fragment is an 8-frame 1 s grid inside a 150 s clip.  This pools each of
the 8 frames through the frozen vision tower (same path as every student head)
and stores one npz per fragment: pooled [8,768] plus the frame timestamps so
the trainer can sample the fragment's 1 Hz saliency label at those times.

Pooled-only (no spatial grid) - QV is a temporal-domain signal; the head ranks
frames, it does not place crops.  ~9,000 fragments x 8 frames = 72 k forwards,
~1 h on one free card.
"""
import argparse, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--index', type=Path, required=True)
ap.add_argument('--frames-root', type=Path, required=True)
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--cards', default='0')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--output-root', type=Path, required=True)
args = ap.parse_args()

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
assert torch.npu.device_count() > 0
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
mine = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
print(f'QV_FRAG_FEAT shard {args.shard}/{args.nshards}: {len(mine)}', flush=True)

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
n_bad = 0
for r in mine:
    vid = r['video_id']
    op = args.output_root / f'{vid}.npz'
    if op.exists():
        continue
    jpgs = sorted((args.frames_root / vid).glob('*.jpg'), key=lambda p: float(p.stem))
    if len(jpgs) < 4:
        # a partial decode left fewer than the 8-grid frames; too short to score
        continue
    # A single frame can come back with an unexpected hidden size (NaFlex tile
    # layout on an extreme aspect ratio), which makes np.stack fail and kills
    # the whole shard.  Score frame by frame and drop the fragment rather than
    # fabricate a time step: a gap in the sequence would corrupt the temporal
    # label alignment this pool exists to provide.
    feats = []
    try:
        for p in jpgs:
            f = pooled_of(Image.open(p).convert('RGB'))
            if f.shape != (768,):
                raise ValueError(f'unexpected pooled shape {f.shape}')
            feats.append(f)
    except Exception as e:  # noqa: BLE001
        n_bad += 1
        continue
    feats = np.stack(feats)
    ts = np.array([float(p.stem) for p in jpgs], dtype=np.float32)
    np.savez(op, pooled=feats, t=ts)
    n_fr += len(jpgs); n_v += 1
    if n_v % 100 == 0:
        el = time.time() - t0
        print(f'QV_FRAG_FEAT {n_v}/{len(mine)} fr={n_fr} {n_fr/el:.1f}fps '
              f'eta={el/max(n_v,1)*(len(mine)-n_v):.0f}s', flush=True)
print('SUMMARY ' + json.dumps({'shard': args.shard, 'videos': n_v,
                             'frames': n_fr, 'skipped_bad_frame': n_bad,
                             'wall_s': round(time.time()-t0, 1)}), flush=True)
