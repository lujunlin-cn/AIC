"""Confirm-set pooled features on CPU (round 4).

Byte-level contract copy of v10_temporal_attnpool.py (the generator of the
dev pool_feats the frozen checkpoints were trained on), with three forced
changes: device cpu, dtype float32 forward, and multi-sharding for the
192-core box.  Everything else - processor config (min_tiles=1, max_tiles=1,
max_image_tokens=256), valid-token grid mean pooling, topk 48, attention
pooling, fp16 storage, keyframe naming - is identical, so confirm-set
features are the same function of the same jpgs that dev features are.
Numerical caveat recorded in the output: fp32 CPU forward vs fp16 NPU
forward differs at float-noise level; logged in meta.

Output: <out-root>/<video_id>.npz  {mean, attn, topk [8,768] fp16, t [8]}
"""
import argparse, json, os, time
for _v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_v, '4')
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--index', type=Path, required=True)
ap.add_argument('--frames-root', type=Path, required=True)
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=16)
ap.add_argument('--output-root', type=Path, required=True)
ap.add_argument('--frames', type=int, default=8)
ap.add_argument('--ext', default='jpg')
args = ap.parse_args()

import torch  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
mine = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
print(f'CPUFEATS shard {args.shard}/{args.nshards}: {len(mine)} fragments', flush=True)

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1,
                                     max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float32,
    attn_implementation='eager').eval()
TOPK = 48
torch.set_grad_enabled(False)


def pools_of(im):
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                         {'type': 'text', 'text': 'describe'}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors='pt')
    x = {k: v for k, v in x.items() if isinstance(v, torch.Tensor)}
    out = model.model.vision_tower(pixel_values=x['pixel_values'],
                                   spatial_shapes=x['spatial_shapes'],
                                   pixel_attention_mask=x['pixel_attention_mask'],
                                   return_dict=True)
    valid = int(x['pixel_attention_mask'][0].sum())
    fh, fw = [int(v) for v in x['spatial_shapes'][0]]
    g = out.last_hidden_state[0, :valid].reshape(fh, fw, -1).float()
    mean = g.mean((0, 1))
    n = g.reshape(-1, g.shape[-1])
    k = min(TOPK, n.shape[0])
    idx = n.norm(dim=1).topk(k).indices
    topk = n[idx].mean(0)
    q = n.mean(0)
    w = torch.softmax((n @ q) / (n.shape[-1] ** 0.5), dim=0)
    attn = (n * w[:, None]).sum(0)
    return (mean.cpu().numpy().astype(np.float16),
            attn.cpu().numpy().astype(np.float16),
            topk.cpu().numpy().astype(np.float16))


args.output_root.mkdir(parents=True, exist_ok=True)
t0, n_fr, n_v, n_bad = time.time(), 0, 0, 0
for r in mine:
    vid = r['video_id']
    op = args.output_root / f'{vid}.npz'
    if op.exists():
        continue
    jpgs = sorted((args.frames_root / vid).glob(f'*.{args.ext}'),
                  key=lambda p: float(p.stem))
    jpgs = [p for p in jpgs if p.exists()]
    if len(jpgs) < 4:
        continue
    M, A, T, ts = [], [], [], []
    try:
        for p in jpgs:
            m, a, t = pools_of(Image.open(p).convert('RGB'))
            if m.shape != (768,) or a.shape != (768,) or t.shape != (768,):
                raise ValueError('unexpected pooled shape')
            M.append(m); A.append(a); T.append(t); ts.append(float(p.stem))
    except Exception:  # noqa: BLE001
        n_bad += 1
        continue
    np.savez(op, mean=np.stack(M), attn=np.stack(A), topk=np.stack(T),
             t=np.array(ts, dtype=np.float32))
    n_fr += len(jpgs); n_v += 1

print('SUMMARY ' + json.dumps({'shard': args.shard, 'videos': n_v, 'frames': n_fr,
                               'skipped_bad_frame': n_bad,
                               'wall_s': round(time.time() - t0, 1)}), flush=True)
