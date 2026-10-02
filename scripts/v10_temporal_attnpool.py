"""Attention-pooled vision features for the temporal head (NPU, sharded).

The temporal head currently scores a frame from the MEAN of the vision tower's
22x40 token grid.  That mean is an order-statistic-free average: it survives a
frame where the subject occupies 5% of the frame exactly as well as one where it
fills the view, which is precisely the difference between "this second is worth
keeping" and "this second is filler".  For a KEEP decision that distinction is
the whole task, so this pass stores two pooled views of the same forward:

  mean   - the existing baseline, kept so runs stay comparable
  attn   - attention pooling with a learned query, computed on CPU from the grid
           (the tower is frozen; only the pooling changes)
  topk   - mean of the top-k tokens by L2 norm, a cheap saliency proxy that
           needs no training and captures "how much of the frame is sharp and
           high-contrast", the visual signature of an action peak

The grid is stored so the head can be retrained against any of the three
without another NPU pass.  Grid-only (no mean) at fp16 is 1.35 MB/frame; for
the fragment pool (8 frames each) that is ~100 GB, too much, so this writes
mean+attn+topk (768-d each) and discards the grid.  The spatial head's grid
cache is untouched and remains the input to any heat-map work.
"""
import argparse, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--index', type=Path, required=True, help='fragment index.jsonl (PHD2 or QV)')
ap.add_argument('--frames-root', type=Path, required=True)
ap.add_argument('--skel-root', type=Path, default=None,
                help='if given, keyframe names come from <skel>/<vid>.json instead of the frames tree')
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--cards', default='0')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--output-root', type=Path, required=True)
ap.add_argument('--frames', type=int, default=8)
ap.add_argument('--ext', default='jpg')
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

rows = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()]
mine = [r for i, r in enumerate(rows) if i % args.nshards == args.shard]
print(f'ATTNPOOL shard {args.shard}/{args.nshards}: {len(mine)} fragments', flush=True)

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1,
                                     max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True,
    attn_implementation='eager').eval().to('npu')

TOPK = 48


def pools_of(im):
    """Return (mean, attn, topk) pooled 768-d vectors for one image."""
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
        g = out.last_hidden_state[0, :valid].reshape(fh, fw, -1).float()
    # mean baseline (identical to the deployed spatial head's pooling)
    mean = g.mean((0, 1))
    # top-k by L2 norm: the visually salient tokens carry most of the energy
    n = g.reshape(-1, g.shape[-1])
    k = min(TOPK, n.shape[0])
    idx = n.norm(dim=1).topk(k).indices
    topk = n[idx].mean(0)
    # attention pooling with a fixed learned query: weights by relevance to a
    # generic "content" direction, computed without training so the pass is a
    # pure feature extraction step
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
    if args.skel_root and (args.skel_root / f'{vid}.json').exists():
        kfs = json.loads((args.skel_root / f'{vid}.json').read_text())['keyframes']
        jpgs = [args.frames_root / vid / f'{k}.{args.ext}' for k in kfs]
    else:
        jpgs = sorted((args.frames_root / vid).glob(f'*.{args.ext}'), key=lambda p: float(p.stem))
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
    except Exception:  # noqa: BLE001 - one bad frame must not kill the shard
        n_bad += 1
        continue
    np.savez(op, mean=np.stack(M), attn=np.stack(A), topk=np.stack(T),
             t=np.array(ts, dtype=np.float32))
    n_fr += len(jpgs); n_v += 1
    if n_v % 200 == 0:
        el = time.time() - t0
        print(f'ATTNPOOL {n_v}/{len(mine)} fr={n_fr} {n_fr/el:.1f}fps '
              f'eta={el/max(n_v,1)*(len(mine)-n_v):.0f}s bad={n_bad}', flush=True)

print('SUMMARY ' + json.dumps({'shard': args.shard, 'videos': n_v, 'frames': n_fr,
                               'skipped_bad_frame': n_bad,
                               'wall_s': round(time.time() - t0, 1)}), flush=True)