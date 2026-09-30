"""V8: dense (all-frame) LFM vision grids for the RetargetVid DIAG sources
031-100.  Sources are DHF1K AVIs (640x360, ~450-590 frames).  Dense features
serve (a) student INTERP / dense-observation submission studies, (b) the
synthetic joint evaluation (temporal value from DHF1K saliency + spatial GT
from RV), and (c) tracking-failure diagnostics.  DIAG is never trained on.
"""
import argparse, json, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--cards', default='2')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--video-dir', default='/data/aic/external_datasets/DHF1K/raw/video')
ap.add_argument('--output-root', type=Path, default=Path('/data/aic/experiments_910a/LFM_V8/diag_feats_dense'))
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

vids = sorted(f'{i:03d}' for i in range(31, 101))
mine = [v for i, v in enumerate(vids) if i % args.nshards == args.shard]
print(f'DEBUG shard {args.shard}/{args.nshards}: {len(mine)} vids', flush=True)

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
    od = args.output_root / vid
    media = Path(args.video_dir) / f'{vid}.AVI'
    n_est = 620
    have = len(list(od.glob('*.npz'))) if od.exists() else 0
    with av.open(media) as cont:
        total = cont.streams.video[0].frames
        if have >= total:
            continue
        od.mkdir(parents=True, exist_ok=True)
        fi = -1
        for frame in cont.decode(cont.streams.video[0]):
            fi += 1  # PyAV 17 VideoFrame has no .index attribute
            outf = od / f'{fi}.npz'
            if outf.exists():
                continue
            im = frame.to_image().convert('RGB')
            g, p = vision(im)
            np.savez_compressed(outf, grid=g, pooled=p,
                                shape=np.array([g.shape[0], g.shape[1]], dtype=np.int32))
            n += 1
    if vi % 5 == 0:
        print(f'DEBUG vid {vi}/{len(mine)} {vid} fw={n} {time.time() - t0:.0f}s', flush=True)

summ = {'shard': args.shard, 'nshards': args.nshards, 'vids': len(mine), 'frames_written': n,
        'wall_s': round(time.time() - t0, 1)}
(args.output_root / f'extract_s{args.shard}_summary.json').write_text(json.dumps(summ, indent=1) + '\n')
print('SUMMARY', json.dumps(summ), flush=True)
