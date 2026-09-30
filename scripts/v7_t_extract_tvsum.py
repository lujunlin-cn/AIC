"""V7 T-line: 1fps LFM pooled features for all TVSum videos (vision tower only).

Alignment: proxy-v2 labels give per-frame 20-rater scores indexed by frame_indices;
we sample 1fps frames, record their source frame index and pull the label row.
Output {vid}.npz: pooled (n,768) fp16, frame_idx (n,), label (n,) float32.
"""
import argparse, json, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--cards', default='2')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--video-dir', type=Path, default=Path('/data/aic/datasets/TVSum/raw/ydata-tvsum50-v1_1/videos/video'))
ap.add_argument('--labels-dir', type=Path, default=Path('/data/aic/datasets/TVSum/labels_v3'))
ap.add_argument('--output', type=Path, default=Path('/data/aic/experiments_910a/LFM_V7/tfeats_tvsum'))
args = ap.parse_args()

import os  # noqa: E402
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
import numpy as np  # noqa: E402
import av  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

vids = sorted(p.name.replace('.proxy-v2.npz', '') for p in args.labels_dir.glob('*.proxy-v2.npz'))
mine = [v for i, v in enumerate(vids) if i % args.nshards == args.shard]
args.output.mkdir(parents=True, exist_ok=True)

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1, max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation='eager').eval().to('npu')


def pooled_of(im):
    msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                         {'type': 'text', 'text': 'describe'}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors='pt')
    x = {k: v.to('npu') for k, v in x.items() if isinstance(v, torch.Tensor)}
    if 'pixel_values' in x:
        x['pixel_values'] = x['pixel_values'].to(torch.float16)
    with torch.no_grad():
        out = model.model.vision_tower(pixel_values=x['pixel_values'], spatial_shapes=x['spatial_shapes'],
                                       pixel_attention_mask=x['pixel_attention_mask'], return_dict=True)
    valid = int(x['pixel_attention_mask'][0].sum())
    return out.last_hidden_state[0, :valid].float().mean(0)


for vid in mine:
    outf = args.output / f'{vid}.npz'
    if outf.exists():
        continue
    lab = np.load(args.labels_dir / f'{vid}.proxy-v2.npz', allow_pickle=True)
    fidx = lab['frame_indices'].astype(np.int64)
    lab_by_frame = {int(f): float(lab['scores'][i].astype(np.float32).mean()) for i, f in enumerate(fidx)}
    with av.open(str(args.video_dir / f'{vid}.mp4')) as c:
        st = c.streams.video[0]
        fps = float(st.average_rate)
        step = max(1, int(round(fps)))  # ~1fps
        pooled, fr_idx = [], []
        for i, fr in enumerate(c.decode(st)):
            if i % step == 0:
                pooled.append(pooled_of(fr.to_image()).cpu().numpy())
                fr_idx.append(i)
    pooled = np.stack(pooled).astype(np.float16)
    fr_idx = np.asarray(fr_idx, dtype=np.int64)
    label = np.array([lab_by_frame.get(int(f), np.nan) for f in fr_idx], dtype=np.float32)
    np.savez_compressed(outf, pooled=pooled, frame_idx=fr_idx, label=label, fps=np.float32(fps))
    print(f'DONE {vid} n={len(fr_idx)} fps={fps:.2f}', flush=True)
print('ALL DONE', flush=True)
