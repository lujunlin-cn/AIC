"""Valid-token pooled features for the 426-video semifinal drop.

Fixes the A0 defect: the deployment cache pooled ALL output tokens (padding
included), the probe head was trained on valid-token means.  Recomputes the
SAME keyframes with the probe's pooling contract - byte-identical pooling to
v10_lora_tower_probe.tower_pooled (valid-token spatial grid mean), stored in
FP32.
"""
import argparse, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--kf-root', type=Path, default=Path('/data/aic/semifinal_20261001/keyframes/keyframes'))
ap.add_argument('--out-features', type=Path, required=True)
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--cards', default='0')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--verbose-every', type=int, default=50)
args = ap.parse_args()

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import torch
import torch_npu
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
assert torch.npu.device_count() > 0, 'no NPU'
import numpy as np
from PIL import Image
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration

DEV = 'npu'
proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1, max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True,
    attn_implementation='eager').eval().to(DEV)

KF = args.kf_root
vids = sorted((d for d in KF.iterdir() if d.is_dir()), key=lambda p: int(p.name))
mine = [v for i, v in enumerate(vids) if i % args.nshards == args.shard]
print(f'VT feats shard {args.shard}/{args.nshards}: {len(mine)} videos', flush=True)

t0, n = time.time(), 0
with torch.no_grad():
    for vi, vd in enumerate(mine):
        pngs = sorted(p for p in vd.glob('*.png'))
        od = args.out_features / vd.name
        need = [p for p in pngs if not (od / (p.stem + '.npy')).exists()]
        if not need:
            continue
        od.mkdir(parents=True, exist_ok=True)
        for p in need:
            im = Image.open(p).convert('RGB')
            msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                                 {'type': 'text', 'text': 'describe'}]}]
            x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                         return_dict=True, return_tensors='pt')
            x = {k: v.to(DEV) for k, v in x.items() if isinstance(v, torch.Tensor)}
            x['pixel_values'] = x['pixel_values'].to(torch.float16)
            out = model.model.vision_tower(pixel_values=x['pixel_values'],
                                           spatial_shapes=x['spatial_shapes'],
                                           pixel_attention_mask=x['pixel_attention_mask'],
                                           return_dict=True)
            valid = int(x['pixel_attention_mask'][0].sum())
            fh, fw = [int(v) for v in x['spatial_shapes'][0]]
            grid = out.last_hidden_state[0, :valid].reshape(fh, fw, -1)
            feat = grid.float().mean((0, 1)).cpu().numpy()
            np.save(od / (p.stem + '.npy'), feat.astype(np.float32))
            n += 1
            if n % args.verbose_every == 0:
                el = time.time() - t0
                print(f'VT {vi + 1}/{len(mine)} frames={n} {el:.0f}s', flush=True)
print('VT_SUMMARY', json.dumps({'shard': args.shard, 'videos': len(mine),
                                'frames': n, 'wall_s': round(time.time() - t0, 1)}), flush=True)
