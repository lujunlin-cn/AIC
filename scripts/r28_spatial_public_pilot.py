#!/usr/bin/env python3
"""Bounded public-only spatial grounding diagnostic; never reads official videos."""
import argparse
import hashlib
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration


def iou(box, gt):
    inter = np.maximum(np.minimum(box[2:], gt[:, 2:]) - np.maximum(box[:2], gt[:, :2]), 0).prod(1)
    union = np.prod(box[2:] - box[:2]) + np.maximum(gt[:, 2:] - gt[:, :2], 0).prod(1) - inter
    return float(np.mean(inter / np.maximum(union, 1e-9)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    cfg = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(cfg['seed'])
    proc = AutoProcessor.from_pretrained(cfg['model'])
    model = Qwen3_5ForConditionalGeneration.from_pretrained(cfg['model'], dtype=torch.bfloat16, attn_implementation='eager').eval().to('cuda:0')
    t0 = time.monotonic()
    rows = []
    for vid in cfg['sources']:
        cached = {int(p.stem): p for p in (Path(cfg['public_keyframe_root']) / vid).glob('*.png')}
        if not cached:
            raise ValueError('missing cached public frames ' + vid)
        gt = {}
        for rw, rh in cfg['ratios']:
            gt[(rw, rh)] = np.stack([np.loadtxt(Path(cfg['annotation_root']) / f'annotator_{a}' / f'{vid}_{rw}-{rh}.txt', delimiter=',') for a in range(1, 7)])
        n = gt[tuple(cfg['ratios'][0])].shape[1]
        if any(v.shape != (6, n, 4) for v in gt.values()):
            raise ValueError('GT geometry/length mismatch')
        for frac in cfg['frame_fractions']:
            frame = min(cached, key=lambda k: (abs(k - int(frac * n)), k))
            if not 0 <= frame < n:
                raise ValueError('cached frame outside annotations')
            img = Image.open(cached[frame]).convert('RGB')
            W, H = img.size
            if abs(W / H - 640 / 360) > .001:
                raise ValueError('unexpected RV aspect ratio')
            png_sha = hashlib.sha256(cached[frame].read_bytes()).hexdigest()
            img.thumbnail((cfg['max_image_side'], cfg['max_image_side']))
            for rw, rh in cfg['ratios']:
                width, height = min(640., 360. * rw / rh), min(360., 640. * rh / rw)
                axis = 'horizontal' if width < 640 - 1e-9 else 'vertical'
                prompt = cfg['prompt_template'].replace('{rw}', str(rw)).replace('{rh}', str(rh)).replace('{axis}', axis)
                msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': prompt}]}]
                inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors='pt', enable_thinking=cfg['thinking']).to('cuda:0')
                ts = time.monotonic()
                with torch.inference_mode():
                    output = model.generate(**inputs, do_sample=cfg['do_sample'], max_new_tokens=cfg['max_new_tokens'])
                raw = proc.batch_decode(output[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                center = None
                for match in re.finditer(r'\{[^{}]*\}', raw):
                    try:
                        c = json.loads(match.group())['center']
                        if type(c) in (int, float) and np.isfinite(c) and 0 <= c <= 1:
                            center = float(c)
                            break
                    except (ValueError, KeyError, TypeError):
                        pass
                valid = center is not None
                if not valid:
                    center = .5
                def box(c):
                    x = np.clip(c * 640 - width / 2, 0, 640 - width) if axis == 'horizontal' else 0.
                    y = np.clip(c * 360 - height / 2, 0, 360 - height) if axis == 'vertical' else 0.
                    return np.array([x, y, x + width, y + height])
                boxes = gt[(rw, rh)][:, frame]
                row = dict(video_id=vid, frame=frame, png_sha256=png_sha, ratio=[rw, rh], axis=axis, raw=raw, parse_valid=valid, center=center, crop=box(center).tolist(), iou=iou(box(center), boxes), center_iou=iou(box(.5), boxes), seconds=time.monotonic() - ts)
                rows.append(row)
                with (args.output / 'predictions.jsonl').open('a') as f:
                    f.write(json.dumps(row) + '\n')
                print(json.dumps(row), flush=True)
    per_source = {v: float(np.mean([r['iou'] - r['center_iou'] for r in rows if r['video_id'] == v])) for v in cfg['sources']}
    report = dict(experiment_id=cfg['experiment_id'], scope='exposed public RV engineering pilot; no fresh holdout and no B3 comparator', config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(), rows=len(rows), sources=len(per_source), mean_iou=float(np.mean([r['iou'] for r in rows])), center_iou=float(np.mean([r['center_iou'] for r in rows])), per_source_delta=per_source, parse_valid=sum(r['parse_valid'] for r in rows), elapsed_seconds=time.monotonic()-t0, peak_vram_allocated=torch.cuda.max_memory_allocated(), runtime_unique_parameters=sum(p.numel() for p in model.parameters()), torch=torch.__version__)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
