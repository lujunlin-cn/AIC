"""Parameter budget of the deployed student: which parts of LFM2.5-VL actually run.

The deployment only calls model.model.vision_tower(...) - no text prompt, no
generate - so the language model is dead weight at inference time, yet it is
still counted by the rules ("all model weights actually loaded").  Knowing the
split decides whether the 500 MB boundary (k_size 0.90 -> 0.95) is reachable by
dropping the LM, by quantising the vision tower, or only by both.
"""
import json
from collections import defaultdict
from pathlib import Path

import torch
from transformers import Lfm2VlForConditionalGeneration

m = Lfm2VlForConditionalGeneration.from_pretrained(
    '/data/aic/pretrained/lfm2_5_vl_450m', dtype=torch.float16,
    device_map='cpu', low_cpu_mem_usage=True)

groups = defaultdict(int)
for n, p in m.named_parameters():
    parts = n.split('.')
    if 'vision_tower' in parts:
        top = 'vision_tower'
    elif 'multi_modal_projector' in parts or 'projector' in parts:
        top = 'projector'
    elif 'language_model' in parts or n.startswith('lm_head'):
        top = 'language_model'
    else:
        top = 'other:' + parts[0]
    groups[top] += p.numel()

tot = sum(groups.values())
print(f'total params {tot:,}')
for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
    print(f'  {k:24s} {v:>12,}  {100*v/tot:5.1f}%  fp16 {v*2/1e6:7.1f} MB  int8 {v/1e6:7.1f} MB')

head = 1511425
yunet = 53104
print()
for name, keep_int8 in (('fp16 all', False), ('vision+proj int8', True)):
    base = 0
    for k, v in groups.items():
        if k == 'language_model':
            continue
        base += v // 2 if keep_int8 else v * 2
    total = base + head * 4 + yunet
    print(f'{name:16s} runtime bytes = {total:,} = {total/1024/1024:.1f} MB  '
          f'(k=1.00 / 0.95 / 0.90 -> '
          f'{1.0 if total <= 100*1024*1024 else (0.95 if total <= 500*1024*1024 else 0.90)})')

Path('/data/aic/experiments_910a/LFM_V9/param_budget.json').write_text(json.dumps(
    {'total': tot, 'groups': dict(groups)}, indent=1))