"""V7 E0: single-factor protocol comparison to explain the E5 0/12 vs 3210/3210 parse conflict.

Same 4 frames E5 used (031-034 frame 0), same 910A FP16/eager/jit-off stack as the
3210/3210 run.  Only the reply schema varies (one factor at a time):
  A JSON01   official grounding JSON array, coords in [0,1]      (the working protocol)
  B BOX1000  E5 P1 verbatim: <box>x1 y1 x2 y2</box>, ints 0-1000
  C JSON1000 official JSON schema but integer 0-1000 coords
  D BOX01    E5 <box> wrapper but [0,1] float coords
Parse each arm with its own protocol parser; also count legal-box rate so parse
recovery is not confused with localisation quality.
"""
import argparse, json, re, time
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--cards', default='2')
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()

import os  # noqa: E402
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import torch  # noqa: E402
import torch_npu  # noqa: E402,F401
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

FR = Path('/data/aic/experiments/QWEN_RV200_KEYFRAMES/d1')
FRAMES = ['031/0.png', '032/0.png', '033/0.png', '034/0.png']

JSON01 = ('Detect all instances of: main subject. Response must be a JSON array: '
          '[{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]. Coordinates are normalized to [0,1].')
JSON1000 = ('Detect all instances of: main subject. Response must be a JSON array: '
            '[{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]. '
            'Coordinates are integers normalized to [0,1000].')
BOX1000 = ('Locate the primary subject that should stay inside the crop. '
           'Reply with exactly one bounding box in the form <box>x1 y1 x2 y2</box> '
           'using integers normalized to 0-1000. No other text.')
BOX01 = ('Locate the primary subject that should stay inside the crop. '
         'Reply with exactly one bounding box in the form <box>x1 y1 x2 y2</box> '
         'using decimal coordinates normalized to [0,1]. No other text.')
ARMS = {'JSON01': JSON01, 'BOX1000': BOX1000, 'JSON1000': JSON1000, 'BOX01': BOX01}

BOX_RE = re.compile(r'<box>\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*</box>')
BB_RE = re.compile(r'"bbox"\s*:\s*\[\s*("(?:[^"]*)"|[-\d.eE+]+)\s*,\s*("(?:[^"]*)"|[-\d.eE+]+)\s*,'
                   r'\s*("(?:[^"]*)"|[-\d.eE+]+)\s*,\s*("(?:[^"]*)"|[-\d.eE+]+)\s*\]')


def parse_json01(reply):
    out = []
    for g in BB_RE.finditer(reply):
        vals = []
        for x in g.groups():
            x = x.strip().strip('"')
            try:
                vals.append(float(x))
            except ValueError:
                return []
        if len(vals) == 4 and all(0.0 <= v <= 1.0 for v in vals):
            out.append(vals)
    return out


def parse_json1000(reply):
    out = []
    for g in BB_RE.finditer(reply):
        vals = []
        for x in g.groups():
            x = x.strip().strip('"')
            try:
                vals.append(float(x))
            except ValueError:
                return []
        if len(vals) == 4 and all(0.0 <= v <= 1000.0 for v in vals):
            out.append(vals)
    return out


def parse_box(reply, scale):
    m = BOX_RE.search(reply)
    if not m:
        return []
    vals = [float(v) for v in m.groups()]
    if all(0.0 <= v <= scale for v in vals):
        return [vals]
    return []


PARSERS = {'JSON01': lambda r: parse_json01(r), 'JSON1000': lambda r: parse_json1000(r),
           'BOX1000': lambda r: parse_box(r, 1000.0), 'BOX01': lambda r: parse_box(r, 1.0)}

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1, max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True, attn_implementation='eager').eval().to('npu')

rows = []
for arm, prompt in ARMS.items():
    for f in FRAMES:
        im = Image.open(FR / f).convert('RGB')
        W, H = im.size
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': im},
                                             {'type': 'text', 'text': prompt}]}]
        x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                     return_dict=True, return_tensors='pt').to('npu')
        if 'pixel_values' in x:
            x['pixel_values'] = x['pixel_values'].to(torch.float16)
        torch.npu.synchronize()
        t0 = time.time()
        with torch.no_grad():
            out = model.generate(**x, max_new_tokens=64, do_sample=False)
        torch.npu.synchronize()
        s = time.time() - t0
        reply = proc.batch_decode(out[:, x['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
        boxes = PARSERS[arm](reply)
        legal = 0
        if boxes:
            # coords -> pixel frame, check in-bounds and ordered
            sc = 1000.0 if arm == 'BOX1000' or arm == 'JSON1000' else 1.0
            px = [[v / sc * (W if i % 2 == 0 else H) for i, v in enumerate(b)] for b in boxes]
            legal = sum(1 for b in px if 0 <= b[0] < b[2] <= W + 1 and 0 <= b[1] < b[3] <= H + 1)
        rows.append({'arm': arm, 'frame': f, 'parse_ok': int(bool(boxes)), 'n_boxes': len(boxes),
                     'legal': legal, 'latency_s': round(s, 2), 'raw': reply[:220]})
        print('ROW', json.dumps(rows[-1], ensure_ascii=False), flush=True)

summ = {}
for arm in ARMS:
    rs = [r for r in rows if r['arm'] == arm]
    summ[arm] = {'parse': f"{sum(r['parse_ok'] for r in rs)}/{len(rs)}",
                 'legal_when_parsed': f"{sum(r['legal'] for r in rs)}/{sum(r['n_boxes'] for r in rs) or 1}"}
out = {'summary': summ, 'rows': rows,
       'protocol': {'model': args.model, 'min_tiles': 1, 'max_tiles': 1, 'max_image_tokens': 256,
                    'dtype': 'fp16', 'attn': 'eager', 'jit': 'off', 'max_new_tokens': 64}}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(out, ensure_ascii=False, indent=1) + '\n')
print('SUMMARY', json.dumps(summ), flush=True)
