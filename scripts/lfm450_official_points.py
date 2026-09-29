"""LFM2.5-VL-450M GROUND subject points on the frozen official index.

Same frozen GROUND prompt / parser / NPU settings as scripts/lfm450_eval_spatial.py
(chosen on the RetargetVid dev manifest, GROUND > GROUNDR there).  Keyframes:
the official 1 s grid + shot starts already extracted for MAX_WINDOW_QWEN_POINT_V1
(QWEN_SUBJECT_POINT_OFFICIAL_V1/keyframes, 3,106 PNG, long side 640) -- only
the frame images and keyframe ids are reused; teacher replies are never read.

Output per video mirrors the qwen points schema consumed by
scripts/max_window_release.py (ratios.t.points = [cx, cy, uncertain]) so the
unchanged release path can package it.
"""
import argparse, json, os, re, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--index', type=Path, default=Path('/data/aic/official_test_20260926/intake/index.enriched.jsonl'))
ap.add_argument('--keyframe-src', type=Path, default=Path('/data/aic/experiments/QWEN_SUBJECT_POINT_OFFICIAL_V1'))
ap.add_argument('--cards', default='2')
ap.add_argument('--shard', type=int, default=0)
ap.add_argument('--nshards', type=int, default=1)
ap.add_argument('--output-dir', type=Path, required=True)
args = ap.parse_args()
os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards

import torch, torch_npu  # noqa: E402
torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

PROMPT = ('Detect all instances of: main subject. Response must be a JSON array: '
          '[{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]. Coordinates are normalized to [0,1].')
BOX_RE = re.compile(r'"bbox"\s*:\s*\[\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)\s*\]')


def parse_boxes(reply):
    found = BOX_RE.findall(reply)
    if not found:
        return None, 'no_bbox'
    valid = []
    for g in found:
        try:
            x1, y1, x2, y2 = [float(v) for v in g]
        except ValueError:
            continue
        if min(x1, y1, x2, y2) < -0.02 or max(x1, y1, x2, y2) > 1.02:
            continue
        x1, y1, x2, y2 = [min(max(v, 0.), 1.) for v in (x1, y1, x2, y2)]
        if x2 - x1 < 0.005 or y2 - y1 < 0.005:
            continue
        valid.append([x1, y1, x2, y2])
    if not valid:
        return None, 'invalid_bbox'
    return max(valid, key=lambda b: (b[2] - b[0]) * (b[3] - b[1])), 'ok'


proc = AutoProcessor.from_pretrained(args.model)
model = Lfm2VlForConditionalGeneration.from_pretrained(args.model, dtype=torch.float16, low_cpu_mem_usage=True,
                                                       attn_implementation='eager').eval().to('npu')
recs = [json.loads(l) for l in args.index.read_text().splitlines() if l.strip()][args.shard::args.nshards]
pdir = args.output_dir / 'points'
pdir.mkdir(parents=True, exist_ok=True)
tq = 0.0
nq = 0
for r in recs:
    vid = r['video_id']
    if (pdir / f'{vid}.json').exists():
        continue
    src = json.loads((args.keyframe_src / 'points' / f'{vid}.json').read_text())
    kfs = src['keyframes']
    pts, st, raw, boxes = [], [], [], []
    t_vid = time.time()
    for kf in kfs:
        img = Image.open(args.keyframe_src / 'keyframes' / vid / f'{kf}.png').convert('RGB')
        x = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image', 'image': img},
                                                                    {'type': 'text', 'text': PROMPT}]}],
                                     tokenize=True, add_generation_prompt=True, return_dict=True,
                                     return_tensors='pt').to('npu')
        x['pixel_values'] = x['pixel_values'].to(model.dtype)
        t = time.time()
        with torch.no_grad():
            o = model.generate(**x, max_new_tokens=64, do_sample=False)
        torch.npu.synchronize()
        tq += time.time() - t
        nq += 1
        reply = proc.batch_decode(o[:, x['input_ids'].shape[1]:], skip_special_tokens=True)[0].strip()
        b, s = parse_boxes(reply)
        pts.append([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2, False] if b else None)
        st.append(s)
        raw.append(reply[:300])
        boxes.append(b)
    out = {'video_id': vid, 'W': src['W'], 'H': src['H'], 'fps': src['fps'], 'step': src['step'],
           'keyframes': kfs, 'model': 'LiquidAI/LFM2.5-VL-450M@fc6221ca597f3315e4f82fc2df606783267b34ba',
           'prompt': 'GROUND', 'prompt_text': PROMPT,
           'ratios': {'t': {'ratio': r['targetRatioWH'], 'points': pts, 'status': st, 'raw': raw, 'boxes': boxes}},
           'seconds': round(time.time() - t_vid, 2)}
    (pdir / f'{vid}.json').write_text(json.dumps(out) + '\n')
    print(f'{vid} kf={len(kfs)} ok={st.count("ok")} avg {tq / max(nq, 1):.3f}s', flush=True)
summ = {'shard': args.shard, 'n_queries': nq, 's_queries': round(tq, 1), 'avg_s_per_query': round(tq / max(nq, 1), 3),
        'peak_mem_gib': round(torch.npu.max_memory_allocated() / 2**30, 3), 't_done': time.strftime('%Y-%m-%d %H:%M:%S')}
(args.output_dir / f'summary_s{args.shard}.json').write_text(json.dumps(summ, indent=1) + '\n')
print('SUMMARY', json.dumps(summ), flush=True)
