"""LFM2.5-VL-450M NPU smoke: dtype/attention validation + parse + timing.

Runs a handful of real keyframes from the RetargetVid dev pool through both
fixed prompt variants.  Reports load time, peak memory, per-query latency and
raw replies so FP16 numeric health can be judged before the full sweep.

  --cards N        single physical NPU id exposed to this process
  --dtype          float16 first (competition rule: do not copy bf16 defaults
                   blindly); bf16 only as an explicit cross-check
  --attn           eager | sdpa (auto-fallback to eager on failure)
"""
import argparse, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--points-dir', type=Path, default=Path('/data/aic/experiments/QWEN_RV200_POINT_D1/points'))
ap.add_argument('--frames-dir', type=Path, default=Path('/data/aic/experiments/QWEN_RV200_KEYFRAMES/d1'))
ap.add_argument('--videos', nargs='*', default=['031', '032', '033'])
ap.add_argument('--per-video', type=int, default=2)
ap.add_argument('--cards', default='6')
ap.add_argument('--dtype', default='float16', choices=['float16', 'bfloat16', 'float32'])
ap.add_argument('--attn', default='eager', choices=['eager', 'sdpa'])
ap.add_argument('--max-tiles', type=int, default=0, help='>0 pins the vision tile budget (uniform shapes, one compile)')
ap.add_argument('--max-new', type=int, default=48)
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards

import torch, torch_npu  # noqa: E402
from PIL import Image  # noqa: E402

t0 = time.time()
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

proc = AutoProcessor.from_pretrained(args.model, **({'max_tiles': args.max_tiles} if args.max_tiles else {}))
kw = dict(dtype=getattr(torch, args.dtype), low_cpu_mem_usage=True,
          attn_implementation=args.attn)
try:
    model = Lfm2VlForConditionalGeneration.from_pretrained(args.model, **kw).eval()
except Exception as e:
    print('LOAD_FAIL', repr(e)[:300], flush=True)
    raise SystemExit(1)
model = model.to('npu')
load_s = time.time() - t0
n_params = sum(p.numel() for p in model.parameters())
print(json.dumps({'load_s': round(load_s, 1), 'n_params': n_params,
                  'n_params_M': round(n_params / 1e6, 2), 'dtype': args.dtype,
                  'attn': args.attn, 'device': str(next(model.parameters()).device)}), flush=True)

SYSTEM_POINT = ('You locate the main subject to keep when reframing a video frame. '
                'Answer with only (x, y): the subject centre on a 0-1000 grid. No other text.')
SYSTEM_BOX = ('You locate the main subject region to keep when reframing a video frame. '
              'Answer with only (x1, y1, x2, y2): the subject bounding box corners on a 0-1000 grid. No other text.')


def user_text(W, H, rw, rh):
    orient = 'portrait (narrower than the frame)' if rw / rh < W / H else 'landscape (wider than the frame)'
    return (f'The source frame is {W}x{H}. The target aspect ratio is {rw}:{rh}, which is {orient}. '
            f'Look at the image and answer with the required tuple only.')


import re
TUP = re.compile(r'\(?\s*(\d+(?:\.\d+)?)\s*[,;]\s*(\d+(?:\.\d+)?)(?:\s*[,;]\s*(\d+(?:\.\d+)?)\s*[,;]\s*(\d+(?:\.\d+)?))?\s*\)?')


def parse_reply(reply):
    m = TUP.findall(reply)
    if not m:
        return None, 'no_tuple'
    v = [float(x) for x in m[0] if x != '']
    if len(v) not in (2, 4):
        return None, 'no_tuple'
    if any(g > 1000 + 1e-6 for g in v):
        return None, 'out_of_range'
    kind = 'point' if len(v) == 2 else 'box'
    return {'kind': kind, 'vals': v}, 'ok'


rows = []
tq = 0.0
nq = 0
for vid in args.videos:
    pj = args.points_dir / f'{vid}.json'
    if not pj.exists():
        continue
    meta = json.loads(pj.read_text())
    W, H = meta['W'], meta['H']
    kfs = meta['keyframes'][:args.per_video]
    for rk, r in meta['ratios'].items():
        rw, rh = r['ratio']
        for kf in kfs:
            img_path = args.frames_dir / vid / f'{kf}.png'
            if not img_path.exists():
                continue
            img = Image.open(img_path).convert('RGB')
            for sys_p in (SYSTEM_POINT, SYSTEM_BOX):
                msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': sys_p}]},
                        {'role': 'user', 'content': [{'type': 'image', 'image': img},
                                                     {'type': 'text', 'text': user_text(W, H, rw, rh)}]}]
                inputs = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                                  return_dict=True, return_tensors='pt').to('npu')
                t = time.time()
                with torch.no_grad():
                    out = model.generate(**inputs, max_new_tokens=args.max_new, do_sample=False)
                torch.npu.synchronize()
                dt = time.time() - t
                tq += dt
                nq += 1
                reply = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)[0]
                p, st = parse_reply(reply)
                rows.append({'vid': vid, 'ratio': rk, 'kf': kf, 'prompt': 'point' if sys_p is SYSTEM_POINT else 'box',
                             's': round(dt, 2), 'status': st, 'reply': reply.strip()[:120], 'parsed': p})
    print(f'{vid} done, running avg {tq / max(nq,1):.2f}s/query', flush=True)

peak = torch.npu.max_memory_allocated() / 2**30
summ = {'dtype': args.dtype, 'attn': args.attn, 'load_s': round(load_s, 1),
        'n_params_M': round(n_params / 1e6, 2), 'n_queries': nq,
        'avg_s_per_query': round(tq / max(nq, 1), 3),
        'peak_mem_gib': round(peak, 2),
        'parse_ok': sum(r['status'] == 'ok' for r in rows), 'n_rows': len(rows)}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({'summary': summ, 'rows': rows}, ensure_ascii=False, indent=1) + '\n')
print('SUMMARY', json.dumps(summ), flush=True)
for r in rows[:14]:
    print('ROW', json.dumps(r, ensure_ascii=False), flush=True)
