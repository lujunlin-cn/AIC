"""LFM2.5-VL-450M latency diagnosis: NPU jit vs binary kernels vs CPU.

Runs the same official-grounding query on a few dev keyframes repeatedly and
reports per-call latency, so compile cost (first call per shape) can be told
apart from steady-state cost.  Also prints the raw replies for format checks.
"""
import argparse, json, os, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/lfm2_5_vl_450m')
ap.add_argument('--device', default='npu', choices=['npu', 'cpu'])
ap.add_argument('--cards', default='4')
ap.add_argument('--dtype', default='float16')
ap.add_argument('--jit', default='default', choices=['default', 'off'])
ap.add_argument('--threads', type=int, default=16)
ap.add_argument('--attn', default='eager', choices=['eager', 'sdpa'])
ap.add_argument('--frames', nargs='*', default=['031/0.png', '031/120.png', '034/0.png'])
ap.add_argument('--repeat', type=int, default=2)
ap.add_argument('--max-new', type=int, default=48)
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards
import torch  # noqa: E402
if args.device == 'npu':
    import torch_npu  # noqa: E402,F401
    if args.jit == 'off':
        torch.npu.set_compile_mode(jit_compile=False)
        torch.npu.config.allow_internal_format = False
else:
    torch.set_num_threads(args.threads)
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

FR = Path('/data/aic/experiments/QWEN_RV200_KEYFRAMES/d1')
PROMPT = ('Detect all instances of: main subject. Response must be a JSON array: '
          '[{"label": ..., "bbox": [x1, y1, x2, y2]}, ...]. Coordinates are normalized to [0,1].')

dt = getattr(torch, args.dtype)
t0 = time.time()
proc = AutoProcessor.from_pretrained(args.model)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=dt, low_cpu_mem_usage=True, attn_implementation=args.attn).eval().to(args.device)
load_s = time.time() - t0


def sync():
    if args.device == 'npu':
        torch.npu.synchronize()


rows = []
for rep in range(args.repeat):
    for f in args.frames:
        img = Image.open(FR / f).convert('RGB')
        msgs = [{'role': 'user', 'content': [{'type': 'image', 'image': img},
                                             {'type': 'text', 'text': PROMPT}]}]
        x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                     return_dict=True, return_tensors='pt').to(args.device)
        if 'pixel_values' in x:
            x['pixel_values'] = x['pixel_values'].to(dt)
        sync()
        t = time.time()
        with torch.no_grad():
            out = model.generate(**x, max_new_tokens=args.max_new, do_sample=False)
        sync()
        s = time.time() - t
        n_new = int(out.shape[1] - x['input_ids'].shape[1])
        reply = proc.batch_decode(out[:, x['input_ids'].shape[1]:], skip_special_tokens=True)[0]
        rows.append({'rep': rep, 'frame': f, 's': round(s, 2), 'new_tokens': n_new,
                     'prompt_tokens': int(x['input_ids'].shape[1]), 'reply': reply.strip()[:200]})
        print('CALL', json.dumps(rows[-1], ensure_ascii=False), flush=True)

peak = (torch.npu.max_memory_allocated() / 2**30) if args.device == 'npu' else None
summ = {'device': args.device, 'dtype': args.dtype, 'jit': args.jit, 'attn': args.attn, 'threads': args.threads,
        'load_s': round(load_s, 1), 'peak_gib': peak,
        'first_pass_s': [r['s'] for r in rows if r['rep'] == 0],
        'last_pass_s': [r['s'] for r in rows if r['rep'] == args.repeat - 1]}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps({'summary': summ, 'rows': rows}, ensure_ascii=False, indent=1) + '\n')
print('SUMMARY', json.dumps(summ), flush=True)
