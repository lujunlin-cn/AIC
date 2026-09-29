"""Batched transformers-eager teacher on the 910A — automated throughput tuning.

vllm-ascend is not runnable on this SoC (no ACLNN kernels for ascend910), so
the production path is batched transformers inference.  This script loads the
teacher once, then sweeps batch sizes: for each B it replicates the official
keyframe query pool, runs chunked greedy decode, records queries/s and checks
every reply against the V100 vLLM references (parse + point distance).

  --cards          comma list of physical NPU ids to expose (default 2,3,4,5)
  --batch-sizes    batch sizes to sweep
  --replicate      total queries for B>=4 sweeps (pool is cycled to this)
"""
import argparse, json, os, sys, time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--model', default='/data/aic/pretrained/qwen3_vl_32b_instruct')
ap.add_argument('--frames-dir', type=Path, default=Path('/data/aic/experiments/QWEN32B_OFFICIAL_V2/keyframes_d2'))
ap.add_argument('--ref-dir', type=Path, default=Path('/data/aic/experiments/QWEN32B_OFFICIAL_V2/point_d2/points'))
ap.add_argument('--videos', nargs='*', default=['0', '1', '2'])
ap.add_argument('--max-keys', type=int, default=6)
ap.add_argument('--cards', default='2,3,4,5')
ap.add_argument('--batch-sizes', nargs='*', type=int, default=[1, 2, 4, 8, 16])
ap.add_argument('--replicate', type=int, default=144)
ap.add_argument('--dtype', default='float16', choices=['float16', 'bfloat16'])
ap.add_argument('--output', type=Path, required=True)
args = ap.parse_args()

os.environ['ASCEND_RT_VISIBLE_DEVICES'] = args.cards

import torch, torch_npu  # noqa: E402
from PIL import Image  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.qwen_subject_point import SYSTEM, USER, parse, prompt_sha  # noqa: E402

assert prompt_sha('point') == '918f0730f5407c7186a3208a9f76662c3121c8ea25475655d8ce3f5dc8acb53d', 'point prompt changed'

from transformers import AutoProcessor, Qwen3VLForConditionalGeneration  # noqa: E402

t0 = time.time()
proc = AutoProcessor.from_pretrained(args.model)
proc.tokenizer.padding_side = 'left'
model = Qwen3VLForConditionalGeneration.from_pretrained(
    args.model, dtype=getattr(torch, args.dtype), device_map='auto',
    low_cpu_mem_usage=True, attn_implementation='eager').eval()
print(json.dumps({'load_s': round(time.time() - t0, 1), 'cards': args.cards,
                  'devices': sorted({str(p.device) for p in model.parameters()})}), flush=True)

pool = []  # (msgs, vid, ratio_key, key, ref_raw)
for vid in args.videos:
    ref = json.loads((args.ref_dir / f'{vid}.json').read_text())
    W, H = ref['W'], ref['H']
    for rk, r in ref['ratios'].items():
        rw, rh = r['ratio']
        orient = 'portrait, narrower than the frame' if rw / rh < W / H else 'landscape, wider than the frame'
        text = USER.format(w=W, h=H, rw=rw, rh=rh, orient=orient)
        for j, key in list(enumerate(ref['keyframes']))[:args.max_keys]:
            img = Image.open(args.frames_dir / vid / f'{key}.png').convert('RGB')
            msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
                    {'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': text}]}]
            pool.append((msgs, vid, rk, key, r['raw'][j]))

results = []
for B in args.batch_sizes:
    n = len(pool) if B < 4 else max(len(pool), args.replicate)
    queries = [pool[i % len(pool)] for i in range(n)]
    # warmup per video resolution: first shapes trigger operator compilation and
    # must NOT pollute the timed region (a full compile can take ~9 minutes)
    vids = []
    for q in queries:
        if q[1] not in vids:
            vids.append(q[1])
    for v in vids:
        warm = [q for q in queries if q[1] == v][:B]
        inputs = proc.apply_chat_template([q[0] for q in warm], tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors='pt', padding=True).to(model.device)
        with torch.no_grad():
            model.generate(**inputs, max_new_tokens=64, do_sample=False)
        torch.npu.synchronize()
    t = time.time()
    rows = []
    for i in range(0, n, B):
        chunk = queries[i:i + B]
        inputs = proc.apply_chat_template([q[0] for q in chunk], tokenize=True, add_generation_prompt=True,
                                          return_dict=True, return_tensors='pt', padding=True).to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        torch.npu.synchronize()
        replies = proc.batch_decode(out[:, inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        for (msgs, vid, rk, key, raw), reply in zip(chunk, replies):
            p, s = parse(reply); q, _ = parse(raw)
            d = None if p is None or q is None else ((p['x'] - q['x']) ** 2 + (p['y'] - q['y']) ** 2) ** .5
            rows.append({'vid': vid, 'ratio': rk, 'key': key, 'exact': reply.strip() == raw.strip(),
                         'status': s, 'dist': d})
    dt = time.time() - t
    ds = [x['dist'] for x in rows if x['dist'] is not None]
    rec = {'B': B, 'n': n, 's': round(dt, 1), 'qps': round(n / dt, 3),
           'parse_ok': sum(x['status'] == 'ok' for x in rows),
           'exact': sum(x['exact'] for x in rows),
           'mean_dist': round(sum(ds) / len(ds), 5) if ds else None,
           'max_dist': round(max(ds), 5) if ds else None}
    results.append(rec)
    print('RESULT', json.dumps(rec), flush=True)

best = max(results, key=lambda r: r['qps'])
summ = {'best': best, 'cards': args.cards, 'dtype': args.dtype,
        'all': results, 't_done': time.strftime('%H:%M:%S')}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(summ, ensure_ascii=False, indent=1) + '\n')
print('SUMMARY', json.dumps(summ), flush=True)
