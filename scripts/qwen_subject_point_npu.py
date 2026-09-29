"""vllm-ascend teacher point runner on the 910A NPUs.

Same SYSTEM/USER prompt (sha-checked) as scripts/qwen_subject_point.py; runs the
Qwen3-VL-32B teacher through vLLM + vllm-ascend on 4 NPUs instead of the V100.
Greedy decoding, fp16, TP4.

  --mode smoke   videos x keyframes vs the V100 vLLM replies in point_d2/points
  --mode bench   replicate the query set to --bench-n prompts, submit in one
                 batch, report end-to-end queries/s (the distillation-throughput
                 number) and reuse the smoke rows for correctness
"""
import argparse, json, sys, time
from pathlib import Path
import torch, torch_npu  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.qwen_subject_point import SYSTEM, USER, parse, prompt_sha
from PIL import Image


def build_query(args, vid, ref, k, key, ratio):
    rw, rh = ratio['ratio']; W, H = ref['W'], ref['H']
    orient = 'portrait, narrower than the frame' if rw / rh < W / H else 'landscape, wider than the frame'
    text = USER.format(w=W, h=H, rw=rw, rh=rh, orient=orient)
    img = Image.open(args.frames_dir / vid / f'{key}.png').convert('RGB')
    msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'image', 'image': img}, {'type': 'text', 'text': text}]}]
    return msgs, (vid, k, key, ratio)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/data/aic/pretrained/qwen3_vl_32b_instruct')
    ap.add_argument('--frames-dir', type=Path, default=Path('/data/aic/experiments/QWEN32B_OFFICIAL_V2/keyframes_d2'))
    ap.add_argument('--ref-dir', type=Path, default=Path('/data/aic/experiments/QWEN32B_OFFICIAL_V2/point_d2/points'))
    ap.add_argument('--videos', nargs='*', default=['0', '1', '2'])
    ap.add_argument('--max-keys', type=int, default=6)
    ap.add_argument('--mode', default='smoke', choices=['smoke', 'bench'])
    ap.add_argument('--bench-n', type=int, default=200)
    ap.add_argument('--tp', type=int, default=4)
    ap.add_argument('--mem', type=float, default=0.85)
    ap.add_argument('--max-seqs', type=int, default=32)
    ap.add_argument('--max-len', type=int, default=8192)
    ap.add_argument('--enforce-eager', action='store_true')
    ap.add_argument('--no-prefix-cache', action='store_true')
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    assert prompt_sha('point') == '918f0730f5407c7186a3208a9f76662c3121c8ea25475655d8ce3f5dc8acb53d', 'point prompt changed'

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=a.model, dtype='float16', tensor_parallel_size=a.tp,
              gpu_memory_utilization=a.mem, max_model_len=a.max_len,
              max_num_seqs=a.max_seqs, enforce_eager=a.enforce_eager,
              enable_prefix_caching=not a.no_prefix_cache, seed=0)
    sp = SamplingParams(temperature=0, max_tokens=64)
    load_s = time.time() - t0

    refs, queries = [], []
    for vid in a.videos:
        ref = json.loads((a.ref_dir / f'{vid}.json').read_text())
        for k, r in list(ref['ratios'].items()):
            for j, key in list(enumerate(ref['keyframes']))[:a.max_keys]:
                msgs, meta = build_query(a, vid, ref, k, key, r)
                refs.append((meta, r['raw'][j]))
                queries.append(msgs)

    if a.mode == 'bench' and len(queries) < a.bench_n:
        base = list(queries)
        while len(queries) < a.bench_n:
            queries.extend(base[:min(a.bench_n - len(queries), len(base))])

    t = time.time()
    outs = llm.chat(queries, sp)
    gen_s = time.time() - t

    rows = []
    for (meta, raw), out in zip(refs, outs[:len(refs)]):
        vid, k, key, _ = meta
        reply = out.outputs[0].text
        p, s = parse(reply); q, _ = parse(raw)
        d = None if p is None or q is None else ((p['x'] - q['x']) ** 2 + (p['y'] - q['y']) ** 2) ** .5
        rows.append({'vid': vid, 'ratio': k, 'key': key, 'npu': reply, 'v100': raw,
                     'status': s, 'exact': reply.strip() == raw.strip(), 'dist': d})

    n_total = len(outs)
    summ = {'mode': a.mode, 'n': n_total, 'parse_ok': sum(x['status'] == 'ok' for x in rows),
            'exact': sum(x['exact'] for x in rows),
            'mean_dist': None if not rows else sum(x['dist'] for x in rows if x['dist'] is not None) / max(1, sum(x['dist'] is not None for x in rows)),
            'load_s': round(load_s, 1), 'gen_s': round(gen_s, 2),
            'qps': round(n_total / gen_s, 2),
            'cfg': {'tp': a.tp, 'mem': a.mem, 'max_seqs': a.max_seqs, 'max_len': a.max_len,
                    'eager': a.enforce_eager, 'prefix_cache': not a.no_prefix_cache}}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({'summary': summ, 'rows': rows}, ensure_ascii=False, indent=1) + '\n')
    print('SUMMARY', json.dumps(summ), flush=True)


if __name__ == '__main__':
    main()
