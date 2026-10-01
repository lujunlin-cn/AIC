"""Production teacher point extraction for the semifinal drop (910A, vLLM TP4).

Same SYSTEM/USER prompt (sha-checked) and greedy decoding as the V100 teacher
run, but iterating EVERY keyframe of every video instead of the smoke subset
that scripts/qwen_subject_point_npu.py offers.  Output schema matches the V100
teacher points so aic.max_window_path.qwen_centres_interp and
scripts/max_window_release.py consume it unchanged.

Resumability: one shard = one video slice, each finished video writes its own
points/{vid}.json; a restart skips videos already present.  Long videos are
chunked so a single llm.chat call never holds the whole drop in memory.
"""
import argparse, json, sys, time
from pathlib import Path

import torch, torch_npu  # noqa: F401
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.qwen_subject_point import SYSTEM, USER, parse, prompt_sha
from PIL import Image


def build_query(frames_dir, ref, ratio, key):
    rw, rh = ratio['ratio']
    W, H = ref['W'], ref['H']
    orient = 'portrait, narrower than the frame' if rw / rh < W / H else 'landscape, wider than the frame'
    text = USER.format(w=W, h=H, rw=rw, rh=rh, orient=orient)
    img = Image.open(frames_dir / f'{key}.png').convert('RGB')
    msgs = [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
            {'role': 'user', 'content': [{'type': 'image', 'image': img},
                                         {'type': 'text', 'text': text}]}]
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/data/aic/pretrained/qwen3_vl_32b_instruct')
    ap.add_argument('--index', type=Path, required=True)
    ap.add_argument('--frames-dir', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--tp', type=int, default=4)
    ap.add_argument('--mem', type=float, default=0.85)
    ap.add_argument('--max-seqs', type=int, default=32)
    ap.add_argument('--max-len', type=int, default=8192)
    ap.add_argument('--chunk', type=int, default=96, help='keyframes per llm.chat call')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()
    assert prompt_sha('point') == '918f0730f5407c7186a3208a9f76662c3121c8ea25475655d8ce3f5dc8acb53d', 'point prompt changed'

    from vllm import LLM, SamplingParams
    t0 = time.time()
    llm = LLM(model=a.model, dtype='float16', tensor_parallel_size=a.tp,
              gpu_memory_utilization=a.mem, max_model_len=a.max_len,
              max_num_seqs=a.max_seqs, enforce_eager=True, enable_prefix_caching=False, seed=0)
    sp = SamplingParams(temperature=0, max_tokens=64)
    print(f'TEACHER_MODEL_LOADED {time.time() - t0:.0f}s', flush=True)

    rows = [json.loads(l) for l in a.index.read_text().splitlines() if l.strip()]
    mine = rows[a.shard::a.nshards]
    if a.limit:
        mine = mine[:a.limit]
    pdir = a.output / 'points'
    pdir.mkdir(parents=True, exist_ok=True)
    todo = [r for r in mine if not (pdir / f"{r['video_id']}.json").exists()]
    print(f'TEACHER_SHARD {a.shard}/{a.nshards} todo={len(todo)} of {len(mine)}', flush=True)

    n_q = n_ok = 0
    t_start = time.time()
    for vi, r in enumerate(todo):
        vid = r['video_id']
        skel = json.loads((a.frames_dir / f'{vid}.json').read_text())
        kfs = skel['keyframes']
        ratio = {'ratio': r['targetRatioWH']}
        pts, st, raws = [], [], []
        t_v = time.time()
        for s in range(0, len(kfs), a.chunk):
            block = kfs[s:s + a.chunk]
            queries = [build_query(a.frames_dir / vid, skel, ratio, k) for k in block]
            outs = llm.chat(queries, sp, use_tqdm=False)
            for out in outs:
                reply = out.outputs[0].text
                p, stat = parse(reply)
                pts.append([p['x'], p['y'], False] if p else [0.5, 0.5, True])
                st.append(stat)
                raws.append(reply)
                n_q += 1
                n_ok += int(stat == 'ok')
        rec = {'video_id': vid, 'W': skel['W'], 'H': skel['H'], 'fps': skel['fps'],
               'step': 1.0, 'densify': 2, 'prompt': 'point',
               'prompt_sha256': prompt_sha('point'),
               'keyframes': kfs, 'queried': len(kfs),
               'ratios': {'t': {'ratio': r['targetRatioWH'], 'points': pts, 'status': st,
                                'raw': raws, 'boxes': []}},
               'seconds': round(time.time() - t_v, 2)}
        (pdir / f'{vid}.json').write_text(json.dumps(rec, ensure_ascii=False) + '\n')
        if vi % 5 == 0:
            el = time.time() - t_start
            print(f'DEBUG {vid} ({vi}/{len(todo)}) q={n_q} ok={n_ok} '
                  f'{el:.0f}s qps={n_q / max(el, 1e-9):.3f}', flush=True)
    print('TEACHER_SUMMARY', json.dumps(
        {'shard': a.shard, 'videos': len(todo), 'queries': n_q, 'parse_ok': n_ok,
         'wall_s': round(time.time() - t_start, 1),
         'qps': round(n_q / max(time.time() - t_start, 1e-9), 4)}), flush=True)


if __name__ == '__main__':
    main()