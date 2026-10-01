"""Production teacher point extraction, semifinal drop (910A, transformers path).

vllm-ascend cannot initialise its engine on this node
(std::logic_error: basic_string::_S_construct null not valid), so the 32B
teacher runs through transformers with device_map='auto' across NPU 2-5, eager
attention (sdpa fails in the vision tower on 910A) and greedy decoding - the same
path scripts/npu_qwen_smoke.py validates against the V100 vLLM replies.

Same frozen SYSTEM/USER prompt (sha-checked) and the same output schema as the
V100 teacher points, so aic.max_window_path.qwen_centres_interp and
scripts/max_window_release.py consume the result unchanged.

Resumable: one video -> one points/{vid}.json; finished videos are skipped.
--batch >1 batches keyframes of the same video (same prompt, different image).
"""
import argparse, json, sys, time
from pathlib import Path

from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.qwen_subject_point import SYSTEM, USER, parse, prompt_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='/data/aic/pretrained/qwen3_vl_32b_instruct')
    ap.add_argument('--index', type=Path, required=True)
    ap.add_argument('--frames-dir', type=Path, required=True)
    ap.add_argument('--skel-dir', type=Path, default=None,
                    help='keyframe skeletons (points/{vid}.json); defaults to frames-dir/../points')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--nshards', type=int, default=1)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--cards', default='4,5,6,7')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--attn', default='eager', choices=['eager', 'sdpa'])
    a = ap.parse_args()
    import os
    # device visibility must be set before torch_npu is imported
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = a.cards
    import torch, torch_npu  # noqa: E402,F401
    # without this the TBE online compiler recompiles per dynamic shape and the
    # process burns CPU with AICore idle (observed 2026-10-01: 0.01 qps)
    torch.npu.set_compile_mode(jit_compile=False)
    torch.npu.config.allow_internal_format = False

    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    t0 = time.time()
    proc = AutoProcessor.from_pretrained(a.model)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        a.model, dtype=torch.float16, device_map='auto', low_cpu_mem_usage=True,
        attn_implementation=a.attn).eval()
    print(json.dumps({'load_s': round(time.time() - t0, 1), 'attn': a.attn,
                      'devices': sorted({str(p.device) for p in model.parameters()})}), flush=True)
    assert prompt_sha('point') == '918f0730f5407c7186a3208a9f76662c3121c8ea25475655d8ce3f5dc8acb53d', 'point prompt changed'

    skel_dir = a.skel_dir or (a.frames_dir.parent / 'points')
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
        skel = json.loads((skel_dir / f'{vid}.json').read_text())
        kfs = skel['keyframes']
        W, H = skel['W'], skel['H']
        rw, rh = r['targetRatioWH']
        orient = 'portrait, narrower than the frame' if rw / rh < W / H else 'landscape, wider than the frame'
        text = USER.format(w=W, h=H, rw=rw, rh=rh, orient=orient)
        pts, st, raws = [], [], []
        t_v = time.time()
        for s in range(0, len(kfs), a.batch):
            block = kfs[s:s + a.batch]
            t_b = time.time()
            imgs = [Image.open(a.frames_dir / vid / f'{k}.png').convert('RGB') for k in block]
            msgs = [[{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]},
                     {'role': 'user', 'content': [{'type': 'image', 'image': im},
                                                 {'type': 'text', 'text': text}]}] for im in imgs]
            inp = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                           return_dict=True, return_tensors='pt', padding=True)
            inp = {k: (v.to(model.device) if isinstance(v, torch.Tensor) else v) for k, v in inp.items()}
            t_tok = time.time()
            with torch.no_grad():
                out = model.generate(**inp, max_new_tokens=64, do_sample=False)
            print(f'DEBUGB {vid} n={len(block)} wall={time.time() - t_b:.1f} '
                  f'proc_s={t_tok - t_b:.1f} gen_s={time.time() - t_tok:.1f}', flush=True)
            gen = proc.batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)
            for reply in gen:
                p, stat = parse(reply)
                pts.append([p['x'], p['y'], False] if p else [0.5, 0.5, True])
                st.append(stat)
                raws.append(reply)
                n_q += 1
                n_ok += int(stat == 'ok')
        rec = {'video_id': vid, 'W': skel['W'], 'H': skel['H'], 'fps': skel['fps'],
               'step': 1.0, 'densify': 2, 'prompt': 'point',
               'prompt_sha256': prompt_sha('point'), 'keyframes': kfs, 'queried': len(kfs),
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