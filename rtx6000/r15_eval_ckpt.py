#!/usr/bin/env python3
"""R15 checkpoint collapse-analysis eval on RTX (vLLM, roll env).

Samples N segments from the RL pool (seed 13), greedy-decodes with the given
adapter, and reports the SFT-v2-style collapse indicators:
  fmt        parse rate
  uniq       unique-output ratio across segments (exact-match on cleaned text)
  pair_r     mean pairwise Pearson between parsed score vectors
             (template collapse drives this toward 1.0; SFT v2 hit 0.92)
  spearman   mean Spearman of pred vs pool target (0.6*teacher+0.4*gif)
  r_mean     mean shaped reward_v4
Same pool slice across adapters = comparable. Run under ROLL_PY.
"""
import argparse
import json
import os
import sys

# vLLM 0.31 flashinfer sampler JIT rejects sm_120/cu13
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r15_common import build_texts, load_frames  # noqa: E402
from r15_grpo_reward import parse_scores, reward_v4, spearman  # noqa: E402


def _mm_meta(n):
    return {"fps": 1.0, "frames_indices": list(range(n)),
            "total_num_frames": n, "do_sample_frames": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--base", default="/data/aic/pretrained/Qwen3.5-4B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--adapter-label", default=None)
    ap.out_label = None
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.20,
                    help="coexists with a running RL trainer")
    ap.add_argument("--max-num-seqs", type=int, default=48,
                    help="must be <= Mamba cache blocks; at low mem-util "
                         "水位 0.18 only ~285 blocks exist (10-13 finding)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--vpb-inject", action="store_true",
                    help="match the RL prompt when the pool carries 'vpb'")
    args = ap.parse_args()
    args.adapter_label = args.adapter_label or os.path.basename(
        os.path.normpath(args.adapter))

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.base, use_fast=False)
    eos_id = proc.tokenizer.eos_token_id
    rows = [json.loads(l) for l in open(args.pool)]
    rng = np.random.default_rng(13)
    idx = rng.permutation(len(rows))[:args.n]
    rows = [rows[i] for i in idx]

    llm = LLM(model=args.base, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              enable_lora=True, max_lora_rank=16,
              max_num_seqs=args.max_num_seqs,
              limit_mm_per_prompt={"video": 1})
    sp = SamplingParams(temperature=0.0, max_tokens=256,
                        stop_token_ids=[eos_id])

    texts = build_texts(proc, rows, vpb_inject=args.vpb_inject)
    prompts = [{"prompt": t,
                "multi_modal_data": {"video": (np.stack(load_frames(r)),
                                                _mm_meta(len(r["files"])))}}
               for t, r in zip(texts, rows)]
    outs = llm.generate(prompts, sp,
                        lora_request=LoRARequest("eval", 1, args.adapter))
    texts_out = [o.outputs[0].text for o in outs]
    parsed = []
    for o, r in zip(outs, rows):
        sc, fmt = parse_scores(o.outputs[0].text, len(r["gif_hard"]))
        parsed.append((sc, fmt))

    fmt_rate = float(np.mean([f for _, f in parsed]))
    cleaned = [t.strip().replace(" ", "") for t in texts_out]
    uniq = len(set(cleaned)) / len(cleaned)
    vecs = [np.asarray(sc, float) for sc, f in parsed if f and sc]
    if len(vecs) >= 2:
        # pad to same length not possible across segments (n differs);
        # compare only same-length pairs (the dominant same-n groups)
        from itertools import combinations
        from collections import defaultdict
        by_n = defaultdict(list)
        for v in vecs:
            by_n[len(v)].append(v)
        prs = []
        for nlen, vs in by_n.items():
            if len(vs) >= 2:
                for a, b in combinations(vs, 2):
                    if np.std(a) > 1e-9 and np.std(b) > 1e-9:
                        prs.append(float(np.corrcoef(a, b)[0, 1]))
        pair_r = float(np.mean(prs)) if prs else float("nan")
    else:
        pair_r = float("nan")
    sps, rms = [], []
    for r, (sc, f) in zip(rows, parsed):
        if not f or sc is None:
            continue
        sps.append(spearman(sc, np.asarray(r["target"], float)))
        sr, _, _ = reward_v4(sc, r["gif_hard"], None, True)
        rms.append(sr)
    # de-template gain (2026-10-13): spearman vs OWN target minus the
    # swap-target baseline (K=8 random other segments). Input-agnostic
    # templates score ~0 here; per-video signal shows up as positive gain.
    rng2 = np.random.default_rng(29)
    tgts = [np.asarray(r["target"], float) for r in rows]
    det = []
    for i, (sc, f) in enumerate(parsed):
        if not f or sc is None:
            continue
        own = spearman(sc, tgts[i])
        sw = []
        for _ in range(8):
            j = int(rng2.integers(len(tgts)))
            while j == i:
                j = int(rng2.integers(len(tgts)))
            sw.append(spearman(sc, tgts[j]))
        det.append(own - float(np.mean(sw)))
    label = args.adapter_label
    m = {"label": label, "n": len(rows), "fmt": round(fmt_rate, 4),
         "uniq": round(uniq, 4), "pair_r": round(pair_r, 4),
         "spearman": round(float(np.mean(sps)), 4) if sps else None,
         "r_mean": round(float(np.mean(rms)), 4) if rms else None,
         "detmpl": round(float(np.mean(det)), 4) if det else None}
    print("METRICS " + json.dumps(m))
    with open(args.out, "a") as f:
        f.write(json.dumps(m) + "\n")


if __name__ == "__main__":
    main()
