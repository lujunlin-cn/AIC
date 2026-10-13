#!/usr/bin/env python3
"""R15 counterfactual template table: greedy-decode the whole pool with the
INIT policy (ckpt_030) and record each segment's template reward. Used by
r15_rollout_vllm.py --cf-table to zero the input-agnostic prior's advantage
(de-prioritization, 2026-10-13). Emits one row per segment per caliber:
  {vid, r}  -- pass --caliber v4|pair (default v4, matches run1-3 reward)
Run under ROLL_PY; standalone job (owns the GPU).
"""
import argparse
import json
import os
import sys

os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r15_common import build_texts, load_frames  # noqa: E402
from r15_grpo_reward import parse_scores, pairwise_reward, reward_v4  # noqa: E402


def _mm_meta(n):
    return {"fps": 1.0, "frames_indices": list(range(n)),
            "total_num_frames": n, "do_sample_frames": False}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--base", default="/data/aic/pretrained/Qwen3.5-4B")
    ap.add_argument("--adapter", required=True,
                    help="the INIT policy whose greedy output defines the "
                         "template (ckpt_030)")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--caliber", choices=["v4", "pair"], default="v4")
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.base, use_fast=False)
    eos_id = proc.tokenizer.eos_token_id
    rows = [json.loads(l) for l in open(args.pool)]
    llm = LLM(model=args.base, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              enable_lora=True, max_lora_rank=16,
              max_num_seqs=args.max_num_seqs,
              limit_mm_per_prompt={"video": 1})
    sp = SamplingParams(temperature=0.0, max_tokens=256,
                        stop_token_ids=[eos_id])
    texts = build_texts(proc, rows)
    prompts = [{"prompt": t,
                "multi_modal_data": {"video": (np.stack(load_frames(r)),
                                                _mm_meta(len(r["files"])))}}
               for t, r in zip(texts, rows)]
    outs = llm.generate(prompts, sp,
                        lora_request=LoRARequest("tmpl", 1, args.adapter))
    n_done = 0
    with open(f"{args.out_prefix}_{args.caliber}.jsonl", "w") as f:
        for o, r in zip(outs, rows):
            sc, fmt = parse_scores(o.outputs[0].text, len(r["gif_hard"]))
            if args.caliber == "pair":
                rr = pairwise_reward(sc, r["target"], fmt)
            else:
                rr, _, _ = reward_v4(sc, r["gif_hard"], None, fmt)
            f.write(json.dumps({"vid": r["vid"], "r": round(float(rr), 6),
                                "parsed": bool(fmt)}) + "\n")
            n_done += 1
    print(f"CF_TABLE_DONE n={n_done} caliber={args.caliber} "
          f"-> {args.out_prefix}_{args.caliber}.jsonl", flush=True)


if __name__ == "__main__":
    main()
