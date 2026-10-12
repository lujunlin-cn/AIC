#!/usr/bin/env python3
"""R15 RL mining sweep on RTX PRO 6000 (CUDA/vLLM).

CUDA counterpart of r14_rl_mining.py: per-segment rollout difficulty map
over the full pool (temp 1.2 production sampling, reward v4 worker-side).
Same output schema (mining_{w}.jsonl: vid / r_mean / r_spread / same / rs /
raws) so downstream difficulty filtering is unchanged.

On one card this is simply a full-pool pass: n=3 samples per segment share
one video prefill via vLLM prefix reuse; 823 segs was ~hours on NPU HF
generate, expect ~minutes-scale per 100 segs here (verify with RSTEP-style
wall prints).
"""
import argparse
import json
import os
import sys
import time

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r15_common import (build_texts, load_frames, parse_scores,  # noqa: E402
                        reward_v4, truncate_at_eos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--base", default="/data/aic/pretrained/Qwen3.5-4B")
    ap.add_argument("--adapter", default=None,
                    help="optional PEFT adapter dir (e.g. ckpt_060 map)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--temp", type=float, default=1.2)
    ap.add_argument("--top-p", type=float, default=0.85)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85,
                    help="standalone job: no trainer to share with")
    ap.add_argument("--max-lora-rank", type=int, default=16)
    ap.add_argument("--chunk", type=int, default=64,
                    help="segments submitted per engine.generate call")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.base, use_fast=False)
    eos_id = proc.tokenizer.eos_token_id

    rows = [json.loads(l) for l in open(args.pool)]
    rows = [r for i, r in enumerate(rows) if i % args.n_workers == args.worker]
    print(f"[m{args.worker}] segs={len(rows)}", flush=True)

    llm = LLM(model=args.base, max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_mem_util,
              enable_lora=args.adapter is not None,
              max_lora_rank=args.max_lora_rank,
              limit_mm_per_prompt={"video": 1})
    lreq = None
    if args.adapter:
        from vllm.lora.request import LoRARequest
        lreq = LoRARequest("mine", 1, args.adapter)
    sp = SamplingParams(n=args.samples, temperature=args.temp,
                        top_p=args.top_p, top_k=-1,
                        max_tokens=args.max_new_tokens,
                        stop_token_ids=[eos_id])
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, f"mining_{args.worker}.jsonl.tmp")
    final_path = os.path.join(args.out, f"mining_{args.worker}.jsonl")
    t0 = time.time()
    with open(out_path, "w") as f:
        for c0 in range(0, len(rows), args.chunk):
            chunk = rows[c0:c0 + args.chunk]
            texts = build_texts(proc, chunk)
            prompts = [{"prompt": t,
                        "multi_modal_data": {"video": (np.stack(
                            load_frames(r)), 1.0)}}
                       for t, r in zip(texts, chunk)]
            outs = llm.generate(prompts, sp, lora_request=lreq)
            for j, r in enumerate(chunk):
                n = len(r["gif_hard"])
                rs, raws = [], []
                for m in range(args.samples):
                    tid = list(outs[j].outputs[m].token_ids)
                    gid = truncate_at_eos(
                        np.array(tid), eos_id,
                        proc.tokenizer.pad_token_id or eos_id)
                    g = proc.tokenizer.decode(gid, skip_special_tokens=True)
                    sc, fmt = parse_scores(g, n)
                    sr, raw, p = reward_v4(sc, r["gif_hard"],
                                           r.get("iou_frame"), fmt)
                    rs.append(sr)
                    raws.append(raw)
                same = max(rs) - min(rs) < 1e-6
                f.write(json.dumps({
                    "vid": r["vid"], "r_mean": round(float(np.mean(rs)), 4),
                    "r_spread": round(float(max(rs) - min(rs)), 4),
                    "same": same, "rs": [round(x, 4) for x in rs],
                    "raws": [round(x, 4) for x in raws]}) + "\n")
            f.flush()
            done = min(c0 + args.chunk, len(rows))
            print(f"[m{args.worker}] {done}/{len(rows)} "
                  f"wall={time.time()-t0:.0f}s", flush=True)
    os.replace(out_path, final_path)
    print(f"[m{args.worker}] DONE {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
