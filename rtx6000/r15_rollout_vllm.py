#!/usr/bin/env python3
"""R15 RL rollout worker on RTX PRO 6000 (CUDA): vLLM engine, file-relay sync.

Drop-in CUDA counterpart of r13_rl_rollout.py. One card, colocated with the
trainer (r15_train_cuda.py): both processes stay resident, phases alternate
via the adapter/traj file relay (strict per-step sync, same protocol).

What changed vs 910A:
  - HF generate -> vLLM offline LLM. GRPO group sampling uses
    SamplingParams(n=samples): the shared video prefill is computed once per
    segment and reused by the whole group (prefix cache), the single biggest
    rollout win on video-heavy prompts.
  - LoRA hot-swap via LoRARequest(adapter dir) with a monotonically rising
    request id (vLLM reloads on new id). Fallback --lora-mode=reload merges
    the adapter into full weights (venv_train subprocess) and rebuilds the
    engine each step (experimental, ~1-2 min/step overhead).
  - NPU patches deleted: safe_chunk, multinomial CPU route, warper where-fix,
    jit_compile=False. vLLM sampling is its own stack.
Rewards (v4), OraRL advantages, tokenization of the oracle tail and the
traj_step{s}_{w}.jsonl schema are byte-compatible with the 910A run
(r15_common / r15_grpo_reward are verbatim ports).

Run with the vLLM venv (see setup_rtx6000.sh); trainer runs in parallel from
venv_train. Keep the /data/aic directory layout identical to 910A so the
pool's absolute frames_dir paths stay valid.
"""
import argparse
import gc
import json
import os
import subprocess
import sys
import time

# vLLM 0.31 flashinfer sampler JIT rejects sm_120/cu13 (10-13 finding);
# native torch sampler is the production path on RTX PRO 6000
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

def _mm_meta(n):
    """vLLM 0.31 qwen3_vl video metadata: frames are pre-sampled at fps=1.0
    (same semantics as r15_common.build_inputs' video_metadata)."""
    return {"fps": 1.0, "frames_indices": list(range(n)),
            "total_num_frames": n, "do_sample_frames": False}


import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r15_common import (build_texts, load_frames, orarl_advantage,  # noqa: E402
                        parse_scores, reward_v4, truncate_at_eos)
from r15_grpo_reward import pairwise_reward  # noqa: E402


def wait_file(path, timeout=1800.0):
    t0 = time.time()
    while not os.path.exists(path):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"wait_file {path}")
        time.sleep(5)


def frames_np(row):
    fl = load_frames(row)
    return np.stack(fl) if len(fl[0].shape) == 3 else fl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", type=int, default=0)
    ap.add_argument("--n-workers", type=int, default=1)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--base", default="/data/aic/pretrained/Qwen3.5-4B")
    ap.add_argument("--adapter-init", required=True,
                    help="PEFT adapter dir (e.g. ckpt_060); user picks which")
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--segs-per-step", type=int, default=16)
    ap.add_argument("--samples", type=int, default=3,
                    help="on-policy samples per segment (G4 = 3+1 oracle)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temp", type=float, default=1.2,
                    help="2026-10-11 production sampling temperature")
    ap.add_argument("--top-p", type=float, default=0.85)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.45,
                    help="colocated with the trainer: leave headroom "
                         "(0.45 on 84GB D / 0.55 on 96GB are sane starts)")
    ap.add_argument("--max-lora-rank", type=int, default=16,
                    help="r13_sft_lora.py trains r=16")
    ap.add_argument("--lora-mode", choices=["lora", "reload"],
                    default="lora",
                    help="lora: vLLM LoRARequest (preferred); reload: merge "
                         "to full weights + engine rebuild (fallback if "
                         "vLLM lacks Qwen3.5 LoRA modules)")
    ap.add_argument("--train-python", default="/opt/venv_train/bin/python",
                    help="venv used for the merge subprocess in reload mode")
    ap.add_argument("--enforce-eager", action="store_true",
                    help="disable CUDA graphs (use on first bring-up)")
    ap.add_argument("--mm-kwargs", default="{}",
                    help="extra vLLM mm_processor_kwargs JSON")
    ap.add_argument("--reward-mode", choices=["v4", "pair"], default="v4")
    ap.add_argument("--cf-table", default=None,
                    help="template reward table jsonl (vid,r) from "
                         "r15_make_cf_table.py; enables counterfactual "
                         "baseline with --cf-lambda")
    ap.add_argument("--cf-lambda", type=float, default=0.5)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.base, use_fast=False)
    eos_id = proc.tokenizer.eos_token_id

    rows = [json.loads(l) for l in open(args.pool)]
    rows = [r for i, r in enumerate(rows) if i % args.n_workers == args.worker]
    print(f"[w{args.worker}] pool={len(rows)}", flush=True)

    # epoch-segment stream: fresh permutation per epoch so multi-epoch runs
    # re-shuffle instead of dying at the pool tail (823 segs = 51 steps/epoch)
    import numpy as _np
    need = args.steps * args.segs_per_step + args.segs_per_step
    stream_idx, seed = [], 13
    while len(stream_idx) < need:
        stream_idx += list(_np.random.default_rng(seed).permutation(len(rows)))
        seed += 4
    stream = [rows[i] for i in stream_idx]
    print(f"[w{args.worker}] seg stream={len(stream)} "
          f"(~{len(stream)/args.segs_per_step:.0f} steps)", flush=True)

    cf_tab, cf_mean = None, 0.0
    if args.cf_table:
        cf_tab = {}
        for l in open(args.cf_table):
            d = json.loads(l)
            cf_tab[d["vid"]] = d["r"]
        cf_mean = float(np.mean(list(cf_tab.values())))
        print(f"[w{args.worker}] cf table n={len(cf_tab)} "
              f"mean={cf_mean:.4f}", flush=True)

    llm = LLM(
        model=args.base,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
        enable_lora=(args.lora_mode == "lora"),
        max_lora_rank=args.max_lora_rank,
        enforce_eager=args.enforce_eager,
        limit_mm_per_prompt={"video": 1},
        mm_processor_kwargs=json.loads(args.mm_kwargs) or None,
    )
    print(f"[w{args.worker}] vLLM ready (mode={args.lora_mode})", flush=True)

    def engine_for_step(step):
        """Return (llm, lora_request_or_None) for the current adapter."""
        nonlocal llm
        if step == 1:
            ad = args.adapter_init
        else:
            d = os.path.join(args.out, f"adapter_step{step - 1}")
            wait_file(os.path.join(d, "adapter_config.json"))
            ad = d
        if args.lora_mode == "lora":
            return llm, LoRARequest(f"adapter_step{step}", step, ad)
        # reload mode: merge with the training venv, then rebuild the engine
        merged = os.path.join(args.out, f"merged_step{max(1, step - 1)}")
        if not os.path.exists(os.path.join(merged, "config.json")):
            subprocess.run([args.train_python,
                            os.path.join(os.path.dirname(
                                os.path.abspath(__file__)),
                                "r15_merge_adapter.py"),
                            "--base", args.base, "--adapter", ad,
                            "--out", merged], check=True)
        del llm
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        llm = LLM(model=merged,
                  max_model_len=args.max_model_len,
                  gpu_memory_utilization=args.gpu_mem_util,
                  enforce_eager=args.enforce_eager,
                  limit_mm_per_prompt={"video": 1},
                  mm_processor_kwargs=json.loads(args.mm_kwargs) or None)
        # disk guard: merged full weights are ~8GB/step; keep only the
        # current one and the previous (rollback window)
        stale = os.path.join(args.out, f"merged_step{step - 2}")
        if step - 2 >= 1 and os.path.isdir(stale):
            subprocess.run(["rm", "-rf", stale], check=False)
        return llm, None

    def lora_dir(step):
        return args.adapter_init if step == 1 else os.path.join(
            args.out, f"adapter_step{step - 1}")

    sp = SamplingParams(
        n=args.samples, temperature=args.temp, top_p=args.top_p,
        top_k=-1, max_tokens=args.max_new_tokens,
        stop_token_ids=[eos_id])

    for step in range(1, args.steps + 1):
        if step >= 2 and args.lora_mode == "lora":
            # adapter dir is produced by tmp-dir+rename; waiting for
            # adapter_config.json == waiting for the rename to have landed
            d = lora_dir(step)
            wait_file(os.path.join(d, "adapter_config.json"))
            print(f"[w{args.worker}] HOTSWAP step{step} -> {d}", flush=True)
        if step >= 2 and args.lora_mode == "reload":
            engine, lreq = engine_for_step(step)
        else:
            engine, lreq = llm, (LoRARequest(
                f"adapter_step{step}", step, lora_dir(step))
                if args.lora_mode == "lora" else None)

        blk = stream[(step - 1) * args.segs_per_step:
                     (step - 1) * args.segs_per_step + args.segs_per_step]
        t_ro = time.time()
        texts = build_texts(proc, blk)
        prompts = [{"prompt": t,
                    "multi_modal_data": {"video": (frames_np(r), _mm_meta(
                        int(frames_np(r).shape[0])))}}
                   for t, r in zip(texts, blk)]
        outs = engine.generate(prompts, sp, lora_request=lreq)

        traj = []
        for j, r in enumerate(blk):
            n = len(r["gif_hard"])
            gids, r_on, tails, info = [], [], [], []
            sc_keep = None
            for m in range(args.samples):
                tid = list(outs[j].outputs[m].token_ids)
                gid = truncate_at_eos(torch.tensor(tid), eos_id,
                                      proc.tokenizer.pad_token_id or eos_id)
                g = proc.tokenizer.decode(gid, skip_special_tokens=True)
                sc, fmt = parse_scores(g, n)
                if args.reward_mode == "pair":
                    sr, raw, p = pairwise_reward(sc, r["target"], fmt), 0.0, {
                        "k": int(sum(1 for v in (sc or []) if v >= args.tau)),
                        "n_gt": int(sum(r["gif_hard"])), "fmt": int(fmt)}
                else:
                    sr, raw, p = reward_v4(sc, r["gif_hard"],
                                           r.get("iou_frame"), fmt, args.tau)
                r_on.append(sr)
                tails.append(gid.tolist())
                info.append(p)
                if m == 0:
                    sc_keep, g_keep = sc, g
            otxt = json.dumps(r["target"])
            otail = proc.tokenizer(otxt, add_special_tokens=False)[
                "input_ids"] + [eos_id]
            sc_o, fmt_o = parse_scores(otxt, n)
            if args.reward_mode == "pair":
                r_gt, _, p_gt = pairwise_reward(sc_o, r["target"], fmt_o), 0.0, {
                    "k": int(sum(1 for v in (sc_o or []) if v >= args.tau)),
                    "n_gt": int(sum(r["gif_hard"])), "fmt": int(fmt_o)}
            else:
                r_gt, _, p_gt = reward_v4(sc_o, r["gif_hard"],
                                          r.get("iou_frame"), fmt_o, args.tau)
            cf_r = cf_tab.get(r["vid"]) if cf_tab else None
            adv_on, adv_gt = orarl_advantage(
                r_on, r_gt, cf_r=cf_r, cf_lambda=args.cf_lambda,
                cf_mean=cf_mean)
            traj.append({"vid": r["vid"], "tails": tails,
                         "advs": adv_on, "adv_gt": adv_gt,
                         "oracle_tail": otail, "r_on": r_on,
                         "r_gt": r_gt, "parts": info,
                         "scores": sc_keep, "gen_text": g_keep})
        dt = time.time() - t_ro
        fp = os.path.join(args.out, f"traj_step{step}_{args.worker}.jsonl.tmp")
        with open(fp, "w") as f:
            for t in traj:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
        os.replace(fp, fp[:-4])
        print(f"[w{args.worker}] RSTEP {step} segs={len(traj)} wall={dt:.0f}s "
              f"r_on_mean={np.mean([np.mean(t['r_on']) for t in traj]):.3f} "
              f"fmt={np.mean([p['fmt'] for t in traj for p in t['parts']]):.2f}",
              flush=True)
    print(f"[w{args.worker}] ROLLDONE", flush=True)


if __name__ == "__main__":
    main()
