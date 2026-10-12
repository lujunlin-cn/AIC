#!/usr/bin/env python3
"""R15 RL trainer on RTX PRO 6000 (CUDA): single card, colocated with rollout.

CUDA counterpart of r13_rl_trainer.py. Same strict file sync: step s waits
for traj_step{s}_{w}.jsonl from every rollout worker, joins pool rows by
vid, accumulates the 16-segment loss (3 on-policy + 1 detached oracle tails
per segment), ONE optimizer update, then publishes the adapter both as a PEFT
directory (adapter_step{s}/ via tmp-dir+rename -- consumed by vLLM
LoRARequest) and as the legacy .pt state_dict (adapter_step{s}.pt).

What changed vs 910A:
  - fp16+eager -> bf16 + attn_implementation="sdpa" (full-attn layers);
    GDN layers use fla triton kernels when flash-linear-attention is
    installed (setup_rtx6000.sh installs it), else the 910A-validated
    safe torch chunk (r15_safe_chunk.apply(), --safe-chunk to force).
  - Single card: plain .to("cuda:0"); no device_map/max_memory juggling,
    no torch_npu, no jit_compile flags, no multinomial/warper patches.
  - --grad-ckpt (use_reentrant=False) available on CUDA; DEFAULT OFF to
    mirror the 910A recipe (there it deadlocked on a per-layer thread leak;
    on CUDA it is safe but changes numerics via dropout-free recompute).
Run with venv_train (see setup_rtx6000.sh).
"""
import argparse
import json
import os
import sys
import time

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r15_common import build_inputs, seq_logprob_loss  # noqa: E402


def wait_files(paths, timeout=1800.0):
    t0 = time.time()
    while not all(os.path.exists(p) for p in paths):
        if time.time() - t0 > timeout:
            raise TimeoutError(f"wait_files {paths}")
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--base", default="/data/aic/pretrained/Qwen3.5-4B")
    ap.add_argument("--adapter-init", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--n-workers", type=int, default=1,
                    help="rollout workers feeding traj_step{s}_{w}.jsonl")
    ap.add_argument("--segs-per-step", type=int, default=16)
    ap.add_argument("--logits-keep", type=int, default=280)
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing (use_reentrant=False); "
                         "default off = 910A-identical numerics")
    ap.add_argument("--safe-chunk", action="store_true",
                    help="force the safe torch GDN chunk even if fla is "
                         "installed (A/B the kernel)")
    args = ap.parse_args()

    if args.safe_chunk:
        import r15_safe_chunk
        r15_safe_chunk.apply()

    dev = "cuda:0"
    assert torch.cuda.is_available(), "no CUDA device visible"
    cap = torch.cuda.get_device_capability(0)
    print(f"trainer device={torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}",
          flush=True)

    rows = {json.loads(l)["vid"]: json.loads(l)
            for l in open(args.pool)}
    os.makedirs(args.out, exist_ok=True)

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from peft import PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict

    proc = AutoProcessor.from_pretrained(args.base, use_fast=False)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to(dev)
    model = PeftModel.from_pretrained(base, args.adapter_init,
                                      is_trainable=True)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    model.print_trainable_parameters()
    print("trainer model ready", flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters()
                             if p.requires_grad], lr=args.lr,
                            weight_decay=0.0)
    eos_id = proc.tokenizer.eos_token_id

    for step in range(1, args.steps + 1):
        paths = [os.path.join(args.out, f"traj_step{step}_{w}.jsonl")
                 for w in range(args.n_workers)]
        wait_files(paths)
        t0 = time.time()
        trajs = []
        for w in range(args.n_workers):
            for l in open(paths[w]):
                trajs.append(json.loads(l))
        assert len(trajs) == args.segs_per_step, \
            f"step {step}: {len(trajs)} segs"
        model.train()
        losses_log = []
        for t in trajs:
            r = rows[t["vid"]]
            inputs, _ = build_inputs(proc, [r], dev)
            prompt_row = {"input_ids": inputs["input_ids"],
                          "attention_mask": inputs["attention_mask"],
                          "pixel_values": inputs.get("pixel_values"),
                          "video_grid_thw": inputs.get("video_grid_thw"),
                          "mm_token_type_ids":
                              inputs.get("mm_token_type_ids")}
            tails = [torch.tensor(x) for x in t["tails"]]
            tails.append(torch.tensor(t["oracle_tail"]))
            advs = list(t["advs"]) + [t["adv_gt"]]
            loss = seq_logprob_loss(model, prompt_row, tails, advs,
                                    eos_id, K=args.logits_keep)
            if loss is not None:
                (loss / float(args.segs_per_step)).backward()
                losses_log.append(float(loss.detach().cpu()))
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1e9)
        opt.step()
        opt.zero_grad()
        dt = time.time() - t0
        n_zero = args.segs_per_step - len(losses_log)

        # publish adapter: PEFT dir (tmp-dir+rename, atomic) for the vLLM
        # LoRARequest path + legacy .pt state_dict (hot-swap verify parity)
        sd = get_peft_model_state_dict(model)
        fp = os.path.join(args.out, f"adapter_step{step}.pt.tmp")
        torch.save(sd, fp)
        os.replace(fp, fp[:-4])
        d_tmp = os.path.join(args.out, f"adapter_step{step}.tmp")
        model.save_pretrained(d_tmp)
        os.replace(d_tmp, d_tmp[:-4])
        peak = torch.cuda.max_memory_allocated() / 2**30
        torch.cuda.reset_peak_memory_stats()
        print(f"TSTEP {step} loss_mean={np.mean(losses_log) if losses_log else float('nan'):.4f} "
              f"zero_adv_segs={n_zero} gnorm={float(gnorm):.3f} "
              f"wall={dt:.0f}s peak_gpu={peak:.1f}G", flush=True)
    print("TRAIN_DONE", flush=True)


if __name__ == "__main__":
    main()
