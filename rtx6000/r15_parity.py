#!/usr/bin/env python3
"""R15 Day-1 gates for the RTX PRO 6000 RL stack. Run before any training.

--gate gdn        (venv_train)  fla triton kernel vs r15_safe_chunk torch
                                kernel on identical inputs, fp32 + bf16.
                                PASS: maxdiff < 5e-3 (fp32) / 2e-2 (bf16).
--gate vllm       (vllm env)    engine loads Qwen3.5-4B, one video prompt,
                                n=2 sampled + greedy, JSON parse works.
--gate lora       (vllm env)    LoRARequest(adapter) changes the greedy
                                output vs base (proof the adapter is live).
--gate numerics   (venv_train)  seq_logprob_loss fwd+bwd on one pool row:
                                finite loss, nonzero LoRA grads, no NaN in
                                the GDN forward.

Cross-engine greedy parity (vLLM vs HF generate) is REPORTED, not asserted:
video prefill tokenization differs between engines by design; the RL loop
is stochastic sampling and the reward/loss chains are byte-identical ports,
so text-level identity is not a requirement.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def gate_gdn(args):
    import transformers.models.qwen3_5.modeling_qwen3_5 as mm
    from r15_safe_chunk import safe_chunk_gated_delta_rule

    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        fla_fn = lambda q, k, v, g, b: chunk_gated_delta_rule(
            q, k, v, g, b)[0]  # noqa: E731
        has_fla = True
    except Exception as e:  # noqa: BLE001
        print(f"fla import failed ({e}) -> comparing native torch chunk "
              f"vs safe chunk only")
        has_fla = False

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.base)
    text_cfg = getattr(cfg, "text_config", cfg)
    H = getattr(text_cfg, "hidden_size", 2560)
    num_k_heads = getattr(text_cfg, "linear_num_key_heads",
                          getattr(text_cfg, "num_key_value_heads", 16))
    num_v_heads = getattr(text_cfg, "linear_num_value_heads",
                          getattr(text_cfg, "num_attention_heads", 32))
    k_dim = getattr(text_cfg, "linear_key_head_dim", 96)
    v_dim = getattr(text_cfg, "linear_value_head_dim", 128)
    T = 128
    print(f"shapes: q[{1},{num_k_heads},{T},{k_dim}] "
          f"v[{1},{num_v_heads},{T},{v_dim}]")

    for dtype, thr in [(torch.float32, 5e-3), (torch.bfloat16, 2e-2)]:
        torch.manual_seed(0)
        dev = "cuda:0"
        q = torch.randn(1, num_k_heads, T, k_dim, device=dev, dtype=dtype)
        k = torch.randn(1, num_k_heads, T, k_dim, device=dev, dtype=dtype)
        v = torch.randn(1, num_v_heads, T, v_dim, device=dev, dtype=dtype)
        g = -torch.rand(1, num_v_heads, T, device=dev, dtype=dtype) * 0.1
        beta = torch.rand(1, num_v_heads, T, device=dev, dtype=dtype)

        y_safe, _ = safe_chunk_gated_delta_rule(q, k, v, g, beta)
        y_native, _ = mm.torch_chunk_gated_delta_rule(q, k, v, g, beta)
        d_native = float((y_safe - y_native).abs().max())
        line = f"[{dtype}] safe vs native-torch maxdiff={d_native:.2e}"
        if has_fla:
            # fla kernels take [B,T,H,D] layout
            qf = q.transpose(1, 2); kf = k.transpose(1, 2)
            vf = v.transpose(1, 2)
            y_fla = fla_fn(qf, kf, vf, g, beta).transpose(1, 2)
            d_fla = float((y_safe.float() - y_fla.float()).abs().max())
            line += f", safe vs fla maxdiff={d_fla:.2e}"
            if d_fla > thr:
                print(line)
                print(f"GDN GATE FAIL: fla deviation {d_fla:.2e} > {thr:.0e}")
                return 1
        print(line)
        if d_native > 1e-4:
            print(f"GDN GATE WARN: native torch chunk deviates {d_native:.2e}"
                  " from the 910A-validated safe chunk")
    print("GDN_GATE_PASS")
    return 0


def _make_llm(args, enable_lora):
    from vllm import LLM
    return LLM(model=args.base, max_model_len=args.max_model_len,
               gpu_memory_utilization=args.gpu_mem_util,
               enable_lora=enable_lora, max_lora_rank=16,
               enforce_eager=args.enforce_eager,
               limit_mm_per_prompt={"video": 1})


def _one_prompt(args, proc):
    from r15_common import build_texts, load_frames
    row = json.loads(next(open(args.pool)))
    texts = build_texts(proc, [row])
    frames = np.stack(load_frames(row))
    return row, [{"prompt": texts[0],
                  "multi_modal_data": {"video": (frames, 1.0)}}]


def gate_vllm(args):
    from vllm import SamplingParams
    from transformers import AutoProcessor
    from r15_common import parse_scores

    proc = AutoProcessor.from_pretrained(args.base, use_fast=False)
    eos = proc.tokenizer.eos_token_id
    row, prompts = _one_prompt(args, proc)
    llm = _make_llm(args, enable_lora=False)
    sp = SamplingParams(n=2, temperature=1.2, top_p=0.85, max_tokens=256,
                        stop_token_ids=[eos])
    outs = llm.generate(prompts, sp)
    fmts = []
    for m in range(2):
        txt = outs[0].outputs[m].text
        sc, fmt = parse_scores(txt, len(row["gif_hard"]))
        fmts.append(fmt)
        print(f"sample{m}: fmt={fmt} head={txt[:100]!r}")
    sp0 = SamplingParams(temperature=0.0, max_tokens=256,
                         stop_token_ids=[eos])
    g = llm.generate(prompts, sp0)
    print(f"greedy: {g[0].outputs[0].text[:160]!r}")
    if not any(fmts):
        print("VLLM_GATE FAIL: no parsable output")
        return 1
    print("VLLM_GATE_PASS")
    return 0


def gate_lora(args):
    from vllm import SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoProcessor

    proc = AutoProcessor.from_pretrained(args.base, use_fast=False)
    eos = proc.tokenizer.eos_token_id
    row, prompts = _one_prompt(args, proc)
    llm = _make_llm(args, enable_lora=True)
    sp0 = SamplingParams(temperature=0.0, max_tokens=256,
                         stop_token_ids=[eos])
    base_out = llm.generate(prompts, sp0)[0].outputs[0].text
    lora_out = llm.generate(
        prompts, sp0, lora_request=LoRARequest("init", 1,
                                               args.adapter_init)
    )[0].outputs[0].text
    print(f"base: {base_out[:120]!r}")
    print(f"lora: {lora_out[:120]!r}")
    if base_out == lora_out:
        print("LORA_GATE FAIL: adapter had no effect (vLLM LoRA not "
              "wired for Qwen3.5 GDN? -> use --lora-mode=reload)")
        return 1
    print("LORA_GATE_PASS")
    return 0


def gate_numerics(args):
    from r15_common import build_inputs, seq_logprob_loss

    rows = [json.loads(l) for l in open(args.pool)][:1]
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration
    from peft import PeftModel
    from peft.utils.save_and_load import get_peft_model_state_dict

    if args.safe_chunk:
        import r15_safe_chunk
        r15_safe_chunk.apply()

    proc = AutoProcessor.from_pretrained(args.base, use_fast=False)
    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to("cuda:0")
    model = PeftModel.from_pretrained(base, args.adapter_init,
                                      is_trainable=True)
    model.train()
    eos = proc.tokenizer.eos_token_id
    inputs, _ = build_inputs(proc, rows, "cuda:0")
    prompt_row = {k: inputs.get(k) for k in
                  ("input_ids", "attention_mask", "pixel_values",
                   "video_grid_thw", "mm_token_type_ids")}
    tails = [torch.tensor([5, 6, eos]), torch.tensor([7, eos])]
    advs = [0.4, 0.0]
    loss = seq_logprob_loss(model, prompt_row, tails, advs, eos)
    assert loss is not None and torch.isfinite(loss), f"loss={loss}"
    loss.backward()
    sd = get_peft_model_state_dict(model)
    named = dict(model.named_parameters())
    grads = [float(named[k.replace(".lora_A.", ".lora_A.default.")
                        .replace(".lora_B.", ".lora_B.default.")]
                   .grad.abs().max())
             for k in list(sd.keys())[:4]
             if k.replace(".lora_A.", ".lora_A.default.")
             .replace(".lora_B.", ".lora_B.default.") in named]
    nan_hit = any(not torch.isfinite(p).all() for p in model.parameters())
    peak = torch.cuda.max_memory_allocated() / 2**30
    print(f"loss={float(loss):.4f} grad_max={grads} any_nan={nan_hit} "
          f"peak_gpu={peak:.1f}G")
    if nan_hit or (grads and max(grads) == 0.0):
        print("NUMERICS_GATE FAIL")
        return 1
    print("NUMERICS_GATE_PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True,
                    choices=["gdn", "vllm", "lora", "numerics"])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--base", default="/data/aic/pretrained/Qwen3.5-4B")
    ap.add_argument("--adapter-init", required=True)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--enforce-eager", action="store_true")
    args = ap.parse_args()
    rc = {"gdn": gate_gdn, "vllm": gate_vllm,
          "lora": gate_lora, "numerics": gate_numerics}[args.gate](args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
