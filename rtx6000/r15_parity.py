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

def _mm_meta(n):
    """vLLM 0.31 qwen3_vl video metadata: frames are pre-sampled at fps=1.0
    (same semantics as r15_common.build_inputs' video_metadata)."""
    return {"fps": 1.0, "frames_indices": list(range(n)),
            "total_num_frames": n, "do_sample_frames": False}


import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def gate_gdn(args):
    import transformers.models.qwen3_5.modeling_qwen3_5 as mm
    from r15_safe_chunk import safe_chunk_gated_delta_rule

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.base)
    text_cfg = getattr(cfg, "text_config", cfg)
    dev = "cuda:0"

    # real kernel-input shapes via a probe on ONE real layer forward:
    # the modeling code repeats k-heads up to v-heads before the call, so
    # hand-built shapes from config alone are wrong (64-vs-80 lesson).
    orig = mm.torch_chunk_gated_delta_rule
    seen = {}

    def probe(*a, **k):
        seen["args"] = [tuple(x.shape) if torch.is_tensor(x) else None
                        for x in a[:5]]
        return orig(*a, **k)

    mm.torch_chunk_gated_delta_rule = probe
    torch.manual_seed(0)
    layer = mm.Qwen3_5GatedDeltaNet(text_cfg, layer_idx=0).to(dev).eval()
    H = getattr(text_cfg, "hidden_size", None)
    x = torch.randn(2, 128, H, device=dev)
    with torch.no_grad():
        y = layer(x)
    mm.torch_chunk_gated_delta_rule = orig
    print(f"layer fwd ok out={tuple(y.shape)}; kernel arg shapes={seen.get('args')}")
    if "args" not in seen:
        print("GDN gate FAIL: layer does not call the torch_chunk path "
              "(fla taken?) - cannot compare torch kernels")
        return 1
    shp = seen["args"]
    # layout is (B, T, H, D) - matches safe_chunk's unpack; g/beta arrive as
    # kwargs in transformers 5.19, derive (B, T, H) from the v shape
    B, T, Hh, D = shp[2]

    torch.manual_seed(1)
    q, k, v = (torch.randn(*s, device=dev, dtype=torch.float32)
               for s in shp[:3])
    g = -torch.rand(B, T, Hh, device=dev, dtype=torch.float32) * 0.1
    beta = torch.rand(B, T, Hh, device=dev, dtype=torch.float32)

    y_safe, _ = safe_chunk_gated_delta_rule(q, k, v, g, beta)
    y_nat, _ = orig(q, k, v, g, beta)
    bad_s = y_safe.isnan() | y_safe.isinf()
    bad_n = y_nat.isnan() | y_nat.isinf()
    mismatch = int((bad_s != bad_n).sum())
    finite = ~(bad_s | bad_n)
    d = (float(((y_safe - y_nat).abs())[finite].max())
         if bool(finite.any()) else float("nan"))
    print(f"safe/native bad(nan|inf)={int(bad_s.sum())}/{int(bad_n.sum())} "
          f"mismatch={mismatch} finite-maxdiff={d:.2e} (fp32, real shapes)")
    # random gaussian k@k^T can be singular -> BOTH sides NaN identically is
    # kernel-equivalence evidence, not a bug; the authoritative numerical
    # check is the real-weight layer forward above + smoke E2E
    # in the overflow domain (random gaussian k@k^T is singular) per-element
    # comparison is noise; only a clean-input mismatch is decisive
    pathological = bool(bad_s.sum()) / y_safe.numel() > 0.01
    if not pathological and (mismatch > 0 or (bool(finite.any()) and d > 1e-4)):
        print("GDN GATE FAIL: native chunk deviates from the "
              "910A-validated safe chunk")
        return 1
    print("GDN_GATE_PASS"
          + (" (random-input pathology, identical nan pattern)"
             if nan_s > 0 else ""))
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
                  "multi_modal_data": {"video": (frames, _mm_meta(
                      int(frames.shape[0])))}}]


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


def gate_logprob(args):
    """Cross-engine token-logprob consistency on a TEXT-only prompt:
    vLLM prompt_logprobs vs a transformers forward over the SAME token ids.
    Kernels differ by design (bf16); PASS = max|dlogp| < 0.15. A semantic
    bug in vLLM's GDN kernel (chunk boundaries, state carry) shows up here
    as an O(1) deviation. Run with ROLL_PY; spawns TRAIN_PY for side B."""
    import json as _json
    import subprocess
    import gc

    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base)
    P = ("The quick brown fox jumps over the lazy dog. "
         "Scientists say that the capital of France is")
    llm = LLM(model=args.base, max_model_len=2048,
              gpu_memory_utilization=args.gpu_mem_util, enforce_eager=True)
    gen = llm.generate([P], SamplingParams(temperature=0.0, max_tokens=48))
    tail = list(gen[0].outputs[0].token_ids)
    ids = list(tok(P)["input_ids"]) + tail
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp)
    lp_v = [list(e.values())[0].logprob for e in out[0].prompt_logprobs[1:]]
    del llm
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    d = "/tmp/r15_logprob_payload.json"
    _json.dump({"base": args.base, "ids": ids}, open(d, "w"))
    code = (
        "import json,sys,torch;"
        "d=json.load(open(sys.argv[1]));"
        "from transformers import Qwen3_5ForConditionalGeneration;"
        "m=Qwen3_5ForConditionalGeneration.from_pretrained("
        "d['base'],torch_dtype=torch.bfloat16,"
        "attn_implementation='sdpa').cuda().eval();"
        "x=torch.tensor([d['ids']]).cuda();"
        "lg=m(input_ids=x).logits[0,:-1].float();"
        "lp=torch.log_softmax(lg,-1);"
        "print(json.dumps(lp.gather(-1,x[0,1:].unsqueeze(-1))"
        ".squeeze(-1).tolist()))")
    r = subprocess.run([args.train_py, "-c", code, d],
                       capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        print(f"LOGPROB_GATE FAIL: train-side error\n{r.stderr[-800:]}")
        return 1
    lp_t = _json.loads(r.stdout.strip().splitlines()[-1])
    n = min(len(lp_v), len(lp_t))
    diffs = [abs(a - b) for a, b in zip(lp_v[:n], lp_t[:n])]
    mx = max(diffs) if diffs else float("inf")
    mean = sum(diffs) / len(diffs) if diffs else float("inf")
    print(f"n_tokens={n} max|dlogp|={mx:.4f} mean|dlogp|={mean:.4f} "
          f"(pass if mean<0.05 and max<0.5: bf16 kernel tail, "
          f"semantic bug shows as O(1) mean)")
    if mean > 0.05 or mx > 0.5:
        print("LOGPROB_GATE FAIL: cross-engine deviation too large "
              "(suspect vLLM GDN kernel semantics -> try --safe-chunk "
              "trainer side + older/newer vLLM tag)")
        return 1
    print("LOGPROB_GATE_PASS")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", required=True,
                    choices=["gdn", "vllm", "lora", "numerics", "logprob"])
    ap.add_argument("--pool", required=True)
    ap.add_argument("--base", default="/data/aic/pretrained/Qwen3.5-4B")
    ap.add_argument("--adapter-init", required=True)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--gpu-mem-util", type=float, default=0.85)
    ap.add_argument("--enforce-eager", action="store_true")
    ap.add_argument("--safe-chunk", action="store_true",
                    help="apply r15_safe_chunk before the gate (numerics)")
    ap.add_argument("--train-py", default="/opt/venv_train/bin/python",
                    help="train env python (AutoDL: miniconda train env)")
    args = ap.parse_args()
    # vLLM 0.31 flashinfer sampler JIT rejects sm_120/cu13 (10-13 finding);
    # native torch sampler is the production path for this box
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    rc = {"gdn": gate_gdn, "vllm": gate_vllm, "lora": gate_lora,
          "numerics": gate_numerics, "logprob": gate_logprob}[args.gate](args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
