#!/usr/bin/env python3
"""R13 T1c: locate the forward fault — CPU fp32 (golden) vs NPU fp32 vs NPU fp16.

Same prompt, top-5 next tokens each. If CPU is sane and NPU fp32 is sane,
the fault is fp16 precision on the GDN reference path; if NPU fp32 is also
garbled, the GDN PyTorch reference itself mis-executes on torch_npu.
"""
import json
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402

MODEL = "/data/aic/pretrained/Qwen3.5-4B"
PROMPT = "The capital of France is"
rep = {}

from transformers import AutoProcessor  # noqa: E402
from transformers import Qwen3_5ForConditionalGeneration  # noqa: E402

tok = AutoProcessor.from_pretrained(MODEL, use_fast=False).tokenizer


def top5(model, device):
    enc = tok(PROMPT, return_tensors="pt").to(device)
    with torch.no_grad():
        lg = model(**enc).logits[0, -1, :].float()
    t = lg.topk(5)
    return [tok.decode([i]) for i in t.indices.tolist()], [round(float(v), 1) for v in t.values.tolist()]


use_npu = "--no-npu" not in sys.argv

# 1. CPU fp32 golden
m = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32)
m.eval()
rep["cpu_fp32_top5"], rep["cpu_fp32_vals"] = top5(m, "cpu")
del m

if use_npu:
    import torch_npu  # noqa: F401
    torch.npu.set_compile_mode(jit_compile=False)

    # 2. NPU fp32
    m = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float32, attn_implementation="eager").to("npu:0")
    m.eval()
    rep["npu_fp32_top5"], rep["npu_fp32_vals"] = top5(m, "npu:0")
    del m
    torch.npu.empty_cache()

    # 3. NPU fp16
    m = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="eager").to("npu:0")
    m.eval()
    rep["npu_fp16_top5"], rep["npu_fp16_vals"] = top5(m, "npu:0")

print(json.dumps(rep, ensure_ascii=False, indent=1))
print("T1C_DONE")
