#!/usr/bin/env python3
"""R13 27B garble verdict: NPU fp16 (4-card device_map) vs CPU fp32 forward.

The 27B no-think lane produced repeated-token garbage; single forward on a
fixed prompt decides: weight/dtype fault (CPU fp32 differs from NPU fp16) or
dispatch fault (CPU also garbled — impossible since weights are correct).
"""
import json
import os
import sys

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402

MODEL = "/data/aic/pretrained/Qwen3.5-27B"
PROMPT = "The capital of France is"
rep = {}

sys.path.insert(0, "/data/aic/scripts")
import r13_qwen35_safe_chunk  # noqa: E402

r13_qwen35_safe_chunk.apply()

from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # noqa: E402

tok = AutoProcessor.from_pretrained(MODEL, use_fast=False).tokenizer


def top5(model, dev):
    enc = tok(PROMPT, return_tensors="pt").to(dev)
    with torch.no_grad():
        lg = model(**enc).logits[0, -1, :].float()
    t = lg.topk(5)
    return [tok.decode([i]) for i in t.indices.tolist()], [round(float(v), 1) for v in t.values.tolist()]


if "--cpu-only" not in sys.argv:
    import torch_npu  # noqa: F401
    torch.npu.set_compile_mode(jit_compile=False)
    lim = {i: "22GiB" for i in range(4)}
    m = Qwen3_5ForConditionalGeneration.from_pretrained(
        MODEL, torch_dtype=torch.float16, attn_implementation="eager",
        device_map="auto", max_memory=lim)
    m.eval()
    rep["npu_fp16_tp4"] = top5(m, m.device)
    del m
    torch.npu.empty_cache()

# CPU fp32 golden (single forward, ~2 min)
m = Qwen3_5ForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32)
m.eval()
rep["cpu_fp32"] = top5(m, "cpu")

print(json.dumps(rep, ensure_ascii=False, indent=1))
print("PROBE_DONE")
