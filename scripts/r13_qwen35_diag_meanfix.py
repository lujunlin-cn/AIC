#!/usr/bin/env python3
"""R13 T1h: (1) shape-scan aten.mean.dim on NPU vs CPU; (2) patch all RMS-norm
means to sum/n and re-check full-model logits."""
import json
import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

torch.npu.set_compile_mode(jit_compile=False)
rep = {"mean_scan": {}}

torch.manual_seed(0)
for shape in [(4096, 128), (128, 2560), (128, 128), (2, 64, 32, 128), (256, 512)]:
    x = torch.randn(*shape)
    ref = x.mean(-1, keepdim=True)
    out = x.to("npu:0").mean(-1, keepdim=True).cpu()
    rep["mean_scan"][str(shape)] = round(float((out - ref).abs().max()), 6)

# also test sum on the broken shape
x = torch.randn(4096, 128)
ref = x.mean(-1, keepdim=True)
out_sum = x.to("npu:0").sum(-1, keepdim=True).cpu() / 128
rep["sum_div_scan_4096x128"] = round(float((out_sum - ref).abs().max()), 6)

# --- patch and full-model check ---
import transformers.models.qwen3_5.modeling_qwen3_5 as mm  # noqa: E402
from transformers.activations import ACT2FN  # noqa: E402


def grms_fwd(self, hidden_states, gate):
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.pow(2).sum(-1, keepdim=True) / hidden_states.shape[-1]
    hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
    hidden_states = self.weight * hidden_states.to(input_dtype)
    hidden_states = hidden_states * ACT2FN[self.activation](gate.to(torch.float32))
    return hidden_states.to(input_dtype)


mm.Qwen3_5RMSNormGated.forward = grms_fwd


def rms_norm_patch(self, x):
    return x * torch.rsqrt(x.pow(2).sum(-1, keepdim=True) / x.shape[-1] + self.eps)


mm.Qwen3_5RMSNorm._norm = rms_norm_patch

from transformers import AutoProcessor  # noqa: E402
from transformers import Qwen3_5ForConditionalGeneration  # noqa: E402

MODEL = "/data/aic/pretrained/Qwen3.5-4B"
tok = AutoProcessor.from_pretrained(MODEL, use_fast=False).tokenizer
m = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="eager").to("npu:0")
m.eval()
enc = tok("The capital of France is", return_tensors="pt").to("npu:0")
with torch.no_grad():
    lg = m(**enc).logits[0, -1, :].float()
t = lg.topk(5)
rep["patched_top5"] = [tok.decode([i]) for i in t.indices.tolist()]
rep["patched_vals"] = [round(float(v), 1) for v in t.values.tolist()]

print(json.dumps(rep, ensure_ascii=False, indent=1))
print("T1H_DONE")
