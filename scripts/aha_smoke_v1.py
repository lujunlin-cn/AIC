#!/usr/bin/env python3
"""AHA smoke v1: base + LoRA merge + three-head existence check + text generate.

Verdict-9 recipe: eager attn, fp16, inference jit_compile=False, NPU card pinned.
Env: /data/aic/venvs/aha49 (transformers 4.49.0 + peft 0.12.0 + torch 2.9.0).

Run: PYTHONPATH=/data/aic/tools/Aha--main:/data/aic/tools/LLaVA-NeXT-main \
  timeout 900 /data/aic/venvs/aha49/bin/python aha_smoke_v1.py
"""
import sys
import time

sys.path.insert(0, "/data/aic/tools/Aha--main")
sys.path.insert(0, "/data/aic/tools/LLaVA-NeXT-main")

import torch  # noqa: E402
import types  # noqa: E402

# bitsandbytes is import-only (quantization unused in this smoke); stub it on aarch64.
try:
    import bitsandbytes  # noqa: F401
except ImportError:
    stub = types.ModuleType("bitsandbytes")
    stub.nn = types.ModuleType("bitsandbytes.nn")
    stub.nn.Linear4bit = object
    stub.functional = types.ModuleType("bitsandbytes.functional")
    import importlib.machinery as _im

    for m in (stub, stub.nn, stub.functional):
        m.__spec__ = _im.ModuleSpec(m.__name__, None)
        m.__path__ = []
    sys.modules["bitsandbytes"] = stub
    sys.modules["bitsandbytes.nn"] = stub.nn
    sys.modules["bitsandbytes.functional"] = stub.functional
    print("[smoke] bitsandbytes stubbed", flush=True)

t0 = time.time()
try:
    import torch_npu  # noqa: F401
except ImportError as e:
    print(f"PROBE_RESULT ok=False err=torch_npu:{e}", flush=True)
    raise SystemExit(1)
assert torch.npu.device_count() > 0, "no NPU"
torch.npu.set_compile_mode(jit_compile=False)
print(f"[smoke] npu ok {time.time() - t0:.1f}s", flush=True)

from transformers import AutoTokenizer  # noqa: E402

from models.live_llava.video_head_live_llava_qwen import (  # noqa: E402
    VideoHeadLiveLlavaQwenConfig,
    VideoHeadLiveLlavaQwenForCausalLM,
)

BASE = "/data/aic/pretrained/llava-onevision-qwen2-7b-ov"
ADAPTER = "/data/aic/pretrained/aha_weights"

model = VideoHeadLiveLlavaQwenForCausalLM.from_pretrained(
    BASE,
    config=VideoHeadLiveLlavaQwenConfig.from_pretrained(BASE),
    torch_dtype=torch.float16,
    attn_implementation="eager",
).to("npu")
model.eval()
print(f"[smoke] base loaded {time.time() - t0:.1f}s", flush=True)

from peft import PeftModel  # noqa: E402

model = PeftModel.from_pretrained(model, ADAPTER, is_trainable=False)
model.eval()
print(f"[smoke] adapter merged-loaded {time.time() - t0:.1f}s", flush=True)

names = [n for n, _ in model.named_parameters()]
heads = {h: sum(1 for n in names if h in n) for h in ("relevance_head", "informative_head", "uncertainty_head")}
n_params = sum(p.numel() for p in model.parameters())
nan_cnt = sum(int(torch.isnan(p.float()).sum()) for p in list(model.parameters())[::50])
print(f"[smoke] params={n_params / 1e9:.2f}B heads={heads} nan_sample={nan_cnt}", flush=True)

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=False)
ids = tokenizer("A person walking a dog. The most exciting moment is", return_tensors="pt").input_ids.to("npu")
t1 = time.time()
with torch.no_grad():
    out = model(input_ids=ids, labels=ids.clone())  # AHA puts logits into tuple only when labels is not None
    cand = out if isinstance(out, tuple) else (out,)
    logits = next(t for t in cand if isinstance(t, torch.Tensor) and t.ndim >= 2)
fwd_s = time.time() - t1
finite = bool(torch.isfinite(logits.float()).all())
print(f"PROBE_RESULT ok=True fwd_s={fwd_s:.1f} total_s={time.time() - t0:.1f} "
      f"logits_shape={tuple(logits.shape)} finite={finite} heads={heads}", flush=True)
