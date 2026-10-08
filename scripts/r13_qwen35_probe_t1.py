#!/usr/bin/env python3
"""R13 T1 probe: Qwen3.5-4B on 910A — GDN (linear_attention) forward verdict.

Judge-nine recipe: use_fast=False, jit_compile=False, eager attention.
Text-only generate is sufficient to hit the 24 linear_attention layers
(3:1 linear:full alternation), so this is the operator verdict for the
whole GDN bet. Visual tower is probed separately in T2.

Run (on 910A host):
  source /data/aic/tools/ascend_teacher.env
  /data/aic/venvs/qwen35/bin/python r13_qwen35_probe_t1.py
"""
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402
import torch_npu  # noqa: F402,E402

torch.npu.set_compile_mode(jit_compile=False)

MODEL = "/data/aic/pretrained/Qwen3.5-4B"
report = {"model": MODEL}

# ---- 0. static check: how is linear_attention implemented? ----
import inspect  # noqa: E402

try:
    import transformers.models.qwen3_5.modeling_qwen3_5 as mm  # noqa: E402
    src = inspect.getsource(mm)
    report["module_import"] = "ok"
    report["uses_fla_lib"] = "from fla" in src or "import fla" in src
    report["gdn_classes"] = [n for n in dir(mm) if ("Delta" in n or "LinearAttention" in n)][:8]
except Exception as e:  # noqa: BLE001
    report["module_import"] = f"FAIL {type(e).__name__}: {str(e)[:300]}"
    print(json.dumps(report, ensure_ascii=False, indent=1))
    print("T1_PROBE_FAILED_AT_IMPORT")
    sys_exit = True
    raise SystemExit(1)

from transformers import AutoProcessor  # noqa: E402
from transformers import Qwen3_5ForConditionalGeneration  # noqa: E402

# ---- 1. load (bf16, eager attn) ----
t0 = time.time()
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.bfloat16, attn_implementation="eager",
).to("npu:0")
model.eval()
report["load_s"] = round(time.time() - t0, 1)
report["mem_gb_after_load"] = round(torch.npu.memory_allocated() / 2**30, 2)

# ---- 2. text generate: the GDN verdict ----
proc = AutoProcessor.from_pretrained(MODEL, use_fast=False)
tok = proc.tokenizer
prompt = ("You score how likely each second of a video is to appear in a "
          "user-generated highlight GIF. In one sentence: what visual cues "
          "make content highlight-worthy?")
ids = tok(prompt, return_tensors="pt").to("npu:0")
t0 = time.time()
try:
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=96, do_sample=False)
    dt = time.time() - t0
    n_new = int(out.shape[1] - ids["input_ids"].shape[1])
    report["text_gen"] = {"s": round(dt, 2), "new_tokens": n_new,
                          "tokens_per_s": round(n_new / dt, 1)}
    report["text_out_head"] = tok.decode(
        out[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)[:300]
    report["mem_gb_peak"] = round(torch.npu.max_memory_allocated() / 2**30, 2)
    verdict = "PASS"
except Exception as e:  # noqa: BLE001
    report["text_gen"] = f"FAIL {type(e).__name__}: {str(e)[:500]}"
    verdict = "FAIL"

report["verdict"] = verdict
print(json.dumps(report, ensure_ascii=False, indent=1))
print("T1_PROBE_DONE" if verdict == "PASS" else "T1_PROBE_FAILED")
