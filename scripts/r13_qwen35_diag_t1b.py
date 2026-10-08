#!/usr/bin/env python3
"""R13 T1b diag: is the garbled output a tokenizer fault or a forward fault?"""
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import torch  # noqa: E402
import torch_npu  # noqa: F402,E402

torch.npu.set_compile_mode(jit_compile=False)

MODEL = "/data/aic/pretrained/Qwen3.5-4B"
rep = {}

from transformers import AutoProcessor  # noqa: E402
from transformers import Qwen3_5ForConditionalGeneration  # noqa: E402

proc = AutoProcessor.from_pretrained(MODEL, use_fast=False)
tok = proc.tokenizer

# 1. tokenizer round-trip
s = "A highlight video should keep the most exciting seconds."
ids = tok(s, return_tensors="pt")["input_ids"][0]
back = tok.decode(ids, skip_special_tokens=False)
rep["roundtrip_ok"] = back.strip() == s.strip()
rep["roundtrip_head"] = back[:120]
rep["tok_type"] = type(tok).__name__
rep["tok_vocab"] = tok.vocab_size

# 2. chat-template prompt sanity
msgs = [{"role": "user", "content": "Say OK."}]
try:
    rendered = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    rep["chat_template_head"] = rendered[:160]
    ct_ids = tok(rendered, return_tensors="pt").to("npu:0")
except Exception as e:  # noqa: BLE001
    rep["chat_template_head"] = f"FAIL {type(e).__name__}: {str(e)[:150]}"
    ct_ids = None

# 3. forward logits sanity
model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="eager",
).to("npu:0")
model.eval()
prompt = "The capital of France is"
enc = tok(prompt, return_tensors="pt").to("npu:0")
with torch.no_grad():
    logits = model(**enc).logits[0, -1, :].float()
top5 = logits.topk(5)
rep["top5_tokens"] = [tok.decode([i]) for i in top5.indices.tolist()]
rep["top5_vals"] = [round(float(v), 2) for v in top5.values.tolist()]
rep["logit_std"] = round(float(logits.std()), 2)

# 4. greedy continuation via chat template (if available)
if ct_ids is not None:
    with torch.no_grad():
        out = model.generate(**ct_ids, max_new_tokens=24, do_sample=False)
    rep["chat_out"] = tok.decode(out[0, ct_ids["input_ids"].shape[1]:],
                                 skip_special_tokens=True)[:200]

print(json.dumps(rep, ensure_ascii=False, indent=1))
print("T1B_DONE")
