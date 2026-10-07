#!/usr/bin/env python3
"""NPU health probe (verdict-8 gold standard): Qwen2.5-VL-7B single-image
generate on one 910A card. Healthy stack: load ~60-90s, generate ~10-30s.
Hang signature (JIT subsystem broken): zero output for 10+ minutes.

Run with: source /data/aic/tools/ascend_teacher.env && timeout 900 \
  /usr/local/python3.11.15/bin/python3 npu_probe_qwen7b_generate.py
"""
import time

import torch

t0 = time.time()
try:
    import torch_npu  # noqa: F401
except ImportError as e:
    print(f"PROBE_RESULT gen_ok=False err=torch_npu_import:{e}", flush=True)
    raise SystemExit(1)
assert torch.npu.device_count() > 0, "torch_npu silent CPU fallback: device_count==0"
print(f"[probe] torch_npu ok devices={torch.npu.device_count()} {time.time() - t0:.1f}s", flush=True)

from transformers import AutoProcessor  # noqa: E402

MODEL = "/data/aic/pretrained/Qwen2.5-VL-7B-Instruct"
IMAGE = "/data/aic/semifinal_20261001/keyframes/keyframes/0/0.png"

model = None
errs = []
for cls_path in ("Qwen2_5_VLForConditionalGeneration", "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
    try:
        if cls_path.startswith("Qwen"):
            from transformers import Qwen2_5_VLForConditionalGeneration as M
        else:
            import transformers
            M = getattr(transformers, cls_path)
        model = M.from_pretrained(MODEL, torch_dtype=torch.float16, local_files_only=True,
                                  attn_implementation="eager").to("npu")
        print(f"[probe] loaded via {cls_path}", flush=True)
        break
    except Exception as e:  # noqa: BLE001
        errs.append(f"{cls_path}: {type(e).__name__}: {str(e)[:100]}")
if model is None:
    print("PROBE_RESULT gen_ok=False err=model_load:" + " | ".join(errs), flush=True)
    raise SystemExit(1)
model.eval()
# 判定九推理配方：fast processor forkserver × TBE 编译线程 fork 锁死 → use_fast=False；
# 推理 jit_compile=False 实测安全且必要（10-04 禁令仅限训练 InplaceNormal bug）。
torch.npu.set_compile_mode(jit_compile=False)
print(f"[probe] model loaded {time.time() - t0:.1f}s npu={model.device} jit_compile=False", flush=True)

proc = AutoProcessor.from_pretrained(MODEL, local_files_only=True, use_fast=False)
messages = [
    {"role": "user", "content": [
        {"type": "image", "image": IMAGE},
        {"type": "text", "text": "用一句话描述这张图片。"},
    ]}
]
prompt = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
from PIL import Image as PILImage  # noqa: E402
inputs = proc(text=[prompt], images=[PILImage.open(IMAGE).convert("RGB")], return_tensors="pt").to("npu")

t1 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=24, do_sample=False)
gen_s = time.time() - t1
text = proc.batch_decode(out, skip_special_tokens=True)[0]
print(f"PROBE_RESULT gen_ok=True gen_s={gen_s:.1f} total_s={time.time() - t0:.1f}", flush=True)
print("[probe] tail:", text[-120:].replace("\n", " "), flush=True)
