#!/usr/bin/env python3
"""R13 T2: Qwen3.5-4B video frame-scoring smoke on 910A.

Feed 16 uniformly-sampled frames of one semifinal video; ask for per-frame
highlight scores; verify the visual tower + deepstack path on NPU and time it.
"""
import json
import os
import time

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import sys  # noqa: E402

sys.path.insert(0, "/data/aic/scripts")
import r13_qwen35_safe_chunk  # noqa: E402

r13_qwen35_safe_chunk.apply()

import torch  # noqa: E402
import torch_npu  # noqa: F401,E402
import cv2  # noqa: E402

torch.npu.set_compile_mode(jit_compile=False)

MODEL = "/data/aic/pretrained/Qwen3.5-4B"
VIDEO = "/data/aic/semifinal_test/复赛-基于视频大模型的通用视频高光剪辑/video/0.mp4"
N_FRAMES = 16
rep = {"video": VIDEO}

from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration  # noqa: E402

cap = cv2.VideoCapture(VIDEO)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
idxs = [int(round(i * (total - 1) / (N_FRAMES - 1))) for i in range(N_FRAMES)]
frames = []
for i in idxs:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ok, fr = cap.read()
    if ok:
        frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
cap.release()
rep["frames_read"] = len(frames)
rep["total_frames"] = total

proc = AutoProcessor.from_pretrained(MODEL, use_fast=False)
messages = [{"role": "user", "content": [
    {"type": "video", "video": [f for f in frames]},
    {"type": "text", "text": (
        "These are 16 frames sampled uniformly from one video, in order. "
        "Rate each frame's likelihood (0.00-1.00) of appearing in a "
        "user-made highlight GIF of this video. Reply ONLY with a JSON "
        "array of 16 numbers.")},
]}]
t0 = time.time()
text = proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = proc(text=[text], videos=[frames], return_tensors="pt").to("npu:0")
rep["prep_s"] = round(time.time() - t0, 1)
rep["input_tokens"] = int(inputs["input_ids"].shape[1])

model = Qwen3_5ForConditionalGeneration.from_pretrained(
    MODEL, torch_dtype=torch.float16, attn_implementation="eager").to("npu:0")
model.eval()
rep["mem_gb"] = round(torch.npu.memory_allocated() / 2**30, 2)

t0 = time.time()
with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=160, do_sample=False)
dt = time.time() - t0
n_new = int(out.shape[1] - inputs["input_ids"].shape[1])
rep["gen_s"] = round(dt, 1)
rep["tokens_per_s"] = round(n_new / dt, 1)
text_out = tok = proc.tokenizer
rep["out"] = proc.tokenizer.decode(out[0, inputs["input_ids"].shape[1]:],
                                   skip_special_tokens=True)[:400]
rep["mem_gb_peak"] = round(torch.npu.max_memory_allocated() / 2**30, 2)

print(json.dumps(rep, ensure_ascii=False, indent=1))
print("T2_DONE")
