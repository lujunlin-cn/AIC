#!/usr/bin/env python3
"""R15 helper: merge a PEFT LoRA into full weights (reload-mode fallback).

Only used by r15_rollout_vllm.py --lora-mode=reload when vLLM cannot inject
LoRA into the Qwen3.5 GDN modules. Run inside venv_train.
"""
import argparse
import os
import sys

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="/data/aic/pretrained/Qwen3.5-4B")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import Qwen3_5ForConditionalGeneration
    from peft import PeftModel

    base = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.base, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa")
    model = PeftModel.from_pretrained(base, args.adapter, is_trainable=False)
    model = model.merge_and_unload()
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    print(f"MERGED {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
