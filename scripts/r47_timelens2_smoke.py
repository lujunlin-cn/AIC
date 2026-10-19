#!/usr/bin/env python3
"""Real visual smoke for TimeLens2-8B on the 12 public annotation clips.

This deliberately evaluates a single, predeclared generic-highlight prompt.
It writes raw model responses and does not turn them into an official package.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model), torch_dtype=torch.float16, device_map="auto",
        attn_implementation="sdpa",
    )
    processor = AutoProcessor.from_pretrained(str(args.model))
    prompt = (
        "This is a generic video with no user query. Identify every time interval "
        "that is most worth keeping in a short highlight edit. Prefer a concrete "
        "action, outcome, reaction, reveal, or visually informative change over "
        "waiting, setup, or static repetition. Return ALL intervals in seconds "
        "as a JSON array of [start, end] pairs and no other text."
    )
    paths = sorted(args.videos.glob("*.mp4"))
    if args.limit:
        paths = paths[:args.limit]
    out = []
    for path in paths:
        messages = [{"role": "user", "content": [
            {"type": "video", "video": path.resolve().as_uri(), "fps": 2.0,
             "min_pixels": 224 * 224, "max_pixels": 320 * 320,
             "total_pixels": 64 * 320 * 320},
            {"type": "text", "text": prompt},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                              add_generation_prompt=True)
        images, videos, video_kwargs = process_vision_info(
            messages, image_patch_size=16, return_video_kwargs=True,
            return_video_metadata=True)
        if videos is not None:
            videos, metadatas = zip(*videos)
            videos, metadatas = list(videos), list(metadatas)
        else:
            metadatas = None
        inputs = processor(text=text, images=images, videos=videos,
                           video_metadata=metadatas, do_resize=False,
                           return_tensors="pt", **video_kwargs)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v
                  for k, v in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=512,
                                    temperature=0.01, top_p=0.001,
                                    top_k=1, repetition_penalty=1.0)
        prompt_len = inputs["input_ids"].shape[1]
        response = processor.batch_decode(output[:, prompt_len:],
                                           skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
        out.append({"video_id": path.stem, "response": response})
        print(path.name, response, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"model": str(args.model),
                                       "prompt": prompt, "results": out},
                                      ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
