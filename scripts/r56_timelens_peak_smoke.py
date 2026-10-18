#!/usr/bin/env python3
"""Run a narrow outcome/peak TimeLens2 prompt on public annotation clips."""
from __future__ import annotations
import argparse, json, gc
from pathlib import Path
import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

PROMPT = (
    "This is a generic video with no user query. Find the most valuable "
    "moments for a short highlight edit. Focus on the decisive action, "
    "visible outcome, reveal, reaction, or completed state; omit setup, "
    "waiting, and static repetition. Return several narrow time intervals "
    "around the decisive moments, with boundaries close to the actual "
    "outcome. Return only a JSON array of [start, end] pairs in seconds."
)

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--videos", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=320 * 320)
    args = ap.parse_args()
    model = AutoModelForImageTextToText.from_pretrained(str(args.model), torch_dtype=torch.float16,
                                                         device_map="auto", attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained(str(args.model))
    paths = sorted(args.videos.glob("*.mp4"))
    if args.limit:
        paths = paths[:args.limit]
    out = []
    for path in paths:
        messages = [{"role": "user", "content": [
            {"type": "video", "video": path.resolve().as_uri(), "fps": 2.0,
             "min_pixels": 112 * 112, "max_pixels": args.max_pixels,
             "total_pixels": 64 * args.max_pixels},
            {"type": "text", "text": PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images, videos, kwargs = process_vision_info(messages, image_patch_size=16,
                                                      return_video_kwargs=True,
                                                      return_video_metadata=True)
        if videos is not None:
            videos, metas = zip(*videos); videos, metas = list(videos), list(metas)
        else:
            metas = None
        inputs = processor(text=text, images=images, videos=videos, video_metadata=metas,
                           do_resize=False, return_tensors="pt", cap_pixels_per_frame=True, **kwargs)
        inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=256, temperature=0.01,
                                    top_p=0.001, top_k=1, repetition_penalty=1.0)
        plen = inputs["input_ids"].shape[1]
        response = processor.batch_decode(output[:, plen:], skip_special_tokens=True,
                                           clean_up_tokenization_spaces=False)[0]
        out.append({"video_id": path.stem, "response": response})
        print(path.name, response, flush=True)
        del inputs, output, videos, images, kwargs, metas
        gc.collect(); torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"model": str(args.model), "prompt": PROMPT,
                                       "fps": 2.0, "max_pixels": args.max_pixels,
                                       "results": out}, ensure_ascii=False, indent=2) + "\n")

if __name__ == "__main__":
    main()
