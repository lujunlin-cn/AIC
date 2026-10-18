#!/usr/bin/env python3
"""Query TimeLens2 for segment-local crop centres.

This is the diagnostic companion to R69's single whole-video crop.  The
temporal frame set is read from an existing parent submission; it is never
changed.  Each contiguous selected run is split into a fixed number of equal
frame ranges and the same frozen TimeLens2 checkpoint is asked for one crop
per range.  Responses are cached verbatim, together with the frame range and
seconds passed to the video reader, so a later package builder can reproduce
the projection without querying the model again.

The script intentionally only writes a response cache.  It does not read
official labels or create a submission package.
"""
from __future__ import annotations

import argparse
import gc
import itertools
import json
from pathlib import Path

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def contiguous_runs(frames: list[int]) -> list[tuple[int, int]]:
    frames = sorted(set(int(x) for x in frames))
    out: list[tuple[int, int]] = []
    for _, group in itertools.groupby(enumerate(frames), lambda item: item[1] - item[0]):
        values = [value for _, value in group]
        out.append((values[0], values[-1]))
    return out


def split_run(start: int, end: int, count: int) -> list[tuple[int, int]]:
    n = end - start + 1
    count = max(1, min(int(count), n))
    bounds: list[tuple[int, int]] = []
    for i in range(count):
        left = start + (i * n) // count
        right = start + ((i + 1) * n) // count - 1
        bounds.append((left, right))
    return bounds


def query_one(model, processor, path: Path, ratio: list[int], start_sec: float,
              end_sec: float, fps: float, max_pixels: int, max_new_tokens: int) -> tuple[str, str]:
    rw, rh = ratio
    prompt = (
        f"This is a temporal segment from a generic video. The target crop aspect ratio is {rw}:{rh}. "
        "Within this segment, identify the main subject or interacting subjects and the best composition "
        "for the valuable action. Return exactly one JSON object and no explanation: "
        '{"crop":[x,y,w]} where x,y,w are normalized integers from 0 to 1000. '
        "x,y are the top-left of a crop and w is its width; height is implied by the requested ratio. "
        "Keep the important subject/action and avoid empty background."
    )
    message = [{"role": "user", "content": [
        {"type": "video", "video": str(path), "video_start": float(start_sec),
         "video_end": float(end_sec), "fps": float(fps), "min_pixels": 224 * 224,
         "max_pixels": int(max_pixels), "total_pixels": 64 * int(max_pixels)},
        {"type": "text", "text": prompt},
    ]}]
    text = processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info(
        message, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
    )
    if videos is not None:
        videos, metas = zip(*videos)
        videos, metas = list(videos), list(metas)
    else:
        metas = None
    inputs = processor(text=text, images=images, videos=videos, video_metadata=metas,
                       do_resize=False, return_tensors="pt", cap_pixels_per_frame=True, **kwargs)
    inputs = {key: value.to(model.device) if hasattr(value, "to") else value
              for key, value in inputs.items()}
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=max_new_tokens, temperature=0.01,
                                top_p=0.001, top_k=1, repetition_penalty=1.0)
    prompt_len = inputs["input_ids"].shape[1]
    response = processor.batch_decode(output[:, prompt_len:], skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0]
    del inputs, output, videos, images, kwargs, metas
    gc.collect()
    torch.cuda.empty_cache()
    return response, prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--video-ids", nargs="+", default=None)
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--context", type=float, default=0.05,
                    help="fraction of the selected run added as context on either side")
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=50176)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    index = {str(row["video_id"]): row for row in read_jsonl(args.index)}
    parent = {str(row["video_id"]): row for row in read_jsonl(args.parent)}
    selected = set(args.video_ids) if args.video_ids else set(parent)
    missing = selected - set(index) - set(parent)
    if missing:
        raise ValueError(f"video ids missing from index or parent: {sorted(missing)}")
    selected &= set(index) & set(parent)

    old: dict[str, dict] = {}
    if args.output.exists():
        payload = json.loads(args.output.read_text(encoding="utf-8"))
        old = {str(row["video_id"]): row for row in payload.get("results", [])}

    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model), torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(str(args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompts: set[str] = set()
    for vid in sorted(selected, key=lambda value: int(value) if value.isdigit() else value):
        if vid in old:
            continue
        meta = index[vid]
        parent_row = parent[vid]
        frames = [int(item["frame"]) for item in parent_row["predictions"]]
        fps = float(meta["fps"])
        ranges: list[dict] = []
        for run_idx, (run_start, run_end) in enumerate(contiguous_runs(frames)):
            run_len = run_end - run_start + 1
            for segment_idx, (seg_start, seg_end) in enumerate(split_run(run_start, run_end, args.segments)):
                pad = int(round(run_len * float(args.context)))
                read_start = max(0, seg_start - pad)
                read_end = min(int(meta["frame_count"]) - 1, seg_end + pad)
                start_sec = read_start / fps
                end_sec = min(float(meta["duration"]), (read_end + 1) / fps)
                response, prompt = query_one(
                    model, processor, args.video_root / f"{vid}.mp4", list(meta["targetRatioWH"]),
                    start_sec, end_sec, args.fps, args.max_pixels, args.max_new_tokens
                )
                prompts.add(prompt)
                ranges.append({
                    "run_index": run_idx,
                    "segment_index": segment_idx,
                    "selected_frame_start": seg_start,
                    "selected_frame_end": seg_end,
                    "read_frame_start": read_start,
                    "read_frame_end": read_end,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "response": response,
                })
                print(json.dumps({"video_id": vid, **ranges[-1]}, ensure_ascii=False), flush=True)
        old[vid] = {"video_id": vid, "targetRatioWH": meta["targetRatioWH"], "fps": fps,
                    "segments": ranges}
        args.output.write_text(json.dumps({"model": str(args.model), "segments": args.segments,
                                           "context": args.context, "fps": args.fps,
                                           "max_pixels": args.max_pixels,
                                           "prompt": sorted(prompts)[0] if prompts else "",
                                           "results": [old[key] for key in sorted(old, key=lambda x: int(x))]},
                                          ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.write_text(json.dumps({"model": str(args.model), "segments": args.segments,
                                       "context": args.context, "fps": args.fps,
                                       "max_pixels": args.max_pixels,
                                       "prompt": sorted(prompts)[0] if prompts else "",
                                       "results": [old[key] for key in sorted(old, key=lambda x: int(x))]},
                                      ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
