#!/usr/bin/env python3
"""Faster resumable R69 segment crop queries using one batched generation/video.

The model is loaded once.  For each video, all segment clips are passed as a
batch so the vision tower sees one launch instead of three serial launches.
The output schema is identical to ``r69_segmented_crop_smoke.py`` and can be
consumed by ``r69_segmented_crop_package.py``.
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


def runs(frames: list[int]) -> list[tuple[int, int]]:
    values = sorted(set(int(x) for x in frames))
    out = []
    for _, group in itertools.groupby(enumerate(values), lambda item: item[1] - item[0]):
        row = [x for _, x in group]
        out.append((row[0], row[-1]))
    return out


def split_run(start: int, end: int, count: int) -> list[tuple[int, int]]:
    n = end - start + 1
    count = max(1, min(count, n))
    return [(start + i * n // count, start + (i + 1) * n // count - 1) for i in range(count)]


def prompt_for(ratio: list[int]) -> str:
    rw, rh = ratio
    return (
        f"This is a temporal segment from a generic video. The target crop aspect ratio is {rw}:{rh}. "
        "Within this segment, identify the main subject or interacting subjects and the best composition "
        "for the valuable action. Return exactly one JSON object and no explanation: "
        '{"crop":[x,y,w]} where x,y,w are normalized integers from 0 to 1000. '
        "x,y are the top-left of a crop and w is its width; height is implied by the requested ratio. "
        "Keep the important subject/action and avoid empty background."
    )


def query_batch(model, processor, path: Path, ratio: list[int], windows: list[dict],
                fps: float, max_pixels: int, max_new_tokens: int) -> list[str]:
    messages = []
    prompt = prompt_for(ratio)
    for window in windows:
        messages.append([{"role": "user", "content": [
            {"type": "video", "video": str(path), "video_start": float(window["start_sec"]),
             "video_end": float(window["end_sec"]), "fps": float(fps),
             "min_pixels": 224 * 224, "max_pixels": int(max_pixels),
             "total_pixels": 64 * int(max_pixels)},
            {"type": "text", "text": prompt},
        ]}])
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    images, videos, kwargs = process_vision_info(
        messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True
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
    responses = processor.batch_decode(output[:, prompt_len:], skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)
    del inputs, output, videos, images, kwargs, metas
    gc.collect()
    torch.cuda.empty_cache()
    return responses


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--video-root", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--segments", type=int, default=3)
    ap.add_argument("--context", type=float, default=0.05)
    ap.add_argument("--fps", type=float, default=1.0)
    ap.add_argument("--max-pixels", type=int, default=50176)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    index = {str(row["video_id"]): row for row in read_jsonl(args.index)}
    parent = {str(row["video_id"]): row for row in read_jsonl(args.parent)}
    old: dict[str, dict] = {}
    if args.output.exists():
        old = {str(row["video_id"]): row for row in json.loads(args.output.read_text()).get("results", [])}
    selected = sorted(set(index) & set(parent), key=lambda value: int(value) if value.isdigit() else value)
    if args.limit:
        selected = selected[:args.limit]
    model = AutoModelForImageTextToText.from_pretrained(
        str(args.model), torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )
    processor = AutoProcessor.from_pretrained(str(args.model))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    prompt = ""
    for vid in selected:
        if vid in old:
            continue
        meta, p = index[vid], parent[vid]
        selected_frames = [int(item["frame"]) for item in p["predictions"]]
        windows: list[dict] = []
        for run_index, (run_start, run_end) in enumerate(runs(selected_frames)):
            run_len = run_end - run_start + 1
            for segment_index, (segment_start, segment_end) in enumerate(split_run(run_start, run_end, args.segments)):
                pad = int(round(run_len * args.context))
                read_start = max(0, segment_start - pad)
                read_end = min(int(meta["frame_count"]) - 1, segment_end + pad)
                windows.append({"run_index": run_index, "segment_index": segment_index,
                                "selected_frame_start": segment_start,
                                "selected_frame_end": segment_end,
                                "read_frame_start": read_start, "read_frame_end": read_end,
                                "start_sec": read_start / float(meta["fps"]),
                                "end_sec": min(float(meta["duration"]), (read_end + 1) / float(meta["fps"]))})
        responses = query_batch(model, processor, args.video_root / f"{vid}.mp4",
                                list(meta["targetRatioWH"]), windows, args.fps,
                                args.max_pixels, args.max_new_tokens)
        if len(responses) != len(windows):
            raise RuntimeError(f"batch response count mismatch for {vid}: {len(responses)} vs {len(windows)}")
        prompt = prompt_for(list(meta["targetRatioWH"]))
        for window, response in zip(windows, responses):
            window["response"] = response
        old[vid] = {"video_id": vid, "targetRatioWH": meta["targetRatioWH"],
                    "fps": float(meta["fps"]), "segments": windows}
        payload = {"model": str(args.model), "segments": args.segments, "context": args.context,
                   "fps": args.fps, "max_pixels": args.max_pixels, "prompt": prompt,
                   "batch_segments": True,
                   "results": [old[key] for key in sorted(old, key=lambda x: int(x))]}
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps({"video_id": vid, "segments": len(windows), "responses": responses}, ensure_ascii=False), flush=True)
    payload = {"model": str(args.model), "segments": args.segments, "context": args.context,
               "fps": args.fps, "max_pixels": args.max_pixels, "prompt": prompt,
               "batch_segments": True,
               "results": [old[key] for key in sorted(old, key=lambda x: int(x))]}
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
