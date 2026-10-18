#!/usr/bin/env python3
"""Build an R69 segment-wise crop package from a cached response file.

The parent temporal predictions are copied byte-for-byte in frame membership.
Only the spatial box is replaced for frames covered by a valid segment query.
Every segment response is parsed and projected to the same legal maximum
target-ratio window used by the original R69 direct-crop package.  Missing or
malformed responses fall back to a supplied whole-video response, then to the
parent box, and are counted in the manifest.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import tempfile
import zipfile
from pathlib import Path


TIME_LENS_PARAMS = 8_767_123_696
CROP_RE = re.compile(r'"crop"\s*:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_crop(response: str, width: int, height: int, ratio: list[float]) -> tuple[list[float] | None, str]:
    match = CROP_RE.search(response or "")
    if not match:
        return None, "missing_crop"
    try:
        xn, yn, wn = (float(value) for value in match.groups())
    except ValueError:
        return None, "non_numeric_crop"
    if not all(math.isfinite(value) for value in (xn, yn, wn)):
        return None, "nonfinite_crop"
    if not (0.0 <= xn <= 1000.0 and 0.0 <= yn <= 1000.0 and 0.0 < wn <= 1000.0):
        return None, "normalized_range"
    rw, rh = map(float, ratio)
    max_w = min(float(width), float(height) * rw / rh)
    raw_w = min(max_w, max(max_w * 0.05, wn / 1000.0 * float(width)))
    raw_h = raw_w * rh / rw
    raw_x = min(max(0.0, xn / 1000.0 * float(width)), float(width) - raw_w)
    raw_y = min(max(0.0, yn / 1000.0 * float(height)), float(height) - raw_h)
    # Preserve the established maximum legal window and transfer the queried
    # center.  This avoids letting a language model silently change budget.
    w, h = max_w, max_w * rh / rw
    cx, cy = raw_x + raw_w / 2.0, raw_y + raw_h / 2.0
    if w < float(width) - 1e-9:
        x, y = min(max(0.0, cx - w / 2.0), float(width) - w), 0.0
    else:
        x, y = 0.0, min(max(0.0, cy - h / 2.0), float(height) - h)
    if w <= 0.0 or h <= 0.0 or x + w > width + 1e-7 or y + h > height + 1e-7:
        return None, "out_of_bounds"
    return [float(x), float(y), float(w)], "ok"


def contiguous_runs(frames: list[int]) -> list[tuple[int, int]]:
    frames = sorted(set(int(frame) for frame in frames))
    runs: list[tuple[int, int]] = []
    for _, group in __import__("itertools").groupby(enumerate(frames), lambda item: item[1] - item[0]):
        values = [value for _, value in group]
        runs.append((values[0], values[-1]))
    return runs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--segments", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--direct-responses", type=Path, default=None)
    args = ap.parse_args()

    parent_rows = read_jsonl(args.parent)
    parent = {str(row["video_id"]): row for row in parent_rows}
    index = {str(row["video_id"]): row for row in read_jsonl(args.index)}
    segment_payload = json.loads(args.segments.read_text(encoding="utf-8"))
    segment_rows = {str(row["video_id"]): row for row in segment_payload.get("results", [])}
    direct_rows: dict[str, dict] = {}
    if args.direct_responses and args.direct_responses.exists():
        direct_payload = json.loads(args.direct_responses.read_text(encoding="utf-8"))
        direct_rows = {str(row["video_id"]): row for row in direct_payload.get("results", [])}
    if set(parent) != set(index):
        raise ValueError("parent/index video coverage mismatch")

    args.output.mkdir(parents=True, exist_ok=False)
    rows: list[dict] = []
    details: list[dict] = []
    status_counts: dict[str, int] = {}
    fallback_frames = 0
    replaced_frames = 0
    for vid in sorted(parent, key=lambda value: int(value) if value.isdigit() else value):
        p = parent[vid]
        meta = index[vid]
        frame_items = [dict(item) for item in p["predictions"]]
        frame_map = {int(item["frame"]): item for item in frame_items}
        seg_meta = segment_rows.get(vid, {})
        queries = {(int(item.get("run_index", 0)), int(item.get("segment_index", 0))): item
                   for item in seg_meta.get("segments", [])}
        direct_response = str(direct_rows.get(vid, {}).get("response", ""))
        direct_crop, direct_status = parse_crop(direct_response, int(meta["width"]), int(meta["height"]),
                                                list(meta["targetRatioWH"]))
        replacement: dict[int, list[float]] = {}
        seg_details: list[dict] = []
        for run_idx, (run_start, run_end) in enumerate(contiguous_runs(list(frame_map))):
            run_len = run_end - run_start + 1
            seg_count = max(1, int(segment_payload.get("segments", 3)))
            for seg_idx in range(min(seg_count, run_len)):
                seg_start = run_start + (seg_idx * run_len) // seg_count
                seg_end = run_start + ((seg_idx + 1) * run_len) // seg_count - 1
                query = queries.get((run_idx, seg_idx))
                response = str(query.get("response", "")) if query else ""
                crop, status = parse_crop(response, int(meta["width"]), int(meta["height"]),
                                          list(meta["targetRatioWH"]))
                fallback = False
                if crop is None and direct_crop is not None:
                    crop, status = direct_crop, f"segment_{status}_direct_fallback"
                    fallback = True
                elif crop is None:
                    status = f"segment_{status}_parent_fallback"
                    fallback = True
                status_counts[status] = status_counts.get(status, 0) + 1
                for frame in range(seg_start, seg_end + 1):
                    if frame not in frame_map:
                        continue
                    if crop is None:
                        fallback_frames += 1
                    else:
                        replacement[frame] = crop
                seg_details.append({"run_index": run_idx, "segment_index": seg_idx,
                                   "frame_start": seg_start, "frame_end": seg_end,
                                   "status": status, "fallback": fallback,
                                   "response": response, "crop": crop})
        out_predictions: list[dict] = []
        for item in frame_items:
            frame = int(item["frame"])
            q = dict(item)
            if frame in replacement:
                q["bboxes"] = replacement[frame]
                if q["bboxes"] != item["bboxes"]:
                    replaced_frames += 1
            out_predictions.append(q)
        rows.append({"video_id": vid, "targetRatioWH": p["targetRatioWH"],
                     "model_size_mb": TIME_LENS_PARAMS / 1_000_000,
                     "model_params_b": TIME_LENS_PARAMS / 1_000_000_000,
                     "predictions": out_predictions})
        details.append({"video_id": vid, "selected_frames": len(out_predictions),
                        "segments": seg_details, "direct_fallback_status": direct_status,
                        "direct_fallback_available": direct_crop is not None})

    prediction_path = args.output / "predictions.jsonl"
    prediction_path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                                      for row in rows), encoding="utf-8")
    zip_path = args.output / f"{args.output.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(prediction_path, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(temp_dir)
        if sha(Path(temp_dir) / "predictions.jsonl") != sha(prediction_path):
            raise RuntimeError("ZIP round-trip mismatch")
    manifest = {
        "submission_id": args.output.name,
        "status": "PACKAGED_NOT_UPLOADED",
        "uploaded": False,
        "official_platform_score": None,
        "parent_submission": "R54_TIMELENS_CENTER_RESIDUAL_V9_B3",
        "mechanism": "TimeLens2 crop query independently per selected temporal segment; R54 frame set retained",
        "temporal_parent_sha256": sha(args.parent),
        "segment_responses_sha256": sha(args.segments),
        "direct_responses_sha256": sha(args.direct_responses) if args.direct_responses else None,
        "video_count": len(rows),
        "prediction_count": sum(len(row["predictions"]) for row in rows),
        "total_parameters": TIME_LENS_PARAMS,
        "model_params_b": TIME_LENS_PARAMS / 1_000_000_000,
        "model_size_mb": TIME_LENS_PARAMS / 1_000_000,
        "prediction_sha256": sha(prediction_path),
        "zip_sha256": sha(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "replaced_frames": replaced_frames,
        "fallback_frames": fallback_frames,
        "segment_status_counts": status_counts,
        "per_video": details,
        "rule_compliance": {
            "official_test_labels_used": False,
            "official_score_used_for_construction": False,
            "total_inference_parameters_le_9b": True,
            "r54_frame_set_preserved": True,
            "segment_queries_are_cached_verbatim": True,
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / f"{zip_path.name}.sha256").write_text(sha(zip_path) + "  " + zip_path.name + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "per_video"},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
