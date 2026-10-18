#!/usr/bin/env python3
"""Build a stability-gated variant of the R69 segmented crop package.

R69 lets every temporal run segment replace the inherited V9 crop.  This
variant only releases a run's segment crops when the three segment centres are
consistent with the complete-video crop and with one another.  Otherwise the
complete-video crop is used for *all* frames in that run.  The gate is fixed
before reading any official score and uses only the cached TimeLens responses.

The default 0.12 thresholds have a concrete interpretation: TimeLens emits
mostly 0.1-quantised normalized coordinates.  A one-step difference is
allowed, while a two-step or larger disagreement is treated as unstable.
The gate is deliberately run-level, so an isolated segment cannot introduce a
single abrupt box jump.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
import tempfile
import zipfile
from pathlib import Path

# Support both ``python scripts/...`` and module invocation from the repo.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from r69_segmented_crop_package import (
    TIME_LENS_PARAMS,
    contiguous_runs,
    parse_crop,
    read_jsonl,
)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def free_axis_center(box: list[float], width: int, height: int,
                     ratio: list[float]) -> float:
    """Return the legal crop centre on its free axis in [0, 1]."""
    x, y, w = (float(v) for v in box)
    rw, rh = (float(v) for v in ratio)
    h = w * rh / rw
    # A target window is either horizontally or vertically movable.  Equal
    # aspect sources have no free axis and are always treated as centre.
    if w < float(width) - 1e-7:
        return (x + w / 2.0) / float(width)
    if h < float(height) - 1e-7:
        return (y + h / 2.0) / float(height)
    return 0.5


def sort_key(value: str):
    return int(value) if value.isdigit() else value


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--segments", type=Path, required=True)
    ap.add_argument("--direct-responses", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-global-delta", type=float, default=0.12)
    ap.add_argument("--max-run-spread", type=float, default=0.12)
    args = ap.parse_args()
    if args.max_global_delta < 0 or args.max_run_spread < 0:
        raise ValueError("gate thresholds must be non-negative")

    parent_rows = read_jsonl(args.parent)
    parent = {str(x["video_id"]): x for x in parent_rows}
    index = {str(x["video_id"]): x for x in read_jsonl(args.index)}
    segment_payload = json.loads(args.segments.read_text(encoding="utf-8"))
    segment_rows = {str(x["video_id"]): x for x in segment_payload.get("results", [])}
    direct_payload = json.loads(args.direct_responses.read_text(encoding="utf-8"))
    direct_rows = {str(x["video_id"]): x for x in direct_payload.get("results", [])}
    if set(parent) != set(index) or set(parent) != set(segment_rows) or set(parent) != set(direct_rows):
        raise ValueError("parent/index/segment/direct coverage mismatch")
    if args.output.exists():
        raise FileExistsError(args.output)

    rows: list[dict] = []
    details: list[dict] = []
    stats = {
        "runs": 0,
        "stable_runs": 0,
        "unstable_runs": 0,
        "segment_frames": 0,
        "global_fallback_frames": 0,
        "changed_frames": 0,
        "changed_videos": 0,
    }
    for vid in sorted(parent, key=sort_key):
        p = parent[vid]
        meta = index[vid]
        width, height = int(meta["width"]), int(meta["height"])
        ratio = list(meta["targetRatioWH"])
        direct_box, direct_status = parse_crop(
            str(direct_rows[vid].get("response", "")), width, height, ratio
        )
        if direct_box is None:
            # This branch is not expected for the current cache, but keeps the
            # package safe if a future direct response is malformed.
            direct_box = None
        frame_items = [dict(item) for item in p["predictions"]]
        frame_map = {int(item["frame"]): item for item in frame_items}
        seg_meta = segment_rows[vid]
        queries = {
            (int(item.get("run_index", 0)), int(item.get("segment_index", 0))): item
            for item in seg_meta.get("segments", [])
        }
        replacement: dict[int, list[float]] = {}
        run_details: list[dict] = []
        video_changed = False
        for run_idx, (run_start, run_end) in enumerate(contiguous_runs(list(frame_map))):
            stats["runs"] += 1
            run_len = run_end - run_start + 1
            seg_count = max(1, int(segment_payload.get("segments", 3)))
            crops: list[list[float] | None] = []
            ranges: list[tuple[int, int]] = []
            responses: list[str] = []
            for seg_idx in range(min(seg_count, run_len)):
                seg_start = run_start + (seg_idx * run_len) // seg_count
                seg_end = run_start + ((seg_idx + 1) * run_len) // seg_count - 1
                q = queries.get((run_idx, seg_idx), {})
                crop, status = parse_crop(str(q.get("response", "")), width, height, ratio)
                crops.append(crop)
                ranges.append((seg_start, seg_end))
                responses.append(str(q.get("response", "")))
            valid = all(crop is not None for crop in crops)
            centers = [free_axis_center(crop, width, height, ratio) for crop in crops if crop is not None]
            global_center = free_axis_center(direct_box, width, height, ratio) if direct_box else None
            global_delta = max((abs(x - global_center) for x in centers), default=float("inf")) if global_center is not None else float("inf")
            spread = max(centers) - min(centers) if centers else float("inf")
            stable = bool(valid and direct_box is not None and
                          global_delta <= args.max_global_delta + 1e-12 and
                          spread <= args.max_run_spread + 1e-12)
            stats["stable_runs" if stable else "unstable_runs"] += 1
            for (seg_start, seg_end), crop in zip(ranges, crops):
                chosen = crop if stable and crop is not None else direct_box
                if chosen is None:
                    continue
                for frame in range(seg_start, seg_end + 1):
                    if frame in frame_map:
                        replacement[frame] = chosen
                        stats["segment_frames"] += 1
            if not stable:
                stats["global_fallback_frames"] += sum(
                    1 for frame in frame_map if run_start <= frame <= run_end
                )
            run_details.append({
                "run_index": run_idx,
                "frame_start": run_start,
                "frame_end": run_end,
                "stable": stable,
                "global_center": global_center,
                "segment_centers": centers,
                "max_global_delta": global_delta,
                "run_spread": spread,
                "segment_crops": crops,
                "responses": responses,
                "direct_status": direct_status,
            })
        out_predictions: list[dict] = []
        for item in frame_items:
            frame = int(item["frame"])
            q = dict(item)
            if frame in replacement and replacement[frame] != item["bboxes"]:
                q["bboxes"] = replacement[frame]
                stats["changed_frames"] += 1
                video_changed = True
            out_predictions.append(q)
        if video_changed:
            stats["changed_videos"] += 1
        rows.append({
            "video_id": vid,
            "targetRatioWH": p["targetRatioWH"],
            "model_size_mb": TIME_LENS_PARAMS / 1_000_000,
            "model_params_b": TIME_LENS_PARAMS / 1_000_000_000,
            "predictions": out_predictions,
        })
        details.append({"video_id": vid, "selected_frames": len(out_predictions), "runs": run_details})

    args.output.mkdir(parents=True, exist_ok=False)
    pred = args.output / "predictions.jsonl"
    pred.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in rows), encoding="utf-8")
    zip_path = args.output / f"{args.output.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(td)
        if sha(Path(td) / "predictions.jsonl") != sha(pred):
            raise RuntimeError("ZIP round-trip mismatch")
    manifest = {
        "submission_id": args.output.name,
        "status": "PACKAGED_NOT_UPLOADED",
        "uploaded": False,
        "official_platform_score": None,
        "parent_submission": "R54_TIMELENS_CENTER_RESIDUAL_V9_B3",
        "mechanism": "R69 segmented crop with run-level stability gate; unstable runs fall back to complete-video crop",
        "gate_policy": {
            "max_global_delta": args.max_global_delta,
            "max_run_spread": args.max_run_spread,
            "coordinate": "legal free-axis normalized center",
            "threshold_rationale": "0.12 permits one 0.1-quantized TimeLens step and rejects larger disagreement",
        },
        "temporal_parent_sha256": sha(args.parent),
        "segment_responses_sha256": sha(args.segments),
        "direct_responses_sha256": sha(args.direct_responses),
        "video_count": len(rows),
        "prediction_count": sum(len(row["predictions"]) for row in rows),
        "total_parameters": TIME_LENS_PARAMS,
        "model_params_b": TIME_LENS_PARAMS / 1_000_000_000,
        "model_size_mb": TIME_LENS_PARAMS / 1_000_000,
        "prediction_sha256": sha(pred),
        "zip_sha256": sha(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "stats": stats,
        "per_video": details,
        "rule_compliance": {
            "official_test_labels_used": False,
            "official_score_used_for_construction": False,
            "total_inference_parameters_le_9b": True,
            "r54_frame_set_preserved": True,
            "segment_responses_are_cached_verbatim": True,
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / f"{zip_path.name}.sha256").write_text(sha(zip_path) + "  " + zip_path.name + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
