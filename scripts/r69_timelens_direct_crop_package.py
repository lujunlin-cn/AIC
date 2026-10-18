#!/usr/bin/env python3
"""Build the high-risk TimeLens2 direct-spatial package.

TimeLens2 is queried once per complete video for a target-ratio crop.  The
R54 temporal frame set is retained exactly; only the spatial box on those
frames is replaced by the parsed video-level crop.  Invalid or malformed
answers fall back to the R54 box for that video and are counted in the
manifest, so a malformed teacher response cannot create an illegal package.
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
DIRECT_MODEL_MB = TIME_LENS_PARAMS / 1_000_000
DIRECT_MODEL_B = TIME_LENS_PARAMS / 1_000_000_000
CROP_RE = re.compile(r'"crop"\s*:\s*\[\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\]')


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_crop(response: str, width: int, height: int, ratio: list[float], projection: str) -> tuple[list[float] | None, str]:
    """Map normalized [x,y,w] to pixels and enforce legal geometry.

    ``max_window`` keeps the established B3 maximum legal crop size and uses
    the direct model only for the free-axis center. ``raw`` is retained as a
    diagnostic mode but is too unconstrained for the final candidate.
    """
    match = CROP_RE.search(response or "")
    if not match:
        return None, "missing_crop"
    try:
        xn, yn, wn = (float(x) for x in match.groups())
    except ValueError:
        return None, "non_numeric_crop"
    if not all(math.isfinite(x) for x in (xn, yn, wn)):
        return None, "nonfinite_crop"
    if not (0.0 <= xn <= 1000.0 and 0.0 <= yn <= 1000.0 and 0.0 < wn <= 1000.0):
        return None, "normalized_range"
    rw, rh = map(float, ratio)
    max_w = min(float(width), float(height) * rw / rh)
    # The prompt defines x/y/w independently on a 0..1000 source canvas.
    raw_w = min(max_w, max(max_w * 0.05, wn / 1000.0 * float(width)))
    raw_h = raw_w * rh / rw
    raw_x = min(max(0.0, xn / 1000.0 * float(width)), float(width) - raw_w)
    raw_y = min(max(0.0, yn / 1000.0 * float(height)), float(height) - raw_h)
    if projection == "raw":
        w, h, x, y = raw_w, raw_h, raw_x, raw_y
    else:
        # Preserve B3's max-window size and transfer only the model's center.
        w, h = max_w, max_w * rh / rw
        cx, cy = raw_x + raw_w / 2.0, raw_y + raw_h / 2.0
        if w < float(width) - 1e-9:
            x, y = min(max(0.0, cx - w / 2.0), float(width) - w), 0.0
        else:
            x, y = 0.0, min(max(0.0, cy - h / 2.0), float(height) - h)
    if w <= 0.0 or h <= 0.0 or x + w > width + 1e-7 or y + h > height + 1e-7:
        return None, "out_of_bounds"
    return [float(x), float(y), float(w)], "ok"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--responses", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--projection", choices=("max_window", "raw"), default="max_window")
    args = ap.parse_args()

    parent_rows = read_jsonl(args.parent)
    parent = {str(x["video_id"]): x for x in parent_rows}
    index = {str(x["video_id"]): x for x in read_jsonl(args.index)}
    payload = json.loads(args.responses.read_text(encoding="utf-8"))
    response_rows = {str(x["video_id"]): x for x in payload.get("results", [])}
    if set(parent) != set(index):
        raise ValueError("parent/index video coverage mismatch")

    rows: list[dict] = []
    details: list[dict] = []
    parse_counts: dict[str, int] = {}
    fallback_videos = 0
    replaced = 0
    for vid in sorted(parent, key=lambda x: int(x) if x.isdigit() else x):
        p = parent[vid]
        meta = index[vid]
        response = str(response_rows.get(vid, {}).get("response", ""))
        crop, status = parse_crop(response, int(meta["width"]), int(meta["height"]), list(meta["targetRatioWH"]), args.projection)
        parse_counts[status] = parse_counts.get(status, 0) + 1
        by_frame = {int(item["frame"]): item for item in p["predictions"]}
        if crop is None:
            fallback_videos += 1
            out_predictions = [dict(item) for item in p["predictions"]]
        else:
            out_predictions = []
            for item in p["predictions"]:
                q = dict(item)
                q["bboxes"] = crop
                if q["bboxes"] != item["bboxes"]:
                    replaced += 1
                out_predictions.append(q)
        row = {
            "video_id": vid,
            "targetRatioWH": p["targetRatioWH"],
            "model_size_mb": DIRECT_MODEL_MB,
            "model_params_b": DIRECT_MODEL_B,
            "predictions": out_predictions,
        }
        rows.append(row)
        details.append({
            "video_id": vid,
            "status": status,
            "response": response,
            "crop": crop,
            "selected_frames": len(out_predictions),
            "fallback": crop is None,
        })

    args.output.mkdir(parents=True, exist_ok=False)
    pred = args.output / "predictions.jsonl"
    pred.write_text("".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) + "\n" for x in rows), encoding="utf-8")
    zip_path = args.output / (args.output.name + ".zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(pred, "predictions.jsonl")
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(zip_path) as archive:
            archive.extractall(td)
        if sha(Path(td) / "predictions.jsonl") != sha(pred):
            raise RuntimeError("ZIP round-trip mismatch")
    manifest = {
        "submission_id": args.output.name,
        "status": "PACKAGED_NOT_UPLOADED",
        "uploaded": False,
        "official_platform_score": None,
        "parent_submission": "R54_TIMELENS_CENTER_RESIDUAL_V9_B3",
        "mechanism": "TimeLens2 complete-video direct target-ratio crop query; R54 temporal set retained",
        "temporal_parent_sha256": sha(args.parent),
        "responses_sha256": sha(args.responses),
        "response_model": payload.get("model"),
        "response_fps": payload.get("fps"),
        "response_max_pixels": payload.get("max_pixels"),
        "projection": args.projection,
        "video_count": len(rows),
        "prediction_count": sum(len(x["predictions"]) for x in rows),
        "total_parameters": TIME_LENS_PARAMS,
        "model_params_b": DIRECT_MODEL_B,
        "model_size_mb": DIRECT_MODEL_MB,
        "prediction_sha256": sha(pred),
        "zip_sha256": sha(zip_path),
        "zip_bytes": zip_path.stat().st_size,
        "replaced_boxes": replaced,
        "fallback_videos": fallback_videos,
        "parse_status_counts": parse_counts,
        "per_video": details,
        "rule_compliance": {
            "official_test_labels_used": False,
            "official_score_used_for_construction": False,
            "total_inference_parameters_le_9b": True,
            "r54_frame_set_preserved": True,
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output / (zip_path.name + ".sha256")).write_text(sha(zip_path) + "  " + zip_path.name + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "per_video"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
