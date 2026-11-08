"""Public dataset readers that preserve source, split, ratio, and label meaning."""

from __future__ import annotations

import ast
import csv
import json
import math
from pathlib import Path
from typing import Iterator, Sequence
from zipfile import ZipFile

from aic.scale_geometry import transpose_xywh


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"JSONL row must be an object at {path}:{line_number}")
            rows.append(row)
    return rows


def attach_frozen_positions(label_rows: Sequence[dict], position_rows: Sequence[dict]) -> list[dict]:
    """Join public labels to exact P1 source/frame centers without split mixing."""
    positions = {}
    for row in position_rows:
        if row.get("split") not in {"train", "val"}:
            raise ValueError("P1 center cache may contain train/val rows only")
        key = (str(row.get("dataset_source")), str(row.get("source_id")), str(row.get("image_id")), tuple(row.get("target_ratio", [])))
        if key in positions:
            raise ValueError(f"duplicate P1 center cache key: {key}")
        positions[key] = row
    result = []
    used = set()
    for label in label_rows:
        if label.get("split") not in {"train", "val"}:
            raise ValueError("position join accepts train/val labels only")
        key = (str(label.get("dataset_source")), str(label.get("source_id")), str(label.get("image_id")), tuple(label.get("target_ratio", [])))
        if key not in positions:
            raise ValueError(f"missing P1 center for labeled sample: {key}")
        position = positions[key]
        for field in ("split", "target_ratio") + tuple(k for k in ("image_sha256", "image_width", "image_height") if k in label):
            if label.get(field) != position.get(field):
                raise ValueError(f"label/P1 center {field} mismatch for {key}")
        center = position.get("center_xy")
        if not isinstance(center, (list, tuple)) or len(center) != 2:
            raise ValueError(f"invalid P1 center for {key}")
        joined = dict(label)
        joined["center_xy"] = list(center)
        joined["position_source"] = "P1_frozen"
        joined["image_width"] = int(position["image_width"])
        joined["image_height"] = int(position["image_height"])
        for field in ("parent_identity_sha256", "p1_aic_bbox_xyw", "position_kind", "position_cache_sha256",
                      "position_protocol_sha256", "selected_frames_sha256", "path_cache_sha256",
                      "qp_floor", "spatial_entrypoint_sha256", "parent_execution_context", "frame_pts"):
            if field in position:
                joined[field] = position[field]
        result.append(joined)
        used.add(key)
    unused = set(positions) - used
    if unused:
        raise ValueError(f"P1 center cache has {len(unused)} unused rows; audit manifest coverage")
    return result


def normalize_train_val_source_split(split_by_source: dict[str, str]) -> dict[str, str]:
    """Normalize source-level train/dev aliases and reject held-out/test rows."""
    if not isinstance(split_by_source, dict) or not split_by_source:
        raise ValueError("a nonempty source-level split manifest is required")
    normalized = {}
    for source_id, split in split_by_source.items():
        source_id = str(source_id)
        if not source_id:
            raise ValueError("source split manifest contains an empty source id")
        if split == "train":
            normalized[source_id] = "train"
        elif split in {"val", "validation", "dev"}:
            normalized[source_id] = "val"
        else:
            raise ValueError(f"LIVE source split must be train or val/dev, got {split!r}")
    return normalized


def live_youtube_crop_rows(
    csv_path: str | Path,
    split_by_source: dict[str, str],
) -> Iterator[dict]:
    """Parse sparse LIVE labels; caller must supply an audited source split."""
    split_by_source = normalize_train_val_source_split(split_by_source)
    seen_sources = set()
    with Path(csv_path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not {"video"}.issubset(reader.fieldnames):
            raise ValueError("unexpected LIVE-YT-VC CSV schema")
        frame_fields = [name for name in reader.fieldnames if name.startswith("frame")]
        for row in reader:
            source_id = str(row["video"])
            if source_id not in split_by_source:
                raise ValueError(f"LIVE source {source_id} is absent from source split manifest")
            if source_id in seen_sources:
                raise ValueError(f"duplicate LIVE source row: {source_id}")
            seen_sources.add(source_id)
            split = split_by_source[source_id]
            for field in frame_fields:
                raw = row.get(field)
                if not raw:
                    continue
                try:
                    box = ast.literal_eval(raw)
                except (ValueError, SyntaxError) as exc:
                    raise ValueError(f"invalid LIVE crop for {source_id}/{field}") from exc
                if not isinstance(box, dict) or not {"left", "right", "top", "bottom"}.issubset(box):
                    raise ValueError(f"invalid LIVE xyxy crop for {source_id}/{field}")
                left, top = float(box["left"]), float(box["top"])
                right, bottom = float(box["right"]), float(box["bottom"])
                if not all(math.isfinite(v) for v in (left, top, right, bottom)) or right <= left or bottom <= top:
                    raise ValueError(f"degenerate LIVE crop for {source_id}/{field}")
                frame_id = int(field.removeprefix("frame"))
                yield {
                    "source_id": source_id,
                    "image_id": f"{source_id}:{frame_id}",
                    "frame_id": frame_id,
                    "dataset_source": "LIVE_YT_VC",
                    "split": split,
                    "target_ratio": [9, 16],
                    "gt_crop_xywh": [left, top, right - left, bottom - top],
                    "label_semantics": "human_video_frame_crop_xyxy_pixels_sparse_6_frame_step",
                }
    missing = set(split_by_source) - seen_sources
    if missing:
        raise ValueError(f"LIVE split manifest references {len(missing)} missing sources")


def parse_ratio_key(ratio: Sequence[int]) -> str:
    if len(ratio) != 2 or any(not isinstance(v, int) or isinstance(v, bool) or v <= 0 for v in ratio):
        raise ValueError("target ratio must contain two positive integers")
    return f"{ratio[0]}:{ratio[1]}"


def gnmc_training_row(
    annotation: dict,
    *,
    split: str,
    image_width: int,
    image_height: int,
    target_ratio: tuple[int, int],
    transpose: bool = False,
) -> dict:
    """Adapt one GNMC annotation; the public test split is never exposed here."""
    if split not in {"train", "val", "validation"}:
        raise ValueError("GNMC test annotations are not available to training adapters")
    split = "val" if split == "validation" else split
    original_ratio = tuple(target_ratio)
    crop_ratio = target_ratio
    boxes = annotation.get("crop_bboxes")
    if not isinstance(boxes, dict):
        raise ValueError("GNMC crop_bboxes missing")
    if transpose:
        crop_ratio = (target_ratio[1], target_ratio[0])
    key = parse_ratio_key(crop_ratio)
    if key not in boxes:
        raise ValueError(f"GNMC has no native {key} crop for {annotation.get('filename')}")
    normalized = boxes[key]
    if not isinstance(normalized, (list, tuple)) or len(normalized) != 4:
        raise ValueError("GNMC crop must be normalized xyxy")
    x1, y1, x2, y2 = (float(value) for value in normalized)
    if (not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x1 < 0 or y1 < 0
            or x2 > 1 or y2 > 1 or x2 <= x1 or y2 <= y1):
        raise ValueError("GNMC crop is outside normalized image coordinates")
    crop = [x1 * image_width, y1 * image_height,
            (x2 - x1) * image_width, (y2 - y1) * image_height]
    output_width, output_height = image_width, image_height
    if transpose:
        crop, dimensions = transpose_xywh(crop, image_width, image_height)
        output_width, output_height = dimensions
    return {
        "source_id": Path(annotation["filename"]).stem,
        "image_id": Path(annotation["filename"]).stem,
        "dataset_source": "GNMC",
        "split": split,
        "target_ratio": list(original_ratio),
        "image_width": output_width,
        "image_height": output_height,
        "gt_crop_xywh": list(crop),
        "transposed_geometry_augmentation": bool(transpose),
        "label_semantics": "single_human_crop_box_native_ratio",
    }


def gaicd_candidate_row(
    media: dict,
    annotation: dict,
    *,
    target_ratio: tuple[int, int],
    center_xy: tuple[float, float],
) -> dict:
    if annotation.get("release") != "journal":
        raise ValueError("GAICD scale adapter only accepts journal annotations")
    if str(annotation.get("image")) != str(media.get("item_id")):
        raise ValueError("GAICD media and annotation image IDs differ")
    boxes = annotation.get("boxes_xywh")
    mos = annotation.get("mos")
    valid = annotation.get("valid")
    if not (isinstance(boxes, list) and isinstance(mos, list) and isinstance(valid, list)
            and len(boxes) == len(mos) == len(valid)):
        raise ValueError("GAICD candidate arrays are misaligned")
    candidates = []
    for box, score, marked_valid in zip(boxes, mos, valid):
        score_valid = (isinstance(score, (int, float)) and not isinstance(score, bool)
                       and math.isfinite(score) and score not in (-2, -1))
        candidates.append({
            "crop_xywh": [float(value) for value in box],
            "mos": float(score) if score_valid else None,
            "valid": bool(marked_valid) and score_valid,
        })
    official_split = media.get("official_split")
    if official_split == "validation":
        official_split = "val"
    return {
        "source_id": str(media.get("group_id") or media.get("media_uid")),
        "image_id": str(media.get("item_id")),
        "dataset_source": "GAICD",
        "release": "journal",
        "split": official_split,
        "target_ratio": list(target_ratio),
        "center_xy": list(center_xy),
        "image_width": int(annotation["W"]),
        "image_height": int(annotation["H"]),
        "candidates": candidates,
        "label_semantics": "same_image_candidate_mean_opinion_score_1_to_5",
    }


def gnmc_split_rows(archive_path: str | Path) -> dict[str, list[dict]]:
    """Read GNMC split membership; returns train/val rows and test IDs only."""
    result: dict[str, list[dict]] = {"train": [], "val": [], "test_source_ids": []}
    with ZipFile(archive_path) as archive:
        result["test_source_ids"] = sorted(Path(n).stem for n in archive.namelist()
                                            if n.startswith("GNMC/test/") and n.lower().endswith((".jpeg", ".jpg", ".png")))
        for split, member in (("train", "GNMC/json/train.json"),
                              ("val", "GNMC/json/validation.json")):
            rows = json.loads(archive.read(member))
            source_ids = [Path(row["filename"]).stem for row in rows]
            if split == "test":
                result["test_source_ids"] = source_ids
            else:
                result[split] = [dict(row, source_id=source_id, split=split)
                                 for row, source_id in zip(rows, source_ids)]
    return result
