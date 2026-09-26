#!/usr/bin/env python3
"""Prepare a provenance-first manifest for the supplied QVHighlights train set.

The AIC training package contains 150 s QVHighlights clips and weak temporal /
spatial annotations.  This utility does not decode frames or train a model. It
indexes the clips, creates frame-indexed soft temporal labels from the supplied
1 Hz timeline, and writes an extractor-compatible manifest. Missing source
clips stay in the inventory with ``download_status=missing`` and are excluded
from split manifests by default.

The generated labels deliberately carry an explicit protocol name. They are
seed weak labels, not TVSum labels or official AIC ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

LABEL_PROTOCOL = "qvh_seed_timeline_linear_v1"
DATASET_VERSION = "aic_qvh_seed_weak_training_v1"


def _sha256(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _probe(path: Path) -> dict[str, Any]:
    import av

    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "video"), None)
        if stream is None:
            raise ValueError(f"no video stream: {path}")
        rate = stream.average_rate or stream.base_rate
        fps = float(rate) if rate else None
        frames = int(stream.frames or 0)
        duration = float(container.duration / 1e6) if container.duration else None
        if duration is None and stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        audio = any(s.type == "audio" for s in container.streams)
        return {
            "duration": duration,
            "fps": fps,
            "frame_count": frames or None,
            "width": int(stream.codec_context.width),
            "height": int(stream.codec_context.height),
            "rotation": 0,
            "has_audio": bool(audio),
        }


def _timeline_labels(annotation: dict[str, Any], nframes: int, fps: float) -> np.ndarray:
    """Interpolate the supplied 1 Hz seed scores to original frame indices."""
    timeline = annotation.get("teacher_signals", {}).get("timeline") or []
    pairs = []
    for item in timeline:
        try:
            t = float(item["time_sec"])
            score = float(item.get("highlight_score", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if np.isfinite(t) and np.isfinite(score):
            pairs.append((max(0.0, t), float(np.clip(score, 0.0, 1.0))))
    if not pairs:
        return np.zeros(nframes, dtype=np.float32)
    pairs.sort(key=lambda x: x[0])
    # Duplicate timeline timestamps are reduced deterministically by keeping
    # the last supplied score, matching JSON annotation order.
    times: list[float] = []
    scores: list[float] = []
    for t, s in pairs:
        if times and t == times[-1]:
            scores[-1] = s
        else:
            times.append(t); scores.append(s)
    x = np.arange(nframes, dtype=np.float64) / float(fps)
    # np.interp extends the endpoint values.  The annotation includes a final
    # zero sentinel for clips whose timeline ends before the media stream.
    return np.interp(x, np.asarray(times), np.asarray(scores),
                     left=scores[0], right=scores[-1]).astype(np.float32)


def _write_labels(annotation: dict[str, Any], probe: dict[str, Any], out: Path) -> None:
    nframes = int(probe["frame_count"] or 0)
    fps = float(probe["fps"] or 0.0)
    if nframes <= 0 or fps <= 0:
        raise ValueError(f"invalid media metadata for {annotation['video_id']}: {probe}")
    labels = _timeline_labels(annotation, nframes, fps)
    # Keep mask true for every decoded source frame. ``segments`` is retained
    # in sidecar JSON for spatial/segment diagnostics; it is not silently used
    # as a different target definition.
    np.savez_compressed(
        out,
        labels=labels,
        mask=np.ones(nframes, dtype=np.bool_),
        frame_indices=np.arange(nframes, dtype=np.int64),
        timestamps=np.arange(nframes, dtype=np.float64) / fps,
        metadata_json=np.asarray(json.dumps({
            "dataset": "QVHighlights",
            "dataset_version": DATASET_VERSION,
            "label_protocol": LABEL_PROTOCOL,
            "annotation_video_id": annotation["video_id"],
            "target_definition": "linear interpolation of supplied teacher_signals.timeline.highlight_score",
            "official_aic_gt": False,
        }, sort_keys=True)),
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="video_highlight root containing train/source_videos and train/annotations")
    ap.add_argument("--annotations", type=Path, default=None)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--sha256", action="store_true", help="hash matched clips; slower for 130 GB")
    args = ap.parse_args()
    root = args.root.resolve()
    ann_path = (args.annotations or root / "train/annotations/train.jsonl").resolve()
    out = args.output.resolve(); out.mkdir(parents=True, exist_ok=True)
    videos = {}
    duplicates: dict[str, list[str]] = {}
    for path in sorted((root / "train/source_videos").rglob("*.mp4")):
        key = path.stem
        if key in videos:
            duplicates.setdefault(key, [str(videos[key])]).append(str(path))
        else:
            videos[key] = path
    rows = [json.loads(line) for line in ann_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    inventory: list[dict[str, Any]] = []
    # ``dataset_manifest.jsonl`` follows ``aic.data.MANIFEST_FIELDS`` and is
    # intentionally separate from the richer inventory/feature manifests.
    # Keeping this canonical source manifest lets the existing feature-cache
    # builder validate provenance without silently accepting ad-hoc fields.
    dataset_manifest: list[dict[str, Any]] = []
    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    counts = {"rows": len(rows), "matched": 0, "missing": 0, "probe_failures": 0}
    for ann in rows:
        source_id = str(ann.get("clip", {}).get("source_vid", ""))
        path = videos.get(source_id)
        split = str(ann.get("dataset_split", "unassigned"))
        rec: dict[str, Any] = {
            "dataset": "QVHighlights", "version": DATASET_VERSION,
            "video_id": str(ann["video_id"]), "source_id": source_id,
            "source_group": source_id, "split": split,
            "annotation_type": "weak_temporal_spatial",
            "label_protocol": LABEL_PROTOCOL,
            "targetRatioWH": ann.get("targetRatioWH"),
            "annotation_json": str(ann_path),
            "source_url": ann.get("video_url") or None,
            "license_gate": "user_authorized_downloadable_source",
            "official_aic_gt": False,
        }
        if path is None:
            counts["missing"] += 1; rec.update({"download_status": "missing", "path": None})
            inventory.append(rec); continue
        try:
            meta = _probe(path)
            label_path = out / "labels" / f"{ann['video_id']}.npz"
            label_path.parent.mkdir(parents=True, exist_ok=True)
            _write_labels(ann, meta, label_path)
        except Exception as exc:
            counts["probe_failures"] += 1; rec.update({"download_status": "failed", "path": str(path), "error": repr(exc)})
            inventory.append(rec); continue
        counts["matched"] += 1
        rec.update({"download_status": "verified", "path": str(path), "video_path": str(path),
                    "labels_path": str(label_path), **meta,
                    "sha256": _sha256(path) if args.sha256 else None,
                    "spatial_annotation_path": str(out / "spatial" / f"{ann['video_id']}.json")})
        spatial = out / "spatial" / f"{ann['video_id']}.json"; spatial.parent.mkdir(parents=True, exist_ok=True)
        spatial.write_text(json.dumps({"video_id": ann["video_id"], "targetRatioWH": ann.get("targetRatioWH"),
                                       "cropRois": ann.get("cropRois", []), "crop_keyframes": ann.get("crop_keyframes", []),
                                       "segments": ann.get("segments", []), "label_protocol": LABEL_PROTOCOL}, ensure_ascii=False), encoding="utf-8")
        inventory.append(rec)
        dataset_manifest.append({
            "dataset": "QVHighlights",
            "version": DATASET_VERSION,
            "video_id": rec["video_id"],
            "source_id": source_id,
            "source_group": source_id,
            "path": str(path),
            "source_url": rec.get("source_url"),
            "license": None,
            "license_url": None,
            "license_text_hash": None,
            "license_gate": "user_authorized_downloadable_source",
            "download_status": "verified",
            "sha256": rec.get("sha256"),
            "duration": rec.get("duration"),
            "fps": rec.get("fps"),
            "frame_count": rec.get("frame_count"),
            "width": rec.get("width"),
            "height": rec.get("height"),
            "rotation": rec.get("rotation", 0),
            "has_audio": rec.get("has_audio"),
            "split": split,
            "annotation_type": "qvh_seed_temporal_spatial_weak",
            "annotation_path": str(label_path),
            "notes": (
                "Challenge-provided QVHighlights-derived seed weak labels; "
                "linear timeline interpolation v1; not native human GT; "
                "official_aic_gt=false"
            ),
        })
        # Feature extractor manifests use these fields and intentionally do not
        # include the large annotation object itself.
        if split in split_rows:
            split_rows[split].append({"video_id": rec["video_id"], "video_path": str(path),
                                      "path": str(out / "features" / f"{ann['video_id']}.npz"),
                                      "labels_path": str(label_path), "split": split,
                                      "source_id": source_id, "source_group": source_id,
                                      "dataset": "QVHighlights", "annotation_type": rec["annotation_type"],
                                      "label_protocol": LABEL_PROTOCOL,
                                      "targetRatioWH": ann.get("targetRatioWH"),
                                      "duration": rec.get("duration"), "fps": rec.get("fps"),
                                      "frame_count": rec.get("frame_count"),
                                      "width": rec.get("width"), "height": rec.get("height"),
                                      "has_audio": rec.get("has_audio"),
                                      "official_aic_gt": False})
    (out / "inventory.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in inventory) + "\n", encoding="utf-8")
    (out / "dataset_manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in dataset_manifest) + "\n",
        encoding="utf-8")
    for split, items in split_rows.items():
        (out / f"{split}.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False, sort_keys=True) for r in items) + "\n", encoding="utf-8")
    summary = {**counts, "source_video_files": len(videos), "duplicate_stems": duplicates,
               "label_protocol": LABEL_PROTOCOL, "root": str(root), "annotation_file": str(ann_path)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "duplicate_stems"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
