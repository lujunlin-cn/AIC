#!/usr/bin/env python3
"""Build a deterministic, training-only QVHighlights-derived subset.

The local challenge package contains weak seed labels rather than native
human GT.  This importer keeps that distinction explicit and produces labels
aligned to decoded source-video frame indices for the existing feature cache
extractors.  It never reads the official test directory.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import av
import numpy as np


def frame_count(path: Path) -> int:
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        if stream.frames:
            return int(stream.frames)
        return sum(1 for _ in container.decode(stream))


def make_labels(row: dict[str, Any], nframes: int, out: Path) -> None:
    labels = np.zeros(nframes, dtype=np.float32)
    for segment in row.get("segments", []):
        start = max(0, int(segment.get("start_frame", 0)))
        end = min(nframes - 1, int(segment.get("end_frame", -1)))
        if end >= start:
            # QVHighlights-derived seed scores are weak supervision.  Keep a
            # continuous segment score but do not call it human importance.
            value = float(segment.get("mean_seed_score", 1.0))
            labels[start : end + 1] = np.clip(value, 0.0, 1.0)
    indices = np.arange(nframes, dtype=np.int64)
    np.savez_compressed(out, labels=labels, frame_indices=indices)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--train-count", type=int, default=40)
    ap.add_argument("--val-count", type=int, default=10)
    args = ap.parse_args()
    ann = args.root / "train" / "annotations" / "train.jsonl"
    video_root = args.root / "train" / "source_videos"
    rows = [json.loads(line) for line in ann.read_text().splitlines() if line.strip()]
    rows.sort(key=lambda r: str(r["video_id"]))
    grouped = {"train": [r for r in rows if r.get("dataset_split") == "train"],
               "val": [r for r in rows if r.get("dataset_split") == "val"]}
    selected = {"train": grouped["train"][: args.train_count],
                "val": grouped["val"][: args.val_count]}
    args.output.mkdir(parents=True, exist_ok=True)
    label_dir = args.output / "labels"
    label_dir.mkdir(exist_ok=True)
    all_rows: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    missing = []
    for split, subset in selected.items():
        for row in subset:
            source_vid = str(row["clip"]["source_vid"])
            matches = list(video_root.rglob(source_vid + ".mp4"))
            if not matches:
                missing.append({"video_id": row["video_id"], "source_vid": source_vid, "split": split})
                continue
            video = matches[0]
            nframes = frame_count(video)
            label_path = label_dir / f"{row['video_id']}.npz"
            make_labels(row, nframes, label_path)
            all_rows[split].append({
                "video_id": str(row["video_id"]), "video_path": str(video),
                "labels_path": str(label_path), "path": str(video),
                "split": split, "source": "QVHighlights_derived_seed_v1",
                "annotation_mode": row.get("annotation_mode"),
                "label_semantics": "weak temporal segment seed score; not human GT",
                "target_ratio_wh": row.get("targetRatioWH"),
                "source_vid": source_vid, "frame_count": nframes,
                "crop_gt_available": bool(row.get("cropRois")),
                "provenance": row.get("provenance", {}),
            })
    for split, records in all_rows.items():
        (args.output / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in records),
            encoding="utf-8")
    summary = {"protocol": "QVH_DERIVED_SEED_SUBSET_V1", "root": str(args.root),
               "selection": {"train_requested": args.train_count, "val_requested": args.val_count,
                             "ordering": "video_id lexical after dataset_split"},
               "counts": {k: len(v) for k, v in all_rows.items()}, "missing": missing,
               "label_semantics": "weak seed segment scores from challenge package; not native human GT",
               "official_test_excluded": True}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
