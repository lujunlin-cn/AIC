#!/usr/bin/env python3
"""Build blind Label Studio image tasks from the fixed AIC manual batch.

The exported JPEGs are one frame per second of an annotation preview clip.
The task records source offsets and timestamps explicitly; they are not
official AIC frame indices and must not be submitted as such. Query,
category, and sampling rationale stay out of the default task JSON so the
first pass remains blind. Use --include-posthoc-metadata only for a separate
post-hoc project.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import quote


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--frames", type=Path, required=True,
                    help="directory containing <video_id>/<second>.jpg")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--image-mode",
        choices=("file-uri", "label-studio-local-files"),
        default="file-uri",
        help=(
            "How to encode image paths. `label-studio-local-files` emits "
            "/data/local-files/?d=... paths and expects the Label Studio "
            "document root to be the repository root."
        ),
    )
    ap.add_argument(
        "--document-root",
        type=Path,
        default=None,
        help=(
            "Document root used with --image-mode label-studio-local-files; "
            "defaults to the repository root inferred from --frames."
        ),
    )
    ap.add_argument(
        "--include-posthoc-metadata",
        action="store_true",
        help=(
            "Include category/query/reason fields in task data. This is "
            "intentionally off for the blind first-pass annotation."
        ),
    )
    ap.add_argument(
        "--video-id",
        action="append",
        dest="video_ids",
        help="Restrict output to one or more manifest video IDs; repeat the flag.",
    )
    args = ap.parse_args()

    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    overlap = set(cfg["annotators"]["overlap_video_ids"])
    single = set(cfg["annotators"]["single_annotator_video_ids"])
    tasks: list[dict] = []
    requested_video_ids = set(args.video_ids or [])
    document_root = args.document_root.resolve() if args.document_root else None
    if args.image_mode == "label-studio-local-files" and document_root is None:
        # artifacts/<batch>/frames/... -> repository root
        document_root = args.frames.resolve().parents[2]
    for item in cfg["items"]:
        vid = item["video_id"]
        if requested_video_ids and vid not in requested_video_ids:
            continue
        frame_dir = args.frames / vid
        paths = sorted(frame_dir.glob("*.jpg"), key=lambda p: float(p.stem))
        if len(paths) != cfg["clip_duration_seconds"]:
            raise SystemExit(f"{vid}: expected 30 jpgs, found {len(paths)}")
        assignment = "overlap_2_annotators" if vid in overlap else "single_annotator"
        if vid not in overlap | single:
            raise SystemExit(f"{vid}: missing annotator assignment")
        for path in paths:
            t = float(path.stem)
            resolved_path = path.resolve()
            if args.image_mode == "file-uri":
                image_value = resolved_path.as_uri()
            else:
                try:
                    rel = resolved_path.relative_to(document_root)
                except ValueError as exc:
                    raise SystemExit(
                        f"{resolved_path} is outside document root {document_root}"
                    ) from exc
                image_value = "/data/local-files/?d=" + quote(rel.as_posix())
            task = {
                "data": {
                    "image": image_value,
                    "video_id": vid,
                    "time_sec": t,
                    "source_offset_sec": item["clip_start_seconds"],
                    "source_time_sec": item["clip_start_seconds"] + t,
                },
            }
            if args.include_posthoc_metadata:
                # These fields are useful for controlled post-hoc analysis but
                # must not be present during the blind first pass. The source
                # manifest remains the authoritative record for them.
                task["data"].update({
                    "source_group": item["source_group"],
                    "category": item["category"],
                    "assignment": assignment,
                    "qvh_query": item.get("qvh_query", ""),
                    "annotation_reason": item["reason"],
                })
            tasks.append(task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if requested_video_ids:
        present = {task["data"]["video_id"] for task in tasks}
        missing = requested_video_ids - present
        if missing:
            raise SystemExit(f"unknown video IDs: {sorted(missing)}")
    args.output.write_text(json.dumps(tasks, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"tasks": len(tasks), "videos": len(cfg["items"]),
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
