#!/usr/bin/env python3
"""Build a cache manifest from verified source-video records.

The output is deliberately a different manifest from the source provenance
manifest: a feature cache can be regenerated, while source paths, licenses and
hashes remain in the dataset manifest.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from aic.data import read_manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--allow-unapproved", action="store_true",
                        help="override the rights gate only after written permission")
    args = parser.parse_args(argv)
    records = read_manifest(args.dataset_manifest, require_path=True)
    cache_dir, output_dir = Path(args.cache_dir), Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = {"train": [], "val": [], "test": []}
    for record in records:
        if record.download_status not in {"verified", "downloaded"}:
            continue
        if record.license_gate != "approved" and not args.allow_unapproved:
            continue
        if not record.path or not record.annotation_path:
            continue
        if record.split not in grouped or (record.split == "test" and not args.include_test):
            continue
        item = {"video_id": record.video_id, "path": str(cache_dir / f"{record.video_id}.npz"),
                "labels_path": record.annotation_path, "split": record.split,
                "source_id": record.source_id, "source_group": record.source_group,
                "dataset": record.dataset, "annotation_type": record.annotation_type,
                "video_path": record.path}
        grouped[record.split].append(item)
    for split, items in grouped.items():
        if split == "test" and not args.include_test:
            continue
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as stream:
            for item in items:
                stream.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({key: len(value) for key, value in grouped.items()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
