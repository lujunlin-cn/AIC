#!/usr/bin/env python3
"""Fit deployment scale bins from GNMC's source-disjoint training split only."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

from PIL import Image

from aic.scale_geometry import fit_scale_bins_from_rows


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fit_gnmc_train_bins(archive_path: Path, count: int = 8) -> dict:
    """Use native 16:9 GNMC train crops; transpose gives the paired 9:16 geometry."""
    archive_hash = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            archive_hash.update(block)

    with ZipFile(archive_path) as archive:
        annotation_bytes = archive.read("GNMC/json/train.json")
        annotations = json.loads(annotation_bytes)
        rows = []
        source_ids = set()
        missing = []
        for annotation in annotations:
            filename = Path(annotation["filename"]).name
            source_id = Path(filename).stem
            if source_id in source_ids:
                raise ValueError(f"duplicate GNMC train source: {source_id}")
            source_ids.add(source_id)
            member = f"GNMC/train/{filename}"
            try:
                with Image.open(io.BytesIO(archive.read(member))) as image:
                    image_width, image_height = image.size
            except KeyError:
                missing.append(member)
                continue
            if image_width <= 0 or image_height <= 0:
                raise ValueError(f"invalid GNMC image dimensions: {member}")
            box = annotation.get("crop_bboxes", {}).get("16:9")
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                raise ValueError(f"GNMC train crop missing native 16:9 box: {source_id}")
            x1, y1, x2, y2 = (float(value) for value in box)
            if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
                raise ValueError(f"invalid normalized GNMC train box: {source_id}")
            maximum_width = min(image_width, image_height * 16 / 9)
            scale_ratio = (x2 - x1) * image_width / maximum_width
            rows.append({"split": "train", "scale_ratio": scale_ratio})

        if missing:
            raise ValueError(f"{len(missing)} GNMC train images are missing; first={missing[0]}")
        if len(rows) != len(annotations):
            raise ValueError("GNMC train image/annotation counts do not match")

    bins = fit_scale_bins_from_rows(rows, count)
    source_manifest = "\n".join(sorted(source_ids)).encode("utf-8")
    return {
        "schema_version": "aic.scale_bins.v1",
        "status": "fitted_from_train_only",
        "values": bins,
        "requested_count": count,
        "actual_count": len(bins),
        "fit_method": "quantiles_of_GNMC_train_native_16_9_crop_width_over_maximum_legal_width",
        "fit_split": "train",
        "fit_sources": {
            "dataset": "GNMC",
            "source_count": len(rows),
            "native_geometry": "16:9",
            "transpose_geometry": "9:16",
            "annotation_sha256": sha256_bytes(annotation_bytes),
            "source_id_set_sha256": sha256_bytes(source_manifest),
            "archive_sha256": archive_hash.hexdigest(),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "validation_or_test_labels_read": False,
        "p1_centers_used": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gnmc-zip", type=Path, required=True)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run-config", type=Path)
    args = parser.parse_args()
    result = fit_gnmc_train_bins(args.gnmc_zip, args.count)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if args.run_config:
        config = json.loads(args.run_config.read_text(encoding="utf-8"))
        scale_bins = config["data"]["scale_bins"]
        scale_bins["values"] = result["values"]
        scale_bins["requested_count"] = result["requested_count"]
        scale_bins["count"] = result["actual_count"]
        scale_bins["status"] = result["status"]
        scale_bins["fit_source_manifest_sha256"] = result["fit_sources"]["source_id_set_sha256"]
        args.run_config.write_text(json.dumps(config, indent=2, ensure_ascii=False) + "\n",
                                   encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
