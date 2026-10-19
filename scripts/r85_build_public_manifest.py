#!/usr/bin/env python3
"""Build a source-disjoint R85 manifest from four trusted local public T5 tables.

Only rv_train/live_train/rv_dev/live_dev.pkl are opened. These are Python pickle
files and must be the project's trusted local tables, never downloaded arbitrary
pickles. Images are joined by table keys[J] and video ID, never R70 shard order.
This command prepares inputs; it does not load models, train, or score officials.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from numbers import Integral
import os
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Any, Mapping

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.crop_observation import (GT_CONVENTION, MANIFEST_VERSION, NC,
                                  OBSERVATION_VERSION, PUBLIC_TABLES, checked_ratio,
                                  image_record, make_row_id, native_dimensions,
                                  numeric_array, validate_boxes, validate_manifest_rows)


def _frame_vector(values: Any, name: str) -> list[int]:
    values = np.asarray(values, dtype=object)
    if values.ndim != 1 or any(isinstance(v, (bool, np.bool_))
                               or not isinstance(v, Integral) or v < 0 for v in values):
        raise ValueError(f"{name} must be a vector of nonnegative integer frame indices")
    result = list(map(int, values))
    if len(set(result)) != len(result):
        raise ValueError(f"duplicate entries in {name}")
    return result


def _source_id(unit: Mapping[str, Any], table: str) -> tuple[str, str, str, str]:
    dataset, split = table.split("_", 1)
    if unit.get("ds") != dataset:
        raise ValueError(f"dataset outside table whitelist: {table} contains ds={unit.get('ds')!r}")
    vid = unit.get("vid")
    if not isinstance(vid, str) or not vid or vid in {".", ".."} or "/" in vid or "\\" in vid:
        raise ValueError("vid must be an explicit nonempty source directory component")
    return dataset, split, vid, f"{dataset}:{vid}"


def _image_path(t5_dir: Path, rv_frames: Path, dataset: str, vid: str, kf: int) -> Path:
    # Both LIVE fold-train and fold-dev are native keyframes_train images.
    root = (rv_frames / "d1") if dataset == "rv" else (t5_dir / "keyframes_train")
    root = root.resolve(strict=True)
    path = (root / vid / f"{kf}.png").resolve(strict=True)
    if not path.is_relative_to(root):
        raise ValueError("image path escapes its explicit public source root")
    return path


def build_public_manifest(t5_dir: str | Path, rv_frames: str | Path) -> tuple[list[dict], dict]:
    """Return validated JSON-ready rows and metadata; do not write output files.

    T5 gt has shape [annotator, gf-frame, 4]. Every stored GT is checked before
    selecting gt[:, index_of(keys[J]), :]. A selected frame without human GT is
    omitted with an explicit count; no pseudo-label is substituted.
    """
    t5_dir, rv_frames = Path(t5_dir).resolve(strict=True), Path(rv_frames).resolve(strict=True)
    table_dir = (t5_dir / "tables").resolve(strict=True)
    source_splits: dict[str, str] = {}
    seen_ids: set[str] = set()
    rows: list[dict] = []
    images: dict[Path, dict] = {}
    table_metadata = []
    for table in PUBLIC_TABLES:
        path = (table_dir / f"{table}.pkl").resolve(strict=True)
        if path.parent != table_dir or path.name != f"{table}.pkl":
            raise ValueError("table path escapes the explicit public table directory")
        payload = path.read_bytes()
        table_sha = hashlib.sha256(payload).hexdigest()
        units = pickle.loads(payload)
        del payload
        if not isinstance(units, (list, tuple)):
            raise ValueError(f"{table} must contain a sequence of T5 unit mappings")
        stats = {"table": table, "table_path": str(path), "table_sha256": table_sha,
                 "units": len(units), "selected_keyframes": 0, "rows": 0,
                 "selected_keyframes_without_gt": 0}
        for unit_index, unit in enumerate(units):
            if not isinstance(unit, Mapping):
                raise ValueError(f"{table} unit {unit_index} must be a mapping")
            dataset, split, vid, source = _source_id(unit, table)
            if source in source_splits and source_splits[source] != split:
                raise ValueError(f"source train/dev leakage: {source}")
            # Check every source, even if no selected frame has a human label.
            source_splits[source] = split
            W, H = native_dimensions(unit["W"], unit["H"])
            ratio = list(checked_ratio(unit["ratio"]))
            keys = _frame_vector(unit["keys"], "keys")
            selected = _frame_vector(unit["J"], "J")
            gf = _frame_vector(unit["gf"], "gf")
            if any(j >= len(keys) for j in selected):
                raise ValueError(f"J indexes outside keys for {source}")
            gt = numeric_array(unit["gt"], "public GT")
            if gt.ndim != 3 or gt.shape[0] == 0 or gt.shape[1:] != (len(gf), 4) or not gf:
                raise ValueError(f"GT must have shape [annotators,len(gf),4] for {source}")
            validate_boxes(gt.reshape(-1, 4), W, H)
            gf_position = {f: i for i, f in enumerate(gf)}
            stats["selected_keyframes"] += len(selected)
            for key_index in selected:
                kf = keys[key_index]
                if kf not in gf_position:
                    stats["selected_keyframes_without_gt"] += 1
                    continue
                row_id = make_row_id(source, kf, ratio)
                if row_id in seen_ids:
                    raise ValueError(f"duplicate row_id: {row_id}")
                seen_ids.add(row_id)
                image_path = _image_path(t5_dir, rv_frames, dataset, vid, kf)
                if image_path not in images:
                    images[image_path] = image_record(image_path)
                image = images[image_path]
                if (image["W"], image["H"]) != (W, H):
                    raise ValueError(f"native image size mismatch for {source}/{kf}: "
                                     f"table={(W, H)}, image={(image['W'], image['H'])}")
                rows.append({
                    "schema_version": MANIFEST_VERSION, "row_id": row_id,
                    "source": source, "dataset": dataset, "video_id": vid,
                    "split": split, "table": table, "frame_id": kf, "kf": kf,
                    "ratio": ratio, "W": W, "H": H,
                    "image_path": image["image_path"], "image_sha256": image["image_sha256"],
                    "table_path": str(path), "table_sha256": table_sha,
                    "table_unit_index": unit_index, "table_key_index": key_index,
                    "table_gt_index": gf_position[kf],
                    "gt_xyxy": gt[:, gf_position[kf], :].tolist(),
                    "gt_coordinate_convention": GT_CONVENTION,
                    "coordinate_scale": "native_pixels", "exif_transpose": False,
                    "label_source": "public_human_gt_from_trusted_T5_table",
                })
                stats["rows"] += 1
        table_metadata.append(stats)
        del units
    rows.sort(key=lambda r: (PUBLIC_TABLES.index(r["table"]), r["source"], r["frame_id"], tuple(r["ratio"])))
    # Images were decoded and hashed above from the same bytes. Consumers can
    # independently repeat image verification with the default verify_images=True.
    validate_manifest_rows(rows, verify_images=False)
    return rows, {
        "schema_version": MANIFEST_VERSION, "status": "public_manifest_built",
        "tables": table_metadata, "whitelist": list(PUBLIC_TABLES),
        "rows": len(rows), "unique_images": len(images),
        "rows_by_table": dict(Counter(r["table"] for r in rows)),
        "sources_by_split": {split: sorted(s for s, part in source_splits.items() if part == split)
                             for split in ("train", "dev")},
        "source_train_dev_overlap": [], "join": "source + keys[J] matched to gf; never R70 shard position",
        "t5_dir": str(t5_dir), "rv_frames": str(rv_frames),
        "gt_coordinate_convention": GT_CONVENTION, "coordinate_scale": "native_pixels",
        "observation_version": OBSERVATION_VERSION, "candidate_count": NC,
        "crop_resampling": "PIL EXTENT exact fractional xyxy; default ceil native output dimensions",
        "exif_transpose": False, "training_executed": False, "official_inputs_requested": False,
        "provenance_scope": "trusted local public T5 tables; no independent media-license audit",
    }


def write_manifest(rows: list[dict], meta: dict, output: str | Path,
                   meta_output: str | Path | None = None, *, overwrite: bool = False) -> dict:
    """Publish JSONL and metadata; refuse existing files unless explicitly requested."""
    output = Path(output).resolve()
    meta_output = Path(meta_output).resolve() if meta_output else output.with_suffix(".meta.json")
    if output == meta_output:
        raise ValueError("manifest and metadata paths must differ")
    for target in (output, meta_output):
        if target.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite {target}")
    validate_manifest_rows(rows, verify_images=True)
    body = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False, sort_keys=True,
                             separators=(",", ":")) + "\n" for r in rows).encode("utf-8")
    result = {**meta, "manifest_path": str(output), "manifest_sha256": hashlib.sha256(body).hexdigest()}
    meta_bytes = (json.dumps(result, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    temporary, created = [], []
    try:
        for target, content in ((output, body), (meta_output, meta_bytes)):
            target.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile("wb", dir=target.parent, prefix=".r85-manifest-", delete=False) as stream:
                temp = Path(stream.name)
                temporary.append(temp)
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            if overwrite:
                os.replace(temp, target)
            else:
                # A hard link publishes complete bytes atomically and refuses
                # a path created after the preflight existence check.
                os.link(temp, target)
                created.append(target)
    except BaseException:
        if not overwrite:
            for target in created:
                target.unlink()
        raise
    finally:
        for temp in temporary:
            temp.unlink(missing_ok=True)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t5-dir", type=Path, required=True)
    parser.add_argument("--rv-frames", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="public manifest JSONL")
    parser.add_argument("--meta-output", type=Path, help="default: output stem + .meta.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    rows, meta = build_public_manifest(args.t5_dir, args.rv_frames)
    result = write_manifest(rows, meta, args.output, args.meta_output, overwrite=args.overwrite)
    print(json.dumps({"manifest": result["manifest_path"], "rows": result["rows"],
                      "manifest_sha256": result["manifest_sha256"], "training_executed": False}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
