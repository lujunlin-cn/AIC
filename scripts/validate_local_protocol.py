#!/usr/bin/env python3
"""Validate the frozen TVSum local validation protocol.

The protocol file is intentionally data, rather than code hidden in a training
script.  This command checks its manifest hash, partition disjointness,
source-group isolation, and the five development-fold partitions.  A changed
manifest or split therefore fails loudly and requires a new protocol version.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict) or not value.get("video_id"):
            raise ValueError(f"{path}:{line_no}: expected an object with video_id")
        rows.append(value)
    return rows


def _assignment(protocol: dict[str, Any]) -> dict[str, Any]:
    return {
        "train": sorted(protocol["train_video_ids"]),
        "dev": sorted(protocol["dev_video_ids"]),
        "lockbox": sorted(protocol["lockbox_video_ids"]),
        "fold_validation": {
            fold["fold_id"]: sorted(fold["validation_video_ids"])
            for fold in protocol["development_cv"]["folds"]
        },
    }


def validate(protocol_path: Path, manifest_path: Path) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("protocol_id") != "tvsum_local_validation_v1":
        raise ValueError("unexpected protocol_id; refusing to validate as v1")
    rows = _load_manifest(manifest_path)
    by_id = {row["video_id"]: row for row in rows}
    if len(by_id) != len(rows):
        raise ValueError("manifest contains duplicate video_id values")
    expected_manifest_hash = protocol.get("manifest_sha256")
    actual_manifest_hash = _sha256(manifest_path)
    if actual_manifest_hash != expected_manifest_hash:
        raise ValueError(
            "manifest hash changed; create a new protocol version instead of mutating v1"
        )

    partitions = {
        "train": list(protocol["train_video_ids"]),
        "dev": list(protocol["dev_video_ids"]),
        "lockbox": list(protocol["lockbox_video_ids"]),
    }
    seen: dict[str, str] = {}
    for name, ids in partitions.items():
        if len(ids) != len(set(ids)):
            raise ValueError(f"{name} partition contains duplicate video IDs")
        for video_id in ids:
            if video_id not in by_id:
                raise ValueError(f"{video_id} in {name} is absent from manifest")
            previous = seen.setdefault(video_id, name)
            if previous != name:
                raise ValueError(f"{video_id} appears in {previous} and {name}")
    if set(seen) != set(by_id):
        missing = sorted(set(by_id) - set(seen))
        extra = sorted(set(seen) - set(by_id))
        raise ValueError(f"partition coverage mismatch missing={missing} extra={extra}")

    # A source group must never straddle a partition.  This catches future
    # manifest changes that duplicate or alias an original source video.
    groups: dict[str, str] = {}
    for video_id, partition in seen.items():
        source_group = str(by_id[video_id].get("source_group", ""))
        if not source_group:
            raise ValueError(f"{video_id} has no source_group")
        previous = groups.setdefault(source_group, partition)
        if previous != partition:
            raise ValueError(
                f"source_group {source_group!r} crosses {previous}/{partition}"
            )

    for video_id in partitions["train"]:
        if by_id[video_id].get("split") != "train":
            raise ValueError(f"{video_id} is not marked train in manifest")
    for video_id in partitions["dev"]:
        if by_id[video_id].get("split") != "val":
            raise ValueError(f"{video_id} is not marked val in manifest")
    for video_id in partitions["lockbox"]:
        if by_id[video_id].get("split") != "test":
            raise ValueError(f"{video_id} is not marked test in manifest")

    development = set(partitions["train"]) | set(partitions["dev"])
    folds = protocol["development_cv"]["folds"]
    fold_ids = [fold["fold_id"] for fold in folds]
    if len(folds) != 5 or len(set(fold_ids)) != 5:
        raise ValueError("development_cv must contain exactly five unique folds")
    fold_union: set[str] = set()
    for fold in folds:
        validation = set(fold["validation_video_ids"])
        training = set(fold["training_video_ids"])
        if not validation or validation & training:
            raise ValueError(f"invalid train/validation overlap in {fold['fold_id']}")
        if validation | training != development:
            raise ValueError(f"{fold['fold_id']} does not partition the development pool")
        if validation & set(partitions["lockbox"]) or training & set(partitions["lockbox"]):
            raise ValueError(f"{fold['fold_id']} contains lockbox records")
        fold_union |= validation
    if fold_union != development:
        raise ValueError("five validation folds do not cover every development record")

    canonical = json.dumps(_assignment(protocol), ensure_ascii=False,
                            sort_keys=True, separators=(",", ":")).encode()
    assignment_hash = hashlib.sha256(canonical).hexdigest()
    if assignment_hash != protocol.get("assignment_hash_sha256"):
        raise ValueError("assignment hash mismatch; protocol file is internally inconsistent")
    return {
        "valid": True,
        "protocol_id": protocol["protocol_id"],
        "manifest_sha256": actual_manifest_hash,
        "counts": {name: len(ids) for name, ids in partitions.items()},
        "fold_counts": {fold["fold_id"]: len(fold["validation_video_ids"])
                        for fold in folds},
        "lockbox_frozen": protocol["lockbox_policy"].get("status") == "frozen",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path,
                        default=Path("splits/local_protocol_v1.json"))
    parser.add_argument("--manifest", type=Path, default=None,
                        help="defaults to manifest_path recorded in the protocol")
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    manifest = args.manifest or Path(protocol["manifest_path"])
    result = validate(args.protocol, manifest)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
