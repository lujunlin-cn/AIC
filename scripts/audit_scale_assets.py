#!/usr/bin/env python3
"""Create a metadata-only Stage-0 asset audit for the Scale SFT/DPO plan."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from aic.scale_data import normalize_train_val_source_split


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tree_fingerprint(root: Path, exclude_names: set[str] | None = None) -> dict:
    exclude_names = exclude_names or set()
    files = sorted(path for path in root.rglob("*") if path.is_file()
                   and path.name not in exclude_names)
    aggregate = hashlib.sha256()
    total_bytes = 0
    for path in files:
        rel = path.relative_to(root).as_posix().encode("utf-8")
        file_hash = sha256_file(path)
        size = path.stat().st_size
        aggregate.update(rel + b"\0" + str(size).encode("ascii") + b"\0" + file_hash.encode("ascii") + b"\n")
        total_bytes += size
    return {"files": len(files), "bytes": total_bytes, "tree_sha256": aggregate.hexdigest()}


def jsonl_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def audit_gaicd(root: Path) -> dict:
    dataset = root / "datasets/GAICD"
    card_path = dataset / "processed/dataset_card.json"
    media_path = dataset / "processed/media.jsonl"
    annotation_manifest_path = dataset / "processed/annotations.jsonl"
    if not all(path.is_file() for path in (card_path, media_path, annotation_manifest_path)):
        return {"present": False, "blockers": ["processed journal manifests missing"]}
    card = json.loads(card_path.read_text(encoding="utf-8"))
    media = jsonl_rows(media_path)
    annotations = jsonl_rows(annotation_manifest_path)
    journal_media = [row for row in media if row.get("release", "").startswith("GAICD journal")
                     or "journal" in row.get("release", "")]
    journal_media_by_id = {str(row.get("item_id")): row for row in journal_media}
    journal_media_by_uid = {str(row.get("media_uid") or row.get("uid")): row for row in journal_media}
    split_sets: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for row in journal_media_by_id.values():
        split = row.get("official_split")
        if split in {"train", "val", "validation", "test"}:
            split = "val" if split == "validation" else split
            split_sets[split].add(str(row.get("group_id") or row.get("media_uid")))
    source_overlap = {
        f"{left}_{right}": sorted(split_sets[left] & split_sets[right])
        for left, right in (("train", "val"), ("train", "test"), ("val", "test"))
    }
    train_candidates = 0
    train_unrated = 0
    train_annotation_files = []
    missing_train_annotations = 0
    for record in annotations:
        media_record = journal_media_by_uid.get(str(record.get("media_uid")))
        if record.get("variant") != "journal" or not media_record or media_record.get("official_split") != "train":
            continue
        raw_path = Path(record.get("annotation_path", ""))
        if not raw_path.is_file():
            local = dataset / "annotations/candidates" / raw_path.name
            raw_path = local
        if not raw_path.is_file():
            missing_train_annotations += 1
            continue
        train_annotation_files.append(raw_path)
        annotation = json.loads(raw_path.read_text(encoding="utf-8"))
        train_candidates += len(annotation.get("boxes_xywh", []))
        train_unrated += sum(
            score in (-2, -1) or not isinstance(score, (int, float))
            for score in annotation.get("mos", [])
        )
    split_disjoint = not any(source_overlap.values())
    return {
        "present": True,
        "source_url": card.get("source"),
        "license_status": "not recorded in local dataset card; verify original data terms before external use",
        "competition_use_authorized": None,
        "media_count": len(journal_media_by_id),
        "unique_source_count": len({str(row.get("group_id") or row.get("media_uid"))
                                     for row in journal_media_by_id.values()}),
        "official_split_source_counts": {key: len(values) for key, values in split_sets.items()},
        "official_split_source_overlap_counts": {key: len(values) for key, values in source_overlap.items()},
        "official_split_source_disjoint": split_disjoint,
        "journal_train_candidate_count": train_candidates,
        "journal_train_unrated_candidate_count": train_unrated,
        "journal_train_annotation_files": len(train_annotation_files),
        "journal_train_annotation_missing": missing_train_annotations,
        "conference_rows_excluded_from_training": sum(row.get("variant") != "journal" for row in annotations),
        "card_summary": card.get("annotation_stats", {}),
        "manifest_hashes": {
            "dataset_card.json": sha256_file(card_path),
            "media.jsonl": sha256_file(media_path),
            "annotations.jsonl": sha256_file(annotation_manifest_path),
        },
        "raw_tree": tree_fingerprint(dataset / "raw") if (dataset / "raw").is_dir() else None,
        "train_data_integrity_ready": bool(
            missing_train_annotations == 0 and split_disjoint and train_candidates > 0
        ),
        "blockers": [] if missing_train_annotations == 0 and split_disjoint else [
            reason for condition, reason in (
                (missing_train_annotations > 0, "journal train annotation files missing"),
                (not split_disjoint, "journal source split overlap"),
            ) if condition
        ],
    }


def _sacd_split(root: Path, split: str) -> tuple[set[str], set[str], int]:
    path = root / f"{split}.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    annotations = set()
    images = set()
    for annotation_path in data.get("jsons", []):
        annotation_name = Path(annotation_path).stem
        image_id = annotation_name.rsplit("_", 1)[0]
        annotations.add(annotation_name)
        images.add(image_id)
    return annotations, images, len(data.get("jsons", []))


def audit_sacd(root: Path) -> dict:
    dataset = root / "datasets/SACD"
    status_path = dataset / "DOWNLOAD_STATUS.json"
    if not status_path.is_file():
        return {"present": False, "blockers": ["SACD metadata missing"]}
    status = json.loads(status_path.read_text(encoding="utf-8"))
    splits = {name: _sacd_split(dataset, name) for name in ("train", "val", "test")}
    overlap = {}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap[f"{left}_{right}"] = len(splits[left][1] & splits[right][1])
    explicit_pair_count = 0
    annotation_files_seen = set()
    for split in ("train",):
        for name in json.loads((dataset / f"{split}.json").read_text(encoding="utf-8"))["jsons"]:
            path = dataset / name
            if path in annotation_files_seen:
                continue
            annotation_files_seen.add(path)
            row = json.loads(path.read_text(encoding="utf-8"))
            for key in ("ranked_pairs", "preference_pairs", "pairwise_preferences"):
                explicit_pair_count += len(row.get(key, []))
    overlap_free = not any(overlap.values())
    return {
        "present": True,
        "provenance": status.get("source"),
        "license_status": "not recorded in local provenance metadata; verify before external use",
        "competition_use_authorized": None,
        "images": status.get("images"),
        "subject_annotations": status.get("subject_annotations"),
        "masks": status.get("masks"),
        "source_tree_sha256_from_prior_validation": status.get("original_tree_sha256"),
        "recomputed_source_payload": tree_fingerprint(dataset, {"DOWNLOAD_STATUS.json"}),
        "split_annotation_counts": {name: value[2] for name, value in splits.items()},
        "split_unique_image_counts": {name: len(value[1]) for name, value in splits.items()},
        "split_image_overlap_counts": overlap,
        "source_image_splits_disjoint": overlap_free,
        "explicit_ranked_pair_count": explicit_pair_count,
        "scale_preference_gate": "blocked_no_confirmed_explicit_ranked_pairs",
        "blockers": ["split files share source images" for _ in [0] if not overlap_free]
                    + ["no explicit subject-linked ranked-pair records" for _ in [0] if explicit_pair_count == 0],
    }


def audit_gnmc(archive_path: Path) -> dict:
    if not archive_path.is_file():
        return {"present": False, "blockers": ["GNMC archive missing"]}
    with ZipFile(archive_path) as archive:
        names = archive.namelist()
        train = json.loads(archive.read("GNMC/json/train.json"))
        validation = json.loads(archive.read("GNMC/json/validation.json"))
        # Read source names only for the held-out split; its crop labels are not
        # used for fitting, candidate selection, or threshold choice.
        test_sources = {Path(name).stem for name in names
                        if name.startswith("GNMC/test/") and name.lower().endswith(".jpeg")}
        train_sources = {Path(row["filename"]).stem for row in train}
        val_sources = {Path(row["filename"]).stem for row in validation}
        train_val_overlap = train_sources & val_sources
        train_test_overlap = train_sources & test_sources
        val_test_overlap = val_sources & test_sources
        train_ratio_counts: dict[str, int] = {}
        for row in train:
            for ratio in row.get("crop_bboxes", {}):
                train_ratio_counts[ratio] = train_ratio_counts.get(ratio, 0) + 1
        license_member = "GNMC/LICENSE.txt"
        license_sha = sha256_file_from_zip(archive, license_member)
        image_counts = {
            split: sum(name.startswith(f"GNMC/{split}/") and name.lower().endswith(".jpeg") for name in names)
            for split in ("train", "validation", "test")
        }
    source_disjoint = not (train_val_overlap or train_test_overlap or val_test_overlap)
    return {
        "present": True,
        "source_url": "https://doi.org/10.5281/zenodo.6228834",
        "license": "PolyForm-Noncommercial-1.0.0",
        "competition_use_authorized": None,
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": sha256_file(archive_path),
        "license_sha256": license_sha,
        "split_image_counts": image_counts,
        "train_annotation_sources": len(train_sources),
        "validation_annotation_sources": len(val_sources),
        "train_source_ratio_coverage": train_ratio_counts,
        "split_source_overlap_counts": {
            "train_val": len(train_val_overlap),
            "train_test": len(train_test_overlap),
            "val_test": len(val_test_overlap),
        },
        "source_split_disjoint": source_disjoint,
        "test_labels_used_for_training_or_selection": False,
        "train_validation_only_adapter": True,
        "blockers": [] if source_disjoint else ["GNMC source split overlap"],
    }


def sha256_file_from_zip(archive: ZipFile, member: str) -> str:
    digest = hashlib.sha256()
    with archive.open(member) as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def audit_live(root: Path, split_manifest: Path | None) -> dict:
    dataset = root / ".cache_videoora_data/LIVE_YT_VC"
    labels = dataset / "video_bbox_labels.csv"
    if not labels.is_file():
        return {"present": False, "blockers": ["LIVE labels missing"]}
    with labels.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    source_ids = {row["video"] for row in rows}
    video_files = list((dataset / "study_videos").glob("*.mp4")) if (dataset / "study_videos").is_dir() else []
    field_names = list(rows[0]) if rows else []
    frame_count = len([field for field in field_names if field.startswith("frame")])
    split_info = {
        "provided": False, "source_counts": None, "source_disjoint": None,
        "invalid_split_sources": 0, "missing_required_splits": ["train", "val"],
    }
    if split_manifest and split_manifest.is_file():
        raw_mapping = json.loads(split_manifest.read_text(encoding="utf-8"))
        invalid_split_sources = sum(
            split not in {"train", "val", "validation", "dev"}
            for split in raw_mapping.values()
        )
        mapping = normalize_train_val_source_split(raw_mapping) if not invalid_split_sources else {}
        source_splits: dict[str, set[str]] = {}
        for source, split in mapping.items():
            source_splits.setdefault(split, set()).add(source)
        missing_required_splits = sorted({"train", "val"} - set(source_splits))
        split_info = {
            "provided": True,
            "source_counts": {name: len(values) for name, values in source_splits.items()},
            "source_disjoint": sum(map(len, source_splits.values())) == len(set().union(*source_splits.values())),
            "unmapped_sources": len(source_ids - set(mapping)),
            "invalid_split_sources": invalid_split_sources,
            "missing_required_splits": missing_required_splits,
        }
    complete = (
        split_info["provided"] and split_info.get("unmapped_sources", 0) == 0
        and split_info.get("invalid_split_sources", 0) == 0
        and not split_info.get("missing_required_splits")
    )
    return {
        "present": True,
        "source_url": "https://github.com/steven413d/LIVE-YT-VideoCropping",
        "license_status": "not confirmed in local asset metadata",
        "competition_use_authorized": None,
        "labeled_sources": len(source_ids),
        "label_rows": len(rows),
        "sparse_frames_per_source": frame_count,
        "video_files_locally_present": len(video_files),
        "label_csv_sha256": sha256_file(labels),
        "source_split_manifest": split_info,
        "source_level_gate": complete and split_info.get("source_disjoint", False),
        "blockers": [] if complete and split_info.get("source_disjoint", False) else [
            reason for condition, reason in (
                (not split_info["provided"], "LIVE source split manifest is missing"),
                (split_info.get("unmapped_sources", 0) > 0, "LIVE labeled sources are unmapped"),
                (split_info.get("invalid_split_sources", 0) > 0, "LIVE source manifest contains non-train/val splits"),
                (bool(split_info.get("missing_required_splits")), "LIVE manifest must contain both train and val sources"),
                (split_info.get("source_disjoint") is False, "LIVE source split overlap"),
            ) if condition
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True,
                        help="root containing datasets/ and .cache_videoora_data/")
    parser.add_argument("--gnmc-zip", type=Path)
    parser.add_argument("--live-split-manifest", type=Path)
    parser.add_argument("--p1-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    gnmc_zip = args.gnmc_zip or args.data_root / ".cache_videoora_data/GNMC/GNMC.zip"
    p1_dir = args.p1_dir
    p1_files = []
    if p1_dir and p1_dir.is_dir():
        p1_files = sorted(path.relative_to(p1_dir).as_posix() for path in p1_dir.rglob("*")
                          if path.is_file() and path.suffix.lower() in {".safetensors", ".bin", ".json"})
    audit = {
        "schema_version": "aic.scale_stage0.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "official_aic_test_data_accessed": False,
        "official_score_used_for_training_or_preference_construction": False,
        "datasets": {
            "GAICD": audit_gaicd(args.data_root),
            "SACD": audit_sacd(args.data_root),
            "LIVE_YT_VC": audit_live(args.data_root, args.live_split_manifest),
            "GNMC": audit_gnmc(gnmc_zip),
        },
        "p1_parent": {
            "path_supplied": str(p1_dir) if p1_dir else None,
            "present": bool(p1_files),
            "manifest_or_weight_files": p1_files,
            "sha256": None,
            "parameter_count": None,
            "blocker": "P1 parent model, tokenizer, and adapter are not present in the local dataset/code checkout",
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "data_audit.json"
    json_path.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    hash_manifest = {
        "GAICD": audit["datasets"]["GAICD"].get("manifest_hashes", {}),
        "SACD": {
            "DOWNLOAD_STATUS.json": sha256_file(args.data_root / "datasets/SACD/DOWNLOAD_STATUS.json")
            if (args.data_root / "datasets/SACD/DOWNLOAD_STATUS.json").is_file() else None,
            "train.json": sha256_file(args.data_root / "datasets/SACD/train.json")
            if (args.data_root / "datasets/SACD/train.json").is_file() else None,
            "val.json": sha256_file(args.data_root / "datasets/SACD/val.json")
            if (args.data_root / "datasets/SACD/val.json").is_file() else None,
            "test.json": sha256_file(args.data_root / "datasets/SACD/test.json")
            if (args.data_root / "datasets/SACD/test.json").is_file() else None,
            "source_payload_tree_sha256": audit["datasets"]["SACD"].get("recomputed_source_payload", {}).get("tree_sha256"),
        },
        "LIVE_YT_VC": {
            "video_bbox_labels.csv": audit["datasets"]["LIVE_YT_VC"].get("label_csv_sha256"),
            "source_split_manifest": sha256_file(args.live_split_manifest)
            if args.live_split_manifest and args.live_split_manifest.is_file() else None,
        },
        "GNMC": {
            "archive_sha256": audit["datasets"]["GNMC"].get("archive_sha256"),
            "license_sha256": audit["datasets"]["GNMC"].get("license_sha256"),
        },
        "P1": {
            "path": str(p1_dir) if p1_dir else None,
            "weights_sha256": None,
            "adapter_sha256": None,
            "tokenizer_sha256": None,
        },
    }
    config_dir = args.output_dir.parent / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "dataset_manifest_hashes.json").write_text(
        json.dumps(hash_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    lines = [
        "# Scale SFT/DPO Asset Inventory",
        "",
        f"Audit time: {audit['created_at_utc']}",
        "",
        "No AIC official test media, labels, predictions, or hidden-test-derived information were read.",
        "Public-dataset test labels were not used for fitting, candidate selection, or threshold choice.",
        "",
    ]
    for name, item in audit["datasets"].items():
        lines += [f"## {name}", ""]
        if not item.get("present"):
            lines += ["Unavailable.", ""]
            continue
        for key in ("source_url", "license", "license_status", "competition_use_authorized", "labeled_sources", "media_count",
                    "unique_source_count", "official_split_source_counts", "images",
                    "subject_annotations", "split_annotation_counts", "split_image_overlap_counts",
                    "explicit_ranked_pair_count", "journal_train_candidate_count",
                    "journal_train_unrated_candidate_count", "journal_train_annotation_missing",
                    "manifest_hashes", "raw_tree", "video_files_locally_present", "source_split_manifest",
                    "split_image_counts", "source_split_disjoint", "source_tree_sha256_from_prior_validation"):
            if key in item:
                lines.append(f"- {key}: `{json.dumps(item[key], ensure_ascii=False)}`")
        lines.append(f"- blockers: `{json.dumps(item.get('blockers', []), ensure_ascii=False)}`")
        lines.append("")
    lines += ["## P1 Parent", "", json.dumps(audit["p1_parent"], indent=2), ""]
    (args.output_dir / "asset_inventory.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"data_audit": str(json_path), "asset_inventory": str(args.output_dir / "asset_inventory.md")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
