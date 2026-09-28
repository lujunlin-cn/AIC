"""Unified index schema.

Three record kinds, each in its own JSONL per dataset
(``<EXT_ROOT>/<dataset>/processed/{media,annotations,anomalies}.jsonl``) and
merged into ``<EXT_ROOT>/_registry/{media,annotations,anomalies}.parquet``:

media       one per media item (video file, image, or image-sequence folder)
annotation  one per annotation unit (one annotator x one media x one type)
anomaly     missing / corrupt / duplicate / misaligned items and the action taken

Supervision types are kept separate (``ANNOTATION_TYPES``); nothing is
converted into a generic "highlight" label.
"""
from __future__ import annotations

from typing import Any

MEDIA_KINDS = ("video_file", "image", "frame_folder", "feature_only", "remote_only")
DOWNLOAD_STATUS = ("pending", "queued", "downloading", "downloaded", "verified", "failed",
                   "missing", "metadata_only", "blocked", "referenced_existing", "sample_only")
PREPROCESS_STATUS = ("pending", "probed", "timeline_ok", "done", "failed", "not_needed")

# supervision type -> description; adapters are keyed by these names
ANNOTATION_TYPES = {
    "crop_box_sparse": "human crop window on annotated frames only (no smooth trajectory)",
    "crop_box_dense": "human crop window on every frame",
    "crop_box_derived": "interpolated/smoothed crop derived from sparse human boxes",
    "fixation_points": "eye-tracking fixation points per frame (binary map)",
    "saliency_map": "continuous saliency density map per frame",
    "salient_object_mask": "binary salient-object mask per annotated frame",
    "temporal_score_1d": "per-time-step importance score (e.g. replay/most-replayed)",
    "temporal_summary_derived": "algorithmic summary derived from a score (not human)",
    "temporal_segment_label": "labelled temporal segments (highlight / non-highlight / soft score)",
    "temporal_user_selection": "user-selected clip (personal highlight); unselected != negative",
    "shot_boundary": "shot transition intervals with type (cut / gradual)",
    "image_crop_candidates": "grid-anchor crop candidates with human MOS",
    "object_track_box": "single-object bounding box track with visibility flags",
    "object_masklet": "object mask track (RLE) with source (manual / auto)",
    "precomputed_features": "precomputed frame features (not raw media)",
}
ANNOTATION_SOURCES = ("human", "auto", "derived", "mixed", "unknown")

MEDIA_REQUIRED = ("uid", "dataset", "release", "item_id", "source_video_id", "group_id",
                  "official_split", "media_kind", "media_path", "download_status",
                  "preprocess_status")
ANNOT_REQUIRED = ("uid", "dataset", "media_uid", "annotation_type", "annotation_source",
                  "annotation_path", "coverage")


def media_record(**kw: Any) -> dict:
    base = {
        "uid": None, "dataset": None, "release": None, "item_id": None,
        "source_video_id": None,      # e.g. YouTube ID or dataset-native video ID
        "source_platform": None,      # youtube / dataset_native / flickr ...
        "group_id": None,             # leakage group; same source video -> same group
        "official_split": None,       # as published (train/val/test/unknown)
        "aic_split": None,            # our leakage-safe split (filled by splits step)
        "media_kind": None, "media_path": None, "media_reused_from": None,
        "download_url": None, "download_source": None, "file_size": None, "sha256": None,
        "download_status": "pending", "preprocess_status": "pending",
        "width": None, "height": None, "duration_s": None, "fps": None, "frame_count": None,
        "rotation": None, "codec": None, "has_audio": None,
        "timeline_path": None,        # npz with real pts/time_s per displayed frame
        "timeline_method": None,
        "frame_index_base": 0,        # frame indices in this index are 0-based display order
        "frame_file_pattern": None,   # for frame folders
        "annotation_types": [],
        "notes": None, "anomalies": [],
    }
    unknown = set(kw) - set(base)
    if unknown:
        raise KeyError(f"unknown media fields: {sorted(unknown)}")
    base.update(kw)
    return base


def annotation_record(**kw: Any) -> dict:
    base = {
        "uid": None, "dataset": None, "media_uid": None, "annotation_type": None,
        "annotation_source": None,    # human / auto / derived / mixed
        "annotator": None, "variant": None,   # e.g. aspect ratio "1-3"
        "annotation_path": None, "annotation_format": None,
        "coverage": None,             # dict: annotated frames / time span / fraction
        "label_semantics": None,      # plain-language meaning of the values
        "coord_format": None,         # original coordinate convention
        "coord_target": None,         # adapter output convention
        "time_reference": None,       # how frame numbers map to time
        "valid_mask": None,           # how unlabeled positions are masked
        "derived_from": None, "derivation": None,
        "notes": None,
    }
    unknown = set(kw) - set(base)
    if unknown:
        raise KeyError(f"unknown annotation fields: {sorted(unknown)}")
    base.update(kw)
    if base["annotation_type"] not in ANNOTATION_TYPES:
        raise ValueError(f"unknown annotation type {base['annotation_type']}")
    if base["annotation_source"] not in ANNOTATION_SOURCES:
        raise ValueError(f"bad annotation source {base['annotation_source']}")
    return base


def anomaly_record(dataset: str, item: str, kind: str, detail: str, action: str, **extra) -> dict:
    return {"dataset": dataset, "item": item, "kind": kind, "detail": detail,
            "action": action, **extra}


def validate(rows: list[dict], required: tuple[str, ...]) -> list[str]:
    errs = []
    seen = set()
    for i, r in enumerate(rows):
        for k in required:
            if k == "media_path" and (r.get("media_kind") in ("feature_only", "remote_only")
                                      or r.get("download_status") in ("missing", "failed", "metadata_only",
                                                                      "blocked", "queued", "pending")):
                continue  # absence of media is recorded by media_kind / download_status
            if r.get(k) in (None, ""):
                errs.append(f"row {i} ({r.get('uid')}): missing {k}")
        if r.get("uid") in seen:
            errs.append(f"row {i}: duplicate uid {r.get('uid')}")
        seen.add(r.get("uid"))
    return errs
