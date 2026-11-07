"""Fixed-center scale candidate generation and Oracle@K diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from aic.scale_geometry import box_iou_xywh, scale_crop_from_center, scale_crop_xywh_exact


def build_scale_candidates(
    *,
    source_id: str,
    image_id: str,
    dataset_source: str,
    split: str,
    image_width: int,
    image_height: int,
    target_ratio: tuple[int, int],
    center_xy: tuple[float, float],
    scale_bins: Sequence[float],
    greedy_scale_id: int,
    stochastic_scale_ids: Sequence[int],
    grid_scale_ids: Sequence[int],
    max_candidates: int = 8,
) -> list[dict]:
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be explicit")
    if not scale_bins:
        raise ValueError("scale_bins cannot be empty")
    if max_candidates < 1:
        raise ValueError("max_candidates must be positive")
    requested = [(greedy_scale_id, "sft_greedy")]
    requested += [(index, "sft_stochastic") for index in stochastic_scale_ids]
    requested += [(index, "fixed_grid") for index in grid_scale_ids]
    if len(requested) > max_candidates:
        raise ValueError("requested candidate count exceeds configured K")

    by_geometry: dict[tuple[int, int, int], dict] = {}
    for scale_id, source in requested:
        if not isinstance(scale_id, int) or isinstance(scale_id, bool) or not 0 <= scale_id < len(scale_bins):
            raise ValueError(f"invalid scale_id: {scale_id}")
        crop = scale_crop_from_center(
            image_width, image_height, target_ratio, center_xy, float(scale_bins[scale_id])
        )
        key = crop[:3]
        if key not in by_geometry:
            by_geometry[key] = {
                "source_id": str(source_id),
                "image_id": str(image_id),
                "dataset_source": dataset_source,
                "split": split,
                "target_ratio": list(target_ratio),
                "position_source": "P1_frozen",
                "position_center_xy": list(center_xy),
                "scale_id": scale_id,
                "scale": float(scale_bins[scale_id]),
                "crop_xywh": list(scale_crop_xywh_exact(
                    image_width, image_height, target_ratio, center_xy, float(scale_bins[scale_id])
                )),
                "aic_bbox_xyw": list(crop[:3]),
                "candidate_sources": [source],
                "requested_count": 1,
            }
        else:
            entry = by_geometry[key]
            entry["requested_count"] += 1
            if source not in entry["candidate_sources"]:
                entry["candidate_sources"].append(source)
    return list(by_geometry.values())


def score_candidate_set(candidates: Sequence[dict], gt_crop_xywh: Sequence[float]) -> dict:
    if not candidates:
        raise ValueError("candidate set cannot be empty")
    candidate_contracts = {
        (row["dataset_source"], row["source_id"], row["image_id"], row["split"],
         tuple(row["target_ratio"]), tuple(row["position_center_xy"]))
        for row in candidates
    }
    if len(candidate_contracts) != 1:
        raise ValueError("candidate set mixes sources, images, ratios, or frozen centers")
    if candidates[0]["split"] == "test":
        raise ValueError("candidate Oracle scoring cannot use test split labels")
    scored = []
    for candidate in candidates:
        iou = box_iou_xywh(candidate["crop_xywh"], gt_crop_xywh)
        scored.append((candidate, iou))
    greedy = [item for item in scored if "sft_greedy" in item[0]["candidate_sources"]]
    if not greedy:
        raise ValueError("candidate set has no greedy action")
    greedy_iou = greedy[0][1]
    oracle_iou = max(score for _, score in scored)
    return {
        "source_id": candidates[0]["source_id"],
        "image_id": candidates[0]["image_id"],
        "dataset_source": candidates[0]["dataset_source"],
        "split": candidates[0]["split"],
        "greedy_deploy_iou": greedy_iou,
        "oracle_at_k_deploy_iou": oracle_iou,
        "oracle_gain": oracle_iou - greedy_iou,
        "unique_candidate_ratio": len(scored) / max(1, sum(
            int(candidate.get("requested_count", 1)) for candidate in candidates
        )),
        "scale_bin_coverage": len({row["scale_id"] for row in candidates}),
        "candidate_count": len(candidates),
        "position_fixed": all(row["position_source"] == "P1_frozen" for row in candidates),
    }


def source_macro_candidate_metrics(rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError("candidate metrics require rows")
    by_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        by_source[(str(row.get("dataset_source", "")), str(row["source_id"]))].append(row)
    result = {}
    for key in ("greedy_deploy_iou", "oracle_at_k_deploy_iou", "oracle_gain"):
        result[key] = sum(sum(row[key] for row in items) / len(items) for items in by_source.values()) / len(by_source)
    result["sources"] = len(by_source)
    result["samples"] = len(rows)
    result["positive_gain_source_fraction"] = sum(
        sum(row["oracle_gain"] for row in items) / len(items) > 0
        for items in by_source.values()
    ) / len(by_source)
    result["macro_unit"] = "source"
    return result


def paired_source_oracle_bootstrap(
    rows: Sequence[dict], seed: int, replicates: int = 5000
) -> dict:
    import random

    by_source: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        by_source[(str(row.get("dataset_source", "")), str(row["source_id"]))].append(
            float(row["oracle_gain"])
        )
    if not by_source or replicates < 100:
        raise ValueError("bootstrap needs source rows and at least 100 replicates")
    values = [sum(v) / len(v) for _, v in sorted(by_source.items())]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("oracle gains must be finite")
    rng = random.Random(seed)
    samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(replicates))
    return {
        "sources": len(values),
        "mean_oracle_gain": sum(values) / len(values),
        "ci95": [samples[math.floor(.025 * (replicates - 1))],
                 samples[math.floor(.975 * (replicates - 1))]],
        "bootstrap_unit": "source",
        "seed": seed,
        "replicates": replicates,
    }
