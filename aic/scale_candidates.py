"""Fixed-center scale candidate generation and Oracle@K diagnostics."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Sequence

from aic.scale_geometry import box_iou_xywh, scale_crop_from_center, scale_crop_xywh_exact, checked_scale_action


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
        checked = checked_scale_action(image_width, image_height, target_ratio, center_xy, float(scale_bins[scale_id]))
        if not checked["valid"]:
            continue
        crop = checked["aic_bbox_xyw"]
        key = tuple(crop[:3])
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
                "position_drift_px": checked["position_drift_px"],
                "total_requested_before_geometry_filter": len(requested),
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
    model_iou = max(score for row, score in scored if any(s in row["candidate_sources"] for s in ("sft_greedy", "sft_stochastic")))
    grid = [score for row, score in scored if "fixed_grid" in row["candidate_sources"]]
    return {
        "target_ratio": candidates[0]["target_ratio"],
        "sft_stochastic_oracle_at_k_deploy_iou": model_iou,
        "sft_stochastic_oracle_gain": model_iou - greedy_iou,
        "fixed_grid_oracle_at_k_deploy_iou": max(grid) if grid else None,
        "max_position_drift_px": max(row.get("position_drift_px", 0) for row in candidates),
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
        "legal_candidate_rate": sum(c.get("requested_count", 1) for c in candidates) / candidates[0].get("total_requested_before_geometry_filter", len(candidates)),
        "model_unique_candidate_count": sum(any(s in c["candidate_sources"] for s in ("sft_greedy", "sft_stochastic")) for c in candidates),
        "grid_unique_candidate_count": sum("fixed_grid" in c["candidate_sources"] for c in candidates),
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
    result["macro_unit"] = "source_within_dataset"
    datasets = sorted({key[0] for key in by_source})
    result["metrics_by_dataset"] = {}
    for dataset in datasets:
        subset = [v for k, v in by_source.items() if k[0] == dataset]
        result["metrics_by_dataset"][dataset] = {
            key: sum(sum(r[key] for r in v) / len(v) for v in subset) / len(subset)
            for key in ("greedy_deploy_iou", "oracle_at_k_deploy_iou", "oracle_gain", "sft_stochastic_oracle_at_k_deploy_iou", "sft_stochastic_oracle_gain")}
        result["metrics_by_dataset"][dataset]["sources"] = len(subset)
    for key in ("greedy_deploy_iou", "oracle_at_k_deploy_iou", "oracle_gain"):
        result[key] = sum(m[key] for m in result["metrics_by_dataset"].values()) / len(datasets)
    result["macro_unit"] = "dataset_macro_of_source_macros"
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
    from aic.scale_objectives import paired_source_bootstrap_ci
    datasets = sorted({d for d,s in by_source})
    grouped = {d:{s:sum(v)/len(v) for (dataset,s),v in by_source.items() if dataset == d} for d in datasets}
    by_dataset = {d:paired_source_bootstrap_ci(v,seed,replicates) for d,v in grouped.items()}
    rng = random.Random(seed)
    samples=[]
    for _ in range(replicates):
        dataset_means=[]
        for d in datasets:
            values=list(grouped[d].values())
            dataset_means.append(sum(rng.choice(values) for _ in values)/len(values))
        samples.append(sum(dataset_means)/len(dataset_means))
    samples.sort()
    return {"sources":len(by_source),"mean_oracle_gain":sum(v["mean_difference"] for v in by_dataset.values())/len(datasets),
            "ci95":[samples[math.floor(.025*(replicates-1))],samples[math.floor(.975*(replicates-1))]],
            "by_dataset":by_dataset,"bootstrap_unit":"source_stratified_by_dataset",
            "macro_unit":"dataset_macro_of_source_macros","seed":seed,"replicates":replicates}
