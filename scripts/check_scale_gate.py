#!/usr/bin/env python3
"""Check a v2 stage gate before any GPU task is initialized."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def evaluate(stage: str, config: dict, audit: dict, preference_audit: dict,
             sft_metrics: dict, candidate_metrics: dict) -> dict:
    blockers = []
    datasets = audit.get("datasets", {})
    parent = config.get("parent", {})
    if stage == "scale_sft":
        sha_pattern = re.compile(r"^[0-9a-fA-F]{64}$")
        if not (sha_pattern.fullmatch(str(parent.get("weights_sha256", "")))
                and sha_pattern.fullmatch(str(parent.get("adapter_sha256", "")))):
            blockers.append("verified P1 base/adapter SHA256 is missing")
        if not sha_pattern.fullmatch(str(parent.get("tokenizer_sha256", ""))):
            blockers.append("verified parent tokenizer SHA256 is missing")
        parameter_count = parent.get("parameter_count")
        if (not isinstance(parameter_count, int) or isinstance(parameter_count, bool)
                or parameter_count <= 0):
            blockers.append("complete active inference parameter count is missing")
        elif parameter_count > 9_000_000_000:
            blockers.append("active inference parameter count exceeds the official 9B cap")
        post_sft_count = config.get("training", {}).get("post_sft_inference_parameter_count")
        if (not isinstance(post_sft_count, int) or isinstance(post_sft_count, bool)
                or post_sft_count <= 0):
            blockers.append("post-SFT active inference parameter count is missing")
        elif post_sft_count > 9_000_000_000:
            blockers.append("post-SFT inference parameter count exceeds the official 9B cap")
        allowlist = config.get("training", {}).get("trainable_parameter_allowlist")
        if not isinstance(allowlist, list) or not allowlist or not all(
            isinstance(name, str) and name for name in allowlist
        ):
            blockers.append("exact spatial adapter trainable-parameter allowlist is missing")
        if not sha_pattern.fullmatch(str(config.get("training", {}).get(
                "target_module_inventory_sha256", ""))):
            blockers.append("P1 adapter target-module inventory SHA256 is missing")
        scale_bins = config.get("data", {}).get("scale_bins", {})
        values = scale_bins.get("values")
        valid_bins = (
            isinstance(values, list) and len(values) >= 2
            and all(isinstance(value, (int, float)) and not isinstance(value, bool)
                    and math.isfinite(value) and 0 < value <= 1 for value in values)
            and values == sorted(set(values)) and abs(values[-1] - 1.0) <= 1e-9
            and scale_bins.get("count") == len(values)
        )
        if not valid_bins:
            blockers.append("scale bins have not been fitted from train-only labels")
        mismatch_threshold = config.get("data", {}).get("minimum_position_conditioned_iou")
        if (not isinstance(mismatch_threshold, (int, float))
                or isinstance(mismatch_threshold, bool) or not math.isfinite(mismatch_threshold)
                or not 0 <= mismatch_threshold <= 1):
            blockers.append("position-mismatch threshold is not locked from training-only analysis")
        if not datasets.get("LIVE_YT_VC", {}).get("source_level_gate"):
            blockers.append("LIVE source-level train/validation manifest gate has not passed")
        if not datasets.get("GNMC", {}).get("source_split_disjoint"):
            blockers.append("GNMC source-disjoint train/validation audit has not passed")
        if not datasets.get("GAICD", {}).get("train_data_integrity_ready"):
            blockers.append("GAICD journal train annotation integrity audit has not passed")
        training_sources = config.get("data", {}).get("training_sources")
        if not isinstance(training_sources, list) or not training_sources:
            blockers.append("Scale-SFT training sources are not explicitly locked")
        else:
            for dataset_name in training_sources:
                if datasets.get(dataset_name, {}).get("competition_use_authorized") is not True:
                    blockers.append(f"{dataset_name} competition-use license review is incomplete")
    elif stage == "candidate_generation":
        if sft_metrics.get("gate_passed") is not True:
            blockers.append("Scale-SFT deploy-path promotion gate has not passed")
        if not sft_metrics.get("checkpoint_sha256"):
            blockers.append("Scale-SFT checkpoint identity is missing")
    elif stage == "preference_build":
        if candidate_metrics.get("gate_passed") is not True:
            blockers.append("candidate OracleGain promotion gate has not passed")
        if not candidate_metrics.get("checkpoint_sha256"):
            blockers.append("candidate metrics are not bound to a checkpoint")
        preference_sources = config.get("data", {}).get("preference_sources")
        if not isinstance(preference_sources, list) or not preference_sources:
            blockers.append("preference data sources are not explicitly locked")
        else:
            for dataset_name in preference_sources:
                if datasets.get(dataset_name, {}).get("competition_use_authorized") is not True:
                    blockers.append(f"{dataset_name} competition-use license review is incomplete")
    elif stage == "spatial_dpo":
        if candidate_metrics.get("gate_passed") is not True:
            blockers.append("candidate gate has not passed")
        if preference_audit.get("status") != "audited":
            blockers.append("preference audit has not completed")
        if int(preference_audit.get("total_pairs") or 0) < 2000:
            blockers.append("valid preference pair count is below the plan minimum 2000")
        if int(preference_audit.get("unique_sources") or 0) < 300:
            blockers.append("unique preference source count is below the plan minimum 300")
        if not sft_metrics.get("checkpoint_sha256"):
            blockers.append("SFT policy/reference checkpoint identity is missing")
        if preference_audit.get("source_split_overlap_checked") is not True:
            blockers.append("preference sources are not bound to the audited source split manifest")
        if preference_audit.get("all_pairs_train_split") is not True:
            blockers.append("preference audit contains non-training pairs")
        if preference_audit.get("pair_limit_passed") is not True:
            blockers.append("preference pair-per-image cap has not passed")
        if preference_audit.get("mos_semantics_preserved") is not True:
            blockers.append("GAICD MOS semantics were not verified")
        if preference_audit.get("sacd_subject_condition_preserved") is not True:
            blockers.append("SACD subject conditions were not verified")
        for dataset_name, pair_count in (preference_audit.get("source_counts_by_dataset") or {}).items():
            if pair_count and datasets.get(dataset_name, {}).get("competition_use_authorized") is not True:
                blockers.append(f"{dataset_name} competition-use license review is incomplete")
    else:
        raise ValueError(f"unknown stage: {stage}")
    return {
        "stage": stage,
        "gate_passed": not blockers,
        "blockers": blockers,
        "cuda_initialization_allowed": not blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("scale_sft", "candidate_generation", "preference_build", "spatial_dpo"), required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--preference-audit", type=Path, required=True)
    parser.add_argument("--sft-metrics", type=Path)
    parser.add_argument("--candidate-metrics", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.stage, load_json(args.run_config), load_json(args.data_audit),
        load_json(args.preference_audit),
        load_json(args.sft_metrics) if args.sft_metrics else {},
        load_json(args.candidate_metrics) if args.candidate_metrics else {},
    )
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0 if result["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
