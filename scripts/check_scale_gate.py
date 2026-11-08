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
        criteria = config.get("promotion", {}).get("preference", {})
        if not criteria.get("evidence_sha256") or not criteria.get("minimum_pairs") or not criteria.get("minimum_sources"):
            blockers.append("preference minimum sample sizes lack locked coverage/stability evidence")
        elif (int(preference_audit.get("total_pairs") or 0) < criteria["minimum_pairs"]
              or int(preference_audit.get("unique_sources") or 0) < criteria["minimum_sources"]):
            blockers.append("preference coverage is below its preregistered evidence-based minimum")
        for field in ("input_contracts_verified", "quality_deltas_above_train_threshold"):
            if preference_audit.get(field) is not True:
                blockers.append("preference audit missing " + field)
        if not sft_metrics.get("checkpoint_sha256"):
            blockers.append("SFT policy/reference checkpoint identity is missing")
        if config.get("data", {}).get("preference_source_balanced_weights") is None:
            blockers.append("DPO dataset weights require train coverage calibration")
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
    # These requirements are common to every execution stage; a downstream
    # boolean cannot silently bypass missing upstream identities or permission.
    if stage in {"scale_sft", "spatial_dpo", "candidate_generation"}:
        training = config.get("training", {})
        for field in ("optimizer_learning_rate", "batch_size", "gradient_accumulation", "max_steps",
                      "precision", "optimizer", "scheduler", "seed", "eval_every"):
            if training.get(field) is None:
                blockers.append("locked training setting is missing: " + field)
        if training.get("sft_objective") != "causal_completion_sft_loss":
            blockers.append("actual SFT objective must be autoregressive JSON completion")
        if training.get("optimizer") != "AdamW" or training.get("scheduler") != "constant":
            blockers.append("executable backend requires the locked AdamW/constant recipe")
        if training.get("precision") != "bfloat16":
            blockers.append("executable Video-ORA backend requires the locked bfloat16 precision")
        if stage in {"scale_sft", "spatial_dpo"}:
            key = "preference_source_balanced_weights" if stage == "spatial_dpo" else "source_balanced_weights"
            weights = config.get("data", {}).get(key)
            if not isinstance(weights, dict) or not weights or any(
                not isinstance(v, (int, float)) or isinstance(v, bool)
                or not math.isfinite(v) or v <= 0 for v in weights.values()
            ):
                blockers.append("train-calibrated positive source weights are missing: " + key)
        if stage == "spatial_dpo":
            beta = training.get("dpo_beta")
            if not isinstance(beta, (int, float)) or isinstance(beta, bool) or not math.isfinite(beta) or beta <= 0:
                blockers.append("positive locked DPO beta is missing")
        if config.get("position_identity", {}).get("frozen_parent_isolation_verified") is not True:
            blockers.append("frozen P1 Position adapter isolation has not passed real-model smoke")
        if config.get("model_smoke", {}).get("real_model_forward_backward_passed") is not True:
            blockers.append("real Video-ORA load/forward/backward/checkpoint smoke has not passed")
        if audit.get("datasets", {}).get("LIVE_YT_VC", {}).get("cross_split_identical_image_groups", 0):
            blockers.append("LIVE byte-identical media crosses train/validation; quarantine sources first")
        coverage = config.get("position_coverage", {})
        if (coverage.get("status") != "audited" or coverage.get("coverage_requirement_locked") is not True
                or coverage.get("gate_passed") is not True):
            blockers.append("P1-conditioned Scale label coverage has not passed its train-locked requirement")
    if stage in {"candidate_generation", "preference_build", "spatial_dpo"}:
        if not assess_sft_metrics(sft_metrics):
            blockers.append("SFT measured deployment metrics are missing or fail promotion")
        expected = config.get("training", {}).get("sft_checkpoint_sha256")
        if not expected or expected != sft_metrics.get("checkpoint_sha256"):
            blockers.append("configured SFT/reference checkpoint differs from the measured SFT identity")
        if stage != "preference_build" and not config.get("training", {}).get("sft_initialization_checkpoint"):
            blockers.append("actual SFT initialization checkpoint path is missing")
        threshold = config.get("data", {}).get("geometry_preference_delta_iou")
        if (not isinstance(threshold, (int, float)) or isinstance(threshold, bool)
                or not math.isfinite(threshold) or threshold <= 0):
            blockers.append("positive train-calibrated geometry preference threshold is missing")
    if stage in {"preference_build", "spatial_dpo"}:
        if not assess_candidate_metrics(candidate_metrics):
            blockers.append("measured model-sampled Oracle gain has not passed; fixed grid is insufficient")
        if candidate_metrics.get("checkpoint_sha256") != sft_metrics.get("checkpoint_sha256"):
            blockers.append("candidate and SFT metrics refer to different checkpoints")
    if stage in {"scale_sft", "spatial_dpo", "candidate_generation"}:
        runtime = config.get("runtime", {})
        for field in ("model_dir", "runtime_dir", "parent_adapter_root", "parent_adapter_hashes",
                      "runtime_source_hashes", "scale_target_modules", "image_root", "arm", "max_image_side"):
            if not runtime.get(field):
                blockers.append("locked runtime field is missing: " + field)
        data = config.get("data", {})
        for field in ("train_manifest", "validation_manifest", "train_manifest_sha256", "validation_manifest_sha256"):
            if not data.get(field):
                blockers.append("hashed data manifest is missing: " + field)
        if data.get("scale_bins", {}).get("status") != "fitted_from_train_only_and_P1_coverage_locked":
            blockers.append("Scale bins are provisional until train P1 coverage analysis is locked")
        if config.get("training", {}).get("majority_scale_upper_bound") is None:
            blockers.append("train-calibrated scale collapse bound is missing")
        if config.get("training", {}).get("position_frozen") is not True:
            blockers.append("Position must stay frozen")
        if config.get("training", {}).get("temporal_path_frozen") is not True:
            blockers.append("temporal path must stay frozen")
    if stage != "scale_sft":
        count = config.get("training", {}).get("post_sft_inference_parameter_count")
        if not isinstance(count, int) or isinstance(count, bool) or not 0 < count <= 9_000_000_000:
            blockers.append("complete downstream inference parameter count has not passed 9B audit")
    return {
        "stage": stage,
        "gate_passed": not blockers,
        "blockers": blockers,
        "cuda_initialization_allowed": not blockers,
    }


def assess_sft_metrics(report: dict) -> bool:
    metrics = report.get("metrics_by_dataset_ratio", {})
    return bool(report.get("checkpoint_sha256")) and bool(metrics) and all(
        m.get("comparison") == "P1_scale1" and m.get("valid_action_rate") == 1.0
        and m.get("bootstrap", {}).get("mean_difference", -1) >= 0
        and m.get("max_position_drift_px", 100) <= 1 and m.get("gate_passed") is True
        for m in metrics.values())


def assess_candidate_metrics(report: dict) -> bool:
    groups = report.get("model_oracle_by_dataset_ratio", {})
    return (bool(report.get("checkpoint_sha256")) and report.get("model_generated") is True
            and report.get("reproducible_seed_manifest") is True and report.get("valid_action_rate") == 1.
            and report.get("max_position_drift_px", 100) <= 1 and bool(groups)
            and all(g.get("ci95", [-1])[0] > 0 and g.get("positive_source_fraction", 0) > 0
                    for g in groups.values()) and report.get("gate_passed") is True)


def assess_dpo_metrics(report: dict) -> bool:
    metrics = report.get("metrics_by_dataset_ratio", {})
    return bool(metrics) and report.get("reference_frozen_verified") is True and all(
        m.get("comparison") == "frozen_SFT" and m.get("bootstrap", {}).get("ci95", [-1])[0] > 0
        and m.get("max_position_drift_px", 100) <= 1 and m.get("valid_action_rate") == 1.
        and m.get("gate_passed") is True for m in metrics.values())


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
