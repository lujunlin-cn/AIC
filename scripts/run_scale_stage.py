#!/usr/bin/env python3
"""Execute one v2 stage after its gate; CUDA workers have a persistent watchdog."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import importlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from aic.scale_budget import STAGE_LIMIT_HOURS, TOTAL_LIMIT_HOURS, empty_budget, validate_budget
from aic.scale_data import read_jsonl
from scripts.check_scale_gate import evaluate, load_json


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load_initialization(backend, path: Path, expected_sha: str):
    import hashlib
    import torch

    manifest = load_json(path / "manifest.json")
    payload_path = path / "trainer.pt"
    actual_sha = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    if actual_sha != expected_sha or manifest.get("trainer_sha256") != actual_sha:
        raise ValueError("initial policy/SFT checkpoint identity mismatch")
    if hasattr(backend, "model_contract") and manifest.get("model_contract") != backend.model_contract:
        raise ValueError("SFT checkpoint changed bins, preprocessing, parent identity or adapter contract")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    named = dict(backend.model.named_parameters())
    if set(payload["parameters"]) != {n for n,p in named.items() if p.requires_grad}:
        raise ValueError("SFT initialization must cover exactly the isolated Scale allowlist")
    with torch.no_grad():
        for name, value in payload["parameters"].items():
            if name not in named or named[name].shape != value.shape:
                raise ValueError("initial policy parameter names/shapes mismatch")
            named[name].copy_(value.to(named[name].device))


def run_preferences(config: dict, rows: list[dict], output: Path):
    from aic.scale_preferences import (bind_preference_input, audit_preference_pairs, gaicd_scale_preferences,
                                       geometry_preference_pairs, sacd_scale_preferences)

    settings = config["data"]
    bins = settings["scale_bins"]["values"]
    if any(row.get("split") != "train" for row in rows):
        raise ValueError("preference construction accepts train manifest rows only")
    pairs = []
    views_with_preferences = 0
    for row in rows:
        dataset = row["dataset_source"]
        if dataset not in settings["preference_sources"]:
            raise ValueError("preference source is not registered in run config")
        if dataset in {"LIVE_YT_VC", "GNMC"}:
            produced = geometry_preference_pairs(row, bins, settings["geometry_preference_delta_iou"],
                                                 settings["max_pairs_per_image"])
        elif dataset == "GAICD":
            produced = gaicd_scale_preferences(row, bins, settings["gaicd_minimum_mos_delta"],
                                               settings["max_pairs_per_image"])
        elif dataset == "SACD":
            produced = sacd_scale_preferences(row, bins, settings["max_pairs_per_image"])
        else:
            raise ValueError("unknown preference source")
        views_with_preferences += bool(produced)
        from aic.scale_videoora import build_prompt
        preprocessing = {"max_image_side": config["runtime"]["max_image_side"], "alignment": 32,
                         "color": "RGB", "resize": "PIL_BICUBIC", "prompt_version": "scale_json_v1"}
        for pair in produced:
            pair["candidate_provenance"] = "fixed_scale_grid_train" if dataset in {"LIVE_YT_VC", "GNMC"} else "human_rated_or_ranked_candidates"
            pairs.append(bind_preference_input(pair, row, build_prompt(row, bins), preprocessing))
    source_splits = load_json(Path(settings["source_split_manifest"]))
    audit = audit_preference_pairs(pairs, source_splits, settings["max_pairs_per_image"])
    audit.update(eligible_input_views=len(rows), views_with_preferences=views_with_preferences,
                 preference_pair_rate=views_with_preferences / len(rows) if rows else 0.,
                 pairs_per_eligible_input_view=len(pairs) / len(rows) if rows else 0.)
    criteria = config["promotion"]["preference"]
    audit["input_contracts_verified"] = all(p.get("input_contract_sha256") for p in pairs)
    audit["quality_deltas_above_train_threshold"] = all(p["score_delta"] >= (
        settings["gaicd_minimum_mos_delta"] if p["score_name"] == "MOS" else settings["geometry_preference_delta_iou"])
        for p in pairs)
    audit["gate_passed"] = (criteria.get("evidence_sha256") is not None
                            and audit["total_pairs"] >= criteria["minimum_pairs"]
                            and audit["unique_sources"] >= criteria["minimum_sources"]
                            and audit["duplicate_pair_count"] == 0 and audit["pair_limit_passed"]
                            and audit["input_contracts_verified"] and audit["quality_deltas_above_train_threshold"])
    output.mkdir(parents=True, exist_ok=False)
    with (output / "preferences.jsonl").open("w") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, allow_nan=False) + "\n")
    write_json(output / "preference_audit.json", audit)
    return audit


def run_candidates(backend, config: dict, rows: list[dict], output: Path):
    from aic.scale_candidates import (build_scale_candidates, paired_source_oracle_bootstrap,
                                      score_candidate_set, source_macro_candidate_metrics)

    settings = config["action_protocol"]
    bins = config["data"]["scale_bins"]["values"]
    seed = config["training"]["seed"]
    noise = config["data"]["geometry_preference_delta_iou"]
    if noise is None or noise <= 0:
        raise ValueError("candidate gain threshold must be fixed from training-only noise analysis")
    candidate_rows, scores = [], []
    failures, attempted, valid = [], 0, 0
    backend.model.eval()
    grid_ids = settings["grid_scale_ids"][:2] if settings["candidate_k"] == 4 else settings["grid_scale_ids"]
    stochastic_count = settings["candidate_k"] - 1 - len(grid_ids)
    if stochastic_count < 1:
        raise ValueError("K must reserve at least one stochastic candidate beyond greedy and grid")
    for index, row in enumerate(rows):
        if row.get("split") != "val" or row["dataset_source"] not in {"LIVE_YT_VC", "GNMC"}:
            raise ValueError("candidate-IoU Oracle requires public LIVE/GNMC validation rows")
        actions = []
        seeds = [seed + index * stochastic_count + k for k in range(stochastic_count)]
        for mode, action_seed in [("greedy", None)] + [("stochastic", s) for s in seeds]:
            attempted += 1
            try:
                action = backend.predict([row], seed=action_seed,
                    temperature=settings["candidate_temperature"] if mode == "stochastic" else None)[0]
                if type(action) is not int or not 0 <= action < len(bins):
                    raise ValueError("invalid Scale Action")
                valid += 1; actions.append(action)
            except ValueError as exc:
                failures.append({"row": index, "mode": mode, "seed": action_seed, "error": str(exc)})
                actions.append(None)
        if actions[0] is None:
            continue
        candidates = build_scale_candidates(
            source_id=row["source_id"], image_id=row["image_id"], dataset_source=row["dataset_source"],
            split="val", image_width=row["image_width"], image_height=row["image_height"],
            target_ratio=tuple(row["target_ratio"]), center_xy=tuple(row["center_xy"]),
            scale_bins=bins, greedy_scale_id=actions[0], stochastic_scale_ids=[a for a in actions[1:] if a is not None],
            grid_scale_ids=grid_ids, max_candidates=settings["candidate_k"])
        try:
            score = score_candidate_set(candidates, row["gt_crop_xywh"])
        except ValueError as exc:
            failures.append({"row": index, "error": str(exc)});continue
        score["generation_seeds"] = seeds
        scores.append(score); candidate_rows.extend(candidates)
    metrics = source_macro_candidate_metrics(scores) if scores else {"metrics_by_dataset": {}}
    by_group = {}
    from aic.scale_objectives import paired_source_bootstrap_ci
    from collections import defaultdict
    for dataset, ratio in sorted({(r["dataset_source"], tuple(r["target_ratio"])) for r in scores}):
        values = defaultdict(list)
        for row in scores:
            if (row["dataset_source"], tuple(row["target_ratio"])) == (dataset, ratio):
                values[row["source_id"]].append(row["sft_stochastic_oracle_gain"])
        deltas = {s: sum(v)/len(v) for s,v in values.items()}
        ci = paired_source_bootstrap_ci(deltas, seed, replicates=1000)
        ci["useful_gain_source_fraction"] = sum(v > noise for v in deltas.values()) / len(deltas)
        by_group[f"{dataset}:{ratio[0]}:{ratio[1]}"] = ci
    coverage = len({r["scale_id"] for r in candidate_rows})
    max_drift = max((r["position_drift_px"] for r in candidate_rows), default=None)
    legal_rate = sum(r["legal_candidate_rate"] for r in scores)/len(scores) if scores else 0.
    metrics.update(legal_candidate_rate=legal_rate, model_oracle_by_dataset_ratio=by_group, scale_bin_coverage=coverage,
        candidate_k=settings["candidate_k"], valid_action_rate=valid/max(1, attempted),
        max_position_drift_px=max_drift, failures=failures, sample_scores=scores,
        checkpoint_sha256=config["training"]["sft_checkpoint_sha256"],
        model_generated=True, reproducible_seed_manifest=True,
        gate_passed=bool(by_group) and all(g["ci95"][0] > 0 and
            g["useful_gain_source_fraction"] >= config["promotion"]["candidates"]["minimum_positive_source_fraction"]
            for g in by_group.values()) and valid == attempted and legal_rate == 1. and coverage >= 3 and max_drift <= 1)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "candidate_manifest.jsonl").open("w") as handle:
        for row in candidate_rows:
            handle.write(json.dumps(row) + "\n")
    write_json(output / "oracle_metrics.json", metrics)
    return metrics


def cuda_watchdog(args, command: list[str]) -> int:
    """Hold one ledger lock throughout the child, including model loading."""
    ledger_path = args.budget
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = load_json(ledger_path) or empty_budget()
        if ledger.get("active_run"):
            raise ValueError("an unfinished GPU session is recorded; reconcile it before launching another")
        validate_budget(ledger)
        stage = ledger["stages"][args.stage]
        remaining_hours = min(STAGE_LIMIT_HOURS[args.stage] - stage["used_hours"],
                              TOTAL_LIMIT_HOURS - ledger["total_used_hours"])
        remaining = remaining_hours * 3600
        if remaining <= 30:
            raise ValueError("insufficient remaining GPU budget, including shutdown reserve")
        try:
            gpu_result = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
                                        capture_output=True, text=True, check=False)
            gpu_name = gpu_result.stdout.strip() if gpu_result.returncode == 0 else "unavailable"
        except OSError:
            gpu_name = "unavailable"
        started = datetime.now(timezone.utc).isoformat()
        ledger["active_run"] = {"stage": args.stage, "started_at": started,
                                "command": command, "output": str(args.output)}
        write_json(ledger_path, ledger)
        start = time.monotonic()
        process = None
        try:
            env = dict(os.environ, AIC_SCALE_BUDGETED_WORKER="1", HF_HUB_OFFLINE="1",
                       TRANSFORMERS_OFFLINE="1")
            process = subprocess.Popen(command, env=env, start_new_session=True)
            try:
                return process.wait(timeout=remaining - 30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                return 124
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            hours = (time.monotonic() - start) / 3600
            stage["used_hours"] += hours
            ledger["total_used_hours"] += hours
            stage["runs"].append({"started_at": started, "ended_at": datetime.now(timezone.utc).isoformat(),
                                  "duration_hours": hours, "exit_code": process.returncode if process else None,
                                  "output": str(args.output), "gpu_name": gpu_name})
            ledger.pop("active_run")
            ledger["budget_exceeded"] = (stage["used_hours"] > STAGE_LIMIT_HOURS[args.stage]
                                          or ledger["total_used_hours"] > TOTAL_LIMIT_HOURS)
            write_json(ledger_path, ledger)


def main() -> int:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--stage", choices=tuple(STAGE_LIMIT_HOURS), required=True)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--data-audit", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--val", type=Path, required=True)
    parser.add_argument("--preference-audit", type=Path, required=True)
    parser.add_argument("--sft-metrics", type=Path)
    parser.add_argument("--candidate-metrics", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--backend", default="aic.scale_videoora:create_backend")
    parser.add_argument("--budget", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = load_json(args.run_config)
    gate = evaluate(args.stage, config, load_json(args.data_audit), load_json(args.preference_audit),
                    load_json(args.sft_metrics) if args.sft_metrics else {},
                    load_json(args.candidate_metrics) if args.candidate_metrics else {})
    if not gate["gate_passed"]:
        print(json.dumps(gate, indent=2))
        return 2
    import hashlib
    for path, key in ((args.train, "train_manifest_sha256"), (args.val, "validation_manifest_sha256")):
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != config["data"].get(key):
            raise ValueError("data manifest is not bound to the locked config: " + str(path))
    train, val = read_jsonl(args.train), read_jsonl(args.val)
    if any(row.get("parent_identity_sha256") != config["parent"].get("identity_sha256")
           for row in train + val):
        raise ValueError("a manifest P1 center is not bound to the locked complete parent identity")
    from aic.scale_execution import run_training, validate_manifests

    validate_manifests(train, val)
    # All stages, including the CPU preference builder, use the same declared
    # geometry arm. Otherwise paired transpose views double the image pair cap
    # and disagree with the SFT/candidate manifests.
    ratio = [9, 16] if config["runtime"]["arm"] == "a" else [16, 9]
    train = [row for row in train if row["target_ratio"] == ratio]
    val = [row for row in val if row["target_ratio"] == ratio]
    if not train or not val:
        raise ValueError("the configured geometry arm requires nonempty train/validation data")
    if args.stage == "preference_build":
        run_preferences(config, train, args.output)
        return 0
    if args.device.startswith("cuda"):
        if config.get("formal_gpu_training_authorized") is not True:
            raise ValueError("formal GPU training is disabled for this CPU repair delivery")
        if args.worker:
            if os.environ.get("AIC_SCALE_BUDGETED_WORKER") != "1":
                raise ValueError("CUDA worker requires its persistent-budget supervisor")
        else:
            if args.budget is None or args.budget.resolve() != Path(config["gpu_budget_ledger"]).resolve():
                raise ValueError("CUDA execution requires the single persistent configured --budget ledger")
            return cuda_watchdog(args, [sys.executable, "-m", "scripts.run_scale_stage", *sys.argv[1:], "--worker"])
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    module, name = args.backend.split(":", 1)
    import random
    import torch
    random.seed(config["training"]["seed"])
    torch.manual_seed(config["training"]["seed"])
    backend = getattr(importlib.import_module(module), name)(config, args.device)
    if args.stage in {"spatial_dpo", "candidate_generation"}:
        load_initialization(backend, Path(config["training"]["sft_initialization_checkpoint"]),
                            config["training"]["sft_checkpoint_sha256"])
    if args.stage == "candidate_generation":
        result = run_candidates(backend, config, val, args.output)
    else:
        result = run_training(backend, train, val, config, args.output, args.stage, args.resume)
    write_json(args.output / "execution.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
