"""A monotonic ledger enforcing the v2 stage and total GPU-hour budgets."""

from __future__ import annotations

from datetime import datetime
import math


STAGE_LIMIT_HOURS = {
    "scale_sft": 4.0,
    "candidate_generation": 3.0,
    "preference_build": 1.0,
    "spatial_dpo": 4.0,
}
TOTAL_LIMIT_HOURS = 12.0


def empty_budget() -> dict:
    return {
        "budget_hours": TOTAL_LIMIT_HOURS,
        "stages": {
            stage: {"limit_hours": limit, "used_hours": 0.0, "runs": []}
            for stage, limit in STAGE_LIMIT_HOURS.items()
        },
        "total_used_hours": 0.0,
    }


def record_gpu_run(
    budget: dict,
    stage: str,
    started_at: str,
    ended_at: str,
    gpu_name: str,
    checkpoint: str,
    log_path: str,
) -> dict:
    if stage not in STAGE_LIMIT_HOURS:
        raise ValueError(f"unknown GPU stage: {stage}")
    start = datetime.fromisoformat(started_at)
    end = datetime.fromisoformat(ended_at)
    seconds = (end - start).total_seconds()
    if seconds <= 0:
        raise ValueError("GPU run end must be after start")
    hours = seconds / 3600.0
    stage_entry = budget["stages"][stage]
    next_stage_hours = float(stage_entry["used_hours"]) + hours
    next_total_hours = float(budget["total_used_hours"]) + hours
    if next_stage_hours > STAGE_LIMIT_HOURS[stage] + 1e-9:
        raise ValueError(f"{stage} GPU budget exceeded")
    if next_total_hours > TOTAL_LIMIT_HOURS + 1e-9:
        raise ValueError("total GPU budget exceeded")
    stage_entry["runs"].append({
        "started_at": started_at,
        "ended_at": ended_at,
        "duration_hours": hours,
        "gpu_name": str(gpu_name),
        "checkpoint": str(checkpoint),
        "log_path": str(log_path),
    })
    stage_entry["used_hours"] = next_stage_hours
    budget["total_used_hours"] = next_total_hours
    return budget
