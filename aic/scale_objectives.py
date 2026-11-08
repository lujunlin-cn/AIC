"""SFT/DPO objective and source-level evaluation primitives."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Iterable, Sequence

import torch
from torch.nn import functional as F


def scale_sft_loss(logits: torch.Tensor, scale_ids: torch.Tensor) -> torch.Tensor:
    if logits.ndim != 2 or scale_ids.ndim != 1 or logits.shape[0] != scale_ids.shape[0]:
        raise ValueError("scale logits must be [batch, bins] and labels [batch]")
    return F.cross_entropy(logits.float(), scale_ids.long())


def causal_completion_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    completion_mask: torch.Tensor,
    *, already_shifted: bool = False,
) -> torch.Tensor:
    if logits.ndim != 3 or labels.ndim != 2 or completion_mask.ndim != 2:
        raise ValueError("expected logits [B,T,V] and labels/mask [B,T]")
    if logits.shape[:2] != labels.shape or labels.shape != completion_mask.shape:
        raise ValueError("logits, labels, and completion mask must align")
    shifted_logits = (logits if already_shifted else logits[:, :-1]).float()
    shifted_labels = (labels if already_shifted else labels[:, 1:]).long()
    shifted_mask = (completion_mask if already_shifted else completion_mask[:, 1:]).bool() & shifted_labels.ne(-100)
    if not torch.any(shifted_mask):
        raise ValueError("batch contains no completion tokens")
    token_losses = F.cross_entropy(
        shifted_logits.reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1), ignore_index=-100, reduction="none",
    ).reshape_as(shifted_labels)
    return token_losses.masked_select(shifted_mask).mean()


def dpo_loss(
    policy_chosen_logps: torch.Tensor,
    policy_rejected_logps: torch.Tensor,
    reference_chosen_logps: torch.Tensor,
    reference_rejected_logps: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    tensors = (policy_chosen_logps, policy_rejected_logps,
               reference_chosen_logps, reference_rejected_logps)
    if any(t.ndim != 1 for t in tensors) or len({tuple(t.shape) for t in tensors}) != 1:
        raise ValueError("DPO log probabilities must be equally shaped [batch] tensors")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be finite and positive")
    policy_margin = policy_chosen_logps.float() - policy_rejected_logps.float()
    reference_margin = reference_chosen_logps.float() - reference_rejected_logps.float()
    return -F.logsigmoid(beta * (policy_margin - reference_margin)).mean()


def causal_completion_logps(
    logits: torch.Tensor,
    labels: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    """Sum next-token log probabilities over completion tokens per sequence."""
    if logits.ndim != 3 or labels.ndim != 2 or completion_mask.ndim != 2:
        raise ValueError("expected logits [B,T,V] and labels/mask [B,T]")
    if logits.shape[:2] != labels.shape or labels.shape != completion_mask.shape:
        raise ValueError("logits, labels, and completion mask must align")
    if labels.shape[1] < 2:
        raise ValueError("sequence must contain a prompt token and a completion token")
    shifted_logits = logits[:, :-1].float()
    shifted_labels = labels[:, 1:].long()
    shifted_mask = completion_mask[:, 1:].bool() & shifted_labels.ne(-100)
    token_logps = F.log_softmax(shifted_logits, dim=-1).gather(
        dim=-1, index=shifted_labels.clamp_min(0).unsqueeze(-1)
    ).squeeze(-1)
    token_logps = token_logps.masked_fill(~shifted_mask, 0.0)
    return token_logps.sum(dim=-1)


def source_macro_mean(rows: Iterable[dict], value_key: str = "iou") -> float:
    by_source: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(value_key)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError(f"{value_key} must be finite")
        by_source[(str(row.get("dataset_source", "")), str(row["source_id"]))].append(float(value))
    if not by_source:
        raise ValueError("cannot macro-average an empty source set")
    return sum(sum(values) / len(values) for values in by_source.values()) / len(by_source)


def paired_source_bootstrap_ci(
    per_source_differences: dict[str, float],
    seed: int,
    replicates: int = 5000,
) -> dict:
    if not per_source_differences:
        raise ValueError("paired bootstrap requires at least one source")
    if replicates < 100:
        raise ValueError("replicates must be >= 100")
    values = [float(value) for _, value in sorted(per_source_differences.items())]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("source differences must be finite")
    rng = random.Random(seed)
    samples = []
    for _ in range(replicates):
        samples.append(sum(rng.choice(values) for _ in values) / len(values))
    samples.sort()
    lower = samples[math.floor(0.025 * (replicates - 1))]
    upper = samples[math.floor(0.975 * (replicates - 1))]
    return {
        "sources": len(values),
        "mean_difference": sum(values) / len(values),
        "ci95": [lower, upper],
        "positive_source_fraction": sum(value > 0 for value in values) / len(values),
        "negative_source_fraction": sum(value < 0 for value in values) / len(values),
        "seed": seed,
        "replicates": replicates,
        "bootstrap_unit": "source",
    }


def scale_collapse_rate(scale_ids: Sequence[int]) -> float:
    if not scale_ids:
        raise ValueError("scale_ids cannot be empty")
    counts: dict[int, int] = defaultdict(int)
    for scale_id in scale_ids:
        counts[int(scale_id)] += 1
    return max(counts.values()) / len(scale_ids)


def position_drift_metrics(before: dict[str, Sequence[float]],
                           after: dict[str, Sequence[float]]) -> dict:
    if set(before) != set(after) or not before:
        raise ValueError("before/after center maps must have identical nonempty source keys")
    distances = []
    max_axis_drift = 0.0
    for source_id in sorted(before):
        left, right = before[source_id], after[source_id]
        if (len(left) != 2 or len(right) != 2
                or not all(math.isfinite(float(v)) for v in (*left, *right))):
            raise ValueError(f"invalid position center for source {source_id}")
        dx, dy = float(right[0]) - float(left[0]), float(right[1]) - float(left[1])
        distances.append(math.hypot(dx, dy))
        max_axis_drift = max(max_axis_drift, abs(dx), abs(dy))
    return {
        "sources": len(distances),
        "mean_euclidean_drift_px": sum(distances) / len(distances),
        "max_axis_drift_px": max_axis_drift,
        "within_one_pixel_tolerance": max_axis_drift <= 1.0,
    }
