"""Exact expected F for independent block actions and fixed box rewards.

This is a reference implementation. It does not load video data or infer GT.
Each cost is a positive ORIGINAL-frame count; each gain is the sum of valid
matched-frame IoUs in the block. GT count includes all GT frames in the video.
"""
from __future__ import annotations
from typing import Sequence
import torch
import torch.nn.functional as F


def expected_f(logits: torch.Tensor, costs: Sequence[int], gains: torch.Tensor,
               gt_count: int, empty_value: float = 1.0) -> torch.Tensor:
    if logits.ndim != 1 or gains.shape != logits.shape or len(costs) != logits.numel():
        raise ValueError('logits, costs, gains must have one entry per block')
    if not logits.is_floating_point() or not gains.is_floating_point():
        raise TypeError('logits and gains must be floating-point tensors')
    if gains.device != logits.device or gains.dtype != logits.dtype:
        raise ValueError('gains and logits must share device and dtype')
    if not isinstance(gt_count, int) or gt_count < 0:
        raise ValueError('gt_count must be a nonnegative integer')
    if not (0.0 <= empty_value <= 1.0):
        raise ValueError('empty_value must be in [0,1]')
    if not torch.isfinite(logits).all() or not torch.isfinite(gains).all():
        raise ValueError('nonfinite inputs')
    if any(not isinstance(d, int) or isinstance(d, bool) or d <= 0 for d in costs):
        raise ValueError('costs must be positive integer frame counts')
    bound = torch.tensor(costs, dtype=gains.dtype, device=gains.device)
    if bool((gains < 0).any()) or bool((gains > bound + 1e-7).any()):
        raise ValueError('block gains must be between zero and their frame counts')
    if float(gains.detach().sum()) > gt_count + 1e-7:
        raise ValueError('sum of gains cannot exceed GT count')
    p = logits.sigmoid()
    if gt_count == 0:
        # All nonempty actions have zero reward. Empty-set convention is explicit.
        return (1.0 - p).prod() * empty_value
    prob = logits.new_ones(1)
    mass = logits.new_zeros(1)
    for i, d in enumerate(costs):
        old_prob, old_mass = prob, mass
        prob = (1 - p[i]) * F.pad(old_prob, (0, d)) + p[i] * F.pad(old_prob, (d, 0))
        mass = ((1 - p[i]) * F.pad(old_mass, (0, d))
                + p[i] * F.pad(old_mass + gains[i] * old_prob, (d, 0)))
    k = torch.arange(prob.numel(), dtype=logits.dtype, device=logits.device)
    return (2.0 * mass / (k + gt_count)).sum()
