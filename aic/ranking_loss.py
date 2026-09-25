"""Bounded video-internal pair ranking; isolated RNG and masked timesteps."""
import torch
from torch.nn import functional as F


def pairwise_logistic(logits, targets, mask, generator, draws=1024, min_gap=.1):
    """Uniform ordered pair draws with replacement, then importance-gap filtering.

    Work/memory O(draws); repeated draws retain their multiplicity. An isolated
    CPU generator avoids changing dropout or training-order randomness.
    """
    if logits.ndim != 1 or targets.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("one matching vector per video required")
    if draws <= 0 or min_gap <= 0:
        raise ValueError("positive draw count and label gap required")
    z, y = logits[mask.bool()], targets[mask.bool()]
    if z.numel() < 2:
        return z.sum() * 0, 0
    pairs = torch.randint(z.numel(), (2, draws), generator=generator, device="cpu").to(z.device)
    i, j = pairs
    gap = y[i] - y[j]
    keep = gap.abs() >= min_gap
    count = int(keep.sum().item())
    if not count:
        return z.sum() * 0, 0
    margin = (z[i[keep]] - z[j[keep]]) * gap[keep].sign()
    return F.softplus(-margin).mean(), count
