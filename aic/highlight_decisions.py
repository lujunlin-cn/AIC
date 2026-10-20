"""Deterministic candidate identities and budget-preserving decisions."""
from __future__ import annotations


def diverse_candidates(scores, k=4, center=64, separation=16):
    """Include head argmax and center, then distinct score peaks without GT."""
    order = sorted(range(len(scores)), key=lambda j: (-float(scores[j]), j))
    take = [order[0]]
    if center not in take:
        take.append(center)
    for j in order:
        if len(take) >= k:
            break
        if all(abs(j-t) >= separation for t in take):
            take.append(j)
    for j in order:
        if len(take) >= k:
            break
        if j not in take:
            take.append(j)
    if len(take) != k:
        raise ValueError("Not enough unique candidates")
    return sorted(take)


def center_window(n, keep=.85):
    k = min(n, max(0, round(n*keep)))
    start = (n-k)//2
    return range(start, start+k)
