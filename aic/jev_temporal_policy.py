"""PTS-safe, bounded temporal policy for the Jev/Valen route.

This module contains only the decision contract.  A Jev backend supplies
candidate scores or a Choice response; this code owns the part that must stay
deterministic across models: PTS validation, half-open interval semantics,
equal-budget phase candidates, centre fallback and mapping back to original
decoded frame IDs.

The contract is deliberately narrower than free-form temporal generation.  It
lets us test whether Jev can correct a strong continuous centre prior without
allowing arbitrary frame-number generation to destroy the competition budget.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Mapping, Sequence

import numpy as np


CHOICES = ("left", "keep", "right")
_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"


@dataclass(frozen=True)
class PTSContract:
    """Validated original-axis timing metadata."""

    pts: np.ndarray
    duration: float


@dataclass(frozen=True)
class PhaseCandidates:
    """Equal-budget candidate frame IDs and their PTS windows."""

    frame_ids: Mapping[str, np.ndarray]
    windows: Mapping[str, tuple[float, float]]
    start_indices: Mapping[str, int]
    budget: int
    shift_frames: int


def validate_pts(pts: Sequence[float], *, min_frames: int = 2) -> PTSContract:
    """Validate and normalize decoded PTS without assuming a constant FPS."""
    values = np.asarray(pts, dtype=np.float64)
    if values.ndim != 1 or len(values) < min_frames:
        raise ValueError("PTS must be a one-dimensional sequence with enough frames")
    if not np.all(np.isfinite(values)):
        raise ValueError("PTS contains non-finite values")
    if np.any(np.diff(values) <= 0):
        raise ValueError("PTS must be strictly increasing")
    values = values - values[0]
    step = float(np.median(np.diff(values)))
    if not math.isfinite(step) or step <= 0:
        raise ValueError("invalid PTS step")
    return PTSContract(values, float(values[-1] + step))


def half_open_indices(pts: Sequence[float], start: float, end: float) -> np.ndarray:
    """Return original frame indices in the half-open interval ``[start,end)``."""
    contract = validate_pts(pts)
    start, end = float(start), float(end)
    if not (math.isfinite(start) and math.isfinite(end) and 0 <= start < end <= contract.duration + 1e-9):
        raise ValueError("invalid half-open interval")
    ids = np.flatnonzero((contract.pts >= start) & (contract.pts < end)).astype(np.int64)
    if not len(ids):
        raise ValueError("interval contains no decoded frame")
    return ids


def merge_events(events: Sequence[Sequence[float]], duration: float) -> list[tuple[float, float]]:
    """Clip, sort and merge model event intervals on the PTS time axis."""
    clean = []
    for item in events:
        if len(item) != 2:
            raise ValueError("event must contain start and end")
        start, end = map(float, item)
        if not (math.isfinite(start) and math.isfinite(end)):
            raise ValueError("event contains non-finite time")
        start, end = max(0.0, start), min(float(duration), end)
        if end > start:
            clean.append((start, end))
    clean.sort()
    merged: list[list[float]] = []
    for start, end in clean:
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(float(a), float(b)) for a, b in merged]


def expand_events_to_frames(pts: Sequence[float], events: Sequence[Sequence[float]]) -> np.ndarray:
    """Expand half-open event intervals to unique, sorted original frame IDs."""
    contract = validate_pts(pts)
    merged = merge_events(events, contract.duration)
    if not merged:
        return np.empty(0, dtype=np.int64)
    mask = np.zeros(len(contract.pts), dtype=bool)
    for start, end in merged:
        mask |= (contract.pts >= start) & (contract.pts < end)
    return np.flatnonzero(mask).astype(np.int64)


def _window_for_indices(pts: np.ndarray, ids: np.ndarray) -> tuple[float, float]:
    start = float(pts[int(ids[0])])
    if int(ids[-1]) + 1 < len(pts):
        end = float(pts[int(ids[-1]) + 1])
    else:
        end = float(pts[-1] + np.median(np.diff(pts)))
    return start, end


def phase_candidates(
    pts: Sequence[float],
    *,
    keep_start: int,
    budget: int,
    shift_seconds: float = 1.0,
) -> PhaseCandidates:
    """Build equal-budget left/keep/right candidates on original frame IDs.

    ``keep_start`` and ``budget`` refer to decoded frame indices, while the
    phase displacement is specified in seconds and converted with the median
    local PTS step.  This preserves equal output budgets and avoids treating
    nominal FPS as the source of truth.
    """
    contract = validate_pts(pts)
    n = len(contract.pts)
    keep_start, budget = int(keep_start), int(budget)
    if budget < 1 or budget > n or not 0 <= keep_start <= n - budget:
        raise ValueError("keep window is outside decoded frame axis")
    if not math.isfinite(float(shift_seconds)) or shift_seconds < 0:
        raise ValueError("shift_seconds must be finite and non-negative")
    step = float(np.median(np.diff(contract.pts)))
    shift = max(1, int(round(float(shift_seconds) / step))) if shift_seconds else 0
    max_start = n - budget
    starts = {
        "left": max(0, keep_start - shift),
        "keep": keep_start,
        "right": min(max_start, keep_start + shift),
    }
    ids = {name: np.arange(start, start + budget, dtype=np.int64) for name, start in starts.items()}
    windows = {name: _window_for_indices(contract.pts, values) for name, values in ids.items()}
    return PhaseCandidates(ids, windows, starts, budget, shift)


def parse_choice(text: str, allowed: Sequence[str] = CHOICES) -> str:
    """Parse a bounded Choice response without accepting arbitrary frame IDs."""
    lower = str(text).strip().lower()
    for choice in allowed:
        if re.search(rf"\b{re.escape(choice)}\b", lower):
            return choice
    raise ValueError("response does not contain a bounded temporal choice")


def choose_phase(
    choice: str | None,
    probabilities: Mapping[str, float] | None,
    *,
    confidence_margin: float = 0.15,
    order_stable: bool = True,
) -> tuple[str, dict[str, object]]:
    """Choose a phase, conservatively falling back to the centre candidate.

    A model response is eligible only when its probability map is complete,
    finite and normalized.  If the top candidate is not separated from the
    runner-up, or the candidate-order control is unstable, the centre prior is
    retained.  This is a deployment safeguard, not a trainable score.
    """
    probs = None
    if probabilities is not None:
        try:
            probs = {name: float(probabilities[name]) for name in CHOICES}
        except (KeyError, TypeError, ValueError):
            probs = None
        if probs is not None and (
            any(not math.isfinite(v) or v < 0 for v in probs.values())
            or not math.isclose(sum(probs.values()), 1.0, abs_tol=1e-5)
        ):
            probs = None
    ranked = sorted(probs, key=lambda name: (-probs[name], name)) if probs else []
    top = ranked[0] if ranked else None
    runner = probs[ranked[1]] if len(ranked) > 1 else 0.0
    margin = (probs[top] - runner) if top else None
    parsed = None
    if choice is not None:
        try:
            parsed = parse_choice(choice)
        except ValueError:
            parsed = None
    accepted = bool(parsed and top == parsed and order_stable and margin is not None and margin >= confidence_margin)
    selected = parsed if accepted else "keep"
    return selected, {
        "accepted": accepted,
        "parsed_choice": parsed,
        "top_choice": top,
        "margin": margin,
        "order_stable": bool(order_stable),
        "fallback": selected == "keep" and parsed != "keep",
    }


def selected_frame_ids(candidates: PhaseCandidates, selected: str) -> np.ndarray:
    """Return a defensive copy of the selected equal-budget original IDs."""
    if selected not in CHOICES:
        raise ValueError(f"unknown phase: {selected}")
    return candidates.frame_ids[selected].copy()
