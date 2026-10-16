#!/usr/bin/env python3
"""Exact synthetic checks of the expected per-video joint-F Bayes decoder.

No learned posterior is supplied here. All examples are declared synthetic.
The public functions accept a finite posterior conditional on deployable video
observations; access to a test GT posterior is neither implemented nor assumed.
Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from fractions import Fraction as Q
from itertools import combinations, product
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import subprocess
import time


@dataclass(frozen=True)
class State:
    probability: Q
    labels: tuple[int, ...]
    # One row per frame, one column per legal crop candidate.
    ious: tuple[tuple[Q, ...], ...]


def validate(posterior: tuple[State, ...]) -> tuple[int, int]:
    if not posterior or sum(s.probability for s in posterior) != 1:
        raise ValueError("Posterior probabilities must sum exactly to one")
    n = len(posterior[0].labels)
    if n == 0 or not posterior[0].ious:
        raise ValueError("Use at least one frame and one candidate")
    c = len(posterior[0].ious[0])
    if n == 0 or c == 0:
        raise ValueError("Use at least one frame and one candidate")
    for state in posterior:
        if state.probability < 0 or len(state.labels) != n:
            raise ValueError("Invalid state probability or label count")
        if any(y not in (0, 1) for y in state.labels):
            raise ValueError("Labels must be binary")
        if len(state.ious) != n or any(len(row) != c for row in state.ious):
            raise ValueError("IoU table shape mismatch")
        if any(not 0 <= value <= 1 for row in state.ious for value in row):
            raise ValueError("IoU outside [0,1]")
    return n, c


def state_f(state: State, action: tuple[tuple[int, int], ...]) -> Q:
    k, m = len(action), sum(state.labels)
    if k + m == 0:
        return Q(1)  # 01 contract; official same-version evaluator not obtained.
    reward = sum((state.labels[f] * state.ious[f][b] for f, b in action), Q(0))
    return 2 * reward / (k + m)


def expected_f(posterior: tuple[State, ...], action: tuple[tuple[int, int], ...]) -> Q:
    return sum((s.probability * state_f(s, action) for s in posterior), Q(0))


def posterior_frame_scores(posterior: tuple[State, ...], k: int) -> list[list[Q]]:
    """Return 2 E[Y_f IoU(b,B_f)/(K+M)|X] for each frame/candidate."""
    n, c = validate(posterior)
    if not 1 <= k <= n:
        raise ValueError("Expected 1 <= K <= N")
    return [
        [sum((2 * s.probability * s.labels[f] * s.ious[f][b]
              / (k + sum(s.labels)) for s in posterior), Q(0))
         for b in range(c)]
        for f in range(n)
    ]


def decode(posterior: tuple[State, ...], k: int | None = None) -> dict:
    """Exact Bayes decision for a supplied finite posterior, independent crops.

    No continuity, shot budget or cross-frame crop constraint is imposed. Such
    constraints invalidate independent top-K selection and require a solver.
    """
    n, _ = validate(posterior)
    if k is not None and not 0 <= k <= n:
        raise ValueError("Expected 0 <= K <= N")
    records = []
    for count in ([k] if k is not None else range(n + 1)):
        if count == 0:
            action = ()
            score = sum((s.probability for s in posterior if not any(s.labels)), Q(0))
        else:
            table = posterior_frame_scores(posterior, count)
            crops = [max(range(len(row)), key=lambda b: (row[b], -b)) for row in table]
            ranking = sorted(range(n), key=lambda f: (-table[f][crops[f]], f))
            action = tuple(sorted((f, crops[f]) for f in ranking[:count]))
            score = sum((table[f][b] for f, b in action), Q(0))
        records.append({"k": count, "action": action, "expected_f": score})
    winner = max(records, key=lambda row: (row["expected_f"], -row["k"]))
    return {"best": winner, "by_k": records}


def actions(n: int, c: int, k: int | None = None):
    for count in ([k] if k is not None else range(n + 1)):
        for subset in combinations(range(n), count):
            for crops in product(range(c), repeat=count):
                yield tuple(zip(subset, crops))


def brute_best(posterior: tuple[State, ...], k: int | None = None) -> tuple[Q, tuple]:
    n, c = validate(posterior)
    evaluated = [(expected_f(posterior, a), a) for a in actions(n, c, k)]
    return max(evaluated, key=lambda row: row[0])


def perfect_state(probability: Q, labels: tuple[int, ...]) -> State:
    return State(probability, labels, tuple((Q(1),) for _ in labels))


def bernoulli_posterior(probabilities: tuple[Q, ...]) -> tuple[State, ...]:
    result = []
    for labels in product((0, 1), repeat=len(probabilities)):
        probability = Q(1)
        for p, y in zip(probabilities, labels):
            probability *= p if y else 1 - p
        result.append(perfect_state(probability, labels))
    return tuple(result)


def decode_independent(probabilities: list[float], crop_utilities: list[float] | None = None) -> dict:
    """Polynomial decoder for an explicitly approximate independent posterior.

    Inputs must be calibrated probabilities inferred WITHOUT target GT. Crop
    utility is deterministic in this approximation (or conditionally independent
    of other labels given this frame's Y=1); do not pass raw VLM text scores.
    Complexity O(N^3), memory O(N^2), no enumeration of 2**N label vectors.
    """
    n = len(probabilities)
    if not n:
        raise ValueError("At least one probability is required")
    qs = [1.0] * n if crop_utilities is None else crop_utilities
    if len(qs) != n or any(not math.isfinite(x) or not 0 <= x <= 1 for x in probabilities + qs):
        raise ValueError("Probabilities/utilities must be finite, in [0,1], with matching lengths")
    leave_one_out = []
    for excluded in range(n):
        masses = [1.0]
        for j, p in enumerate(probabilities):
            if j == excluded:
                continue
            updated = [0.0] * (len(masses) + 1)
            for m, mass in enumerate(masses):
                updated[m] += mass * (1 - p)
                updated[m + 1] += mass * p
            masses = updated
        leave_one_out.append(masses)
    records = [{"k": 0, "selected_frames": [],
                "expected_f": math.prod(1 - p for p in probabilities)}]
    for k in range(1, n + 1):
        values = [2 * p * q * sum(mass / (k + 1 + m) for m, mass in enumerate(masses))
                  for p, q, masses in zip(probabilities, qs, leave_one_out)]
        selected = sorted(range(n), key=lambda f: (-values[f], f))[:k]
        records.append({"k": k, "selected_frames": sorted(selected),
                        "expected_f": sum(values[f] for f in selected)})
    return {"best": max(records, key=lambda row: (row["expected_f"], -row["k"])),
            "by_k": records, "posterior_assumption": "conditionally independent Bernoulli"}


def plugin_f(posterior: tuple[State, ...], action: tuple[tuple[int, int], ...]) -> Q:
    if not action:
        # Give the plugin its exact empty posterior, avoiding a straw man.
        return expected_f(posterior, action)
    m_mean = sum((s.probability * sum(s.labels) for s in posterior), Q(0))
    reward_mean = sum((s.probability * sum(
        (s.labels[f] * s.ious[f][b] for f, b in action), Q(0)
    ) for s in posterior), Q(0))
    return 2 * reward_mean / (len(action) + m_mean)


def iou_interval(x: Q, gt_x: Q, width: Q = Q(1, 3)) -> Q:
    intersection = max(Q(0), min(x + width, gt_x + width) - max(x, gt_x))
    return intersection / (2 * width - intersection)


def serialize(value):
    if isinstance(value, Q):
        return {"fraction": str(value), "decimal": float(value)}
    if isinstance(value, State):
        return serialize({"probability": value.probability, "labels": value.labels, "ious": value.ious})
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serialize(v) for v in value]
    return value


def counterexamples() -> dict:
    ranking = (perfect_state(Q(3, 5), (1, 1, 1, 0)),
               perfect_state(Q(2, 5), (0, 0, 0, 1)))
    independent = bernoulli_posterior((Q(9, 10), Q(2, 5)))
    crop = (
        State(Q(13, 20), (1, 1, 1, 1), ((Q(1), Q(0)),) + ((Q(1), Q(1)),) * 3),
        State(Q(7, 20), (1, 0, 0, 0), ((Q(0), Q(1)),) + ((Q(1), Q(1)),) * 3),
    )
    joint = tuple(State(s.probability, s.labels, ((Q(1, 5),), (Q(4, 5),)))
                  for s in bernoulli_posterior((Q(9, 10), Q(3, 5))))
    assert decode(ranking, 1)["best"]["action"] == ((3, 0),)
    assert expected_f(ranking, ((0, 0),)) == Q(3, 10)
    assert expected_f(ranking, ((3, 0),)) == Q(2, 5)
    assert decode(independent)["best"]["k"] == 1
    plugin_winner = max(actions(2, 1), key=lambda a: plugin_f(independent, a))
    assert plugin_winner == ((0, 0), (1, 0))
    assert expected_f(independent, ((0, 0),)) == Q(39, 50)
    assert expected_f(independent, plugin_winner) == Q(56, 75)
    assert posterior_frame_scores(crop, 1)[0][1] > posterior_frame_scores(crop, 1)[0][0]
    assert posterior_frame_scores(crop, 4)[0][0] > posterior_frame_scores(crop, 4)[0][1]
    assert decode(joint, 1)["best"]["action"] == ((1, 0),)
    return {
        "fixed_k_ranking_reversal": {
            "posterior": ranking, "marginal_p": [Q(3, 5)] * 3 + [Q(2, 5)],
            "fixed_k_1": decode(ranking, 1), "all_k": decode(ranking),
            "select_frame_0_exact": expected_f(ranking, ((0, 0),)),
            "select_frame_3_exact": expected_f(ranking, ((3, 0),)),
            "select_frame_0_plugin": plugin_f(ranking, ((0, 0),)),
            "select_frame_3_plugin": plugin_f(ranking, ((3, 0),)),
        },
        "independent_budget_mismatch": {
            "posterior": independent, "marginal_p": [Q(9, 10), Q(2, 5)],
            "exact": decode(independent), "plugin_action": plugin_winner,
            "plugin_scores": [plugin_f(independent, tuple((f, 0) for f in range(k))) for k in range(3)],
            "regret_exact_f": expected_f(independent, ((0, 0),)) - expected_f(independent, plugin_winner),
        },
        "crop_choice_depends_on_k": {
            "posterior": crop,
            "frame_0_scores_k1": posterior_frame_scores(crop, 1)[0],
            "frame_0_scores_k4": posterior_frame_scores(crop, 4)[0],
            "interpretation": "Crop left/right are nonoverlapping legal equal-size windows; the latent GT frame-0 box is correlated with video M.",
        },
        "highlight_probability_conflicts_with_joint_utility": {
            "posterior": joint, "fixed_k1": decode(joint, 1),
            "marginal_p": [Q(9, 10), Q(3, 5)],
            "crop_iou_conditional_positive": [Q(1, 5), Q(4, 5)],
            "select_frame_0_exact": expected_f(joint, ((0, 0),)),
            "select_frame_1_exact": expected_f(joint, ((1, 0),)),
            "interpretation": "The lower-probability frame wins after crop utility is included.",
        },
    }


def proxy_counterexamples() -> dict:
    # Fixed two videos, source-macro and pooled-frame metrics choose differently.
    macro_a, macro_b = Q(1, 2), (Q(160, 180) + 1) / 2
    pooled_a, pooled_b = Q(200, 201), Q(162, 182)
    assert macro_b > macro_a and pooled_a > pooled_b
    # GT [1,1,0,0,0], descending ranks A=[0,2,3,4,1], B=[2,0,1,3,4].
    # Binary-label ties get their usual average rank. Pearson of these ranks is
    # Spearman; numerator 0 and 2.5, common squared norm 75.
    scores_a, scores_b = (5, 1, 4, 3, 2), (4, 3, 5, 2, 1)
    label_ranks = (Q(9, 2), Q(9, 2), Q(2), Q(2), Q(2))
    numerator_a = sum((Q(s - 3) * (r - 3) for s, r in zip(scores_a, label_ranks)), Q(0))
    numerator_b = sum((Q(s - 3) * (r - 3) for s, r in zip(scores_b, label_ranks)), Q(0))
    assert numerator_a == 0 and numerator_b == Q(5, 2)
    return {
        "macro_vs_pooled": {
            "videos": [{"n_frames": 100, "m": 100}, {"n_frames": 1, "m": 1}],
            "A": {"per_video_k": [100, 0], "per_video_reward": [100, 0], "macro_f": macro_a, "pooled_f": pooled_a},
            "B": {"per_video_k": [80, 1], "per_video_reward": [80, 1], "macro_f": macro_b, "pooled_f": pooled_b},
        },
        "spearman_vs_f": {
            "labels": [1, 1, 0, 0, 0], "k": 1,
            "A": {"scores": scores_a, "spearman": 0.0, "f": Q(2, 3)},
            "B": {"scores": scores_b, "spearman": float(numerator_b) / (75 ** .5), "f": Q(0)},
        },
        "all_frame_crop_mean_vs_joint_f": {
            "labels": [1, 0], "selected_frames": [0, 1],
            "A": {"ious": [Q(9, 10), Q(1, 10)], "mean_iou": Q(1, 2), "joint_f": Q(3, 5)},
            "B": {"ious": [Q(8, 10), Q(9, 10)], "mean_iou": Q(17, 20), "joint_f": Q(8, 15)},
            "note": "Auxiliary crop GT exists for both synthetic frames; the competition rewards only Y=1. This is not a multiplication of separate datasets' metrics.",
        },
    }


def exhaustive_checks(seed: int = 20261016) -> dict:
    rng = random.Random(seed)
    decoder_comparisons = marginal_checks = known_m_checks = independent_rank_checks = independent_dp_checks = 0
    max_difference = Q(0)
    for _ in range(40):
        n, c = 4, 3
        label_rows = list(product((0, 1), repeat=n))
        raw = [rng.randrange(1, 10) for _ in label_rows]
        states = []
        for labels, weight in zip(label_rows, raw):
            gt_positions = [Q(rng.randrange(5), 6) for _ in range(n)]
            ious = tuple(tuple(iou_interval(Q(b, 3), gt) for b in range(c)) for gt in gt_positions)
            states.append(State(Q(weight, sum(raw)), labels, ious))
        posterior = tuple(states)
        for k in [None, *range(n + 1)]:
            exact = decode(posterior, k)["best"]["expected_f"]
            brute, _ = brute_best(posterior, k)
            difference = abs(exact - brute)
            max_difference = max(max_difference, difference)
            assert difference == 0
            decoder_comparisons += 1
        # Distinct validation: recompute realized add/delete differences across
        # every truth state, set and candidate, including zero-IoU rewards.
        for state in posterior:
            m = sum(state.labels)
            for a in actions(n, 1):
                k, reward = len(a), sum((state.labels[f] * state.ious[f][b] for f, b in a), Q(0))
                d = k + m
                if d == 0:
                    continue
                selected = {f for f, _ in a}
                for f in range(n):
                    if f not in selected:
                        z = state.labels[f] * state.ious[f][0]
                        difference = state_f(state, a + ((f, 0),)) - state_f(state, a)
                        assert difference == 2 * (d * z - reward) / (d * (d + 1))
                        marginal_checks += 1
                    elif d > 1:
                        z = state.labels[f] * state.ious[f][0]
                        reduced = tuple(item for item in a if item[0] != f)
                        difference = state_f(state, reduced) - state_f(state, a)
                        assert difference == 2 * (reward - d * z) / (d * (d - 1))
                        marginal_checks += 1
        # Condition the same posterior on each M. The plugin must then be exact
        # for every action, even when crop utility is correlated with labels.
        for m in range(n + 1):
            subset = [s for s in posterior if sum(s.labels) == m]
            mass = sum((s.probability for s in subset), Q(0))
            conditional = tuple(State(s.probability / mass, s.labels, s.ious) for s in subset)
            for a in actions(n, c):
                assert expected_f(conditional, a) == plugin_f(conditional, a)
                known_m_checks += 1
    # Independent Bernoulli + equal perfect crops preserves marginal ordering,
    # ruling out the overclaim that uncertainty ALWAYS changes frame ranking.
    for ps in [(Q(2, 10), Q(4, 10), Q(7, 10)),
               (Q(9, 10), Q(2, 5), Q(1, 10)),
               (Q(1, 3), Q(1, 3), Q(2, 3))]:
        posterior = bernoulli_posterior(ps)
        for k in range(1, len(ps) + 1):
            values = posterior_frame_scores(posterior, k)
            for i in range(len(ps)):
                for j in range(len(ps)):
                    assert (values[i][0] > values[j][0]) == (ps[i] > ps[j])
                    assert (values[i][0] == values[j][0]) == (ps[i] == ps[j])
                    independent_rank_checks += 1
        for qs in [(Q(1),) * len(ps), (Q(1, 4), Q(3, 4), Q(1, 2))]:
            weighted = tuple(State(s.probability, s.labels, tuple((q,) for q in qs)) for s in posterior)
            polynomial = decode_independent(list(map(float, ps)), list(map(float, qs)))
            for row in polynomial["by_k"]:
                exact = decode(weighted, row["k"])["best"]["expected_f"]
                assert abs(float(exact) - row["expected_f"]) < 1e-12
                action = tuple((f, 0) for f in row["selected_frames"])
                assert abs(float(expected_f(weighted, action)) - row["expected_f"]) < 1e-12
                independent_dp_checks += 1
    return {"seed": seed, "random_posteriors": 40, "frames": 4, "crop_candidates": 3,
            "decoder_vs_all_actions_comparisons": decoder_comparisons,
            "known_m_plugin_equalities": known_m_checks,
            "realized_add_delete_equalities": marginal_checks,
            "independent_bernoulli_pair_order_checks": independent_rank_checks,
            "polynomial_independent_decoder_vs_exact_checks": independent_dp_checks,
            "maximum_exact_difference": max_difference,
            "status": "PASS"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/r29/decision_theory.json"))
    args = parser.parse_args()
    start = time.monotonic()
    result = {"experiment_id": "R29_DECISION_THEORY", "evidence_level": "synthetic exact validation",
              "examples": counterexamples(), "proxy_counterexamples": proxy_counterexamples(),
              "verification": exhaustive_checks()}
    profile_start = time.monotonic()
    independent_profile = decode_independent([0.1 + .8 * i / 47 for i in range(48)])
    result["independent_decoder_profile"] = {"synthetic_probabilities": 48,
        "seconds": time.monotonic() - profile_start, "best": independent_profile["best"],
        "note": "CPU decoder timing only; these are invented probabilities, not 48 video observations."}
    script = Path(__file__).resolve()
    result["reproducibility"] = {"python": platform.python_version(), "platform": platform.platform(),
        "script_sha256": hashlib.sha256(script.read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(Path("reports/r29/decision_theory_protocol.json").read_bytes()).hexdigest(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "command": "timeout 5m python3 scripts/r29_decision_theory.py --output reports/r29/decision_theory.json",
        "wall_seconds": time.monotonic() - start, "exit_status": 0, "gpu_used": False,
        "note": "Git commit identifies parent state; script SHA identifies this initially uncommitted implementation."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(serialize(result), ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(serialize(result["verification"]), ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
