#!/usr/bin/env python3
"""R9 finite-action temporal audit. Public labeled data only.

Input JSONL: {source_id, fragment_id, labels: [0|1,...], scores:[float,...]}.
The caller must pass the ACTUAL integer K used by the evaluated decoder.
Fit positions only on train. Evaluation labels are used for metrics/oracles,
never for deployable decisions. The reported local statistic is fragment-macro.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any
import numpy as np


def temporal_f(mask: np.ndarray, y: np.ndarray) -> float:
    k, g = int(mask.sum()), int(y.sum())
    return 1.0 if k + g == 0 else float(2 * np.dot(mask, y) / (k + g))


def masks(n: int, k: int) -> np.ndarray:
    out = np.zeros((math.comb(n, k), n), dtype=np.int8)
    for idx, selected in enumerate(itertools.combinations(range(n), k)):
        out[idx, list(selected)] = 1
    return out


def choose(scores: np.ndarray, k: int) -> np.ndarray:
    # Stable, label-free tie-break: lower slot index first.
    m = np.zeros(scores.size, dtype=np.int8)
    m[np.argsort(-scores, kind='stable')[:k]] = 1
    return m


def bounds(n: int, k: int, g: int) -> tuple[float, float, float]:
    if k + g == 0:
        return 1.0, 1.0, 1.0
    return (2 * max(0, k + g - n) / (k + g),
            2 * k * g / (n * (k + g)),
            2 * min(k, g) / (k + g))


def load_jsonl(path: Path, n: int) -> list[dict[str, Any]]:
    rows = []
    seen = set()
    for lineno, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
        if not line.strip():
            continue
        r = json.loads(line)
        for key in ('source_id', 'fragment_id', 'labels', 'scores'):
            if key not in r:
                raise ValueError(f'{path}:{lineno}: missing {key}')
        sid, fid = str(r['source_id']), str(r['fragment_id'])
        if (sid, fid) in seen:
            raise ValueError(f'duplicate row {(sid, fid)}')
        seen.add((sid, fid))
        y = np.asarray(r['labels'])
        s = np.asarray(r['scores'], dtype=np.float64)
        if y.shape != (n,) or s.shape != (n,):
            raise ValueError(f'{path}:{lineno}: need exactly {n} slots')
        if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
            raise ValueError(f'{path}:{lineno}: invalid label/score')
        rows.append(dict(source_id=sid, fragment_id=fid,
                         labels=y.astype(np.int8), scores=s))
    if not rows:
        raise ValueError(f'empty dataset: {path}')
    return rows


def fit_priors(rows: list[dict[str, Any]], k: int) -> tuple[np.ndarray, np.ndarray]:
    y = np.stack([r['labels'] for r in rows]).astype(np.float64)
    marginal = y.mean(axis=0)
    denom = k + y.sum(axis=1)
    weighted = np.divide(2 * y, denom[:, None],
                         out=np.zeros_like(y), where=denom[:, None] > 0).mean(axis=0)
    return marginal, weighted


def source_bootstrap(values: np.ndarray, sources: list[str], reps: int, seed: int) -> list[float]:
    # Resample entire sources; retain fragment-macro within each bootstrap draw.
    unique = sorted(set(sources))
    if len(unique) < 2:
        raise ValueError('need at least 2 evaluation sources for a CI')
    groups = {s: [] for s in unique}
    for s, v in zip(sources, values):
        groups[s].append(float(v))
    sums = np.array([sum(groups[s]) for s in unique])
    counts = np.array([len(groups[s]) for s in unique])
    rng = np.random.default_rng(seed)
    samples = np.empty(reps)
    for b in range(reps):
        ix = rng.integers(0, len(unique), len(unique))
        samples[b] = sums[ix].sum() / counts[ix].sum()
    return np.quantile(samples, [0.025, 0.975]).tolist()


def audit(train: list[dict[str, Any]], evaluation: list[dict[str, Any]], k: int,
          n: int, reps: int = 2000, seed: int = 20261006) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    overlap = {r['source_id'] for r in train} & {r['source_id'] for r in evaluation}
    if overlap:
        raise ValueError(f'train/eval source overlap: {sorted(overlap)[:8]}')
    mp, wp = fit_priors(train, k)
    marginal_mask, weighted_mask = choose(mp, k), choose(wp, k)
    out = []
    for r in evaluation:
        y, scores = r['labels'], r['scores']
        hm = choose(scores, k)
        g = int(y.sum())
        low, random_f, ceiling = bounds(n, k, g)
        h, w, m = temporal_f(hm, y), temporal_f(weighted_mask, y), temporal_f(marginal_mask, y)
        incoming = np.where((hm == 1) & (weighted_mask == 0))[0]
        outgoing = np.where((hm == 0) & (weighted_mask == 1))[0]
        out.append(dict(source_id=r['source_id'], fragment_id=r['fragment_id'],
                        n=n, k=k, g=g, head_f=h, marginal_prior_f=m, macro_prior_f=w,
                        oracle_f=ceiling, random_expected_f=random_f, worst_f=low,
                        delta_head_macro_prior=h-w, oracle_gap_from_prior=ceiling-w,
                        same_mask=bool(np.array_equal(hm, weighted_mask)),
                        same_hits=bool(np.dot(hm, y) == np.dot(weighted_mask, y)),
                        swaps=len(incoming), incoming_positive=int(y[incoming].sum()),
                        outgoing_positive=int(y[outgoing].sum()),
                        head_mask=hm.tolist()))
    def avg(key: str) -> float:
        return float(np.mean([r[key] for r in out]))
    delta = np.array([r['delta_head_macro_prior'] for r in out])
    ci = source_bootstrap(delta, [r['source_id'] for r in out], reps, seed)
    summary = dict(
        scope='local_binary_equal_cost_fragment_macro; NOT official joint F',
        status='COMPUTED_FROM_SUPPLIED_PUBLIC_RECORDS', n=n, actual_k=k,
        actual_keep_fraction=k/n, feasible_sets=math.comb(n, k),
        train_fragments=len(train), train_sources=len({r['source_id'] for r in train}),
        eval_fragments=len(out), eval_sources=len({r['source_id'] for r in out}),
        eval_g_hist=dict(sorted(Counter(r['g'] for r in out).items())),
        train_marginal_rates=mp.tolist(), train_macro_weights=wp.tolist(),
        marginal_prior_mask=marginal_mask.tolist(), macro_prior_mask=weighted_mask.tolist(),
        metrics={name: avg(name) for name in ('head_f','marginal_prior_f','macro_prior_f',
                  'oracle_f','random_expected_f','worst_f','delta_head_macro_prior',
                  'oracle_gap_from_prior','same_mask','same_hits','swaps')},
        paired_source_bootstrap_delta_ci95=ci,
        ci_scope='conditional on frozen fitted priors/checkpoints; no seed-selection or domain-shift uncertainty',
        oracle_warning='Evaluation GT oracles are diagnostics only and must not generate a deployment mask.'
    )
    return summary, out


def self_test() -> dict[str, Any]:
    max_error = 0.0
    cases = 0
    n = 8
    # Exhaust all binary ground truths and all cardinalities. Uniform-mask mean
    # must equal the analytic random baseline; max/min must match exact bounds.
    for bits in itertools.product([0, 1], repeat=n):
        y = np.asarray(bits, dtype=np.int8)
        for k in range(n+1):
            mm = masks(n, k)
            denominator = k + int(y.sum())
            f = (np.ones(len(mm)) if denominator == 0
                 else 2 * (mm @ y).astype(float) / denominator)
            low, rand, high = bounds(n, k, int(y.sum()))
            error = max(abs(f.min()-low), abs(f.mean()-rand), abs(f.max()-high))
            max_error = max(max_error, float(error))
            assert error < 1e-12
            cases += 1
    # A marginal-rate prior and a macro-F-optimal prior can choose differently.
    # Slot 0 is positive in 3/5 dense rows; slot 1 in 2/5 sparse rows.
    yy = np.array([[1,0,1,1]]*3 + [[0,1,0,0]]*2, dtype=np.int8)
    toy = [dict(labels=y) for y in yy]
    marginal, macro = fit_priors(toy, 1)
    assert choose(marginal, 1).argmax() == 0
    assert choose(macro, 1).argmax() == 1
    empirical = np.array([np.mean([temporal_f(m, y) for y in yy]) for m in masks(4,1)])
    selected = np.mean([temporal_f(choose(macro,1), y) for y in yy])
    assert abs(selected - empirical.max()) < 1e-12
    # Rank changes inside a selected set do not alter F.
    y = np.array([1,0,1,0,1,0,0,0], dtype=np.int8)
    a = np.array([8,7,6,5,4,3,2,1.])
    b = np.array([3,8,4,7,5,6,2,1.])
    assert np.array_equal(choose(a,6), choose(b,6))
    assert temporal_f(choose(a,6),y) == temporal_f(choose(b,6),y)
    # One positive-for-negative exchange, n=8,k=6,g=3: exactly 2/9.
    m1 = np.array([1,1,1,1,0,1,1,0], dtype=np.int8)
    m2 = np.array([1,1,1,1,1,1,0,0], dtype=np.int8)
    assert abs(temporal_f(m2,y)-temporal_f(m1,y)-2/9) < 1e-12
    # A positive-weighted mean at source level must NOT replace fragment-macro.
    vals = np.array([1.,1.,1.,0.])
    assert abs(vals.mean()-0.75) < 1e-12
    return dict(status='PASS', exhaustive_label_cardinality_cases=cases,
                max_bound_error=max_error, additional_properties=4,
                aic_real_experiments='NOT_RUN',
                notes='Synthetic/analytic tests only. No AIC predictions, training, official inputs, or NPU calls.')


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path)
    p.add_argument('--eval', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--n', type=int, default=8)
    p.add_argument('--k', type=int)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--self-test', action='store_true')
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.self_test:
        result = self_test()
    else:
        if args.train is None or args.eval is None or args.k is None:
            p.error('--train, --eval and explicit --k are required')
        if not 0 <= args.k <= args.n or not 1 <= args.n <= 16 or args.bootstrap < 100:
            p.error('require 0 <= k <= n <= 16 and bootstrap >= 100')
        result, per = audit(load_jsonl(args.train,args.n), load_jsonl(args.eval,args.n),
                            args.k,args.n,args.bootstrap)
        result['inputs_sha256'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest()
                                   for f in (args.train,args.eval)}
        per_path = args.out.with_suffix('.per_fragment.jsonl')
        per_path.write_text('\n'.join(json.dumps(r) for r in per)+'\n',encoding='utf-8')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__ == '__main__':
    main()
