"""Local binary, equal-cost signal-census diagnostics. Not an official scorer.

No model or official video is accessed. Scores must be frozen before evaluation.
Usage: python signal_census_audit.py --self-test --output result.json
Dependencies: numpy, scipy. All statistical intervals are conditional on the
frozen candidates and sampled source population; they do not cover domain shift.
"""
from __future__ import annotations
import argparse
import itertools
import json
import math
from typing import Sequence
import numpy as np
from scipy.stats import norm


def _labels(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.ndim != 2 or not y.size or not np.isfinite(y).all():
        raise ValueError('labels must be a finite nonempty [fragment, slot] array')
    if not np.isin(y, [0, 1]).all():
        raise ValueError('this diagnostic accepts binary equal-cost labels only')
    return y


def topk(scores: np.ndarray, k: int) -> np.ndarray:
    s = np.asarray(scores, dtype=float)
    if s.ndim != 2 or not np.isfinite(s).all():
        raise ValueError('finite 2D scores required')
    if not isinstance(k, (int, np.integer)) or not 0 <= k <= s.shape[1]:
        raise ValueError('k must be an actual integer action count')
    out = np.zeros(s.shape, dtype=bool)
    order = np.argsort(-s, axis=1, kind='stable')[:, :k]
    np.put_along_axis(out, order, True, axis=1)
    return out


def temporal_f(y: np.ndarray, keep: np.ndarray) -> np.ndarray:
    y = _labels(y)
    m = np.asarray(keep, dtype=bool)
    if m.shape != y.shape:
        raise ValueError('mask shape mismatch')
    den = y.sum(1) + m.sum(1)
    return np.divide(2 * (m*y).sum(1), den, out=np.ones(len(y)), where=den>0)


def fit_prior_on_train(y_train: np.ndarray, k: int) -> dict:
    y = _labels(y_train)
    g = y.sum(1)
    if k <= 0:
        raise ValueError('positive k required for the prior fit')
    p = np.clip(y.mean(0), 1e-8, 1-1e-8)
    u = 2*y/(k + g[:,None])
    return {'probabilities':p, 'logit':np.log(p/(1-p)),
            'utility':u.mean(0), 'fit_fragments':len(y)}


def exchange_diagnostics(y: np.ndarray, prior_keep: np.ndarray,
                         candidate_keep: np.ndarray) -> dict:
    y = _labels(y)
    a, b = np.asarray(prior_keep, bool), np.asarray(candidate_keep, bool)
    if a.shape != y.shape or b.shape != y.shape:
        raise ValueError('mask shape mismatch')
    if not np.all(a.sum(1)==b.sum(1)):
        raise ValueError('fixed-budget exchange formula requires equal per-row K')
    k = a.sum(1)
    den = k+y.sum(1)
    incoming, outgoing = ((b&~a)*y).sum(1), ((a&~b)*y).sum(1)
    delta = temporal_f(y,b)-temporal_f(y,a)
    exchange_delta = np.divide(2*(incoming-outgoing),den,
                              out=np.zeros(len(y)),where=den>0)
    oracle = np.divide(2*np.minimum(k,y.sum(1)),den,
                       out=np.ones(len(y)),where=den>0)
    return {'delta':delta,'exchange_delta':exchange_delta,
            'swaps':(b&~a).sum(1), 'incoming_hits':incoming,
            'outgoing_hits':outgoing, 'oracle':oracle,
            'prior_f':temporal_f(y,a),'candidate_f':temporal_f(y,b)}


def fragment_log_likelihood(y: np.ndarray, probability: np.ndarray) -> np.ndarray:
    y = _labels(y)
    p = np.asarray(probability,float)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('probabilities must match labels and lie in [0,1]')
    p = np.clip(p,1e-8,1-1e-8)
    return (y*np.log(p)+(1-y)*np.log1p(-p)).mean(1)


def cluster_influence(delta: Sequence[float], source: Sequence[str]) -> dict:
    d, ids = np.asarray(delta,float), np.asarray(source,str)
    if d.ndim != 1 or ids.shape != d.shape or len(d)<2 or not np.isfinite(d).all():
        raise ValueError('matching finite delta and source vectors required')
    names, inv = np.unique(ids,return_inverse=True)
    if len(names)<2:
        raise ValueError('at least two independent sources required')
    counts = np.bincount(inv)
    sums = np.bincount(inv,weights=d)
    mean = d.mean()
    psi = (sums-counts*mean)/counts.mean()
    se = psi.std(ddof=1)/math.sqrt(len(names))
    p = float(norm.sf(mean/se)) if se>0 else (0.0 if mean>0 else 1.0)
    # Zero variance is a property of the supplied candidate, not proof that an
    # entire feature family is null. Generalization still requires a valid split.
    return {'delta_fragment_macro':float(mean),'n_sources':len(names),
            'n_fragments':len(d),'sigma_influence':float(psi.std(ddof=1)),
            'se_asymptotic':float(se),'one_sided_p_asymptotic':p,
            'ci95_asymptotic':[float(mean-1.96*se),float(mean+1.96*se)],
            'source_counts':counts.tolist()}


def holm_adjust(pvalues: Sequence[float]) -> np.ndarray:
    p=np.asarray(pvalues,float)
    if p.ndim!=1 or not len(p) or not np.isfinite(p).all() or np.any((p<0)|(p>1)):
        raise ValueError('valid p-values required')
    order=np.argsort(p,kind='stable'); adjusted=np.empty(len(p))
    adjusted[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return adjusted


def required_sources(sigma: float, effect: float=.007,
                     alpha: float=.01, power: float=.8) -> int:
    if sigma<=0 or effect<=0 or not 0<alpha<.5 or not 0<power<1:
        raise ValueError('invalid planning arguments')
    return math.ceil(((norm.ppf(1-alpha)+norm.ppf(power))*sigma/effect)**2)


def self_test() -> dict:
    count=0; max_error=0.0
    rng=np.random.default_rng(20261007)
    # Fixed-budget formula and exhaustive oracle over all binary labels at n=8.
    for bits in itertools.product([0.,1.],repeat=8):
        y=np.array([bits])
        for k in range(1,9):
            a=topk(rng.normal(size=(1,8)),k)
            b=topk(rng.normal(size=(1,8)),k)
            r=exchange_diagnostics(y,a,b)
            err=float(np.max(np.abs(r['delta']-r['exchange_delta'])))
            max_error=max(max_error,err)
            assert err<1e-12
            oracle=max(temporal_f(y,np.isin(np.arange(8),subset)[None])[0]
                       for subset in itertools.combinations(range(8),k))
            assert abs(oracle-r['oracle'][0])<1e-12
            count+=1
    # Marginal probability contains no X signal, yet weighted utility does.
    # X=0: [1,0,0] or [0,1,1]; X=1: [0,1,0] or [1,0,1].
    y0=np.array([[1,0,0],[0,1,1]],float)
    y1=np.array([[0,1,0],[1,0,1]],float)
    assert np.all(y0.mean(0)==y1.mean(0))
    q0=(2*y0/(1+y0.sum(1)[:,None])).mean(0)
    q1=(2*y1/(1+y1.sum(1)[:,None])).mean(0)
    base=max((q0+q1)/2); optimum=(max(q0)+max(q1))/2
    assert abs(optimum-base-1/12)<1e-12; count+=1
    # Common logit offset leaves every top-K mask unchanged.
    s=rng.normal(size=(20,8))
    assert np.array_equal(topk(s,6),topk(s+rng.normal(size=(20,1)),6)); count+=1
    # The constant-probability LL is unchanged in the above correlation example.
    assert np.allclose(fragment_log_likelihood(y0,np.full_like(y0,.5)),
                       fragment_log_likelihood(y1,np.full_like(y1,.5))); count+=1
    # Fragment macro differs from source macro when source sizes differ.
    r=cluster_influence([1,0,0,0],['a','b','b','b'])
    assert r['delta_fragment_macro']==.25 and r['n_sources']==2; count+=1
    assert np.allclose(holm_adjust([.01,.04,.03]),[.03,.06,.06]); count+=1
    assert required_sources(.05)==513; count+=1
    # Train-only fitting and zero residual preserve the baseline.
    y=rng.integers(0,2,size=(40,8))
    p=fit_prior_on_train(y,6)
    assert np.array_equal(topk(p['utility'][None],6),
                          topk((p['utility']+np.zeros(8))[None],6)); count+=1
    return {'status':'PASS','tests':count,'max_exchange_error':max_error,
            'correlation_only_signal_example':{'baseline_F':float(base),
              'legal_X_policy_F':float(optimum),'marginal_LL_gain':0.0,
              'F_gain':float(optimum-base)},
            'scope':'synthetic/local arithmetic only; no AIC model or NPU executed'}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    parser.add_argument('--output',default=None)
    args=parser.parse_args()
    if not args.self_test:
        parser.error('import the functions for pre-registered public-data diagnostics, or pass --self-test')
    result=self_test(); text=json.dumps(result,indent=2,allow_nan=False)
    if args.output:
        from pathlib import Path
        Path(args.output).write_text(text+'\n',encoding='utf-8')
    print(text)

if __name__=='__main__':
    main()
