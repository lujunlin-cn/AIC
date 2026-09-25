import itertools
import numpy as np
import pytest
from scipy.stats import kendalltau
from aic.temporal_metrics import (kendall_tau,ndcg,ranking_report,spearman,average_precision,
                                 fixed_segments,knapsack,summary_mask,summary_report)
def test_ranking_perfect_reverse_and_ties():
    x=[0.,1.,2.,3.]
    assert spearman(x,x)==pytest.approx(1)
    assert kendall_tau(x,x)==1
    assert spearman(x,x[::-1])==pytest.approx(-1)
    assert ndcg(x,x)==1
    assert kendall_tau([1,1,3],[1,2,3])==kendalltau([1,1,3],[1,2,3]).statistic
    assert spearman([1,1],[1,2]) is None
    assert ndcg([1,1],[1,0],1)==.5
    assert average_precision([.5,.5],[1,0])==.5
    assert average_precision([1,0],[0,0]) is None
def test_monotone_invariance_and_invalid():
    p=np.array([.11,.4,.1,.5]);y=np.array([.2,.6,.2,.8])
    assert ranking_report(p,y)==ranking_report(p*2+3,y)
    with pytest.raises(ValueError):ranking_report([float("nan")],[1])
    with pytest.raises(ValueError):ranking_report([],[])
def test_knapsack_exhaustive_and_ties():
    rng=np.random.default_rng(91)
    for _ in range(30):
        w=rng.integers(1,10,8);v=rng.integers(0,10,8);capacity=13
        actual=knapsack(w,v,capacity)
        best=max(sum(v*np.array(s)) for s in itertools.product([0,1],repeat=8)
                 if sum(w*np.array(s))<=capacity)
        assert sum(w*actual)<=capacity
        assert sum(v*actual)==best
    assert knapsack([1,1],[1,1],1).tolist()==[True,False]
def test_author_remainder_and_fallback():
    assert fixed_segments(130).tolist()==[[0,60],[60,130]]
    assert fixed_segments(121).tolist()==[[0,60],[60,121]]
    assert fixed_segments(30).tolist()==[[0,30]]
    assert summary_mask(np.zeros(20)).sum()==3
    assert summary_mask(np.arange(3)).sum()==1
    with pytest.raises(ValueError):summary_mask([1,2],[[0,1]])
def test_summary_identity():
    p=np.arange(600,dtype=float);mask=summary_mask(p)
    assert summary_report(p,np.tile(mask,(20,1)))["summary_f1"]==1
