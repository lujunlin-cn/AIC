import numpy as np
from aic.temporal_metrics import kendall_tau, ndcg, ranking_report, spearman

def test_perfect_ranking():
    x = [0., 1., 2., 3.]
    assert spearman(x, x) == 1.0
    assert kendall_tau(x, x) == 1.0
    assert ndcg(x, x) == 1.0

def test_report_is_finite_on_ties():
    r = ranking_report([.5, .5, .1], [1., 0., 0.])
    assert set(r) == {"spearman", "kendall_tau", "ndcg", "ndcg_at_15pct", "top15_ap"}
    assert all(np.isfinite(v) for v in r.values())
