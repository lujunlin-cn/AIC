import numpy as np
from aic.foundation import anchor_windows
from scripts.foundation_candidate_train import ndcg


def test_context_keeps_same_anchors_and_replicates_only_edges():
    a,w=anchor_windows(7,8)
    np.testing.assert_array_equal(a,[0,4])
    np.testing.assert_array_equal(w[0],[0,0,0,0,0,1,2,3])
    assert w.max()==6
    for length in [1,16]:
        b,_=anchor_windows(7,length)
        np.testing.assert_array_equal(a,b)


def test_ndcg_ties_are_permutation_invariant():
    scores=np.array([.5,.5,.2]);targets=np.array([0.,1.,.5])
    assert np.isclose(ndcg(scores,targets),ndcg(scores,targets[[1,0,2]]))
    assert np.isclose(ndcg(targets,targets),1.)
