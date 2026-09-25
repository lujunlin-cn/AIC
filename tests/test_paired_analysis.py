import pytest
from scripts.analyze_nested_cv import paired_bootstrap

def test_paired_bootstrap_identical_and_known_constant_delta():
    a=[.1,.2,.4]
    same=paired_bootstrap(a,a,nboot=100)
    assert same['ci95']==[0.,0.]
    changed=paired_bootstrap(a,[x+.2 for x in a],nboot=100)
    assert changed['delta']==pytest.approx(.2)
    assert changed['ci95']==pytest.approx([.2,.2])
    assert changed['n_videos']==3
