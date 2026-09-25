import torch
from aic.models import A0Model, temporal_shift
from scripts.benchmark_representations import load_head

def test_full_bundle_and_bare_head_logits_match(tmp_path):
    torch.manual_seed(23)
    model=A0Model(backbone=torch.nn.Identity(),temporal_shift_enabled=True).eval()
    x=torch.randn(1,17,512)
    full=tmp_path/'full.pt';bare=tmp_path/'bare.pt'
    torch.save({'model':model.state_dict(),'config':{'temporal_shift':True}},full)
    torch.save({'model':model.temporal.state_dict()},bare)
    a,shift,_=load_head(full,'cpu');b,no_shift,_=load_head(bare,'cpu')
    assert shift and not no_shift
    with torch.inference_mode():
        expected=model(x)
        assert torch.equal(a(temporal_shift(x)),expected)
        assert torch.equal(b(temporal_shift(x)),expected)
