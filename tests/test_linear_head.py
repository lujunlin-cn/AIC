import torch
from aic.linear_head import LinearScorer
from scripts.benchmark_representations import load_head


def test_linear_checkpoint_and_padding(tmp_path):
    torch.manual_seed(1)
    model=LinearScorer(5).eval()
    x=torch.randn(1,7,5)
    padded=torch.cat([x,torch.randn(1,6,5)],dim=1)
    assert torch.equal(model(x),model(padded)[:,:7])
    p=tmp_path/'linear.pt';torch.save({'model':model.state_dict()},p)
    loaded,shift,_=load_head(p,'cpu')
    assert not shift
    assert torch.equal(model(x),loaded(x))


def test_linear_can_fit_simple_feature_target():
    torch.manual_seed(2)
    model=LinearScorer(2)
    x=torch.randn(1,64,2);y=(x[...,0]>0).float()
    optimizer=torch.optim.Adam(model.parameters(),lr=.1)
    start=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y).item()
    for _ in range(50):
        optimizer.zero_grad();loss=torch.nn.functional.binary_cross_entropy_with_logits(model(x),y)
        loss.backward();optimizer.step()
    assert loss.item()<start*.4
