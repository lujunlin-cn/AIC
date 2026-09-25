import torch
from aic.b0 import B0ViTTemporal

def test_b0_temporal_head_shape():
    model=B0ViTTemporal().eval()
    with torch.no_grad():
        out=model(torch.randn(1,5,768))
    assert out.shape==(1,5) and torch.isfinite(out).all()
