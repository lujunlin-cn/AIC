import numpy as np
import torch
from aic.features import extract_video_cache, load_feature_cache


def test_non_resnet_cache_retains_encoder_identity(tmp_path, monkeypatch):
    from aic import video
    monkeypatch.setattr(video, 'iter_sampled_frames', lambda *a, **k: iter([
        (0, 0., np.zeros((224,224,3),dtype=np.uint8)),
        (15, .5, np.full((224,224,3),127,dtype=np.uint8))]))
    net=torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1),torch.nn.Flatten())
    dest=tmp_path/'cache.npz'
    extract_video_cache(net,'synthetic',dest,metadata={'backbone':'explicit_semantic_encoder'})
    cache=load_feature_cache(dest)
    assert cache['metadata']['backbone']=='explicit_semantic_encoder'
    assert cache['features'].shape==(2,3)
    np.testing.assert_array_equal(cache['frame_indices'],[0,15])
