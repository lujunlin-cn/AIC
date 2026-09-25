from fractions import Fraction
from types import SimpleNamespace
import numpy as np
import pytest
from aic.annotated_video import iter_annotated_cfr_frames

def fake_video(monkeypatch, n=20, fps=10):
    import av
    class Container:
        streams = SimpleNamespace(video=[SimpleNamespace(average_rate=Fraction(fps))])
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def decode(self, stream):
            for i in range(n):
                # Deliberately scrambled timestamps; decoded frame content is ordinal.
                yield SimpleNamespace(pts=(i*7)%n, to_ndarray=lambda format,i=i: np.full((8,8,3),i,np.uint8))
    monkeypatch.setattr(av,"open",lambda p:Container())

def test_explicit_annotation_reader_preserves_decoded_order(monkeypatch):
    fake_video(monkeypatch)
    rows=list(iter_annotated_cfr_frames("fixture",annotation_fps=10,annotation_count=20,size=8))
    assert [r[0] for r in rows]==[0,5,10,15]
    assert [r[1] for r in rows]==[0,.5,1,1.5]
    assert [r[2][0,0,0] for r in rows]==[0,5,10,15]

@pytest.mark.parametrize("fps,count",[(20,20),(10,19),(10,21)])
def test_annotation_reader_rejects_mismatch(monkeypatch,fps,count):
    fake_video(monkeypatch)
    with pytest.raises(ValueError):
        list(iter_annotated_cfr_frames("fixture",annotation_fps=fps,annotation_count=count))
