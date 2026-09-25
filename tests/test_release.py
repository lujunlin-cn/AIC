import hashlib
import pytest
from aic.release import verify_weights


def test_release_audits_detector_and_temporal_weights(tmp_path):
    manifest = {'weights': {}}
    for name, data in [('temporal', b'weights'), ('detector', b'face')]:
        (tmp_path / name).write_bytes(data)
        manifest['weights'][name] = {'filename': name, 'bytes': len(data),
                                    'sha256': hashlib.sha256(data).hexdigest()}
    candidate = {'temporal_weight': 'temporal', 'detector_weight': 'detector',
                 'expected_loaded_bytes': 11}
    assert verify_weights(manifest, candidate, tmp_path)[1] == 11
    (tmp_path / 'detector').write_bytes(b'FAKE')
    with pytest.raises(ValueError, match='integrity'):
        verify_weights(manifest, candidate, tmp_path)


def test_release_rejects_omitted_auxiliary_bytes(tmp_path):
    (tmp_path / 'x').write_bytes(b'abc')
    manifest = {'weights': {'x': {'filename': 'x', 'bytes': 3,
                                  'sha256': hashlib.sha256(b'abc').hexdigest()}}}
    with pytest.raises(ValueError, match='total'):
        verify_weights(manifest, {'temporal_weight': 'x', 'expected_loaded_bytes': 2}, tmp_path)
