"""Budget and sampling unit tests; these do not claim real model success."""
import json
from types import SimpleNamespace

import pytest

from scripts.run_final_p1_cache import elapsed_budget, select_frames, schedule_sources, supervise


def source_rows(source='s', split='train'):
    return [dict(source_id=source, split=split, frame_id=i * 17,
                 transposed_geometry_augmentation=False, image_width=1920,
                 image_height=1080, gt_crop_xywh=[10, 10, 250 + i * 7, 600]) for i in range(30)]


def test_frames_cover_six_disjoint_temporal_strata():
    rows = source_rows()
    ids = select_frames(rows)
    assert len(ids) == len(set(ids)) == 6
    assert all(j * 5 * 17 <= f < (j + 1) * 5 * 17 for j, f in enumerate(ids))
    assert ids[-1] - ids[0] > 20 * 17
    assert select_frames(list(reversed(rows))) == ids


def test_source_schedule_preserves_split_and_alternates():
    groups = {s: source_rows(s, split) for s, split in
              [('a', 'train'), ('b', 'train'), ('c', 'val'), ('d', 'val')]}
    order = schedule_sources(groups)
    assert len(set(order)) == 4
    assert [groups[s][0]['split'] for s in order] == ['train', 'val', 'train', 'val']


def test_restarts_cannot_reset_shared_gpu_budget():
    ledger = {'runs': [{'elapsed_seconds': 1700, 'ended_at': 'done'},
                       {'elapsed_seconds': 1840, 'ended_at': 'done'}]}
    assert elapsed_budget(ledger) == 3540
    ledger['runs'].append({'elapsed_seconds': None, 'ended_at': None})
    with pytest.raises(RuntimeError, match='unclosed'):
        elapsed_budget(ledger)


def test_exhausted_budget_rejects_before_any_model_load(tmp_path, monkeypatch):
    (tmp_path / 'gpu_budget_ledger.json').write_text(json.dumps({
        'cap_seconds': 3600., 'runs': [{'elapsed_seconds': 3599., 'ended_at': 'done'}]}))
    monkeypatch.setattr('subprocess.Popen', lambda *a, **k: pytest.fail('must not load GPU model'))
    with pytest.raises(RuntimeError, match='exhausted'):
        supervise(SimpleNamespace(output=tmp_path, workers=2))
