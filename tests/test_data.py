import json
from pathlib import Path
import pytest
from aic.data import ManifestRecord, grouped_splits, read_manifest, write_manifest, stable_split
from scripts.acquire_tvsum import audit_annotation_record

def rec(**kwargs):
    d=dict(dataset='TVSum',version='tvsum50_v1.1',video_id='v1',source_id='v1',source_group='tvsum:v1',download_status='missing',license_gate='unknown',split='train',annotation_type='summary_importance_2s')
    d.update(kwargs); return ManifestRecord(**d)

def test_manifest_roundtrip(tmp_path):
    p=tmp_path/'m.jsonl'; write_manifest([rec()],p)
    got=read_manifest(p); assert got[0].video_id=='v1'; assert got[0].annotation_type.startswith('summary')

def test_manifest_duplicate_rejected(tmp_path):
    with pytest.raises(ValueError, match='duplicate'):
        write_manifest([rec(),rec()],tmp_path/'m.jsonl')

def test_unknown_and_invalid_rejected():
    with pytest.raises(ValueError): rec(download_status='weird').validate()
    with pytest.raises(ValueError): rec(foo='x') if False else __import__('aic.data',fromlist=['record_from_mapping']).record_from_mapping({'foo':1})

def test_license_gate_is_explicit():
    rec(license_gate='blocked').validate()
    with pytest.raises(ValueError, match='license_gate'):
        rec(license_gate='maybe').validate()


def test_tvsum_annotation_audit_preserves_shape_and_protocol():
    report = audit_annotation_record({"nframes": 1000, "fps": 25,
                                      "duration": 40,
                                      "user_anno": __import__('numpy').zeros((20, 20))}, "vid")
    assert report["annotation_rows"] == 20
    assert report["annotator_columns"] == 20
    assert report["rows_equal_nframes"] is False
    assert report["alignment_convention"].startswith("uniform_edges")
    assert report["binary_target_protocol"].endswith("ge_0.5_v1")

def test_group_split_is_deterministic_and_same_group():
    groups=['source:a','source:b','source:a']
    a=grouped_splits(groups); b=grouped_splits(reversed(groups))
    assert a==b and a['source:a']==stable_split('source:a')
