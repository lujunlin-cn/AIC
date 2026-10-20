"""A user's duplicate or overlapping GIFs must never create extra votes."""
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

SCRIPT_DIR=Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0,str(SCRIPT_DIR))
from dinov3_extract_phd2_frame_cache import load_annotations, labels


def test_overlaps_merge_per_user(tmp_path):
    path=tmp_path/'train.json'
    path.write_text(json.dumps({'v': {'u1': [{'t0':1,'t1':4},{'t0':1,'t1':4},{'t0':3,'t1':5}],
                                     'u2': [{'t0':4,'t1':6}]}}))
    intervals=load_annotations(path)['v']
    u,s,n=labels(np.asarray([0.,1.,3.9,4.,5.,6.]),intervals)
    assert n==2
    np.testing.assert_array_equal(u,[0,1,1,1,1,0])
    np.testing.assert_array_equal(s,[0,.5,.5,1,.5,0])


def test_no_intervals_remain_unobserved():
    u,s,n=labels(np.asarray([0.,1.]),[])
    assert n==0
    np.testing.assert_array_equal(s,[0,0])
    np.testing.assert_array_equal(u,[0,0])
