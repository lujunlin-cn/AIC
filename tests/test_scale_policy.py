"""Contract tests for the P2-only, parameter-free deployment intervention."""
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from aic.scale_policy import (POLICIES, budget_ledger, center, drift,
                              frozen_qp_limits, legal, route, scale_box,
                              scale_schedule, sha, shot_slices)
from scripts.build_scale_policy_candidates import package
from scripts.finalize_scale_policy_optimization import official_results


@pytest.mark.parametrize('ratio,width,height', [([9,16],1920,1080),([16,9],720,1280)])
@pytest.mark.parametrize('location', ['center','left','right','top','bottom','tl','tr','bl','br'])
@pytest.mark.parametrize('scale', [.90,.93,.95,.97,1.])
def test_fixed_final_parent_center_all_edges(ratio, width, height, location, scale):
    w = 320; h = w*ratio[1]/ratio[0]
    x = 0 if location in ['left','tl','bl'] else width-w if location in ['right','tr','br'] else round((width-w)/2)
    y = 0 if location in ['top','tl','tr'] else int(height-h) if location in ['bottom','bl','br'] else round((height-h)/2)
    parent = [x,y,w]; before = parent.copy()
    out, reason = scale_box(parent,scale,width,height,ratio)
    assert reason is None and parent == before
    assert legal(out,width,height,ratio) and len(out) == 3
    assert drift(center(parent,ratio),center(out,ratio)) <= .5+1e-9
    assert out[2] == int(w*scale+1e-9)


def test_infeasible_integer_shrink_reverts_without_clamp():
    assert scale_box([0,0,1],.90,1,2,[9,16]) == ([0,0,1],'nonpositive_integer_width')
    with pytest.raises(ValueError,match='original P2'):
        scale_box([-1,0,320],.9,1920,1080,[9,16])
    for s in [0,1.01,float('nan')]:
        with pytest.raises(ValueError): scale_box([0,0,100],s,500,500,[1,1])


def test_original_route_thresholds():
    assert route(1920,1080,[9,16]) == 'a'
    assert route(720,1280,[16,9]) == 'b'
    assert route(1600,902,[16,9]) == 'native'
    assert route(1600,903,[16,9]) == 'b'
    assert route(1080,1920,[9,16]) == 'native'
    assert route(1920,1080,[16,9]) == 'native'


def test_qp_singleton_position_constraint_is_restored_on_exception():
    def limits(base,*args):
        return np.zeros(2),np.full(2,20),np.array([4,5])
    module=SimpleNamespace(limits=limits)
    with pytest.raises(RuntimeError):
        with frozen_qp_limits(module):
            lo,hi,overlap=module.limits(np.array([1,19]),None,None,1,20)
            np.testing.assert_array_equal(lo,hi)
            np.testing.assert_array_equal(lo,[1,19])
            np.testing.assert_array_equal(overlap,[4,5])
            raise RuntimeError('test cleanup')
    assert module.limits is limits


def test_segment_boundaries_use_original_shots_and_events():
    cache={'pts':list(range(8)), 'events':[[0,4],[6,8]],
           'anchors':[{'time':2,'scene_cut':True},{'time':6,'scene_cut':True}]}
    pred=[{'frame':i} for i in [0,1,2,3,6,7]]
    shots=shot_slices(cache,pred,lambda a,e:a)
    assert [s['positions'] for s in shots] == [[0,1],[2,3],[4,5]]
    np.testing.assert_array_equal(scale_schedule(shots,6,.9,'frame'),scale_schedule(shots,6,.9,'segment'))
    assert shot_slices(dict(cache,events=[]),[],lambda a,e:a) == []
    with pytest.raises(ValueError,match='coverage'):
        shot_slices(cache,[{'frame':4}],lambda a,e:a)


def test_budget_is_shared_persistent_and_fails_closed(tmp_path):
    path=tmp_path/'budget.json'; initial=budget_ledger(path)
    assert initial['cumulative_GPU_seconds'] == 0
    initial.update(cumulative_GPU_seconds=100,runs=[{'ended_at':'done','elapsed_GPU_seconds':100}])
    path.write_text(json.dumps(initial))
    assert budget_ledger(path)['cumulative_GPU_seconds'] == 100
    for seconds in [36001,-1,float('nan')]:
        initial.update(cumulative_GPU_seconds=seconds,runs=[{'ended_at':'done','elapsed_GPU_seconds':seconds}])
        path.write_text(json.dumps(initial))
        with pytest.raises(RuntimeError):budget_ledger(path)
    initial.update(cumulative_GPU_seconds=0,runs=[{'ended_at':None,'elapsed_GPU_seconds':0}])
    path.write_text(json.dumps(initial))
    with pytest.raises(RuntimeError,match='unclosed'):budget_ledger(path)


def test_packaging_is_resumable_and_never_overwrites_conflicts(tmp_path):
    data=[{'video_id':'x','targetRatioWH':[16,9],'model_params_b':8.999771458,'predictions':[]}]
    parent=tmp_path/'parent.jsonl'; parent.write_text(json.dumps(data[0])+'\n')
    packed=tmp_path/'parent.zip'
    with zipfile.ZipFile(packed,'w') as archive:archive.writestr('predictions.jsonl',parent.read_bytes())
    protocol={'parent_predictions':str(parent),'parent_package':str(packed)}
    for name in POLICIES:
        first=package(tmp_path,name,data,protocol)
        second=package(tmp_path,name,data,protocol)
        assert first == second
        if name == 'C0':assert first['zip_sha256'] == sha(packed)
    target=tmp_path/'candidates/C1/predictions.jsonl'; target.write_text('corruption')
    with pytest.raises(ValueError,match='conflicts'):package(tmp_path,'C1',data,protocol)


def test_promotion_requires_real_zip_bound_official_receipts_not_historical_score(tmp_path):
    protocol={'owner_reported_baseline_score':44.63,'historical_score_conflict':{},
              'input_hashes':{'inputs/index.jsonl':'index'}}
    manifest={'candidates':{k:{'zip_sha256':k+'_hash'} for k in POLICIES}}
    pending=official_results(protocol,manifest)
    assert pending['best_improving_measured_candidate'] is None
    assert pending['by_candidate']['C0']['official_f1'] is None
    assert not pending['historical_score_substituted_for_new_C0_evaluation']
    evidence=tmp_path/'test_receipt.txt'; evidence.write_text('synthetic unit-test fixture; not a platform result')
    def item(k,s):return {'official_f1':s,'zip_sha256':k+'_hash','evidence_file':evidence.name,'evidence_sha256':sha(evidence)}
    feedback={'stage':'semifinal','metric':'official_f1_percent','index_sha256':'index',
              'scores':{'C0':item('C0',44),'C1':item('C1',44.1),'C2':item('C2',43.9)}}
    actual=official_results(protocol,manifest,feedback,tmp_path)
    assert actual['best_improving_measured_candidate']=='C1'
    assert actual['by_candidate']['C1']['promotion_eligible']
    assert not actual['by_candidate']['C2']['promotion_eligible']
    assert not actual['all_four_official_results_available']
    feedback['scores'].pop('C0')
    assert official_results(protocol,manifest,feedback,tmp_path)['best_improving_measured_candidate'] is None
    feedback['scores']['C1']['zip_sha256']='wrong'
    with pytest.raises(ValueError,match='different ZIP'):official_results(protocol,manifest,feedback,tmp_path)
