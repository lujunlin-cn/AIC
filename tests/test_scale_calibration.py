import copy
import numpy as np
import pytest

from aic.scale_calibration import deploy_iou_matrix, source_weights, calibrate, weighted_quantiles
from aic.scale_geometry import box_iou_xywh, checked_scale_action


def rows():
    return [dict(dataset_source=d,source_id=f's{i}',image_id=f's{i}:0',split='train',
                 position_kind='P1_final_deployed',image_width=640,image_height=360,
                 target_ratio=[9,16],center_xy=[100+i*10,180],
                 gt_crop_xywh=[60+i*10,95,60+i*2,140+i])
            for d in ['LIVE_YT_VC','GNMC'] for i in range(12)]


def test_vectorized_deployment_matches_exact_scalar_geometry_at_boundaries():
    r=rows()
    for row,center in zip(r,[[20,20],[620,20],[20,340],[620,340]]):row['center_xy']=center
    scales=[.3,.5,.75,1.]
    values,drift=deploy_iou_matrix(r,scales)
    for j,row in enumerate(r):
        for k,scale in enumerate(scales):
            action=checked_scale_action(640,360,row['target_ratio'],row['center_xy'],scale)
            expected=box_iou_xywh(action['crop_xywh'],row['gt_crop_xywh']) if action['valid'] else -1
            assert values[j,k]==pytest.approx(expected)
            assert drift[j,k]==pytest.approx(action['position_drift_px'])


def test_source_weights_do_not_overweight_frames_or_shared_source_ids():
    r=rows();weights=source_weights(r)
    assert weights.sum()==pytest.approx(1)
    repeated=r+[dict(r[0],image_id=f'dup{i}') for i in range(30)]
    w=source_weights(repeated)
    assert sum(v for row,v in zip(repeated,w) if row['dataset_source']=='LIVE_YT_VC')==pytest.approx(.5)
    assert sum(v for row,v in zip(repeated,w) if row['dataset_source']=='LIVE_YT_VC' and row['source_id']=='s0')==pytest.approx(.5/12)


def test_train_only_bin_calibration_keeps_fit_and_calibration_sources_disjoint():
    r=rows();result=calibrate(r,[.3,.4,.5,.6,.7,.8,.9,1.])
    assert not set(map(tuple,result['fit_source_keys'])) & set(map(tuple,result['calibration_source_keys']))
    assert result['values'][-1]==1.
    assert result['count'] in {8,16}
    assert result['validation_or_test_used'] is False
    r[0]['split']='val'
    with pytest.raises(ValueError,match='Train'):calibrate(r,[.3,.4,.5,.6,.7,.8,.9,1.])


def test_weighted_quantiles_use_independent_source_mass():
    assert weighted_quantiles([.1,.9],[.5,.5],[.5])[0]==pytest.approx(.5)
