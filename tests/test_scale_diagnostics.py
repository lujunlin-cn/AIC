import copy

import pytest

from aic.scale_diagnostics import (
    constant_and_oracle, join_final_positions, paired_interval, source_partition,
    validate_labels,
)


def rows():
    return [dict(dataset_source='LIVE_YT_VC',source_id=f's{i}',image_id=f's{i}:0',
                 image_sha256=f'hash{i}',image_width=160,image_height=90,
                 target_ratio=[9,16],split='val',gt_crop_xywh=[65,20,30,50],
                 center_xy=[80,45],transposed_geometry_augmentation=False)
            for i in range(8)]


def test_source_macro_is_invariant_to_duplicating_frames_inside_one_source():
    r=rows();bins=[.3,.4,.5,.6,.7,.8,.9,1.]
    a,b=constant_and_oracle(r,[.85,1.],bins,repeats=100)
    repeated=r+[dict(r[0],image_id=f'dup{i}') for i in range(9)]
    c,d=constant_and_oracle(repeated,[.85,1.],bins,repeats=100)
    assert a['best_constant']['scale']==c['best_constant']['scale']
    assert a['best_constant']['confirmation_source_macro_iou']==pytest.approx(c['best_constant']['confirmation_source_macro_iou'])
    for k in ('1','2','4','8'):
        assert b['oracle_at_k'][k]['source_macro_iou']==pytest.approx(d['oracle_at_k'][k]['source_macro_iou'])
    vals=[b['oracle_at_k'][str(k)]['source_macro_iou'] for k in (1,2,4,8)]
    assert vals==sorted(vals)
    assert b['model_greedy_iou'] is None
    assert b['maximum_position_drift_px']<=1


def test_constant_selection_does_not_use_confirmation_outcomes():
    r=rows();bins=[.3,.4,.5,.6,.7,.8,.9,1.]
    a,_=constant_and_oracle(r,[.85],bins,repeats=100)
    tune,confirm=source_partition([x['source_id'] for x in r])
    altered=copy.deepcopy(r)
    for x in altered:
        if x['source_id'] in confirm:x['gt_crop_xywh']=[5,5,110,80]
    b,_=constant_and_oracle(altered,[.85],bins,repeats=100)
    assert a['best_constant']['scale']==b['best_constant']['scale']
    assert not set(tune)&set(confirm)


def test_test_labels_and_identical_train_val_media_rejected():
    r=rows();r[0]['split']='test'
    with pytest.raises(ValueError,match='Train/Val'):validate_labels(r)
    r=rows();r[0]['split']='train';r[1]['image_sha256']=r[0]['image_sha256']
    with pytest.raises(ValueError,match='identical image'):validate_labels(r)


def test_final_position_identity_and_full_source_coverage_are_required():
    r=rows()
    p=[dict(x,p1_aic_bbox_xyw=[55,0,50],parent_identity_sha256='parent',
            position_kind='P1_final_deployed',selected_frames_sha256='frames',
            path_cache_sha256='path',qp_floor=.15,p1_validation_source_disjoint_verified=True) for x in r]
    joined=join_final_positions(r,p,'parent')
    assert joined[0]['center_xy']==[80,50*16/9/2]
    changed=copy.deepcopy(p);changed[0]['parent_identity_sha256']='other'
    with pytest.raises(ValueError,match='identity'):join_final_positions(r,changed,'parent')
    changed=copy.deepcopy(p);changed[0]['p1_aic_bbox_xyw']=[55,0,49]
    with pytest.raises(ValueError,match='exactly reproduce'):join_final_positions(r,changed,'parent')
    extra=dict(r[0],image_id='s0:1')
    with pytest.raises(ValueError,match='coverage'):join_final_positions(r+[extra],p,'parent')
    changed=copy.deepcopy(p);changed[0].pop('p1_validation_source_disjoint_verified')
    with pytest.raises(ValueError,match='exposure'):join_final_positions(r,changed,'parent')


def test_no_A_B_pooling_or_training_label_selection():
    r=rows();r[0]['target_ratio']=[16,9]
    with pytest.raises(ValueError,match='mix'):constant_and_oracle(r,[1.],[.3,.4,.5,.6,.7,.8,.9,1.])
    r=rows();r[0]['split']='train'
    with pytest.raises(ValueError,match='Val'):constant_and_oracle(r,[1.],[.3,.4,.5,.6,.7,.8,.9,1.])


def test_paired_bootstrap_uses_source_effects():
    out=paired_interval([.1,.1,.1],100)
    assert out['paired_source_bootstrap_ci95']==pytest.approx([.1,.1])
