"""Semantic CPU checks for Train-selected, source-balanced deployment metrics."""
import copy

import numpy as np
import pytest

from aic.scale_final_preflight import (
    CONSTANTS, action_table, center_errors, center_jitter, constants_oracle,
    macro_array, validate_join,
)
from aic.scale_geometry import checked_scale_action

BINS = [.3, .55, .7, .8, .87, .9, .97, 1.]


def row(source, split, scale=.9, ratio=(9, 16)):
    width, height = (1600, 900) if ratio == (9, 16) else (900, 1600)
    center = [width/2, height/2]
    crop = checked_scale_action(width, height, ratio, center, scale)
    base = checked_scale_action(width, height, ratio, center, 1.)
    return dict(dataset_source='LIVE_YT_VC', source_id=source, image_id=source,
                split=split, target_ratio=list(ratio), image_width=width, image_height=height,
                frame_id=0, image_sha256='image-'+source, center_xy=center,
                gt_crop_xywh=crop['crop_xywh'], p1_aic_bbox_xyw=base['aic_bbox_xyw'])


def test_macro_does_not_weight_video_frames_as_independent_sources():
    rows = [row('many', 'val')]*30 + [row('one', 'val')]
    ids, means = macro_array(rows, [1.]*30+[0.])
    assert len(ids) == 2 and means.mean() == .5


def test_constant_is_selected_only_on_train_even_when_val_disagrees():
    rows = [row('t1', 'train', .9), row('t2', 'train', .9),
            row('v1', 'val', .97), row('v2', 'val', .97)]
    constant, oracle = constants_oracle(rows, BINS, repeats=100)
    assert constant['best_constant_scale'] == .9
    assert set(c['scale'] for c in constant['all_constants']) == set(CONSTANTS)
    assert oracle['GT_visible_upper_bound']
    assert oracle['learned_model_greedy'] is None
    assert oracle['candidate_order'][0] == .9
    assert oracle['oracle_at_k']['8']['val_source_macro_iou']['mean'] == 1.


def test_datasets_and_routes_cannot_share_an_opaque_macro():
    rows = [row('t', 'train'), row('v', 'val', ratio=(16, 9))]
    with pytest.raises(ValueError, match='separate'):
        constants_oracle(rows, BINS)


def test_center_error_is_real_prediction_minus_gt_and_source_based():
    rows = [row('t', 'train'), row('v', 'val')]
    for r in rows:
        r['gt_crop_xywh'][0] = r['center_xy'][0] - r['gt_crop_xywh'][2]/2
    rows[1]['center_xy'][0] += 32
    result = center_errors(rows, repeats=100)
    assert result['val']['absolute_x_px']['mean'] == 32
    assert result['val']['absolute_x_fraction_width']['mean'] == .02
    assert result['train_validation_gap']['causal_in_sample_optimism_proven'] is False


def test_jitter_does_not_move_original_centers_or_rescue_infeasible_windows():
    rows = [row('t', 'train'), row('v1', 'val'), row('v2', 'val')]
    snapshot = copy.deepcopy(rows)
    result = center_jitter(rows, BINS, repeats=100)
    assert rows == snapshot
    assert len(result['conditions']) == 7
    # Vertical perturbation of a maximum-height crop must be rejected instead
    # of silently changing the requested center back to the legal midline.
    moved = dict(rows[0], center_xy=[800, 10])
    q, _, geometry = action_table([moved], [1.])
    assert np.isnan(q[0, 0])
    assert geometry['invalid_actions']['position_shift_invalid'] == 1


def test_validation_residual_fit_requires_another_val_source():
    r = row('only', 'val')
    result = center_jitter([r], BINS, repeats=100)
    assert result['conditions']['validation_error_matched']['result'] is None


def test_raw_center_or_changed_identity_cannot_masquerade_as_deployment():
    r = row('x', 'train')
    p = dict(r, position_kind='raw_model_center', qp_floor=.15,
             parent_identity_sha256='p', p1_base_sha256='b', tokenizer_sha256='t',
             postprocessing_config_sha256='c', spatial_entrypoint_sha256='s',
             p1_adapter_sha256='a', path_cache_sha256='path', selected_frames_sha256='frames')
    m = {k: p[k] for k in ('parent_identity_sha256', 'p1_base_sha256', 'tokenizer_sha256',
                           'postprocessing_config_sha256', 'spatial_entrypoint_sha256')}
    m['p1_adapter_sha256_by_route'] = {'portrait': 'a'}
    with pytest.raises(ValueError, match='final deployed'):
        validate_join([r], [p], m)
    p['position_kind'] = 'P1_final_deployed'
    assert len(validate_join([r], [p], m)) == 1
    p['tokenizer_sha256'] = 'other'
    with pytest.raises(ValueError, match='identity'):
        validate_join([r], [p], m)
