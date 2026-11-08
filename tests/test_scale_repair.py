import copy
import hashlib
import json
from pathlib import Path
import pytest
import torch
from torch import nn
from aic.scale_audit import audit_media, gaicd_retention, position_conditioned_coverage, repair_sacd_splits
from aic.scale_geometry import checked_scale_action, deployed_center
from aic.scale_objectives import causal_completion_sft_loss, source_macro_mean, scale_collapse_rate
from aic.scale_preferences import bind_preference_input, fixed_center_sft_target, geometry_preference_pairs, validate_bound_preference
from aic.scale_budget import empty_budget, record_gpu_run
from scripts.check_scale_gate import assess_candidate_metrics, assess_dpo_metrics


@pytest.mark.parametrize('ratio',[(9,16),(16,9)])
@pytest.mark.parametrize('center',[(320,180),(20,180),(620,180),(320,5),(320,355),
                                   (0,0),(640,0),(0,360),(640,360)])
@pytest.mark.parametrize('scale',[.1,.3,.8,1.])
def test_boundary_position_projection_never_silently_becomes_valid(ratio,center,scale):
    action=checked_scale_action(640,360,ratio,center,scale)
    actual=deployed_center(action['aic_bbox_xyw'],ratio)
    drift=max(abs(a-b) for a,b in zip(center,actual))
    assert action['valid'] == (drift<=1)
    assert action['status']==('ok' if drift<=1 else 'position_shift_invalid')


def row(center=(320,180)):
    return {'source_id':'same','image_id':'same:0','dataset_source':'LIVE_YT_VC','split':'train',
            'image_width':640,'image_height':360,'target_ratio':[16,9], 'center_xy':list(center),
            'gt_crop_xywh':[160,90,320,180],'position_source':'P1_frozen',
            'input_image':'frame.png','image_sha256':'a'*64}


def test_invalid_positions_are_excluded_from_sft_and_preference():
    r=row((20,20))
    label=fixed_center_sft_target(r,[.5,1.],.1)
    assert label['label_status']=='position_shift_invalid' and label['scale_id'] is None
    assert geometry_preference_pairs(r,[.5,1.],.01,10)==[]


def test_position_quality_report_distinguishes_geometry_from_optimizable_scale():
    result=position_conditioned_coverage([row(),dict(row((0,0)),source_id='bad')],[.5,1.],.5)
    metrics=result['by_dataset_ratio']['LIVE_YT_VC:16:9']
    assert metrics['valid_scale_sft_labels']==1
    assert metrics['diagnostic_categories']['fixed_position_scale_can_improve']==1
    assert metrics['position_shift_invalid_samples']==1
    with pytest.raises(ValueError,match='train-only'):
        position_conditioned_coverage([dict(row(),split='val')],[.5,1.],.5)


def test_live_missing_media_is_explicit_not_a_shorter_dataset(tmp_path):
    metrics=audit_media([row()],tmp_path)
    assert metrics['expected']==1 and metrics['missing_count']==1 and not metrics['complete']


def test_gaicd_funnel_rejects_position_changes_and_unrated_candidates():
    r=dict(row(),dataset_source='GAICD',release='journal',candidates=[
        {'crop_xywh':[160,90,320,180],'mos':4.,'valid':True},
        {'crop_xywh':[0,0,640,360],'mos':2.,'valid':True},
        {'crop_xywh':[0,0,320,180],'mos':5.,'valid':True},
        {'crop_xywh':[0,0,320,180],'mos':-2.,'valid':False}])
    report=gaicd_retention([r],[.5,1.],1.)
    assert report['total_candidates']==4 and report['rated_candidates']==3
    assert report['position_center_matched']==2 and report['scale_bin_matched']==2
    assert report['valid_preference_pairs']==1
    assert report['filter_reasons']['position_center_mismatch']['count']==1
    missing=gaicd_retention([dict(r,center_xy=None)],[.5,1.],1.)
    assert missing['valid_preference_pairs'] is None


def test_human_candidate_coordinate_tolerance_cannot_hide_center_drift():
    from aic.scale_geometry import map_box_to_scale_id
    # Every coordinate is within one pixel of the .5 action, but the center
    # moved 1.35 px. That MOS must not supervise the fixed-center action.
    assert map_box_to_scale_id([160.9,90,320.9,180],640,360,(16,9),(320,180),[.5,1.]) is None


def test_final_inference_rechecks_original_p1_deployment_provenance():
    from scripts.infer_scale import validate_position_provenance
    r = dict(row(), parent_identity_sha256='p1', position_kind='P1_final_deployed',
             p1_aic_bbox_xyw=[0,0,640])
    validate_position_provenance([r], 'p1')
    with pytest.raises(ValueError, match='identity mismatch'):
        validate_position_provenance([r], 'different_p1')
    with pytest.raises(ValueError, match='original deployed P1 bbox'):
        validate_position_provenance([dict(r, center_xy=[319,180])], 'p1')


def test_final_inference_rejects_unpromoted_dpo_even_with_lower_loss(tmp_path):
    from scripts.infer_scale import validate_checkpoint_promotion
    (tmp_path / 'evaluation.json').write_text(json.dumps({
        'checkpoint_sha256':'dpo','reference_frozen_verified':True,'loss':.01,'gate_passed':True,
        'metrics_by_dataset_ratio':{'LIVE:9:16':{'comparison':'frozen_SFT',
            'bootstrap':{'ci95':[-.02,-.01]},'max_position_drift_px':0,
            'valid_action_rate':1.,'gate_passed':False}}}))
    with pytest.raises(ValueError, match='retain the frozen SFT'):
        validate_checkpoint_promotion(tmp_path,'dpo')


def test_sacd_image_group_repair_keeps_all_subjects_heldout(tmp_path):
    for name,rels in {'train':['annotations/image_1.json','annotations/other_1.json'],
                      'val':['annotations/image_2.json'],'test':['annotations/image_3.json']}.items():
        (tmp_path/(name+'.json')).write_text(json.dumps({'jsons':rels}))
    result=repair_sacd_splits(tmp_path)
    group=next(g for g in result['groups'] if g['image_id']=='image')
    assert group['split']=='test' and len(group['subjects'])==3
    assert result['source_counts']=={'test':1,'train':1}


def test_dpo_input_contract_detects_changed_image_geometry_or_prompt():
    r=row();pair=geometry_preference_pairs(r,[.5,1.],.1,1)[0]
    bound=bind_preference_input(pair,r,'same prompt',{'side':640})
    validate_bound_preference(bound,'same prompt',{'side':640})
    for mutated in [dict(bound,image_sha256='b'*64),dict(bound,center_xy=[319,180])]:
        with pytest.raises(ValueError,match='contract changed'):
            validate_bound_preference(mutated,'same prompt',{'side':640})
    with pytest.raises(ValueError,match='contract changed'):
        validate_bound_preference(bound,'new prompt',{'side':640})


def test_cross_dataset_source_ids_remain_independent_and_collapse_is_reported():
    rows=[{'dataset_source':'LIVE','source_id':'x','iou':1.}]*30+[
        {'dataset_source':'GNMC','source_id':'x','iou':0.}]
    assert source_macro_mean(rows)==.5
    assert scale_collapse_rate([1]*99+[0])==.99


def test_selected_completion_loss_matches_full_causal_loss_and_ignores_prompt():
    logits=torch.randn(1,8,13,requires_grad=True);labels=torch.tensor([[-100]*4+[2,3,4,5]])
    mask=labels.ne(-100)
    full=causal_completion_sft_loss(logits,labels,mask)
    selected=causal_completion_sft_loss(logits[:,3:7],labels[:,4:],mask[:,4:],already_shifted=True)
    assert torch.allclose(full,selected)
    full.backward();assert logits.grad[:,:3].abs().sum()==0


def test_budget_overrun_does_not_reset_stage_or_total():
    budget=empty_budget();budget['stages']['scale_sft']['used_hours']=4.
    budget['total_used_hours']=4.
    before=copy.deepcopy(budget)
    with pytest.raises(ValueError,match='exceeded'):
        record_gpu_run(budget,'scale_sft','2026-11-08T00:00:00+00:00','2026-11-08T00:01:00+00:00','GPU','ckpt','log')
    assert budget==before


def test_gate_rejects_forged_grid_only_gain_and_dpo_loss_only_claim():
    assert not assess_candidate_metrics({'gate_passed':True,'oracle_gain':.5,'checkpoint_sha256':'a'*64})
    assert not assess_dpo_metrics({'gate_passed':True,'loss':.1})


def test_reference_dropout_is_eval_and_dpo_swapping_has_expected_behavior():
    from aic.scale_objectives import dpo_loss
    class Ref(nn.Module):
        def __init__(self):
            super().__init__();self.drop=nn.Dropout(.8);self.weight=nn.Parameter(torch.ones(3))
        def forward(self,x):return self.drop(x)*self.weight
    ref=Ref().requires_grad_(False);ref.eval();before=ref.weight.clone()
    assert torch.equal(ref(torch.ones(3)),ref(torch.ones(3)))
    margin=torch.tensor([0.],requires_grad=True);zeros=torch.zeros(1)
    loss=dpo_loss(margin,zeros,zeros,zeros,.1)
    assert loss.item()==pytest.approx(.69314718)
    loss.backward();assert margin.grad.abs().item()>0
    with torch.no_grad():margin-=margin.grad
    assert dpo_loss(margin,zeros,zeros,zeros,.1)<loss
    assert dpo_loss(zeros,margin,zeros,zeros,.1)>loss
    assert torch.equal(ref.weight,before)


def test_final_integration_preserves_frame_selection_and_p1_centers():
    from aic.scale_runtime import apply_scale_to_p1_predictions
    record={'predictions':[{'frame':4,'bboxes':[0,0,640]},{'frame':7,'bboxes':[0,0,640]}]}
    before=copy.deepcopy(record)
    result=apply_scale_to_p1_predictions(record,{4:0,7:1},640,360,[16,9],[.5,1.])
    assert record==before
    assert [r['frame'] for r in result['predictions']]==[4,7]
    for r in result['predictions']:
        assert deployed_center(r['bboxes'],[16,9])==(320.,180.) and len(r['bboxes'])==3


def test_official_scale_export_uses_validator_and_preserves_p1_selection(tmp_path):
    from aic.contract import VideoMetadata, load_jsonl
    from scripts.infer_scale import write_scaled_submission, validate_submission_positions
    record={'video_id':'video','targetRatioWH':[16,9], 'model_params_b':8.99,
            'predictions':[{'frame':4,'bboxes':[0,0,640]}]}
    index={'video':VideoMetadata('video',640,360,10,(16,9))}
    inputs=[dict(row(),source_id='video',frame_id=4,p1_aic_bbox_xyw=[0,0,640])]
    validate_submission_positions([record],inputs,index)
    with pytest.raises(ValueError,match='selected frames/bboxes'):
        validate_submission_positions([record],[dict(inputs[0],p1_aic_bbox_xyw=[1,0,638])],index)
    actions=[{'source_id':'video','frame':4,'scale_id':0}]
    path=tmp_path/'predictions.jsonl'
    report=write_scaled_submission([record],actions,index,path,[.5,1.],8_999_976_258)
    assert report.valid
    saved=load_jsonl(path)[0]
    assert saved['model_params_b']==8.999976258
    assert saved['predictions']==[{'frame':4,'bboxes':[160,90,320]}]
    assert record['predictions'][0]['bboxes']==[0,0,640]
    with pytest.raises(ValueError,match='outside the original P1 selection'):
        write_scaled_submission([record],actions+[dict(actions[0],frame=5)],index,path,[.5,1.],8_999_976_258)


def test_previous_budget_overrun_and_inconsistent_total_block_next_stage():
    from aic.scale_budget import validate_budget
    b=empty_budget();b['budget_exceeded']=True
    with pytest.raises(ValueError,match='overrun'):validate_budget(b)
    b=empty_budget();b['stages']['scale_sft']['used_hours']=1
    with pytest.raises(ValueError,match='inconsistent'):validate_budget(b)


def test_preference_position_changes_are_rejected_even_with_matching_input_metadata():
    from aic.scale_preferences import validate_preference_pair
    pair=geometry_preference_pairs(row(),[.5,1.],.1,1)[0]
    pair['chosen']['aic_bbox_xyw'][0]+=10
    with pytest.raises(ValueError,match='position_shift_invalid'):validate_preference_pair(pair)


def test_dpo_step_forces_dropout_reference_to_eval_and_checks_image_tensors():
    from types import SimpleNamespace
    from aic.scale_trainer import dpo_train_step
    class Net(nn.Module):
        def __init__(self):
            super().__init__();self.emb=nn.Embedding(8,4);self.drop=nn.Dropout(.8);self.head=nn.Linear(4,8)
        def forward(self,input_ids,pixel_values=None):
            return SimpleNamespace(logits=self.head(self.drop(self.emb(input_ids))))
    model=Net().eval();ref=copy.deepcopy(model).requires_grad_(False).train()
    def example(ids):
        ids=torch.tensor([ids]);labels=ids.clone();labels[:,:2]=-100
        return {'model_inputs':{'input_ids':ids,'pixel_values':torch.ones(1,3)},'labels':labels,'completion_mask':labels.ne(-100)}
    pair={'chosen':example([1,2,3,4]),'rejected':example([1,2,4,3]),'position_source':'P1_frozen'}
    for name,value in [('prompt_sha256','prompt'),('source_id','video'),('image_id','frame'),('image_sha256','hash'),
                       ('position_xy',[320,180]),('target_ratio',[16,9])]:
        pair['chosen_'+name]=value;pair['rejected_'+name]=value
    before=copy.deepcopy(ref.state_dict());opt=torch.optim.SGD(model.parameters(),lr=.01)
    report=dpo_train_step(model,ref,opt,pair,.1)
    assert report['loss']==pytest.approx(.69314718) and report['grad_norm']>0
    assert not ref.training and not ref.drop.training
    assert all(torch.equal(v,ref.state_dict()[k]) for k,v in before.items())
    assert all(p.grad is None for p in ref.parameters())
    bad=copy.deepcopy(pair);bad['rejected']['model_inputs']['pixel_values']*=2
    with pytest.raises(ValueError,match='image input tensors differ'):dpo_train_step(model,ref,opt,bad,.1)
