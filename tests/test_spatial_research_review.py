"""CPU contract tests. Synthetic tensors here are never research evidence."""
from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from aic.scale_policy import canonical_sha, legal, sha
from aic.spatial_readout import (UtilityHead, batch, feature_path, fit_train_normalization,
    read_feature, save_checkpoint, train_readout, typed_loss)
from aic.spatial_research import (PARENT_ELEMENTS, candidate_boxes, crop_iou,
    preference_field, preference_pairs, promotion_gate, shot_dp, source_stat, validate_manifest, xywh)
from aic.spatial_research_data import recover_explicit_pairs, recover_sacd_selection_pairs, split_source
from aic.spatial_shared_vision import pool_candidates, unmerge_grid
from scripts.run_spatial_research import ledger_remaining, validate_lease


@pytest.mark.parametrize("size", [(1920,1080),(1080,1920),(640,480),(17,13),(3840,2160)])
@pytest.mark.parametrize("ratio", [(9,16),(16,9),(1,3),(3,1)])
def test_129_deployed_integer_windows(size, ratio):
    boxes, mask = candidate_boxes(*size, ratio)
    assert boxes.shape == (129,3) and mask.sum() == len(set(map(tuple, boxes)))
    assert all(legal(list(map(int,b)), *size, ratio) for b in boxes)
    assert np.all(xywh(boxes,ratio)[:,3] == boxes[:,2]*ratio[1]/ratio[0])


def sample(split="train", source="v", image="im"):
    return {"dataset_source":"LIVE_YT_VC","source_id":source,"image_id":image,"image_sha256":canonical_sha(image),
            "split":split,"target_ratio":[9,16],"label_type":"crop_iou","route":"A","media_kind":"video",
            "image_width":1920,"image_height":1080}


def test_public_contract_leakage_and_parent_identity():
    a=sample(); b=sample("dev")
    with pytest.raises(ValueError,match="source leakage"): validate_manifest([a,b])
    b=sample("dev","other")
    with pytest.raises(ValueError,match="image-byte leakage"): validate_manifest([a,b])
    for split in ("test","val","hidden_test"):
        with pytest.raises(ValueError,match="Test"): validate_manifest([sample(split)])
    a["dataset_source"]="AIC"
    with pytest.raises(ValueError,match="official"): validate_manifest([a])
    a=sample(); a.update(position_kind="P1_final_deployed",qp_floor=.15)
    with pytest.raises(ValueError,match="P2"): validate_manifest([a],True)


def test_cross_dataset_source_id_collision_and_source_bootstrap():
    s=[("LIVE","same")]*30+[("GNMC","same")]
    result=source_stat([0.]*30+[1.],s,300)
    assert result["sources"]==2 and result["mean"]==.5
    assert result["ci95"] is not None
    assert source_stat([.2],["one"],300)["ci95"] is None


def test_iou_annotators_averaged_not_cherry_picked():
    assert crop_iou([[0,0,10]], [[0,0,10,10],[20,20,10,10]],[1,1])[0]==.5


def test_sacd_no_best_box_or_stage_order_pairs():
    assert recover_explicit_pairs({"best_box":[[0,0,10,10]],"phase1_box":[[0,0,20,20]],"index":5})==[]
    with pytest.raises(ValueError,match="direction"):
        recover_explicit_pairs({"ranking_pairs":[[0,1]]})
    d={"subject_id":"person1","pair_semantics":"chosen_crop_id_preferred_to_rejected_crop_id_same_subject",
       "candidates":{"a":[0,0,1,1],"b":[0,0,2,2]},
       "ranking_pairs":[{"chosen_crop_id":"a","rejected_crop_id":"b","subject_id":"person1"}]}
    assert recover_explicit_pairs(d)==[("a","b")]
    d["ranking_pairs"][0]["subject_id"]="person2"
    with pytest.raises(ValueError,match="cross-subject"):recover_explicit_pairs(d)


def test_sacd_documented_selection_partial_orders_not_best_box():
    # A 640x427 image uses cell height floor(427/5)=85, not round-to-image-boundary.
    grid=[(i*128,j*85,(i+dx)*128,(j+dy)*85) for dx in (2,3,4,5) for dy in (2,3,4,5)
          for i in range(6-dx) for j in range(6-dy)]
    a={"phase1_box":grid[:20],"phase2_box":grid[:16],"best_box":[[999,999,9999,9999]],"select_from_phase2_box":list(reversed(grid[:8]))}
    boxes,pairs,levels=recover_sacd_selection_pairs(a,640,427)
    assert len(boxes)==100 and len(pairs)==20*80+16*4
    assert all((levels[i]>=1 and levels[j]==0) or (levels[i]==2 and levels[j]==1) for i,j in pairs)
    before=pairs.copy();a["best_box"]=[[0,0,1,1]];a["select_from_phase2_box"]=[]
    assert recover_sacd_selection_pairs(a,640,427)[1]==before
    a["phase2_box"]=[(0,0,640,427)]*16
    with pytest.raises(ValueError,match="contract"):recover_sacd_selection_pairs(a,640,427)


def test_subject_directed_pair_loss_and_mask_ablation():
    p=torch.zeros(1,3,requires_grad=True);m=torch.ones_like(p,dtype=torch.bool)
    q=torch.tensor([[2.,1.,0.]])
    loss=typed_loss(p,q,m,"subject_ranking","pairwise",explicit_pairs=[np.array([[0,2],[1,2]])]);loss.backward()
    assert p.grad[0,2]>0 and p.grad[0,0]<0
    model=UtilityHead(15,"linear",subject_conditioning=False)
    x=torch.randn(2,3,15);y=x.clone();y[:,:,-3:]+=50
    assert torch.equal(model(x),model(y))


def test_resume_after_interruption_matches_uninterrupted(tmp_path,monkeypatch):
    import aic.spatial_readout as module
    torch.set_num_threads(1)
    left=tmp_path/"full";right=tmp_path/"resumed"
    rows,identity=make_fixture(left);other,_=make_fixture(right)
    spec={"name":"unit_only","label_type":"crop_iou","kind":"linear","objective":"regression",
          "seed":31,"learning_rate":.01,"weight_decay":.001,"steps":4,"batch_size":2,"eval_every":2,"temperature":.05,"MOS_tie_margin":.25,"unit_fixture_only":True}
    train_readout(rows,left,identity,spec,"cpu")
    original=module.save_checkpoint
    def interrupt(path,payload):
        original(path,payload)
        if path.name=="last.pt" and payload["step"]==2:raise RuntimeError("unit simulated disconnect")
    monkeypatch.setattr(module,"save_checkpoint",interrupt)
    with pytest.raises(RuntimeError,match="disconnect"):train_readout(other,right,identity,spec,"cpu")
    monkeypatch.setattr(module,"save_checkpoint",original)
    train_readout(other,right,identity,spec,"cpu",resume=True)
    a=torch.load(left/"runs/unit_only/last.pt",weights_only=False);b=torch.load(right/"runs/unit_only/last.pt",weights_only=False)
    assert a["history"]==b["history"]
    for k in a["state"]:assert torch.equal(a["state"][k],b["state"][k])


def test_gaicd_pairs_keep_mos_ties_and_invalid_out():
    pairs=preference_pairs([1,1.1,4,-2],[True,True,True,False],.25)
    assert set(map(tuple,pairs))=={(2,0),(2,1)}


def test_preference_field_multipeak_not_pairs():
    result=preference_field([[0,0,10,10],[30,0,10,10],[1,1,10,10]],[5,5,1],[1,1])
    assert result["peak_count"]==2
    assert "pairs" not in result


def test_qwen_group_major_unmerge_is_not_reshape():
    grid=np.arange(4*6).reshape(4,6,1)
    tokens=grid.reshape(2,2,3,2,1).transpose(0,2,1,3,4).reshape(-1,1)
    assert not np.array_equal(tokens.reshape(4,6,1),grid)
    assert np.array_equal(unmerge_grid(tokens,4,6),grid)


def test_roi_fractional_original_coordinates_and_subject_presence():
    grid=np.array([[[0.],[10.]],[[20.],[30.]]],np.float32)
    f=pool_candidates(grid,[[0,0,50,100],[25,0,50,100]],100,100)
    assert f.shape==(2,15) and f[0,0]==10 and f[1,0]==15
    mask=np.array([[1,0],[1,0]],float)
    conditioned=pool_candidates(grid,[[0,0,50,100]],100,100,mask)
    assert conditioned[0,-3:].tolist()==[1.,1.,1.]
    assert f[0,-1]==0
    with pytest.raises(ValueError,match="bounds"):pool_candidates(grid,[[-1,0,50,100]],100,100)


@pytest.mark.parametrize("kind",["linear","mlp32"])
def test_real_dimension_complete_inference_parameter_cap(kind):
    model=UtilityHead(3*1152+12,kind)
    assert PARENT_ELEMENTS+model.extra_elements()<=9_000_000_000
    assert model(torch.ones(2,129,3468)).shape==(2,129)


@pytest.mark.parametrize("objective,label",[("regression","crop_iou"),("soft_target","crop_iou"),("pairwise","mos"),("regression","mos")])
def test_objectives_nonzero_gradients_masked_padding(objective,label):
    p=torch.zeros(2,4,requires_grad=True); y=torch.tensor([[1.,.2,0.,100.],[4.,1.,3.,999.]])
    mask=torch.tensor([[True,True,True,False],[True,True,True,False]])
    loss=typed_loss(p,y,mask,label,objective)
    assert torch.isfinite(loss); loss.backward()
    assert p.grad[:,:3].abs().sum()>0 and p.grad[:,3].abs().sum()==0
    if objective=="pairwise":
        good=torch.tensor([[5.,1.,0.,0.],[5.,1.,4.,0.]])
        assert typed_loss(good,y,mask,label,objective)<typed_loss(-good,y,mask,label,objective)


def test_losses_do_not_silently_mix_supervision():
    p=torch.zeros(1,2); mask=torch.ones_like(p,dtype=torch.bool)
    for label,objective in [("crop_iou","pairwise"),("mos","soft_target"),("subject_ranking","regression")]:
        with pytest.raises(ValueError):typed_loss(p,p,mask,label,objective)


def test_shot_dp_resets_instead_of_smoothing_across_cuts():
    scores=np.array([[10.,0],[0,10.],[10.,0],[0,10.]])
    boxes=np.tile([[[0,0,10],[90,0,10]]],(4,1,1))
    chosen=shot_dp(scores,boxes,[0,0,1,2],[0,1,2,3],penalty=100)
    assert chosen[0]==chosen[1] and chosen[2]==0 and chosen[3]==1


def test_budget_open_attempt_and_restart_cannot_reset_cap(tmp_path):
    assert ledger_remaining({"cap_seconds":100,"runs":[{"seconds":90,"ended_utc":"done"}]})==10
    with pytest.raises(ValueError,match="exhausted"):ledger_remaining({"cap_seconds":100,"runs":[{"seconds":100,"ended_utc":"done"}]})
    with pytest.raises(ValueError,match="unclosed"):ledger_remaining({"cap_seconds":100,"runs":[{"seconds":None,"ended_utc":None}]})
    from types import SimpleNamespace
    with pytest.raises(ValueError,match="supervisor"):validate_lease(SimpleNamespace(device="cuda:0",output=tmp_path,stage="extract"))


def test_oracle_or_static_or_transposed_B_cannot_clear_gate():
    result={"LIVE":{"media_kind":"video","split":"confirm","evaluation_geometry":"pointwise_integer_windows","policy_minus_P2":{"ci95":[.01,.02]}}}
    assert not promotion_gate(result,PARENT_ELEMENTS,{})["passed"]
    assert not promotion_gate({},9_000_000_001,{})["passed"]


def make_fixture(tmp_path):
    # Explicit synthetic unit fixture for executable/resume testing only.
    rows=[]; identity="unit_fixture_NOT_real_encoder"
    for i,split in enumerate(["train","train","dev","confirm"]):
        r=sample(split,str(i),str(i));rows.append(r)
        x=np.array([[0.,1.],[1.,0.],[.5,.5]],np.float16);y=np.array([0.,1.,.5],np.float32)
        p=feature_path(tmp_path,r);p.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(p,x=x,y=y,boxes=np.array([[0,0,10,10]]*3),mask=np.ones(3,bool))
        p.with_suffix(".json").write_text(json.dumps({"image_sha256":r["image_sha256"],"feature_identity_sha256":identity,
                                                   "manifest_row_sha256":canonical_sha(r),"npz_sha256":sha(p)}))
    return rows,identity


def test_checkpoint_loop_restore_and_provenance_fail_closed(tmp_path):
    torch.set_num_threads(1)
    rows,identity=make_fixture(tmp_path)
    spec={"name":"unit_only","label_type":"crop_iou","kind":"linear","objective":"regression",
          "seed":8,"learning_rate":.01,"weight_decay":0.,"steps":4,"batch_size":2,"eval_every":2,"temperature":.05,"MOS_tie_margin":.25,"unit_fixture_only":True}
    best=train_readout(rows,tmp_path,identity,spec,"cpu")
    before=sha(best);train_readout(rows,tmp_path,identity,spec,"cpu",resume=True);assert sha(best)==before
    changed=deepcopy(rows);changed[0]["image_id"]="wrong"
    with pytest.raises(FileNotFoundError):read_feature(tmp_path,changed[0],identity)
    with pytest.raises(ValueError,match="provenance"):read_feature(tmp_path,rows[0],"wrong_encoder")
    model=UtilityHead(2)
    with pytest.raises(ValueError,match="Train only"):fit_train_normalization(model,tmp_path,rows,identity)


def test_split_assignment_is_by_source_and_deterministic():
    assert split_source("same",True)==split_source("same",True)
    assert all(split_source(str(i),True) in {"dev","confirm"} for i in range(100))


def test_disk_preflight_blocks_before_gpu(tmp_path,monkeypatch):
    from PIL import Image
    from types import SimpleNamespace
    import scripts.run_spatial_research as runner
    p=tmp_path/"unit_image.png";Image.new("RGB",(16,16)).save(p)
    r=sample();r.update(input_image=str(p),image_sha256=sha(p),image_width=16,image_height=16,media_size=[16,16])
    manifest=tmp_path/"public_manifest.jsonl";manifest.write_text(json.dumps(r)+"\n")
    monkeypatch.setattr(runner,"verify_prepared",lambda *a:[r])
    monkeypatch.setattr(runner.shutil,"disk_usage",lambda *a:SimpleNamespace(free=0))
    with pytest.raises(ValueError,match="insufficient disk"):
        runner.audit_media(SimpleNamespace(output=tmp_path,asset_root=tmp_path),{"remote_asset_aliases":{}})
    result=json.loads((tmp_path/"media_audit.json").read_text())
    assert not result["passed"] and not result["GPU_used"]


def test_live_utility_train_extension_is_label_blind_source_level(tmp_path,monkeypatch):
    import scripts.prepare_spatial_live_train as module
    from aic.spatial_research import read_rows
    measured=sample();measured.update(source_id="cached",p2_bbox_xyw=[0,0,9])
    existing=tmp_path/"existing.jsonl";existing.write_text(json.dumps(measured)+"\n")
    data=[]
    for src in ['cached','extra','validation']:
        for frame in [0,36,72,108,144,174]:
            for ratio in [[9,16],[16,9]]:
                r=sample();r.update(source_id=src,frame_id=frame,target_ratio=ratio,
                    gt_crop_xywh=[0,0,5,9],split="val" if src=="validation" else "train")
                data.append(r)
    full=tmp_path/"full.jsonl";full.write_text(''.join(json.dumps(r)+'\n' for r in data))
    monkeypatch.setattr(module,'sha',lambda p:'a98aa78ada1f3f6b2a349c29c4bad15b65ede955dee761c586ced83f9d7d96c0')
    out=tmp_path/"extra.jsonl";module.sample(full,existing,out,target_sources=2)
    rows=read_rows(out)
    assert len(rows)==12 and {r['source_id'] for r in rows}=={'extra'}
    assert all(r['split']=='train' and r['p2_bbox_xyw'] is None for r in rows)
