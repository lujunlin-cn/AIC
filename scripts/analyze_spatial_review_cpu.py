#!/usr/bin/env python3
"""E6/E8 public P2 diagnostics; consumes no GPU and trains no model."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from aic.scale_geometry import checked_scale_action
from aic.scale_policy import atomic_json, center, legal, sha
from aic.spatial_research import crop_iou, read_rows, source_stat
from aic.spatial_research_data import load_runtime, replay


def action_quality(row, requested_center, bins):
    scores=[]; statuses=[]
    for value in bins:
        try:
            action=checked_scale_action(row["image_width"],row["image_height"],row["target_ratio"],requested_center,value)
        except ValueError:
            action={"valid":False,"status":"center_out_of_image"}
        statuses.append(action["status"])
        scores.append(float(crop_iou([action["aic_bbox_xyw"]],row["gt_boxes_xywh"],row["target_ratio"])[0]) if action["valid"] else -np.inf)
    return np.asarray(scores),statuses


def jitter(rows,bins,bootstrap):
    groups=defaultdict(list);out={};rng=np.random.default_rng(20261108)
    for r in rows:
        if r.get("p2_bbox_xyw"):
            groups[(r["dataset_source"],r["route"],r["split"])].append(r)
    for key,group in groups.items():
        axis=0 if key[1]=="A" else 1
        # Source-normalized Val residuals only support a diagnostic perturbation,
        # never new Train labels or a trained cross-fitted parent claim.
        residuals=[]
        for r in rows:
            if r.get("p2_bbox_xyw") and r["dataset_source"]==key[0] and r["route"]==key[1] and r["split"]!="train":
                truth=np.mean(r["gt_boxes_xywh"],axis=0);actual=center(r["p2_bbox_xyw"],r["target_ratio"])
                residuals.append((truth[axis]+truth[axis+2]/2-actual[axis])/[r["image_width"],r["image_height"]][axis])
        conditions={}
        for name,percent in [("minus_1pct",-.01),("plus_1pct",.01),("minus_3pct",-.03),("plus_3pct",.03),("minus_5pct",-.05),("plus_5pct",.05),("Val_residual_analysis_only",None)]:
            values=defaultdict(list);invalid=0;checks=0;flip=[];regret=[];src=[]
            for r in group:
                actual=np.array(center(r["p2_bbox_xyw"],r["target_ratio"]));q,_=action_quality(r,actual,bins)
                shifted=actual.copy();amount=float(rng.choice(residuals)) if percent is None else percent
                shifted[axis]+=amount*[r["image_width"],r["image_height"]][axis]
                change,status=action_quality(r,shifted,bins);checks+=len(change);invalid+=int((~np.isfinite(change)).sum())
                # Never average only convenient feasible rows without coverage.
                if not np.isfinite(change).any():
                    continue
                best=int(q.argmax());newbest=int(change.argmax());flip.append(float(best!=newbest))
                # If the old action is invalid under jitter, the regret is not
                # assigned a made-up finite value.
                regret.append(float(change[newbest]-change[best]) if np.isfinite(change[best]) else None)
                src.append((r["dataset_source"],r["source_id"]))
            valid=[i for i,v in enumerate(regret) if v is not None]
            conditions[name]={"best_scale_flip_rate":source_stat(flip,src,bootstrap),
                "perturbed_old_action_regret":source_stat([regret[i] for i in valid],[src[i] for i in valid],bootstrap),
                "invalid_action_checks":invalid,"action_checks":checks,"rows_without_any_window":len(group)-len(flip),
                "invalid_old_action_rows":len(group)-len(valid),"max_allowed_integer_position_drift_px":1}
        out["/".join(key)]={"conditions":conditions,"scale_bins":bins,"bins_refitted":False,
            "true_cross_fitted_Position_model":False,"blocker":"P2 Train parent was trained on these sources; no fold-specific Position checkpoints",
            "Val_residual_used_for_training_or_label_construction":False}
    return out


def motion(boxes,cache):
    values={"velocity":[],"acceleration":[],"jerk":[]}
    cuts=sorted(a["time"] for a in cache["anchors"] if a.get("scene_cut"))
    for start,end in cache["events"]:
        boundaries=[start]+[t for t in cuts if start<t<end]+[end]
        for left,right in zip(boundaries,boundaries[1:]):
            frames=sorted(f for f in boxes if left<=cache["pts"][f]<right)
            t=np.array([cache["pts"][f] for f in frames]);p=np.array([center(boxes[f],cache["ratio"]) for f in frames])
            for name in ("velocity","acceleration","jerk"):
                if len(t)<2:break
                delta=np.diff(t)
                if (delta<=0).any():raise ValueError("non-increasing frame PTS")
                p=np.diff(p,axis=0)/delta[:,None];t=(t[1:]+t[:-1])/2
                values[name].extend(np.linalg.norm(p,axis=1).tolist())
    return {k:float(np.mean(v)) if v else None for k,v in values.items()}


def trajectory(rows,output,repo,bootstrap):
    protocol=json.loads((repo/"artifacts/scale_policy_optimization/protocol.json").read_text())
    mods=load_runtime(repo/"artifacts/scale_policy_optimization/inputs/runtime",protocol["runtime_hashes"])
    paths=defaultdict(list);report=defaultdict(list)
    for r in rows:
        if r.get("p2_path_cache"):paths[r["p2_path_cache"]].append(r)
    for path,group in paths.items():
        cache=json.loads((output/path).read_text());base=replay(cache,mods,.6)
        b={r["frame"]:r["bboxes"] for r in base};w,h=cache["W"],cache["H"];ratio=cache["ratio"]
        # Registered finite CPU control: symmetric 3-frame within-shot median.
        # No GT determines a center, window size or smoothing parameter.
        cuts=sorted(a["time"] for a in cache["anchors"] if a.get("scene_cut"));changed={}
        for event_start,event_end in cache["events"]:
            bounds=[event_start]+[t for t in cuts if event_start<t<event_end]+[event_end]
            for left,right in zip(bounds,bounds[1:]):
                frames=[p["frame"] for p in base if left<=cache["pts"][p["frame"]]<right]
                for j,f in enumerate(frames):
                    neighbors=frames[max(0,j-1):min(len(frames),j+2)]
                    target=np.median(np.array([b[n] for n in neighbors]),axis=0).round().astype(int).tolist()
                    changed[f]=target if legal(target,w,h,ratio) else b[f]
        if set(changed)!=set(b):raise ValueError("shot control changed P2 time coverage")
        original_motion=motion(b,cache);control_motion=motion(changed,cache)
        for r in group:
            f=r["frame_id"]
            old=float(crop_iou([b[f]],r["gt_boxes_xywh"],ratio)[0]);new=float(crop_iou([changed[f]],r["gt_boxes_xywh"],ratio)[0])
            report[(r["dataset_source"],r["route"],r["split"])].append((r["source_id"],old,new,original_motion,control_motion))
    result={}
    for key,v in report.items():
        src=[(key[0],x[0]) for x in v];old=np.array([x[1] for x in v]);new=np.array([x[2] for x in v])
        result["/".join(key)]={"P2_deployed_IoU":source_stat(old,src,bootstrap),
            "registered_post_QP_shot_median_IoU":source_stat(new,src,bootstrap),"absolute_IoU_delta":source_stat(new-old,src,bootstrap),
            "P2_velocity_px_per_second":source_stat([x[3]["velocity"] for x in v],src,bootstrap),
            "control_velocity_px_per_second":source_stat([x[4]["velocity"] for x in v],src,bootstrap),
            "P2_acceleration_px_per_second2":source_stat([x[3]["acceleration"] for x in v],src,bootstrap),
            "control_acceleration_px_per_second2":source_stat([x[4]["acceleration"] for x in v],src,bootstrap),
            "P2_jerk_px_per_second3":source_stat([x[3]["jerk"] for x in v],src,bootstrap),
            "control_jerk_px_per_second3":source_stat([x[4]["jerk"] for x in v],src,bootstrap),
            "maximum_sampled_frame_IoU_regret":float((old-new).max()),"tracking_lag_seconds":None,
            "temporal_identity":1.,"cross_shot_smoothing":False,"optimizer_uses_GT":False,
            "control_is_not_promoted":True}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("artifacts/spatial_research_review"));a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];rows=read_rows(a.output/"public_manifest.jsonl")
    bins=json.loads((repo/"artifacts/scale_sft_dpo/config/scale_bins.json").read_text())["values"]
    atomic_json(a.output/"E6_P2_center_jitter.json",jitter(rows,bins,2000))
    atomic_json(a.output/"E8_shot_stability_control.json",trajectory(rows,a.output,repo,2000))


if __name__=="__main__":main()
