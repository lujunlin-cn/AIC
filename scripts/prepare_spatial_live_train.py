#!/usr/bin/env python3
"""CPU-only: source-balanced LIVE utility supervision needs no P2 Train center.

Keep the 31 measured Train sources for diagnostics, add label-blind source
coverage up to 400, six fixed temporal frames per new source, both A/B views.
Reads only the previously verified public Train/Val manifest, never Test data.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

from aic.scale_policy import atomic_json,sha
from aic.spatial_research import read_rows,write_rows


def sample(public_labels,existing_manifest,output,target_sources=400):
    expected="a98aa78ada1f3f6b2a349c29c4bad15b65ede955dee761c586ced83f9d7d96c0"
    if sha(public_labels)!=expected:
        raise ValueError("full public label manifest differs from verified clean source")
    existing={r["source_id"] for r in read_rows(existing_manifest) if r["dataset_source"]=="LIVE_YT_VC" and r["split"]=="train" and r.get("p2_bbox_xyw")}
    sources=set()
    with Path(public_labels).open() as f:
        for line in f:
            row=json.loads(line)
            if row["dataset_source"]=="LIVE_YT_VC" and row["split"]=="train":sources.add(row["source_id"])
    if not existing<=sources or len(existing)>target_sources:
        raise ValueError("cached measured training sources do not match clean public split")
    ordered=sorted(sources-existing,key=lambda s:hashlib.sha256(f"utility-train-20261108:{s}".encode()).hexdigest())
    selected=set(ordered[:target_sources-len(existing)])
    if len(selected)+len(existing)!=target_sources:
        raise ValueError("insufficient actual clean LIVE sources")
    frames={0,36,72,108,144,174};rows=[]
    with Path(public_labels).open() as f:
        for line in f:
            r=json.loads(line)
            if r["dataset_source"]!="LIVE_YT_VC" or r["split"]!="train" or r["source_id"] not in selected or r["frame_id"] not in frames:continue
            r.update(label_type="crop_iou",media_kind="video",route="A" if r["target_ratio"]==[9,16] else "B",
                gt_boxes_xywh=[r["gt_crop_xywh"]],p2_bbox_xyw=None,position_kind=None,parent_baseline_pending=True,
                sampling="Train-only label-blind source hash and fixed temporal endpoints; no GT-scale/source-quality selection",
                frame_pts=None,subject_count=None,occluded=None,interaction=None)
            rows.append(r)
    if len(rows)!=len(selected)*12:
        raise ValueError("missing original frame IDs/A-B views; no silent source skip")
    write_rows(output,rows)
    atomic_json(Path(output).with_suffix(".audit.json"),{"full_public_labels_sha256":expected,"cached_Train_sources":len(existing),
        "additional_Train_sources":len(selected),"utility_Train_sources":target_sources,"additional_views":len(rows),
        "frame_ids":sorted(frames),"sampling_uses_GT_quality":False,"P2_center_inference_needed":False,
        "native_B_claimed":False,"test_labels_read":False,"additional_manifest_sha256":sha(output),
        "additional_source_ids":sorted(selected),"source_count_rationale":"preregistered 400-source first pilot within six-hour/24GiB-current-free-disk envelope, not a significance threshold"})


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--public-labels",type=Path,required=True);p.add_argument("--existing-manifest",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True);a=p.parse_args();sample(a.public_labels,a.existing_manifest,a.output)
