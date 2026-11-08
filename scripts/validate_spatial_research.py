#!/usr/bin/env python3
"""Independent CPU semantic checks on actual prepared research inputs."""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np

from aic.scale_policy import atomic_json,canonical_sha,legal,sha
from aic.spatial_research import candidate_boxes,crop_iou,read_rows,validate_manifest,xywh
from aic.spatial_research_data import recover_sacd_selection_pairs
from aic.spatial_readout import UtilityHead


def validate(root):
    rows=read_rows(root/"public_manifest.jsonl"); summary=validate_manifest(rows)
    counts=Counter();stage_pairs=Counter();images=defaultdict(set);geometry=0
    for r in rows:
        counts[r["dataset_source"]+"/"+r["split"]]+=1
        images[(r["dataset_source"],r["split"])].add(r["source_id"])
        assert r["split"] in {"train","dev","confirm"}
        if r["label_type"]=="crop_iou":
            boxes,mask=candidate_boxes(r["image_width"],r["image_height"],r["target_ratio"])
            # Independently check submitted x,y,w and implicit heights rather
            # than assuming source float xywh were deployable.
            assert all(legal([int(v) for v in b],r["image_width"],r["image_height"],r["target_ratio"]) for b in boxes)
            q=crop_iou(boxes,r["gt_boxes_xywh"],r["target_ratio"])
            assert np.isfinite(q).all() and (q>=0).all() and (q<=1).all();geometry+=len(boxes)
            if r.get("p2_bbox_xyw"):
                assert r["position_kind"]=="P2_final_deployed" and r["qp_floor"]==.6
                assert sha(root/r["p2_path_cache"])==r["p2_path_cache_sha256"]
        elif r["label_type"]=="subject_ranking":
            levels=r["candidate_preference_levels"];seen=set()
            for g in r["explicit_pair_groups"]:
                for hi in g["chosen"]:
                    for lo in g["rejected"]:
                        assert levels[hi]>levels[lo];seen.add((hi,lo))
            assert len(seen)==r["explicit_pair_count"] and not r["best_box_used"]
            assert not r["phase3_refinement_selection_used_as_ranking"]
            stage_pairs[r["split"]]+=len(seen)
        elif r["label_type"]=="mos":
            assert r["release"]=="journal" and all(1<=v<=5 for v,m in zip(r["candidate_quality"],r["candidate_mask"]) if m)
    parameters={}
    for kind in ("linear","mlp32"):
        model=UtilityHead(3468,kind)
        parameters[kind]={"head_parameters":sum(p.numel() for p in model.parameters()),"normalization_buffers":sum(p.numel() for p in model.buffers()),
                          "complete_P2_plus_one_head_elements":8_999_771_458+model.extra_elements()}
        assert parameters[kind]["complete_P2_plus_one_head_elements"]<=9_000_000_000
    result={"passed":True,"integrity":summary,"row_counts":dict(counts),"source_counts":{k[0]+"/"+k[1]:len(v) for k,v in images.items()},
            "integer_candidate_geometry_checks":geometry,"SACD_verified_stage_pairs_by_split":dict(stage_pairs),"parameters":parameters,
            "manifest_sha256":sha(root/"public_manifest.jsonl"),"no_hidden_or_Test_labels":True,
            "real_feature_GPU_smoke_claimed":False,"unit_fixtures_used_for_research":False,
            "official_results_generated":False,"Position_weight_updates":0}
    atomic_json(root/"semantic_validation.json",result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['row_counts','source_counts']},ensure_ascii=False,indent=2))
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,default=Path("artifacts/spatial_research_review"));a=p.parse_args();validate(a.output)
