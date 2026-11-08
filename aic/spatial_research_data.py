"""Public-only data adapters for the spatial review's E1--E8 experiments."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

from aic.spatial_research import (SEED, candidate_boxes, crop_iou, preference_field,
    preference_pairs, read_rows, source_stat, validate_manifest, write_rows)
from aic.scale_policy import atomic_json, canonical_sha, center, route, sha


def split_source(identifier, validation=False):
    value = int(hashlib.sha256(f"spatial-review:{SEED}:{identifier}".encode()).hexdigest()[:8], 16) % 10
    return ("dev" if value < 5 else "confirm") if validation else ("dev" if value == 0 else "confirm" if value == 1 else "train")


def load_runtime(path, hashes):
    """CPU modules only. Does not import a historical script that hides CUDA."""
    path = Path(path).resolve()
    for name, digest in hashes.items():
        if sha(path / name) != digest:
            raise ValueError("original P2 CPU runtime identity changed: " + name)
    sys.path.insert(0, str(path))
    modules = {n: importlib.import_module(n) for n in ("contracts", "adaptive_crop", "clock_rule", "recovery_path_projection", "generation")}
    for name, module in modules.items():
        if Path(module.__file__).resolve() != path / (name + ".py"):
            raise ValueError("another runtime has already been imported: " + name)
    return modules


def replay(cache, modules, qp_floor=.6, replacement=None):
    """Original P2 smoothing, max-speed, integer export and QP. No model/GT.

    replacement contains normalized, image-derived candidate centers as spatial
    anchors. It registers a new spatial observation, preserving the original
    subjects, cuts, events, routes, full PTS and target frame membership.
    """
    c, a = modules["contracts"], modules["adaptive_crop"]
    adjusted = modules["clock_rule"].adjusted
    w, h, ratio = cache["W"], cache["H"], cache["ratio"]
    if cache["branch"] != "subject":
        return c.integer_predictions(c.frame_predictions(cache["pts"], cache["events"], c.crop_box(w, h, ratio)), w, h, ratio)
    baseline, _ = a.smooth_crops(np.asarray(cache["pts"]), cache["events"], adjusted(cache["anchors"], cache["events"]), w, h, ratio, max_speed=.15)
    baseline = c.integer_predictions(baseline, w, h, ratio)
    if cache["route"] == "native":
        return baseline
    anchors = cache["proposed"] if replacement is None else replacement
    candidate, _ = a.smooth_crops(np.asarray(cache["pts"]), cache["events"], adjusted(anchors, cache["events"]), w, h, ratio, max_speed=.15)
    candidate = c.integer_predictions(candidate, w, h, ratio)
    function = modules["recovery_path_projection"].apply_events if cache["route"] == "a" else modules["generation"].apply_vertical
    result, _ = function({"pts": cache["pts"], "width": w, "height": h}, cache["events"], cache["anchors"], baseline, candidate, qp_floor=qp_floor)
    if [p["frame"] for p in result] != [p["frame"] for p in baseline]:
        raise ValueError("spatial path changed time members")
    return result


def live_cached_rows(repo, output, config):
    preflight = repo / config["inputs"]["old_preflight"]
    labels = {(r["source_id"], r["image_id"], tuple(r["target_ratio"])): r for r in read_rows(preflight / "inputs/cohort.jsonl")}
    exposure = json.loads((preflight / "parent_exposure_audit.json").read_text())
    identity = json.loads((repo / "artifacts/scale_policy_optimization/parent_identity.json").read_text())
    protocol = json.loads((repo / "artifacts/scale_policy_optimization/protocol.json").read_text())
    runtime_path = repo / "artifacts/scale_policy_optimization/inputs/runtime"
    mods = load_runtime(runtime_path, protocol["runtime_hashes"])
    selected_sources = {r["source_id"] for r in read_rows(preflight / "positions.jsonl")}
    rows, audit = [], {"complete_cached_sources": len(selected_sources), "replayed_paths": 0, "P1_control_matches": 0, "excluded_exposed_val_views": 0}
    for path in sorted((preflight / "workers").glob("*/videos/*.json")):
        original = json.loads(path.read_text())
        if path.stem not in selected_sources:
            continue
        for detail in original["details"]:
            cache = detail["original_runtime_cache"]; ratio = cache["ratio"]
            old = {p["frame"]: p["bboxes"] for p in replay(cache, mods, .15)}
            expected = [r for r in original["records"] if r["target_ratio"] == ratio]
            if any(old[r["frame_id"]] != r["p1_aic_bbox_xyw"] for r in expected):
                raise ValueError("original P1 control cannot be reproduced before P2 conversion")
            audit["P1_control_matches"] += len(expected)
            current = {p["frame"]: p["bboxes"] for p in replay(cache, mods, .6)}
            cache_name = f"public_p2_paths/{path.stem}_{ratio[0]}_{ratio[1]}.json"
            atomic_json(output / cache_name, cache)
            allowed = exposure["branches"]["A" if ratio == [9, 16] else "B"]["independent_validation_sources"]
            for r in expected:
                key = (r["source_id"], r["image_id"], tuple(ratio)); label = dict(labels[key])
                if label["split"] == "val" and label["source_id"] not in allowed:
                    audit["excluded_exposed_val_views"] += 1
                    continue
                label["split"] = "train" if label["split"] == "train" else split_source(label["source_id"], True)
                label.update(label_type="crop_iou", media_kind="video", route=cache["route"].upper(),
                    p2_bbox_xyw=current[r["frame_id"]], position_kind="P2_final_deployed", qp_floor=.6,
                    p2_path_cache=cache_name, p2_path_cache_sha256=sha(output / cache_name),
                    p2_base_identity_sha256=canonical_sha(identity["model_evidence"]),
                    parent_cache_sha256=sha(path), source_video_sha256=original["source_video_sha256"],
                    p2_training_source_disjoint_verified=label["split"] != "train",
                    project_unseen_lockbox=False, evaluation_history="prior_scale_diagnostic_exposed; heldout_to_new_readout_only",
                    frame_pts=r["frame_pts"], subject_count=None, occluded=None, interaction=None)
                label["gt_boxes_xywh"] = [label["gt_crop_xywh"]]
                rows.append(label)
            audit["replayed_paths"] += 1
    validate_manifest(rows, require_baseline=True)
    audit.update(retained_rows=len(rows), by_split=dict(Counter(r["split"] for r in rows)),
                 metric_scope="actual P2 public video path, not relabelled P1")
    return rows, audit


def gnmc_rows(root):
    """Read Train/Validation only; quarantine complete byte aliases from Train."""
    root = Path(root); rows = []; by_hash = defaultdict(list)
    for split, name in (("train", "train"), ("val", "validation")):
        data = json.loads((root / "json" / (name + ".json")).read_text())
        for annotation in data:
            image_path = root / name / annotation["filename"]
            if not image_path.exists():
                image_path = root / "images" / annotation["filename"]
            if not image_path.exists():
                raise FileNotFoundError("GNMC missing media: " + str(image_path))
            digest = sha(image_path)
            with Image.open(image_path) as im:
                width, height = im.size
            item = {"source_id": digest, "image_id": Path(annotation["filename"]).stem,
                    "dataset_source": "GNMC", "split": "train" if split == "train" else split_source(digest, True),
                    "image_sha256": digest, "image_width": width, "image_height": height, "media_size": [width, height],
                    "input_image": str(image_path), "media_kind": "image", "label_type": "crop_iou",
                    "position_kind": None, "p2_bbox_xyw": None, "parent_baseline_pending": True}
            # Real native target ratios only. No LIVE transpose is called native B.
            for ratio in ([9, 16], [16, 9]):
                key = f"{ratio[0]}:{ratio[1]}"
                if key not in annotation["crop_bboxes"]:
                    continue
                raw = np.asarray(annotation["crop_bboxes"][key], float)
                clipped = np.clip(raw, 0, 1)
                if not np.isfinite(raw).all() or clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
                    raise ValueError("GNMC invalid crop")
                box = (clipped * [width, height, width, height]).tolist()
                box = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
                row = dict(item, target_ratio=ratio, gt_boxes_xywh=[box], raw_normalized_gt=raw.tolist(),
                           annotation_boundary_clip=bool((raw != clipped).any()), route=route(width, height, ratio).upper(),
                           transposed_geometry_augmentation=False)
                rows.append(row); by_hash[digest].append(row)
    quarantined = {h for h, group in by_hash.items() if len({r["split"] for r in group}) > 1}
    result = [r for r in rows if not (r["image_sha256"] in quarantined and r["split"] == "train")]
    # Two original IDs for identical bytes stay one independent source. Keep a
    # single view/ratio so source identity and feature caches are unambiguous.
    unique = {}
    for r in result:
        unique.setdefault((r["image_sha256"], tuple(r["target_ratio"])), r)
    result = list(unique.values()); validate_manifest(result)
    return result, {"raw_views": len(rows), "retained_views": len(result), "byte_alias_groups_quarantined_from_train": sorted(quarantined),
                    "test_labels_read": False, "P2_baseline_measured": False}


def gaicd_rows(root):
    root = Path(root); rows = []; count = Counter(); fields = []; per_image = []
    media = read_rows(root / "processed/media.jsonl")
    for item in media:
        if item["official_split"] not in {"train", "val", "validation"}:
            count["excluded_Test_image_ids_without_reading_labels"] += 1
            continue
        image_id = item["item_id"]
        candidate_file = root / "annotations/candidates" / f"{image_id}__journal.json"
        d = json.loads(candidate_file.read_text())
        if d["release"] != "journal":
            raise ValueError("GAICD conference release forbidden")
        relative = Path(item["media_path"]).parts
        at = relative.index("raw"); image_path = root.joinpath(*relative[at:])
        if not image_path.exists() or sha(image_path) != item["sha256"]:
            raise ValueError("GAICD media missing/hash changed")
        boxes = np.asarray(d["boxes_xywh"], float); scores = np.asarray(d["mos"], float)
        mask = np.asarray(d["valid"], bool) & np.isfinite(scores) & (scores >= 1) & (scores <= 5)
        geometry = (boxes[:, :2] >= 0).all(1) & (boxes[:, 2:] > 0).all(1)
        geometry &= ((boxes[:, :2] + boxes[:, 2:]) <= [d["W"], d["H"]]).all(1)
        valid = mask & geometry
        count["total_candidates"] += len(boxes); count["rated_candidates"] += int(mask.sum())
        count["valid_geometry_candidates"] += int(valid.sum())
        for ratio in ([9, 16], [16, 9]):
            match = valid & (np.abs(boxes[:, 2] / boxes[:, 3] - ratio[0] / ratio[1]) <= .005)
            count[f"target_ratio_matched_{ratio[0]}_{ratio[1]}"] += int(match.sum())
        pairs = preference_pairs(scores, valid, .25)
        count["valid_MOS_pairs_before_sampling"] += len(pairs)
        split = split_source(item["sha256"], item["official_split"] != "train")
        count[split+"_MOS_pairs_before_sampling"] += len(pairs)
        count[split+"_valid_candidates"] += int(valid.sum())
        row = {"dataset_source": "GAICD", "source_id": item["sha256"], "image_id": image_id,
               "image_sha256": item["sha256"], "split": split, "label_type": "mos", "media_kind": "image",
               "target_ratio": None, "ratio_condition_kind": "mixed_native_candidate_ratios", "route": "MOS_STATIC",
               "image_width": int(d["W"]), "image_height": int(d["H"]), "input_image": str(image_path),
               "candidate_boxes_xywh": boxes.tolist(), "candidate_quality": np.where(valid, scores, 0).tolist(),
               "candidate_mask": valid.tolist(), "candidate_annotation_sha256": sha(candidate_file),
               "release": "journal", "pair_semantics": "same_image_human_MOS_strict_delta_gt_0.25",
               "subject_id": None, "P2_center_matched_candidates": None}
        if valid.sum() >= 2:
            rows.append(row)
        if split == "train" and valid.any():
            fields.append(dict(image_id=image_id, **preference_field(boxes[valid], scores[valid], [1, 1])))
        per_image.append({"image_id": image_id, "split": split, "valid_candidates": int(valid.sum()), "MOS_pairs": len(pairs)})
    validate_manifest(rows)
    scored = [q for r in rows for q, m in zip(r["candidate_quality"], r["candidate_mask"]) if m]
    return rows, {**dict(count), "images": len(rows), "per_image": per_image,
        "MOS_distribution": {"mean": float(np.mean(scored)), "p5": float(np.quantile(scored, .05)), "p95": float(np.quantile(scored, .95))},
        "position_center_matched": None, "reason": "generic ranker deliberately allows Position; no actual P2 GAICD centers",
        "MOS_never_renamed_IoU": True, "Test_labels_read": False}, fields


def recover_explicit_pairs(annotation):
    """Accept only directed crop-ID pairs with explicitly defined semantics.

    phase1_box/phase2_box/order/index/best_box do not establish pair direction.
    This adapter is executable if the original ranking export is later supplied.
    """
    if "ranking_pairs" not in annotation:
        return []
    if annotation.get("pair_semantics") != "chosen_crop_id_preferred_to_rejected_crop_id_same_subject":
        raise ValueError("SACD pair direction/subject semantics unverified")
    crops = annotation.get("candidates", {})
    result = []
    for pair in annotation["ranking_pairs"]:
        if not {"chosen_crop_id", "rejected_crop_id", "subject_id"} <= set(pair):
            raise ValueError("incomplete explicit SACD pair")
        if pair["subject_id"] != annotation["subject_id"]:
            raise ValueError("SACD cross-subject pair")
        hi, lo = str(pair["chosen_crop_id"]), str(pair["rejected_crop_id"])
        if hi == lo or hi not in crops or lo not in crops:
            raise ValueError("SACD missing/distinct crop-ID contract")
        result.append((hi, lo))
    return result


def recover_sacd_selection_pairs(annotation, width, height):
    """Recover ONLY paper section 3.3's documented stage-1/2 human choices.

    Section 3.3 defines 100 windows on a 5x5 grid; phase1 retains reasonable
    crops and phase2 selects the best 16 among phase1. Direction is therefore
    established by original human selections, not by best_box or list order.
    select_from_phase2_box is a refinement choice and is deliberately unused.
    No within-stage order or cardinal MOS is invented.
    """
    if width < 5 or height < 5:
        raise ValueError("SACD image too small for its native grid")
    sx,sy=width//5,height//5
    grid=[(i*sx,j*sy,(i+dx)*sx,(j+dy)*sy) for dx in (2,3,4,5) for dy in (2,3,4,5)
          for i in range(6-dx) for j in range(6-dy)]
    first=set(map(tuple,annotation["phase1_box"])); second=set(map(tuple,annotation["phase2_box"]))
    if len(grid)!=100 or not second<=first<=set(grid) or len(second)<16 or len(first)<20:
        raise ValueError("SACD original staged selection/grid contract does not match paper")
    index={b:i for i,b in enumerate(grid)}
    # Explicit two-stage comparisons; union deduplicates only identical directed
    # pairs. No additional transitive or refinement preferences are synthesized.
    pairs={ (index[a],index[b]) for a in first for b in set(grid)-first }
    pairs.update((index[a],index[b]) for a in second for b in first-second)
    level=[2 if b in second else 1 if b in first else 0 for b in grid]
    boxes=[[x,y,x2-x,y2-y] for x,y,x2,y2 in grid]
    return boxes,sorted(pairs),level


def sacd_rows(root, split_manifest):
    root=Path(root); saved=json.loads(Path(split_manifest).read_text()); rows=[];counts=Counter(); hashes={}
    for group in saved["groups"]:
        if group["split"]=="test":
            continue
        image=next((root/"images").glob(group["image_id"]+".*")); image_hash=sha(image)
        if image_hash!=group["image_sha256"]:
            raise ValueError("SACD image bytes differ from image-disjoint split")
        with Image.open(image) as im:width,height=im.size
        split="train" if group["split"]=="train" else split_source(group["image_id"],True)
        for subject in group["subjects"]:
            annotation_path=root/subject["annotation"]; d=json.loads(annotation_path.read_text())
            boxes,pairs,levels=recover_sacd_selection_pairs(d,width,height)
            stem=Path(subject["annotation"]).stem
            mask=next((root/"masks").glob(stem+".*"))
            with Image.open(mask) as im:
                if im.size!=(width,height):raise ValueError("SACD subject mask/image geometry mismatch")
            counts[split+"_explicit_stage_selection_pairs"]+=len(pairs);counts[split+"_subject_rows"]+=1
            hashes[subject["annotation"]]=sha(annotation_path)
            rows.append({"dataset_source":"SACD","source_id":image_hash,"image_id":group["image_id"],
                "subject_id":subject["subject_id"],"annotated_subject_count":len(group["subjects"]),
                "split":split,"label_type":"subject_ranking","media_kind":"image","route":"SUBJECT_STATIC",
                "target_ratio":None,"image_sha256":image_hash,"image_width":width,"image_height":height,
                "input_image":str(image),"subject_mask":str(mask),"subject_mask_sha256":sha(mask),
                "candidate_boxes_xywh":boxes,"explicit_pair_groups":[
                    {"stage":"human_reasonable_selection","chosen":[i for i,v in enumerate(levels) if v>=1],"rejected":[i for i,v in enumerate(levels) if v==0]},
                    {"stage":"human_best_subset_selection","chosen":[i for i,v in enumerate(levels) if v==2],"rejected":[i for i,v in enumerate(levels) if v==1]}],
                "explicit_pair_count":len(pairs),"candidate_preference_levels":levels,
                "candidate_annotation_sha256":sha(annotation_path),"candidate_mask":[True]*100,
                "pair_semantics":"documented_same_subject_human_phase1_and_phase2_selection_only",
                "pair_definition_evidence":"SACD original paper section 3.3, Computational Visual Media DOI 10.1007/s41095-021-0263-3",
                "best_box_used":False,"phase3_refinement_selection_used_as_ranking":False})
    validate_manifest(rows)
    return rows,{**dict(counts),"subject_rows":len(rows),"image_splits_mutually_exclusive":True,
        "pair_contract_verified":True,"independent_human_supervision":True,"best_box_pairs_created":0,
        "raw_explicit_pair_fields":0,"recovered_from":"documented original two-stage human selections on exact native grid",
        "annotation_manifest_sha256":canonical_sha(hashes),"Test_annotation_content_read":False,
        "not_general_unconditional_MOS":True}


def sacd_audit(root, split_manifest):
    root = Path(root); saved = json.loads(Path(split_manifest).read_text())
    result = Counter(); schema = Counter(); hashes = {}; recovered = []
    images = set(); by_split = defaultdict(set)
    for group in saved["groups"]:
        split = group["split"]; image_id = group["image_id"]
        if image_id in images:
            raise ValueError("SACD image-level split is not mutually exclusive")
        images.add(image_id); by_split[split].add(image_id)
        if split == "test":
            continue  # Test membership known; its annotation content is never read.
        for subject in group["subjects"]:
            p = root / subject["annotation"]; d = json.loads(p.read_text()); hashes[subject["annotation"]] = sha(p)
            schema.update(d.keys()); pairs = recover_explicit_pairs(d)
            result[f"{split}_subject_annotations"] += 1; result["explicit_ranked_pairs"] += len(pairs)
            if pairs:
                recovered.append({"image_id": image_id, "subject_id": subject["subject_id"], "pairs": pairs, "annotation_sha256": sha(p)})
    return {**dict(result), "image_splits": {k: len(v) for k, v in by_split.items()}, "schema_keys": dict(schema),
            "image_disjoint": True, "split_manifest_sha256": sha(split_manifest), "annotation_manifest_sha256": canonical_sha(hashes),
            "annotation_hashes": hashes, "pairs": recovered, "DPO_eligible": False,
            "best_box_pairs_created": 0, "Test_annotation_content_read": False,
            "blocker": None if recovered else "installed SACD export lacks explicit directed subject ranking; stage membership is not preference direction"}


def position_and_oracle(rows, bootstrap=2000):
    """E1/E4 CPU baseline and GT-visible integer-window ceiling, route-separated."""
    out = {}
    groups = defaultdict(list)
    for r in rows:
        if r["label_type"] == "crop_iou" and r.get("p2_bbox_xyw"):
            groups[(r["dataset_source"], r["route"], r["split"])].append(r)
    for key, group in groups.items():
        sources, base, oracle, error_x, error_y, edge = [], [], [], [], [], []
        for r in group:
            boxes, mask = candidate_boxes(r["image_width"], r["image_height"], r["target_ratio"])
            quality = crop_iou(boxes, r["gt_boxes_xywh"], r["target_ratio"])
            sources.append((r["dataset_source"], r["source_id"]))
            base.append(float(crop_iou([r["p2_bbox_xyw"]], r["gt_boxes_xywh"], r["target_ratio"])[0]))
            oracle.append(float(quality[mask].max()))
            gt = np.mean(r["gt_boxes_xywh"], axis=0)
            cx, cy = center(r["p2_bbox_xyw"], r["target_ratio"])
            error_x.append(abs(cx - gt[0] - gt[2]/2) / r["image_width"])
            error_y.append(abs(cy - gt[1] - gt[3]/2) / r["image_height"])
            edge.append(min(gt[0] / r["image_width"], gt[1] / r["image_height"],
                            1-(gt[0]+gt[2])/r["image_width"], 1-(gt[1]+gt[3])/r["image_height"]) < .05)
        b, o = np.asarray(base), np.asarray(oracle)
        out["/".join(key)] = {"P2_deployed_IoU": source_stat(b, sources, bootstrap),
            "GT_visible_integer_Oracle129": source_stat(o, sources, bootstrap),
            "GT_visible_integer_Oracle_minus_P2": source_stat(o-b, sources, bootstrap),
            "normalized_center_abs_x": source_stat(error_x, sources, bootstrap),
            "normalized_center_abs_y": source_stat(error_y, sources, bootstrap),
            "GT_edge_proxy_source_macro": source_stat(np.array(edge, float), sources, bootstrap),
            "unknown_subject_count_rows": len(group), "unknown_occlusion_rows": len(group),
            "GT_edge_is_not_manual_subject_taxonomy": True,
            "oracle_scope": "pointwise integer candidate GT ceiling, not learned or full-QP oracle"}
    return out
