#!/usr/bin/env python3
"""Execute the spatial review: public audit -> real shared features -> readouts.

prepare/audit/status are CPU-only. run requires explicit invocation after GPU
activation and supervises a single persistent six-hour cumulative ledger.
No Video-ORA training, DPO, official upload or automatic package generation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time

import numpy as np

from aic.scale_policy import atomic_json, canonical_sha, center, legal, sha
from aic.spatial_research import (PARENT_ELEMENTS, candidate_boxes, crop_iou, promotion_gate,
                                read_rows, source_stat, validate_manifest, write_rows, xywh)

REPO = Path(__file__).resolve().parents[1]


def config(path):
    c = json.loads(Path(path).read_text())
    if c["parent"] != "P2" or c["candidate_count"] != 129 or c["train_videoora"] or c["use_DPO"]:
        raise ValueError("review scope changed")
    if any(s.get("unit_fixture_only") for s in c["readouts"]):
        raise ValueError("unit fixtures are forbidden in research config")
    return c


def prepare(args, c):
    from aic.spatial_research_data import gaicd_rows, gnmc_rows, live_cached_rows, position_and_oracle, sacd_audit, sacd_rows
    out = args.output
    if (out/"prepared.json").exists():
        verify_prepared(out, c)
        print("Prepared inputs already verified; not overwritten.", flush=True)
        return
    out.mkdir(parents=True, exist_ok=True)
    live, live_audit = live_cached_rows(REPO, out, c)
    extra_path=REPO/c["inputs"]["additional_live_train"]
    if sha(extra_path)!=c["inputs"]["additional_live_train_sha256"]:
        raise ValueError("additional Train-only LIVE input identity changed")
    extra=read_rows(extra_path)
    if any(r["dataset_source"]!="LIVE_YT_VC" or r["split"]!="train" or r.get("p2_bbox_xyw") for r in extra):
        raise ValueError("additional utility samples must be actual Train labels, not fabricated P2 positions")
    if len({r["source_id"] for r in live+extra if r["split"]=="train"})!=c["utility_LIVE_train_sources"]:
        raise ValueError("preregistered LIVE training source coverage differs")
    live_audit["utility_training"]={"sources":c["utility_LIVE_train_sources"],"additional_views":len(extra),
        "actual_P2_Train_center_sources":31,"P2_Position_inference_added":False,
        "sampling":"clean Train source hash, six fixed temporal frames, both A/B views; no GT-scale quality selection"}
    gnmc, gnmc_audit = gnmc_rows(c["local_assets"]["GNMC"])
    gaicd, gaicd_audit, fields = gaicd_rows(c["local_assets"]["GAICD"])
    sacd = sacd_audit(c["local_assets"]["SACD"], REPO/c["inputs"]["sacd_split"])
    subject_rows, recovery=sacd_rows(c["local_assets"]["SACD"],REPO/c["inputs"]["sacd_split"])
    sacd["historical_explicit_field_blocker"]=sacd.pop("blocker")
    sacd.update(selection_pair_recovery=recovery,blocker=None,ranker_eligible=True)
    rows = live + extra + gnmc + gaicd + subject_rows
    # Global byte leakage check covers datasets too. If a new alias is found,
    # stop for an explicit whole-source quarantine, never silently drop a row.
    integrity = validate_manifest(rows)
    write_rows(out/"public_manifest.jsonl", rows)
    atomic_json(out/"data_audit.json", {"LIVE": live_audit, "GNMC": gnmc_audit,
        "GAICD": gaicd_audit, "SACD": sacd, "integrity": integrity,
        "RetargetVid": {"annotations_exist": True, "recoverable_video_media_verified": False,
                        "result": None, "blocker": "no original RetargetVid video/old R70 tables in current clone"},
        "authorization": "owner_completed; no repeated license investigation"})
    atomic_json(out/"E1_P2_position_and_Oracle129.json", position_and_oracle(live, c["bootstrap"]))
    atomic_json(out/"E5_GAICD_train_preference_fields.json", {"label_type": "mos", "images": fields,
        "SACD_field": None, "SACD_blocker": "stage-selection partial orders have no cardinal MOS field; do not invent interpolation values",
        "continuous_field_validated": False, "created_ranked_pairs_from_fields": 0})
    atomic_json(out/"prepared.json", {"config_sha256": canonical_sha(c), "manifest_sha256": sha(out/"public_manifest.jsonl"),
        "row_count": len(rows), "sources_by_dataset_and_split": dict(Counter(r["dataset_source"]+"/"+r["split"] for r in rows)),
        "created_utc": datetime.now(timezone.utc).isoformat(), "P2_only": True,
        "path_caches": {str(p.relative_to(out)): sha(p) for p in sorted((out/"public_p2_paths").glob("*.json"))},
        "code_sha256": code_hashes(), "raw_data_modified": False, "GPU_started": False})
    status(out, c)


def code_hashes():
    names = ["aic/spatial_research.py", "aic/spatial_research_data.py", "aic/spatial_shared_vision.py",
             "aic/spatial_readout.py", "scripts/run_spatial_research.py", "scripts/analyze_spatial_review_cpu.py",
             "scripts/validate_spatial_research.py", "scripts/run_spatial_research_gpu.sh", "scripts/prepare_spatial_live_train.py"]
    return {name: sha(REPO/name) for name in names}


def verify_prepared(out, c):
    lock = json.loads((out/"prepared.json").read_text())
    if lock["config_sha256"] != canonical_sha(c) or lock["manifest_sha256"] != sha(out/"public_manifest.jsonl"):
        raise ValueError("prepared input/config changed; preserve old run and prepare a new output directory")
    if sha(REPO/c["inputs"]["additional_live_train"])!=c["inputs"]["additional_live_train_sha256"]:
        raise ValueError("additional Train source manifest changed")
    for path, digest in lock["path_caches"].items():
        if sha(out/path) != digest:
            raise ValueError("P2 public path cache changed")
    if lock["code_sha256"] != code_hashes():
        raise ValueError("prepared code changed; use a new immutable run directory")
    rows = read_rows(out/"public_manifest.jsonl"); validate_manifest(rows)
    return rows


def aliases(c, asset_root):
    root = str(asset_root)
    return {**{old: root+"/"+new for old, new in c["remote_asset_aliases"].items()}, "repro": root+"/repro"}


def audit_media(args, c):
    from aic.spatial_shared_vision import load_image,load_subject_mask
    rows = verify_prepared(args.output, c); missing, passed = [], 0
    seen = set()
    for r in rows:
        key = (r["image_sha256"], r.get("transposed_geometry_augmentation", False))
        try:
            if key not in seen:
                image = load_image(r, aliases(c, args.asset_root)); image.close(); passed += 1;seen.add(key)
            if r.get("subject_mask"):
                load_subject_mask(r,(2,2),aliases(c,args.asset_root))
        except (FileNotFoundError, ValueError) as exc:
            missing.append({"image_id": r["image_id"], "dataset": r["dataset_source"], "reason": str(exc)})
    from aic.spatial_readout import feature_path
    pending_bytes=0
    for r in rows:
        if feature_path(args.output,r).exists():continue
        k=129 if r["label_type"]=="crop_iou" else len(r["candidate_boxes_xywh"])
        pending_bytes+=k*(3468*2+4*4+4+1)+r.get("explicit_pair_count",0)*16+4096
    free=shutil.disk_usage(args.output).free
    required=int(pending_bytes*1.15)+2*1024**3
    enough=free>=required
    atomic_json(args.output/"media_audit.json", {"passed_images": passed, "missing_or_invalid": missing,
        "passed": not missing and enough, "silent_skips": 0, "GPU_used": False,
        "manifest_sha256":sha(args.output/"public_manifest.jsonl"),
        "disk_free_bytes":free,"conservative_remaining_feature_and_reserve_bytes":required,"disk_capacity_passed":enough})
    if missing:
        raise ValueError(f"missing/invalid public media: {len(missing)}; see media_audit.json")
    if not enough:
        raise ValueError("insufficient disk for real feature cache; GPU not started; preserve all historical data")


def extract(args, c):
    import torch
    from aic.spatial_readout import feature_path, read_feature
    from aic.spatial_shared_vision import SharedP2Vision, load_image, pool_candidates, load_subject_mask
    rows = verify_prepared(args.output, c)
    vision = SharedP2Vision(args.asset_root/"models/compact", REPO/c["inputs"]["parent_identity"], args.device)
    atomic_json(args.output/"feature_identity.json", {"identity": vision.identity, "sha256": vision.identity_sha256})
    remaining = []
    complete_keys=set()
    for r in rows:
        p = feature_path(args.output, r)
        if p.exists() and p.with_suffix(".json").exists():
            read_feature(args.output, r, vision.identity_sha256)
            complete_keys.add(canonical_sha(r))
        else:
            remaining.append(r)
    batch_size = c["vision_batch_size"]; started = time.monotonic(); completed = 0
    for start in range(0, len(rows), batch_size):
        # Keep batch context stable across a partially completed/resumed chunk.
        batch = rows[start:start+batch_size]
        if all(canonical_sha(r) in complete_keys for r in batch):continue
        images = [load_image(r, aliases(c, args.asset_root)) for r in batch]
        try:
            grids = vision.encode(images)
            smoke=args.output/"feature_smoke.json"
            if not smoke.exists():
                singleton=vision.encode(images[:1])[0]
                maximum=float(np.max(np.abs(singleton-grids[0])))
                matches=bool(np.allclose(singleton,grids[0],rtol=.02,atol=.02))
                atomic_json(smoke,{"passed":matches,"real_P2_visual_weights":True,"feature_identity_sha256":vision.identity_sha256,
                    "first_image_sha256":batch[0]["image_sha256"],"singleton_batch_max_abs":maximum,
                    "frozen_eval":True,"GPU":torch.cuda.get_device_name(0),"optimizer_steps":0,
                    "scope":"new shared visual feature path; not a repeated 9B SFT/DPO smoke"})
                if not matches:raise ValueError("visual batch/singleton contract mismatch; see feature_smoke.json")
            else:
                previous=json.loads(smoke.read_text())
                if not previous["passed"] or previous["feature_identity_sha256"]!=vision.identity_sha256:
                    raise ValueError("existing shared-vision smoke is failed or drifted")
        finally:
            for im in images:
                im.close()
        for r, grid in zip(batch, grids):
            if canonical_sha(r) in complete_keys:continue
            if r["label_type"] == "crop_iou":
                boxes, mask = candidate_boxes(r["image_width"], r["image_height"], r["target_ratio"])
                boxes = xywh(boxes, r["target_ratio"])
                quality = crop_iou(boxes, r["gt_boxes_xywh"], r["target_ratio"])
            elif r["label_type"] in {"mos","subject_ranking"}:
                boxes = np.asarray(r["candidate_boxes_xywh"], float)
                mask = np.asarray(r["candidate_mask"], bool)
                quality = np.asarray(r["candidate_quality"] if r["label_type"]=="mos" else r["candidate_preference_levels"],float)
            else:
                raise ValueError("subject ranking requires recovered explicit schema")
            # Invalid MOS rows are not pooled, and receive a zero feature with
            # mask=False; arbitrary invalid coordinates never reach ROI code.
            x = np.zeros((len(boxes), grid.shape[-1]*3+12), np.float32)
            subject=load_subject_mask(r,grid.shape[:2],aliases(c,args.asset_root)) if r.get("subject_mask") else None
            x[mask] = pool_candidates(grid, boxes[mask], r["image_width"], r["image_height"],subject)
            pairs=[]
            for group in r.get("explicit_pair_groups",[]):
                pairs.extend((hi,lo) for hi in group["chosen"] for lo in group["rejected"])
            pairs=np.asarray(pairs,dtype=np.int64).reshape(-1,2)
            if r["label_type"]=="subject_ranking" and len(pairs)!=r["explicit_pair_count"]:
                raise ValueError("SACD directed pair reconstruction changed")
            p = feature_path(args.output, r); p.parent.mkdir(parents=True, exist_ok=True)
            temp = p.with_suffix(".tmp")
            with temp.open("wb") as f:
                np.savez_compressed(f, x=x.astype(np.float16), y=quality.astype(np.float32), boxes=boxes.astype(np.float32), mask=mask,pairs=pairs)
            temp.replace(p)
            atomic_json(p.with_suffix(".json"), {"feature_identity_sha256": vision.identity_sha256,
                "image_sha256": r["image_sha256"], "manifest_row_sha256": canonical_sha(r), "npz_sha256": sha(p)})
            complete_keys.add(canonical_sha(r));completed+=1
        atomic_json(args.output/"feature_progress.json", {"this_attempt_completed": completed,
            "pending_this_attempt": len(remaining)-completed, "seconds": time.monotonic()-started})
        print(json.dumps({"feature_completed_this_attempt": completed, "remaining": len(remaining)-completed}), flush=True)
    atomic_json(args.output/"features_complete.json", {"manifest_sha256": sha(args.output/"public_manifest.jsonl"),
        "rows": len(rows), "real_P2_visual_tower": True, "feature_identity_sha256": vision.identity_sha256,
        "feature_files": {feature_path(args.output, r).name: sha(feature_path(args.output, r)) for r in rows}})


def train(args, c):
    import torch
    from aic.spatial_readout import train_readout
    rows = verify_prepared(args.output, c)
    complete = json.loads((args.output/"features_complete.json").read_text())
    if complete["manifest_sha256"] != sha(args.output/"public_manifest.jsonl"):
        raise ValueError("feature manifest differs")
    torch.set_num_threads(c["torch_cpu_threads"])
    for spec in c["readouts"]:
        path = args.output/"runs"/spec["name"]/"training_receipt.json"
        if path.exists() and args.resume:
            receipt = json.loads(path.read_text())
            if sha(path.parent/"best.pt") != receipt["best_checkpoint_sha256"]:
                raise ValueError("completed readout checkpoint changed")
            continue
        train_readout(rows, args.output, complete["feature_identity_sha256"], spec, args.device, args.resume)


def path_evaluate(predictions, out, c):
    from aic.spatial_research_data import load_runtime, replay
    p = json.loads((REPO/"artifacts/scale_policy_optimization/protocol.json").read_text())
    mods = load_runtime(REPO/"artifacts/scale_policy_optimization/inputs/runtime", p["runtime_hashes"])
    groups = defaultdict(list); report = defaultdict(list)
    for record in predictions:
        if record["row"].get("p2_path_cache"):
            groups[record["row"]["p2_path_cache"]].append(record)
    for cache_path, records in groups.items():
        cache = json.loads((out/cache_path).read_text())
        proposed = []; oracle_proposed = []
        # Copy original scene cuts into the new anchor set. The chosen windows
        # are image-model outputs; no public GT is passed into the policy path.
        for record in records:
            r = record["row"]; x, y, w, h = record["bbox_xywh"]
            proposed.append({"time": r["frame_pts"], "sample_time": r["frame_pts"], "sample_frame_id": r["frame_id"],
                "subject_box": [(x+w/2)/r["image_width"], (y+h/2)/r["image_height"], (x+w/2)/r["image_width"], (y+h/2)/r["image_height"]], "scene_cut": False})
        for anchor in cache["proposed"]:
            if anchor.get("scene_cut"):
                q = dict(anchor)
                # Original cut anchor is retained to reset interpolation. Its
                # original P2 position is a registered fallback at the cut.
                proposed.append(q)
        proposed.sort(key=lambda r: (r["time"], not r.get("scene_cut", False)))
        result = {p["frame"]: p["bboxes"] for p in replay(cache, mods, .6, proposed)}
        parent = {p["frame"]: p["bboxes"] for p in replay(cache, mods, .6)}
        if set(result) != set(parent):
            raise ValueError("path evaluation changed P2 time members")
        for record in records:
            r = record["row"]; f = r["frame_id"]
            if parent[f] != r["p2_bbox_xyw"] or not legal(result[f], r["image_width"], r["image_height"], r["target_ratio"]):
                raise ValueError("P2 baseline/geometry replay failed")
            new = float(crop_iou([result[f]], r["gt_boxes_xywh"], r["target_ratio"])[0])
            old = float(crop_iou([parent[f]], r["gt_boxes_xywh"], r["target_ratio"])[0])
            displacement = max(abs(a-b) for a, b in zip(center(result[f], r["target_ratio"]), center(parent[f], r["target_ratio"])))
            report[(r["dataset_source"], r["route"], r["split"])].append((r["source_id"], new, old, displacement))
    output = {}
    for key, values in report.items():
        src = [(key[0], v[0]) for v in values]; new = np.array([v[1] for v in values]); old = np.array([v[2] for v in values])
        output["/".join(key)] = {"media_kind": "video", "split": key[2], "evaluation_geometry": "original_P2_video_path",
            "P2_IoU": source_stat(old, src, c["bootstrap"]), "readout_deployed_IoU": source_stat(new, src, c["bootstrap"]),
            "policy_minus_P2": source_stat(new-old, src, c["bootstrap"]),
            "parent_center_displacement_px": source_stat([v[3] for v in values], src, c["bootstrap"]),
            "temporal_identity": 1., "original_QP_floor": .6, "Native_unchanged": True,
            "source_ids":sorted({v[0] for v in values}),
            "native_domain":not any(r["row"].get("transposed_geometry_augmentation") for records in groups.values() for r in records if r["row"]["dataset_source"]==key[0] and r["row"]["route"]==key[1] and r["row"]["split"]==key[2]),
            "B_scope": "transposed LIVE, not native B" if key[1] == "B" else "native LIVE A",
            "spatial_observation_change": "readout candidates at sampled public annotation frames, original cut anchors retained"}
    return output


def evaluate(args, c):
    import torch
    from aic.spatial_readout import UtilityHead, evaluate_head
    rows = verify_prepared(args.output, c); all_results = {}
    identity = json.loads((args.output/"feature_identity.json").read_text())["sha256"]
    for spec in c["readouts"]:
        directory = args.output/"runs"/spec["name"]; done = directory/"confirmation.json"
        if done.exists():
            recorded = json.loads(done.read_text())
            if recorded["best_checkpoint_sha256"] != sha(directory/"best.pt"):
                raise ValueError("Confirm already opened with another model; no retuning")
            all_results[spec["name"]] = recorded; continue
        state = torch.load(directory/"best.pt", map_location="cpu", weights_only=False)
        receipt=json.loads((directory/"training_receipt.json").read_text())
        if sha(directory/"best.pt")!=receipt["best_checkpoint_sha256"] or state["feature_identity"]!=identity or state["spec"]!=spec:
            raise ValueError("model/feature/spec does not match its training receipt")
        model = UtilityHead(state["dim"], spec["kind"],subject_conditioning=spec.get("subject_conditioning",True)); model.load_state_dict(state["state"])
        selected = [r for r in rows if r["split"] in {"dev", "confirm"} and r["label_type"] == spec["label_type"]]
        report, predictions = evaluate_head(model, selected, args.output, identity, "cpu", spec["batch_size"], c["bootstrap"])
        deployed = path_evaluate(predictions, args.output, c) if spec["label_type"] == "crop_iou" else {}
        result = {"run": spec["name"], "label_type": spec["label_type"], "best_checkpoint_sha256": sha(directory/"best.pt"),
            "pointwise_metrics_NOT_video_deployment": report, "P2_video_deployment": deployed,
            "confirmation_opened_once_after_checkpoint_lock": True, "official_F1": None,
            "complete_inference_elements": state["total_inference_elements"]}
        atomic_json(done, result); all_results[spec["name"]] = result
    atomic_json(args.output/"readout_results.json", all_results)
    subject_comparison={}
    for kind in ("linear","mlp32"):
        conditioned=all_results.get(f"E3_{kind}_subject",{}).get("pointwise_metrics_NOT_video_deployment",{})
        control=all_results.get(f"E3_{kind}_zero_subject_control",{}).get("pointwise_metrics_NOT_video_deployment",{})
        for key,report in conditioned.items():
            if key not in control:continue
            actual=report["pairwise_accuracy"]["per_source"];zero=control[key]["pairwise_accuracy"]["per_source"]
            if set(actual)!=set(zero):raise ValueError("subject/control source pairing changed")
            sources=sorted(actual)
            subject_comparison[kind+"/"+key]={"conditioned_minus_zero_subject_image_macro_accuracy":source_stat([actual[s]-zero[s] for s in sources],sources,c["bootstrap"]),
                "same_images_candidates_splits":True,"condition_permutation_control":None,
                "occlusion_interaction_human_labels_available":False,"eligible_for_AIC_direct_deployment":False}
    atomic_json(args.output/"E7_subject_conditioning_ablation.json",subject_comparison)
    status(args.output, c)


def ledger_remaining(value, now=None):
    used = sum(r["seconds"] for r in value["runs"] if r.get("ended_utc"))
    if any(not r.get("ended_utc") for r in value["runs"]):
        raise ValueError("unclosed GPU attempt; reconcile wall time, never reset the ledger")
    remaining = value["cap_seconds"] - used
    if remaining <= 1:
        raise ValueError("cumulative GPU cap exhausted")
    return remaining


def supervised(args, c, stage):
    """Exclusive shared ledger; GPU import/load/warmup all count. Fail closed on crash."""
    out = args.output; ledger_path = out/"gpu_budget_ledger.json"
    with (out/"gpu_budget.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {"cap_seconds": c["GPU_cap_seconds"], "runs": []}
        if ledger["cap_seconds"] != c["GPU_cap_seconds"]:
            raise ValueError("budget cap changed")
        remaining = ledger_remaining(ledger)
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,utilization.gpu", "--format=csv"], capture_output=True, text=True, check=True).stdout
        print(gpu, flush=True)
        started = time.monotonic()
        lease = canonical_sha({"pid": os.getpid(), "started": time.monotonic_ns(), "stage": stage})
        run = {"stage": stage, "started_utc": datetime.now(timezone.utc).isoformat(), "ended_utc": None, "lease": lease,
               "GPU_inventory": gpu, "seconds": None, "config_sha256": canonical_sha(c)}
        ledger["runs"].append(run); atomic_json(ledger_path, ledger)
        cmd = [sys.executable, "-u", str(Path(__file__).resolve()), stage, "--config", str(args.config), "--output", str(out),
               "--asset-root", str(args.asset_root), "--device", args.device]
        if args.resume:
            cmd.append("--resume")
        process = None
        try:
            process = subprocess.Popen(cmd, start_new_session=True, env={**os.environ, "AIC_SPATIAL_BUDGET_LEASE": lease})
            try:
                code = process.wait(timeout=max(1, remaining-6))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL); process.wait()
                raise TimeoutError("shared GPU budget stopped stage; partial caches/checkpoints retained")
            if code:
                raise RuntimeError(f"GPU stage {stage} exited {code}; preserve logs and resume after diagnosis")
        finally:
            if process is not None and process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL); process.wait()
            run["ended_utc"] = datetime.now(timezone.utc).isoformat(); run["seconds"] = time.monotonic()-started
            run["exit_code"] = process.returncode if process else None
            atomic_json(ledger_path, ledger)


def status(out, c):
    audit = json.loads((out/"data_audit.json").read_text()) if (out/"data_audit.json").exists() else {}
    results = json.loads((out/"readout_results.json").read_text()) if (out/"readout_results.json").exists() else {}
    gates = {}
    for name, result in results.items():
        source_pools=defaultdict(set);native_routes={}
        for key,metric in result["P2_video_deployment"].items():
            if metric["split"]!="confirm":continue
            dataset,branch,_=key.split("/");source_pools[dataset].update((dataset,s) for s in metric["source_ids"])
            if metric.get("native_domain"):
                ci=metric["policy_minus_P2"]["ci95"]
                native_routes[branch]=bool(ci and ci[0]>=0)
        pools=list(source_pools.values())
        gates[name] = promotion_gate(result["P2_video_deployment"], result["complete_inference_elements"], {
            "source_pools_disjoint_verified":len(pools)>=2 and all(not a.intersection(b) for i,a in enumerate(pools) for b in pools[i+1:]),
            "native_A_and_B_noninferiority_passed":native_routes.get("A",False) and native_routes.get("B",False),
            "exact_temporal_identity": True, "original_qp_replayed": bool(result["P2_video_deployment"]),
            "integer_geometry_valid": True, "inputs_and_code_locked": True})
    value = {"P2_baseline_owner_reported": 44.65, "P2_baseline_source": "current_review_owner_instruction",
             "C1": 42.23, "C3": 42.87, "VideoORA_training_started": False, "SFT_DPO_started": False,
             "feature_extraction_complete": (out/"features_complete.json").exists(), "readout_results": results or None,
             "promotion_gates": gates, "full_official_candidate_allowed": any(g["passed"] for g in gates.values()),
             "SACD": audit.get("SACD", {}).get("blocker"), "native_B_video_result": None,
             "RetargetVid_video_result": None, "GPU_smoke_new_feature_path": None if not (out/"features_complete.json").exists() else True,
             "missing_B3_replaced_by": "explicit new P2-shared-Qwen feature contract, never B3 reproduction"}
    atomic_json(out/"research_readiness.json", value)
    if results:
        lines=["# GPU 研究链执行结果","","只训练小型候选读出头，P2 Base/Position 与时间集合未更新。官方 F1 均未测量。","",
               "| 实验 | 监督 | 推理元素总数（P2+一个头） | 视频部署门 |","|---|---|---:|---|"]
        for name,result in results.items():
            gate=gates[name]
            lines.append(f"| {name} | {result['label_type']} | {result['complete_inference_elements']:,} | {'通过工程门，仍待官方分' if gate['passed'] else '未通过'} |")
        lines.extend(["","各数据集/画幅/源级区间见 `readout_results.json`。SACD/MOS 指标不是 IoU 或官方分。","",
            "## 未晋级原因",""])
        for name,gate in gates.items():
            lines.append("- "+name+": "+("; ".join(gate["reasons"]) or "公开工程门通过，仍需负责人决定完整视频推理并取得官方反馈"))
        lines.extend(["","没有自动制作或上传官方包，没有覆盖当前最佳 P2。"])
        (out/"gpu_results_report.md").write_text("\n".join(lines)+"\n")
    return value


def validate_lease(args):
    if not args.device.startswith("cuda"):
        return
    ledger_path = args.output/"gpu_budget_ledger.json"
    if not ledger_path.exists():
        raise ValueError("GPU stage requires the shared supervisor; invoke run")
    value = json.loads(ledger_path.read_text())
    active = [r for r in value["runs"] if r.get("ended_utc") is None]
    if len(active) != 1 or active[0].get("lease") != os.environ.get("AIC_SPATIAL_BUDGET_LEASE") or active[0]["stage"] != args.stage:
        raise ValueError("GPU stage has no valid persistent budget lease")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "audit", "run", "extract", "train", "evaluate", "status"])
    parser.add_argument("--config", type=Path, default=REPO/"configs/SPATIAL_RESEARCH_REVIEW_20261108.json")
    parser.add_argument("--output", type=Path, default=REPO/"artifacts/spatial_research_review")
    parser.add_argument("--asset-root", type=Path, default=Path("/root/autodl-tmp/videoora42_clone"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(); c = config(args.config); args.output = args.output.resolve()
    if args.stage == "prepare":
        prepare(args, c)
    elif args.stage == "audit":
        audit_media(args, c)
    elif args.stage == "run":
        verify_prepared(args.output, c); audit_media(args, c)
        for stage in ("extract", "train"):
            if stage == "extract" and (args.output/"features_complete.json").exists():
                continue
            supervised(args, c, stage)
        evaluate(args, c)
    elif args.stage == "extract":
        validate_lease(args)
        extract(args, c)
    elif args.stage == "train":
        validate_lease(args)
        train(args, c)
    elif args.stage == "evaluate":
        evaluate(args, c)
    else:
        print(json.dumps(status(args.output, c), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
