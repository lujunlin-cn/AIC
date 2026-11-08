"""Runnable source-balanced Scale training with resumable adapter checkpoints.

The backend owns model-specific image processing and causal completion scores.
No weights, CUDA state, or remote dependencies are loaded at module import.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.utils.data import Dataset, DataLoader

from aic.scale_geometry import box_iou_xywh, scale_crop_xywh_exact
from aic.scale_objectives import dpo_loss, paired_source_bootstrap_ci
from aic.scale_preferences import fixed_center_sft_target, validate_preference_pair
from aic.scale_trainer import set_trainable_parameter_allowlist


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(",", ":")).encode()).hexdigest()


def row_key(row: dict) -> tuple:
    return (row["dataset_source"], row["source_id"], row["image_id"],
            tuple(row["target_ratio"]))


def validate_manifests(train: list[dict], val: list[dict]) -> None:
    if not train or not val:
        raise ValueError("both training and source-held-out validation rows are required")
    source_splits = {}
    image_splits = {}
    for rows, split in ((train, "train"), (val, "val")):
        for row in rows:
            if row.get("split") != split or row.get("dataset_source") not in {
                "LIVE_YT_VC", "GNMC", "GAICD", "SACD"
            }:
                raise ValueError("manifest must contain public train/val rows with explicit source")
            key = (row["dataset_source"], row["source_id"])
            if key in source_splits and source_splits[key] != split:
                raise ValueError(f"source crosses train/validation: {key}")
            source_splits[key] = split
            if not all(isinstance(row.get(key), int) and row[key] > 0 for key in ("image_width", "image_height")):
                raise ValueError("manifest image dimensions must be positive integer pixels")
            image_hash = row.get("image_sha256")
            if not isinstance(image_hash, str) or len(image_hash) != 64:
                raise ValueError("every row requires an input image SHA256")
            if image_hash in image_splits and image_splits[image_hash] != split:
                raise ValueError("identical image bytes cross train/validation")
            image_splits[image_hash] = split
            if row.get("position_source") != "P1_frozen" or not row.get("input_image"):
                raise ValueError("manifest must bind an image to a frozen P1 center")
            if not all(0 <= row["center_xy"][i] <= row[("image_width", "image_height")[i]] for i in range(2)):
                raise ValueError("P1 center outside image")



def source_balanced_indices(rows: list[dict], weights: dict[str, float],
                            count: int, seed: int) -> list[int]:
    by_dataset = defaultdict(lambda: defaultdict(list))
    for index, row in enumerate(rows):
        by_dataset[row["dataset_source"]][row["source_id"]].append(index)
    if set(weights) != set(by_dataset) or any(
        not isinstance(v, (float, int)) or not math.isfinite(v) or v <= 0
        for v in weights.values()
    ):
        raise ValueError("positive, preregistered weights must cover exactly the training datasets")
    rng = random.Random(seed)
    datasets = sorted(by_dataset)
    sources = {dataset: sorted(by_dataset[dataset]) for dataset in datasets}
    result = []
    for _ in range(count):
        dataset = rng.choices(datasets, weights=[weights[d] for d in datasets], k=1)[0]
        source = rng.choice(sources[dataset])
        result.append(rng.choice(by_dataset[dataset][source]))
    return result


def prepare_sft_rows(rows: list[dict], bins: list[float], threshold: float) -> tuple[list, list]:
    accepted, excluded = [], []
    for row in rows:
        target = fixed_center_sft_target(row, bins, threshold)
        if target["label_status"] == "ok":
            accepted.append(dict(row, scale_id=target["scale_id"]))
        else:
            excluded.append(dict(target, reason=target["label_status"]))
    if not accepted:
        raise ValueError("no train rows passed the fixed-position SFT target contract")
    return accepted, excluded


def preference_accuracy_report(observations: list[dict]) -> dict:
    """Source-balanced online diagnostics, explicitly not validation evidence."""
    grouped = defaultdict(lambda: defaultdict(list))
    for row in observations:
        grouped[(row["dataset_source"], row["score_name"])][row["source_id"]].append(row["correct"])
    metrics = {}
    for (dataset, score_name), sources in sorted(grouped.items()):
        metrics[f"{dataset}:{score_name}"] = {
            "source_macro_preference_accuracy": sum(sum(v) / len(v) for v in sources.values()) / len(sources),
            "source_count": len(sources),
            "sampled_pair_evaluations": sum(len(v) for v in sources.values()),
            "quality_score_name": score_name,
        }
    return {"split": "train", "scope": "optimization_batches_since_previous_evaluation",
            "used_for_model_selection": False, "metrics_by_dataset_and_supervision": metrics}


def evaluate_geometry(backend, rows: list[dict], bins: list[float], seed: int,
                      majority_upper_bound: float | None = None, reference_predictions=None) -> dict:
    """Report each geometry dataset/ratio separately; never blend MOS with IoU."""
    reference = {row_key(r): r for r in reference_predictions or []}
    per_source = defaultdict(list)
    predictions = []
    groups = defaultdict(list)
    backend.model.eval()
    with torch.no_grad():
        for row in rows:
            if row["split"] != "val" or row["dataset_source"] not in {"LIVE_YT_VC", "GNMC"}:
                raise ValueError("deployment-IoU evaluation requires public LIVE/GNMC validation rows")
            action = backend.predict([row])[0]
            if not isinstance(action, int) or not 0 <= action < len(bins):
                raise ValueError("invalid greedy scale action; do not hide parser failure as a fallback")
            args = (row["image_width"], row["image_height"], tuple(row["target_ratio"]),
                    tuple(row["center_xy"]))
            crop = scale_crop_xywh_exact(*args, bins[action])
            baseline = scale_crop_xywh_exact(*args, 1.0)
            iou = box_iou_xywh(crop, row["gt_crop_xywh"])
            p1_iou = box_iou_xywh(baseline, row["gt_crop_xywh"])
            base_iou = reference[row_key(row)]["iou"] if reference_predictions is not None else p1_iou
            drift = max(abs(crop[i] + crop[i + 2] / 2 - row["center_xy"][i])
                        for i in range(2))
            record = dict(source_id=row["source_id"], dataset_source=row["dataset_source"],
                          image_id=row["image_id"], scale_id=action, iou=iou,
                          baseline_iou=base_iou, p1_scale1_iou=p1_iou, position_drift_px=drift,
                          crop_xywh=list(crop), target_ratio=row["target_ratio"])
            predictions.append(record)
            key = (row["dataset_source"], row["source_id"], tuple(row["target_ratio"]))
            per_source[key].append(record)
            groups[(row["dataset_source"], tuple(row["target_ratio"]))].append(key)
    metrics = {}
    for (dataset, ratio), keys in sorted(groups.items()):
        keys = sorted(set(keys))
        differences = {str(key): sum(r["iou"] - r["baseline_iou"] for r in per_source[key])
                       / len(per_source[key]) for key in keys}
        ci = paired_source_bootstrap_ci(differences, seed=seed, replicates=1000)
        distribution = defaultdict(float)
        for key in keys:
            for record in per_source[key]:
                distribution[record["scale_id"]] += 1 / len(per_source[key]) / len(keys)
        collapse = max(distribution.values())
        drift = max(r["position_drift_px"] for key in keys for r in per_source[key])
        iou = sum(sum(r["iou"] for r in per_source[key]) / len(per_source[key]) for key in keys) / len(keys)
        metrics[f"{dataset}:{ratio[0]}:{ratio[1]}"] = {
            "source_macro_deploy_iou_proxy": iou, "bootstrap": ci,
            "source_count": len(keys), "scale_distribution_source_macro": dict(distribution),
            "comparison": "frozen_SFT" if reference_predictions is not None else "P1_scale1",
            "valid_action_rate": 1.0, "scale_bin_coverage": len(distribution),
            "positive_source_fraction": ci["positive_source_fraction"],
            "negative_source_fraction": ci["negative_source_fraction"],
            "scale_collapse_rate": collapse, "max_position_drift_px": drift,
            "gate_passed": (ci["ci95"][0] > 0 if reference_predictions is not None else ci["mean_difference"] >= 0) and drift <= 1
            and majority_upper_bound is not None and collapse <= majority_upper_bound,
        }
    return {"metrics_by_dataset_ratio": metrics, "predictions": predictions,
            "gate_passed": bool(metrics) and all(m["gate_passed"] for m in metrics.values()),
            "official_metric": False}


def save_checkpoint(path: Path, backend, optimizer, step: int, identity: str,
                    best_metric: float | None) -> str:
    path.mkdir(parents=True, exist_ok=False)
    trainable = {name: p.detach().cpu() for name, p in backend.model.named_parameters()
                 if p.requires_grad}
    payload = {"parameters": trainable, "optimizer": optimizer.state_dict(), "step": step,
               "identity": identity, "best_metric": best_metric,
               "torch_rng_state": torch.get_rng_state()}
    if torch.cuda.is_initialized():
        payload["cuda_rng_states"] = torch.cuda.get_rng_state_all()
    torch.save(payload, path / "trainer.pt")
    backend.save_adapter(path / "adapter")
    digest = hashlib.sha256((path / "trainer.pt").read_bytes()).hexdigest()
    (path / "manifest.json").write_text(json.dumps({
        "step": step, "identity": identity, "trainer_sha256": digest,
        "trainable_parameters": list(trainable), "best_metric": best_metric,
        "model_contract": getattr(backend, "model_contract", None),
    }, indent=2) + "\n")
    return digest


def resume_checkpoint(path: Path, backend, optimizer, identity: str) -> dict:
    manifest = json.loads((path / "manifest.json").read_text())
    payload_path = path / "trainer.pt"
    if hashlib.sha256(payload_path.read_bytes()).hexdigest() != manifest["trainer_sha256"]:
        raise ValueError("checkpoint payload hash mismatch")
    payload = torch.load(payload_path, map_location="cpu", weights_only=True)
    if payload["identity"] != identity:
        raise ValueError("resume config/data/bin identity mismatch")
    named = dict(backend.model.named_parameters())
    expected = {name for name, p in named.items() if p.requires_grad}
    if expected != set(payload["parameters"]):
        raise ValueError("checkpoint trainable-parameter allowlist mismatch")
    with torch.no_grad():
        for name, tensor in payload["parameters"].items():
            named[name].copy_(tensor.to(named[name].device))
    optimizer.load_state_dict(payload["optimizer"])
    torch.set_rng_state(payload["torch_rng_state"])
    if torch.cuda.is_initialized() and "cuda_rng_states" in payload:
        torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    return payload


def run_training(backend, train: list[dict], val: list[dict], config: dict,
                 output: Path, stage: str, resume: Path | None = None) -> dict:
    if stage not in {"scale_sft", "spatial_dpo"}:
        raise ValueError("unknown training stage")
    validate_manifests(train, val)
    settings = config["training"]
    bins = config["data"]["scale_bins"]["values"]
    for field in ("max_steps", "batch_size", "optimizer_learning_rate", "eval_every", "seed"):
        if settings.get(field) is None:
            raise ValueError(f"preregistered training setting is missing: {field}")
    steps, batch_size = settings["max_steps"], settings["batch_size"]
    if steps <= 0 or batch_size <= 0 or settings["eval_every"] <= 0:
        raise ValueError("training steps, batch size and eval interval must be positive")
    output.mkdir(parents=True, exist_ok=True)
    original_train = train
    if stage == "scale_sft":
        train, excluded = prepare_sft_rows(train, bins, config["data"]["minimum_position_conditioned_iou"])
        (output / "position_mismatch.json").write_text(json.dumps(excluded, indent=2) + "\n")
    else:
        for pair in train:
            validate_preference_pair(pair)
            if hasattr(backend, "preprocessing_contract"):
                from aic.scale_preferences import validate_bound_preference
                from aic.scale_videoora import build_prompt
                validate_bound_preference(pair, build_prompt(pair, bins), backend.preprocessing_contract)
                threshold = config["data"].get("gaicd_minimum_mos_delta" if pair["score_name"] == "MOS" else "geometry_preference_delta_iou")
                if threshold is None or pair["score_delta"] < threshold:
                    raise ValueError("DPO quality delta is below the locked train threshold")
    trainable = set_trainable_parameter_allowlist(backend.model, settings["trainable_parameter_allowlist"])
    if settings.get("scheduler", "constant") != "constant" or settings.get("optimizer", "AdamW") != "AdamW":
        raise ValueError("this executable recipe supports constant-schedule AdamW only")
    optimizer = torch.optim.AdamW([p for p in backend.model.parameters() if p.requires_grad],
                                 lr=settings["optimizer_learning_rate"],
                                 weight_decay=settings.get("weight_decay", 0.01))
    identity = fingerprint({"config": config, "train": train, "val": val, "stage": stage})
    batch_size *= settings.get("gradient_accumulation", 1)
    weights_key = "preference_source_balanced_weights" if stage == "spatial_dpo" else "source_balanced_weights"
    schedule = source_balanced_indices(train, config["data"].get(weights_key, config["data"]["source_balanced_weights"]),
                                       steps * batch_size, settings["seed"])
    start_step, best_metric = 0, None
    if resume:
        state = resume_checkpoint(resume, backend, optimizer, identity)
        start_step, best_metric = state["step"], state["best_metric"]
    reference = None
    if stage == "spatial_dpo":
        reference_path = output / "frozen_reference_logps.pt"
        if reference_path.exists():
            integrity = json.loads(reference_path.with_suffix(".json").read_text())
            if hashlib.sha256(reference_path.read_bytes()).hexdigest() != integrity["sha256"]:
                raise ValueError("frozen reference cache hash mismatch")
            stored = torch.load(reference_path, weights_only=True, map_location="cpu")
            if stored["identity"] != identity:
                raise ValueError("frozen reference cache identity mismatch")
            if stored.get("sft_checkpoint_sha256") != settings["sft_checkpoint_sha256"]:
                raise ValueError("reference cache is not bound to the initialized SFT checkpoint")
            reference = stored["values"]
        else:
            if resume:
                raise ValueError("cannot reconstruct SFT reference from an already-updated DPO policy")
            backend.model.eval()
            backend.model.requires_grad_(False)
            backend.reference_frozen = True
            values = []
            with torch.no_grad():
                for row in train:
                    chosen, _ = backend.completion_logps([row], [row["chosen"]["scale_id"]])
                    rejected, _ = backend.completion_logps([row], [row["rejected"]["scale_id"]])
                    values.append([float(chosen[0]), float(rejected[0])])
            backend.reference_frozen = False
            set_trainable_parameter_allowlist(backend.model, settings["trainable_parameter_allowlist"])
            reference = torch.tensor(values)
            torch.save({"identity": identity, "values": reference,
                        "sft_checkpoint_sha256": config["training"]["sft_checkpoint_sha256"]}, reference_path)
            reference_path.with_suffix(".json").write_text(json.dumps({"sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(), "identity": identity})+"\n")
    reference_predictions = None
    if stage == "spatial_dpo":
        reference_evaluation = output / "frozen_sft_validation.json"
        if reference_evaluation.exists():
            cached = json.loads(reference_evaluation.read_text())
            if cached["identity"] != identity:
                raise ValueError("frozen SFT evaluation identity mismatch")
            reference_predictions = cached["predictions"]
        else:
            if resume:
                raise ValueError("cannot rebuild SFT validation from an updated DPO checkpoint")
            reference_predictions = evaluate_geometry(backend, val, bins, settings["seed"])["predictions"]
            reference_evaluation.write_text(json.dumps({"identity": identity, "predictions": reference_predictions}) + "\n")
    majority_bound = settings.get("majority_scale_upper_bound")
    history, final_checkpoint = [], None
    preference_observations = []
    stopped = None
    accumulation = settings.get("gradient_accumulation", 1)
    if not isinstance(accumulation, int) or accumulation < 1:
        raise ValueError("gradient_accumulation must be positive")
    # The batch sampler is persisted implicitly by the seeded source schedule.
    batches = [schedule[i:i+batch_size] for i in range(0, len(schedule), batch_size)]
    loader = DataLoader(ScaleRowsDataset(train), batch_sampler=batches[start_step:],
                        collate_fn=list, num_workers=0)
    for step, indexed_rows in enumerate(loader, start_step + 1):
        ids, rows = [item[0] for item in indexed_rows], [item[1] for item in indexed_rows]
        backend.train_mode()
        optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        logs = {}
        micro_size = settings["batch_size"]
        for offset in range(0, len(rows), micro_size):
            micro = rows[offset:offset + micro_size]
            micro_ids = ids[offset:offset + micro_size]
            context = backend.autocast_context() if hasattr(backend, "autocast_context") else nullcontext()
            with context:
                if stage == "scale_sft":
                    if hasattr(backend, "completion_loss"):
                        loss = backend.completion_loss(micro, [r["scale_id"] for r in micro])
                    else:
                        scores, counts = backend.completion_logps(micro, [r["scale_id"] for r in micro])
                        loss = -scores.sum() / counts.sum()
                else:
                    from aic.scale_preferences import validate_bound_preference
                    from aic.scale_videoora import build_prompt
                    if hasattr(backend, "preprocessing_contract"):
                        for row in micro:
                            validate_bound_preference(row, build_prompt(row, bins), backend.preprocessing_contract)
                    chosen, _ = backend.completion_logps(micro, [r["chosen"]["scale_id"] for r in micro])
                    rejected, _ = backend.completion_logps(micro, [r["rejected"]["scale_id"] for r in micro])
                    ref = reference[micro_ids].to(chosen.device)
                    loss = dpo_loss(chosen, rejected, ref[:, 0], ref[:, 1], settings["dpo_beta"])
                    for row, correct in zip(micro, (chosen.detach() > rejected.detach()).tolist()):
                        preference_observations.append({"dataset_source": row["dataset_source"],
                            "source_id": row["source_id"], "score_name": row["score_name"], "correct": correct})
                    logs.update(chosen_logp=float(chosen.detach().mean()),
                                rejected_logp=float(rejected.detach().mean()),
                                reference_log_ratio_margin=float(((chosen-rejected).detach()-(ref[:,0]-ref[:,1])).mean()))
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite training loss")
            weight = len(micro) / len(rows)
            (loss * weight).backward()
            total_loss += float(loss.detach()) * weight
        logs["loss"] = total_loss
        grad = torch.nn.utils.clip_grad_norm_([p for p in backend.model.parameters() if p.requires_grad],
                                             1.0, error_if_nonfinite=True)
        optimizer.step()
        logs.update(step=step, grad_norm=float(grad),
                    sampled_source_ids=[r["source_id"] for r in rows])
        with (output / "train.jsonl").open("a") as handle:
            handle.write(json.dumps(logs, allow_nan=False) + "\n")
        if step % settings["eval_every"] == 0 or step == steps:
            report = evaluate_geometry(backend, val, bins, settings["seed"], majority_bound, reference_predictions)
            if stage == "spatial_dpo":
                report["training_preference_accuracy"] = preference_accuracy_report(preference_observations)
                preference_observations.clear()
            primary = [m["source_macro_deploy_iou_proxy"] for name, m in report["metrics_by_dataset_ratio"].items()
                       if name.startswith("LIVE_YT_VC:")]
            if not primary:
                raise ValueError("LIVE validation is required for checkpoint selection")
            if len(primary) != 1:
                raise ValueError("select one preregistered native LIVE geometry for primary selection")
            metric = primary[0]
            improved = best_metric is None or metric > best_metric
            if improved:
                best_metric = metric
            final_checkpoint = output / f"checkpoint-{step:06d}"
            checkpoint_sha = save_checkpoint(final_checkpoint, backend, optimizer, step, identity, best_metric)
            report.update(checkpoint_sha256=checkpoint_sha, step=step, best_metric=best_metric,
                          reference_frozen_verified=stage == "spatial_dpo",
                          is_best=improved, trainable_inventory=trainable,
                          training_manifest_sha256=fingerprint(original_train))
            (final_checkpoint / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
            history.append(report)
            (output / "last.json").write_text(json.dumps({"checkpoint": str(final_checkpoint),
                                                          "sha256": checkpoint_sha}) + "\n")
            if improved:
                (output / "best.json").write_text(json.dumps({"checkpoint": str(final_checkpoint),
                                                              "sha256": checkpoint_sha}) + "\n")
            metrics = report["metrics_by_dataset_ratio"].values()
            if any(m["max_position_drift_px"] > 1 for m in metrics):
                stopped = "position_drift"
            elif majority_bound is not None and any(m["scale_collapse_rate"] > majority_bound for m in metrics):
                stopped = "scale_collapse"
            if stopped:
                break
    return {"stage": stage, "identity": identity, "evaluations": history,
            "stopped_reason": stopped, "last_checkpoint": str(final_checkpoint) if final_checkpoint else None}


class ScaleRowsDataset(Dataset):
    """A materialized, hashed public manifest; returns stable row indices for reference lookup."""
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return index, self.rows[index]


class ScaleSFTDataset(ScaleRowsDataset):
    def __init__(self, rows, bins, threshold):
        accepted, self.exclusions = prepare_sft_rows(rows, bins, threshold)
        super().__init__(accepted)


class PreferenceDataset(ScaleRowsDataset):
    def __init__(self, rows):
        for row in rows:
            validate_preference_pair(row)
        super().__init__(rows)


def evaluate_human_supervision(backend, rows: list[dict], bins: list[float]) -> dict:
    """Separate MOS and subject-conditioned validation, with no IoU relabeling."""
    from aic.scale_geometry import map_box_to_scale_id
    by_source = defaultdict(list)
    backend.model.eval()
    with torch.no_grad():
        for row in rows:
            if row.get("split") != "val":
                raise ValueError("human validation accepts public validation rows only")
            dataset = row["dataset_source"]
            if dataset == "GAICD":
                if row.get("release") != "journal":
                    raise ValueError("GAICD validation must use journal release")
                action = backend.predict([row])[0]
                mos = [c["mos"] for c in row["candidates"] if c.get("valid", True)
                       and isinstance(c.get("mos"), (int,float)) and 1 <= c["mos"] <= 5
                       and map_box_to_scale_id(c["crop_xywh"], row["image_width"],row["image_height"],
                           tuple(row["target_ratio"]),tuple(row["center_xy"]), bins) == action]
                if mos:
                    by_source[(dataset,row["source_id"],None)].append(sum(mos)/len(mos))
            elif dataset == "SACD":
                if not row.get("subject_id") or not row.get("ranked_pairs"):
                    raise ValueError("SACD validation needs explicit same-subject human ranked pairs")
                for pair in row["ranked_pairs"]:
                    if pair["subject_id"] != row["subject_id"] or pair["image_id"] != row["image_id"]:
                        raise ValueError("SACD subject/image mismatch")
                    ids = [map_box_to_scale_id(pair[k],row["image_width"],row["image_height"],
                           tuple(row["target_ratio"]),tuple(row["center_xy"]),bins)
                           for k in ("chosen_box_xywh","rejected_box_xywh")]
                    if None not in ids and ids[0] != ids[1]:
                        chosen,_=backend.completion_logps([row],[ids[0]])
                        rejected,_=backend.completion_logps([row],[ids[1]])
                        by_source[(dataset,row["source_id"],row["subject_id"])].append(float(chosen[0]>rejected[0]))
            else:
                raise ValueError("human supervision evaluation is GAICD/SACD only")
    result={}
    for dataset in {r["dataset_source"] for r in rows}:
        groups={k:sum(v)/len(v) for k,v in by_source.items() if k[0]==dataset}
        # First average subjects inside an image, then images independently.
        images=defaultdict(list)
        for (_,source,subject),value in groups.items():images[source].append(value)
        result[dataset]={"metric":"image_macro_MOS" if dataset=="GAICD" else "subject_conditioned_preference_accuracy_image_macro",
                         "value":sum(sum(v)/len(v) for v in images.values())/len(images) if images else None,
                         "valid_sources":len(images),"evaluated_groups":len(groups),
                         "requested_sources":len({r["source_id"] for r in rows if r["dataset_source"]==dataset}),
                         "missing_exact_action_MOS_or_pair_not_imputed":True}
    return result
