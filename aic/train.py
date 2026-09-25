"""Reproducible A0 temporal-head training from NPZ feature caches.

This entrypoint deliberately reports a *temporal proxy F1*; it is not the
competition F_video because caches do not contain spatial crop ground truth.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .features import FeatureCacheDataset, collate_feature_batch
from .models import A0Model, export_inference


# TVSum scores are stored as normalized continuous summary-importance values
# ((mean rater score - 1) / 4).  This is a project proxy protocol, not AIC GT.
# Keep the target definition fixed while prediction_threshold is tuned on the
# development split.  In particular, never use the prediction threshold to
# binarize labels.
TVSUM_LABEL_PROTOCOL = "tvsum_summary_mean_norm_ge_0.5_v1"
TVSUM_TARGET_THRESHOLD = 0.5


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def finite_or_raise(name: str, tensor: torch.Tensor) -> None:
    if not torch.isfinite(tensor).all():
        raise FloatingPointError(f"Non-finite {name}")


def tvsum_target_binary(labels: torch.Tensor) -> torch.Tensor:
    """Return the fixed TVSum binary proxy target.

    ``labels`` remain continuous for the regression/ranking diagnostics.  The
    binary threshold is deliberately a versioned constant and is independent
    of a model's prediction threshold.
    """
    return labels >= TVSUM_TARGET_THRESHOLD


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    valid = mask.bool()
    if not valid.any():
        # Keep a differentiable zero so distributed/single GPU loops are alike.
        return logits.sum() * 0.0
    loss = nn.functional.binary_cross_entropy_with_logits(logits[valid], labels[valid])
    finite_or_raise("loss", loss)
    return loss


def masked_regression(logits: torch.Tensor, labels: torch.Tensor,
                      mask: torch.Tensor, kind: str = "smooth_l1") -> torch.Tensor:
    valid = mask.bool()
    if not valid.any():
        return logits.sum() * 0.0
    scores = logits.sigmoid()[valid]
    target = labels[valid]
    if kind == "mse":
        loss = nn.functional.mse_loss(scores, target)
    elif kind in {"smooth_l1", "huber"}:
        loss = nn.functional.smooth_l1_loss(scores, target)
    else:
        raise ValueError(f"Unknown regression loss: {kind}")
    finite_or_raise("loss", loss)
    return loss


def frame_f1(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
             threshold: float = .5) -> tuple[float, int, int, int]:
    valid = mask.bool()
    if not valid.any():
        return float("nan"), 0, 0, 0
    pred = logits.sigmoid() >= threshold
    truth = tvsum_target_binary(labels)
    tp = int((pred & truth & valid).sum())
    fp = int((pred & ~truth & valid).sum())
    fn = int((~pred & truth & valid).sum())
    f1 = (2 * tp / (2 * tp + fp + fn)) if (tp + fp + fn) else 1.0
    return f1, tp, fp, fn


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Small dependency-free average-rank implementation for Spearman rho."""
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def _spearman(prediction: np.ndarray, target: np.ndarray) -> float | None:
    if len(prediction) < 2:
        return None
    p, t = _rankdata(prediction.astype(np.float64)), _rankdata(target.astype(np.float64))
    p -= p.mean(); t -= t.mean()
    denominator = float(np.sqrt(np.dot(p, p) * np.dot(t, t)))
    return float(np.dot(p, t) / denominator) if denominator > 0 else 0.0


def video_proxy_metrics(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor,
                        prediction_threshold: float = .5,
                        video_id: str | None = None) -> dict[str, Any]:
    """Compute one video's metrics under the fixed TVSum proxy protocol."""
    valid = mask.bool()
    scores = logits.detach().float().sigmoid()[valid].cpu().numpy()
    targets = labels.detach().float()[valid].cpu().numpy()
    if not len(scores):
        return {"video_id": video_id, "valid_frames": 0, "f1": None,
                "tp": 0, "fp": 0, "fn": 0,
                "precision": None, "recall": None, "prediction_rate": None,
                "target_rate": None, "empty_prediction": True,
                "score_quantiles": {}, "continuous_mae": None,
                "spearman": None}
    pred = scores >= prediction_threshold
    truth = targets >= TVSUM_TARGET_THRESHOLD
    tp = int(np.count_nonzero(pred & truth)); fp = int(np.count_nonzero(pred & ~truth)); fn = int(np.count_nonzero(~pred & truth))
    denom = 2 * tp + fp + fn
    return {"video_id": video_id, "valid_frames": int(len(scores)),
            "tp": tp, "fp": fp, "fn": fn,
            "f1": float(2 * tp / denom) if denom else 1.0,
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "recall": float(tp / (tp + fn)) if tp + fn else None,
            "prediction_rate": float(pred.mean()), "target_rate": float(truth.mean()),
            "empty_prediction": bool(not pred.any()),
            "score_quantiles": {key: float(np.quantile(scores, q)) for key, q in
                                (("q05", .05), ("q25", .25), ("q50", .5),
                                 ("q75", .75), ("q95", .95))},
            "continuous_mae": float(np.mean(np.abs(scores - targets))),
            "spearman": _spearman(scores, targets)}


def aggregate_proxy_metrics(per_video: list[dict[str, Any]], tp: int, fp: int,
                            fn: int, valid_frames: int) -> dict[str, Any]:
    """Aggregate per-video reports; micro values remain diagnostic only."""
    valid = [row for row in per_video if row.get("f1") is not None]
    macro = float(np.mean([row["f1"] for row in valid])) if valid else float("nan")
    micro_denom = 2 * tp + fp + fn
    micro = float(2 * tp / micro_denom) if micro_denom else float("nan")
    return {
        "temporal_proxy_f1": macro,
        "video_macro_f1": macro,
        "micro_f1_diagnostic": micro,
        "tp": tp, "fp": fp, "fn": fn, "valid_frames": valid_frames,
        "video_count": len(valid),
        "empty_prediction_rate": (float(np.mean([row["empty_prediction"] for row in valid]))
                                   if valid else None),
        "mean_prediction_rate": (float(np.mean([row["prediction_rate"] for row in valid]))
                                  if valid else None),
        "mean_target_rate": (float(np.mean([row["target_rate"] for row in valid]))
                              if valid else None),
        "mean_precision": (float(np.mean([row["precision"] for row in valid
                                           if row["precision"] is not None]))
                            if any(row["precision"] is not None for row in valid) else None),
        "mean_recall": (float(np.mean([row["recall"] for row in valid
                                        if row["recall"] is not None]))
                         if any(row["recall"] is not None for row in valid) else None),
        "mean_continuous_mae": (float(np.mean([row["continuous_mae"] for row in valid
                                                if row["continuous_mae"] is not None]))
                                if any(row["continuous_mae"] is not None for row in valid) else None),
        "mean_spearman": (float(np.mean([row["spearman"] for row in valid
                                          if row["spearman"] is not None]))
                          if any(row["spearman"] is not None for row in valid) else None),
        "per_video": valid,
    }


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
            config = yaml.safe_load(text)
        except ImportError as exc:
            raise RuntimeError("YAML config requires PyYAML; use JSON instead") from exc
    else:
        config = json.loads(text)
    if not isinstance(config, dict):
        raise ValueError("Config must be an object")
    defaults = {"run_id": "A0_001", "output_dir": "experiments/A0_001",
                "epochs": 20, "batch_size": 2, "num_workers": 0,
                "learning_rate": 1e-3, "weight_decay": 1e-4, "seed": 20260925,
                "patience": 5, "min_delta": 1e-4, "max_hours": 11.0,
                "device": "auto", "amp": True, "grad_clip": 1.0,
                "feature_dim": 512, "freeze_backbone": True,
                "prediction_threshold": .5, "threshold": .5}
    defaults.update(config)
    # ``threshold`` is a legacy alias and only controls prediction selection.
    # Target binarization is fixed by TVSUM_TARGET_THRESHOLD.
    if "prediction_threshold" not in config and "threshold" in config:
        defaults["prediction_threshold"] = config["threshold"]
    for required in ("train_manifest", "val_manifest"):
        if required not in defaults:
            raise ValueError(f"Config missing {required}")
    return defaults


def environment_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"hostname": socket.gethostname(), "python": os.sys.version,
                              "torch": torch.__version__, "cuda": torch.version.cuda,
                              "cuda_available": torch.cuda.is_available(),
                              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    if torch.cuda.is_available():
        result["device"] = torch.cuda.get_device_name(0)
    try:
        result["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                         text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        result["git_commit"] = None
    return result


def _move(batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(batch[key].to(device, non_blocking=True)
                 for key in ("features", "aux", "labels", "mask", "lengths"))


def run_epoch(model: A0Model, loader: DataLoader, optimizer: torch.optim.Optimizer | None,
              scaler: torch.cuda.amp.GradScaler, device: torch.device, amp: bool,
              grad_clip: float = 0.0, threshold: float = .5,
              loss_name: str = "bce") -> dict[str, Any]:
    train = optimizer is not None
    model.train(train)
    if train and model.backbone is not None:
        # Caches already contain frozen ResNet features for A0.
        model.backbone.eval()
    losses: list[float] = []; tp = fp = fn = valid_count = 0
    per_video: list[dict[str, Any]] = []
    autocast = torch.cuda.amp.autocast
    for batch in loader:
        features, aux, labels, mask, lengths = _move(batch, device)
        valid_count += int(mask.sum())
        if train:
            optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            # Pass true sequence lengths so GroupNorm/pooling never observe
            # right-padding from another video in this batch.
            logits = model(features, aux if model.feature_bank_enabled else None,
                           lengths=lengths)
            finite_or_raise("logits", logits)
            loss = (masked_bce(logits, labels, mask) if loss_name == "bce"
                    else masked_regression(logits, labels, mask, loss_name))
        if train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # Detect NaN/Inf before an optimizer update; the last checkpoint
            # remains recoverable and the failure is explicit in the log.
            for name, parameter in model.named_parameters():
                if parameter.requires_grad and parameter.grad is not None:
                    finite_or_raise(f"gradient:{name}", parameter.grad)
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer); scaler.update()
        losses.append(float(loss.detach().cpu()))
        _, btp, bfp, bfn = frame_f1(logits.detach(), labels, mask, threshold)
        tp += btp; fp += bfp; fn += bfn
        video_ids = batch.get("video_ids", [None] * logits.shape[0])
        for row_index, video_id in enumerate(video_ids):
            per_video.append(video_proxy_metrics(logits[row_index], labels[row_index],
                                                 mask[row_index], threshold,
                                                 str(video_id) if video_id is not None else None))
    if not losses:
        raise ValueError("DataLoader yielded no batches")
    metrics = aggregate_proxy_metrics(per_video, tp, fp, fn, valid_count)
    metrics["loss"] = float(np.mean(losses))
    # Compatibility key; this is now video-macro F1.  Micro F1 is explicitly
    # retained as a diagnostic and is never the selection metric.
    metrics["label_protocol"] = TVSUM_LABEL_PROTOCOL
    metrics["prediction_threshold"] = float(threshold)
    return metrics


def save_checkpoint(path: Path, model: A0Model, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler._LRScheduler, scaler: Any,
                    epoch: int, best_metric: float, bad_epochs: int,
                    elapsed_seconds: float, config: dict[str, Any], metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format_version": 1, "epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(), "best_metric": best_metric,
                "bad_epochs": bad_epochs, "elapsed_seconds": elapsed_seconds,
                "config": config, "metrics": metrics}, path)


def train(config: dict[str, Any]) -> dict[str, Any]:
    set_seed(int(config["seed"]))
    requested = str(config.get("device", "auto"))
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested)
    if requested == "auto" and not torch.cuda.is_available():
        device = torch.device("cpu")
    output = Path(config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    if config.get("_command"):
        (output / "command.txt").write_text(str(config["_command"]) + "\n")
    (output / "environment.json").write_text(json.dumps(environment_snapshot(), indent=2, default=str))
    train_data = FeatureCacheDataset(config["train_manifest"])
    val_data = FeatureCacheDataset(config["val_manifest"])
    loader_args = {"batch_size": int(config["batch_size"]), "num_workers": int(config["num_workers"]),
                   "collate_fn": collate_feature_batch, "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    model = A0Model(int(config.get("feature_dim", 512)),
                    temporal_shift_enabled=bool(config.get("temporal_shift", False)),
                    feature_bank_enabled=bool(config.get("feature_bank", False)))
    backbone_state = config.get("backbone_state")
    if backbone_state:
        state = torch.load(backbone_state, map_location="cpu", weights_only=True)
        if "state_dict" in state:
            state = state["state_dict"]
        state = {key.removeprefix("backbone."): value for key, value in state.items()}
        model.backbone.load_state_dict(state, strict=False)
    if config.get("freeze_backbone", True):
        for parameter in model.backbone.parameters():
            parameter.requires_grad_(False)
    model.to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"]))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(config["epochs"])))
    amp = bool(config.get("amp", True) and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    start_epoch = 0; best_metric = -float("inf"); bad_epochs = 0; elapsed = 0.0
    resume = config.get("resume")
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state.get("scaler", {}))
        start_epoch = int(state["epoch"]) + 1; best_metric = float(state["best_metric"])
        bad_epochs = int(state.get("bad_epochs", 0)); elapsed = float(state.get("elapsed_seconds", 0.0))
    log_path = output / "epoch_log.jsonl"
    max_seconds = float(config["max_hours"]) * 3600.0
    started = time.monotonic()
    history: list[dict[str, Any]] = []
    for epoch in range(start_epoch, int(config["epochs"])):
        if elapsed + (time.monotonic() - started) >= max_seconds:
            break
        prediction_threshold = float(config.get("prediction_threshold", config.get("threshold", .5)))
        loss_name = str(config.get("loss", "bce"))
        train_metrics = run_epoch(model, train_loader, optimizer, scaler, device, amp,
                                  float(config["grad_clip"]), prediction_threshold, loss_name)
        val_metrics = run_epoch(model, val_loader, None, scaler, device, amp,
                                0.0, prediction_threshold, loss_name)
        scheduler.step()
        epoch_elapsed = elapsed + time.monotonic() - started
        metric = val_metrics["temporal_proxy_f1"]
        score = metric if math.isfinite(metric) else -val_metrics["loss"]
        record = {"epoch": epoch, "elapsed_seconds": epoch_elapsed,
                  "lr": optimizer.param_groups[0]["lr"], "train": train_metrics,
                  "val": val_metrics, "metric_used_for_selection": score}
        history.append(record)
        with log_path.open("a") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        save_checkpoint(output / "last.pt", model, optimizer, scheduler, scaler, epoch,
                        best_metric, bad_epochs, epoch_elapsed, config, record)
        if score > best_metric + float(config["min_delta"]):
            best_metric = score; bad_epochs = 0
            save_checkpoint(output / "best.pt", model, optimizer, scheduler, scaler, epoch,
                            best_metric, bad_epochs, epoch_elapsed, config, record)
        else:
            bad_epochs += 1
        if bad_epochs >= int(config["patience"]):
            break
    # Export the best model when available; export keeps FP32/FP16 as alternatives.
    best_path = output / "best.pt"
    if best_path.exists():
        state = torch.load(best_path, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
    audit = export_inference(model, output / "export", config,
                             {"source": "A0 cached feature training", "weights": config.get("backbone_state"),
                              "warning": "If no backbone_state was supplied, backbone weights are uninitialized."})
    observed_f1 = [row["val"]["temporal_proxy_f1"] for row in history
                   if math.isfinite(row["val"]["temporal_proxy_f1"])]
    summary = {"run_id": config["run_id"], "status": "completed", "epochs_completed": len(history),
               "best_selection_metric": best_metric,
               "best_temporal_proxy_f1": max(observed_f1) if observed_f1 else None,
               "elapsed_seconds": elapsed + time.monotonic() - started,
               "official_f_video": None, "official_like_metric_note": "Temporal proxy only; no spatial GT/official evaluator.",
               "audit": audit}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="A0 ResNet18 + temporal U-Net cached-feature trainer")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config["_command"] = " ".join(sys.argv)
    print(json.dumps(train(config), indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
