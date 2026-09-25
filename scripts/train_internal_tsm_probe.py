#!/usr/bin/env python3
"""Train/evaluate a temporal head on canonical internal-TSM feature caches.

This deliberately avoids exporting an A0 inference bundle: the ResNet encoder
is represented by an intermediate-TSM cache, so a raw-video bundle would need
the matching online encoder.  The output is a controlled temporal comparison
artifact (checkpoint, per-epoch metrics and cache provenance).
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from aic.features import FeatureCacheDataset, collate_feature_batch
from aic.models import TemporalUNet
from aic.train import video_proxy_metrics


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260925)
    ap.add_argument("--threshold", type=float, default=.5)
    a = ap.parse_args(argv)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device(a.device if a.device != "auto" or torch.cuda.is_available() else "cpu")
    train_data, val_data = FeatureCacheDataset(a.train), FeatureCacheDataset(a.val)
    loader_args = {"batch_size": 1, "num_workers": 2,
                   "collate_fn": collate_feature_batch, "pin_memory": dev.type == "cuda"}
    train_loader = DataLoader(train_data, shuffle=True, **loader_args)
    val_loader = DataLoader(val_data, shuffle=False, **loader_args)
    input_dim = int(train_data[0]["features"].shape[1])
    model = TemporalUNet(input_dim).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = (-float("inf"), None)
    history = []
    started = time.time()

    def epoch(loader, training):
        model.train(training); losses = []; rows = []
        for batch in loader:
            x = batch["features"].to(dev); y = batch["labels"].to(dev)
            mask = batch["mask"].to(dev); lengths = batch["lengths"].to(dev)
            z = model(x, lengths=lengths)
            loss = nn.functional.binary_cross_entropy_with_logits(z[mask], y[mask])
            if training:
                opt.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach().cpu()))
            for i, video_id in enumerate(batch["video_ids"]):
                rows.append(video_proxy_metrics(z[i], y[i], mask[i], a.threshold, str(video_id)))
        valid = [r for r in rows if r.get("f1") is not None]
        return {"loss": float(np.mean(losses)),
                "video_macro_f1": float(np.mean([r["f1"] for r in valid])),
                "mean_spearman": float(np.mean([r["spearman"] for r in valid
                                                  if r.get("spearman") is not None])),
                "empty_prediction_rate": float(np.mean([r["empty_prediction"] for r in valid])),
                "mean_prediction_rate": float(np.mean([r["prediction_rate"] for r in valid])),
                "per_video": valid}

    for epoch_index in range(a.epochs):
        train_metrics = epoch(train_loader, True)
        val_metrics = epoch(val_loader, False)
        record = {"epoch": epoch_index, "train": train_metrics, "val": val_metrics}
        history.append(record)
        if val_metrics["video_macro_f1"] > best[0]:
            best = (val_metrics["video_macro_f1"], epoch_index)
            Path(a.output).with_suffix(".pt").parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "input_dim": input_dim,
                        "seed": a.seed, "encoder_variant": "canonical_internal_tsm_layer1_fold8_v1"},
                       Path(a.output).with_suffix(".pt"))
    out = {"run_id": "TSM_internal_probe_001", "input_dim": input_dim,
           "target_protocol": "tvsum_summary_mean_norm_ge_0.5_v1",
           "prediction_threshold": a.threshold, "best_video_macro_f1": best[0],
           "best_epoch": best[1], "elapsed_seconds": time.time() - started,
           "history": history, "submission_candidate": False,
           "reason": "temporal-head probe on intermediate feature-map TSM cache; raw encoder bundle not exported"}
    Path(a.output).parent.mkdir(parents=True, exist_ok=True)
    Path(a.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("best_video_macro_f1", "best_epoch", "elapsed_seconds")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
