#!/usr/bin/env python3
"""Evaluate zero-parameter temporal post-processing on a feature manifest.

This is a diagnostic sweep.  It never chooses a model checkpoint and it does
not estimate AIC ``F_video``: only the versioned TVSum temporal proxy is
reported.  Use a development manifest to select a policy, then run the same
config once on the immutable lockbox.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from aic.features import load_feature_cache
from aic.models import load_inference_model
from aic.postprocess import PostprocessConfig, select_temporal
from aic.train import TVSUM_LABEL_PROTOCOL, TVSUM_TARGET_THRESHOLD


def _records(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")]


def _metric(selected: np.ndarray, labels: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    truth = (labels >= TVSUM_TARGET_THRESHOLD) & valid
    pred = selected & valid
    tp = int(np.count_nonzero(pred & truth)); fp = int(np.count_nonzero(pred & ~truth))
    fn = int(np.count_nonzero(~pred & truth)); denom = 2 * tp + fp + fn
    return {"f1": float(2 * tp / denom) if denom else 1.0,
            "tp": tp, "fp": fp, "fn": fn,
            "precision": float(tp / (tp + fp)) if tp + fp else None,
            "recall": float(tp / (tp + fn)) if tp + fn else None,
            "prediction_rate": float(pred.sum() / valid.sum()) if valid.any() else None,
            "target_rate": float(truth.sum() / valid.sum()) if valid.any() else None,
            "empty_prediction": bool(not pred.any()),
            "valid_frames": int(valid.sum())}


def evaluate(model_path: str | Path, manifest: str | Path,
             configs: list[PostprocessConfig], device: str = "cpu") -> dict[str, Any]:
    model, metadata = load_inference_model(model_path, device=device)
    model.eval(); rows: list[dict[str, Any]] = []
    # Scores are generated once per video, making policy comparisons exactly
    # paired and avoiding any hidden post-processing/model interaction.
    for record in _records(manifest):
        cache = load_feature_cache(record["path"])
        features = torch.from_numpy(cache["features"]).to(device)
        aux = torch.from_numpy(cache["aux"]).to(device) if cache["aux"].shape[1] else None
        # Training/validation checkpoints were scored on each whole feature
        # sequence with explicit lengths.  Do not silently switch to the
        # overlapping-window inference helper here: that is a separate
        # policy and would make post-processing gains incomparable with the
        # repaired A0 threshold sweeps.
        with torch.inference_mode():
            lengths = torch.tensor([len(features)], dtype=torch.long, device=device)
            logits = model(features.unsqueeze(0), aux.unsqueeze(0) if aux is not None else None,
                           lengths=lengths)[0, :len(features)]
            probabilities = logits.float().sigmoid().cpu().numpy()
        rows.append({"video_id": str(record.get("video_id", Path(record["path"]).stem)),
                     "scores": probabilities, "labels": cache["labels"],
                     "valid": cache["mask"].astype(bool)})
    result: list[dict[str, Any]] = []
    for config in configs:
        per_video = []
        for row in rows:
            selected = select_temporal(row["scores"], config)
            per_video.append({"video_id": row["video_id"],
                              **_metric(selected, row["labels"], row["valid"])})
        valid = [r for r in per_video if r["f1"] is not None]
        result.append({"config": config.as_dict(),
                       "video_macro_f1": float(np.mean([r["f1"] for r in valid])),
                       "std_video_f1": float(np.std([r["f1"] for r in valid])),
                       "mean_prediction_rate": float(np.mean([r["prediction_rate"] for r in valid])),
                       "mean_target_rate": float(np.mean([r["target_rate"] for r in valid])),
                       "empty_prediction_rate": float(np.mean([r["empty_prediction"] for r in valid])),
                       "per_video": per_video})
    result.sort(key=lambda r: r["video_macro_f1"], reverse=True)
    return {"metric": "TVSum temporal proxy", "label_protocol": TVSUM_LABEL_PROTOCOL,
            "target_threshold": TVSUM_TARGET_THRESHOLD, "model_path": str(model_path),
            "model_bytes": Path(model_path).stat().st_size, "manifest": str(manifest),
            "video_count": len(rows), "results": result}


def _default_configs() -> list[PostprocessConfig]:
    configs = []
    # Coarse dev-only grid: each family changes one mechanism at a time.
    for threshold in (.30, .35, .40, .45, .50):
        configs.append(PostprocessConfig(threshold=threshold))
    for method, window in (("median", 5), ("median", 9), ("gaussian", 5), ("ema", 1)):
        kwargs = {"smoothing": method, "smoothing_window": window}
        if method == "ema": kwargs["ema_alpha"] = .35
        for threshold in (.30, .35, .40, .45):
            configs.append(PostprocessConfig(threshold=threshold, **kwargs))
    for high, low in ((.35, .25), (.40, .30), (.45, .35), (.50, .40)):
        configs.append(PostprocessConfig(threshold=high, low_threshold=low))
    for gap in (1, 2, 4):
        configs.append(PostprocessConfig(threshold=.40, gap=gap, min_duration=2))
    for norm, threshold in (("rank", .90), ("rank", .95), ("zscore", .5), ("robust", .5)):
        configs.append(PostprocessConfig(threshold=threshold, normalization=norm))
    # De-duplicate configs while retaining stable order.
    seen = set(); unique = []
    for cfg in configs:
        key = json.dumps(cfg.as_dict(), sort_keys=True)
        if key not in seen: seen.add(key); unique.append(cfg)
    return unique


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config-json", help="single frozen PostprocessConfig JSON; no sweep")
    parser.add_argument("--protocol-role", choices=("dev", "lockbox"), default="dev",
                        help="lockbox requires --config-json and never runs the dev sweep")
    args = parser.parse_args()
    if args.protocol_role == "lockbox" and not args.config_json:
        parser.error("--protocol-role lockbox requires --config-json; lockbox is not tunable")
    if args.config_json:
        config = PostprocessConfig(**json.loads(Path(args.config_json).read_text()
                                                 if Path(args.config_json).is_file()
                                                 else args.config_json))
        configs = [config]
    else:
        configs = _default_configs()
    report = evaluate(args.model, args.manifest, configs, args.device)
    report["protocol_role"] = args.protocol_role
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": args.output, "video_count": report["video_count"],
                      "best": report["results"][0]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
