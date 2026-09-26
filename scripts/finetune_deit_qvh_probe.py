#!/usr/bin/env python3
"""Bounded raw-video DeiT-S finetuning probe on QVHighlights weak labels.

This is intentionally a small diagnostic, not a competition checkpoint
trainer.  It compares a frozen DeiT representation with unfreezing the final
transformer block while keeping the same Temporal U-Net and sampled frames.
The input manifest must be training-only and carry frame-indexed NPZ labels.
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

from aic.models import TemporalUNet
from aic.video import iter_sampled_frames


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def load_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def load_video(row: dict, sample_fps: float, device: torch.device):
    sampled = list(iter_sampled_frames(row["video_path"], sample_fps=sample_fps, size=224))
    if not sampled:
        raise ValueError(f"no frames: {row['video_id']}")
    with np.load(row["labels_path"], allow_pickle=False) as data:
        source_indices = np.asarray(data["frame_indices"], dtype=np.int64)
        source_labels = np.asarray(data["labels"], dtype=np.float32)
    lookup = {int(i): float(v) for i, v in zip(source_indices, source_labels)}
    images = np.stack([item[2] for item in sampled]).astype(np.float32) / 255.0
    x = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    x = ((x - mean) / std).to(device)
    y = torch.tensor([lookup.get(int(item[0]), 0.0) for item in sampled], dtype=torch.float32, device=device)
    return x, y


def encode(model, images: torch.Tensor, chunk: int, grad: bool) -> torch.Tensor:
    out = []
    context = torch.enable_grad() if grad else torch.no_grad()
    with context:
        for start in range(0, len(images), chunk):
            out.append(model.backbone(images[start:start + chunk]))
    return torch.cat(out, dim=0)


def video_step(model, row, device, sample_fps, image_chunk, train, optimizer=None, amp=False):
    images, labels = load_video(row, sample_fps, device)
    with torch.cuda.amp.autocast(enabled=amp):
        features = encode(model, images, image_chunk, grad=train and any(p.requires_grad for p in model.backbone.parameters()))
        logits = model.temporal(features.unsqueeze(0), lengths=torch.tensor([len(features)], device=device))[0]
        loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
    if train:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    scores = logits.detach().float().sigmoid().cpu().numpy()
    truth = labels.detach().cpu().numpy() >= 0.5
    pred = scores >= 0.5
    tp = int(np.count_nonzero(pred & truth)); fp = int(np.count_nonzero(pred & ~truth)); fn = int(np.count_nonzero(~pred & truth))
    denom = 2 * tp + fp + fn
    f1 = float(2 * tp / denom) if denom else 1.0
    return float(loss.detach().cpu()), f1, float(np.mean(scores)), int(len(scores))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--unfreeze", choices=["none", "last1"], default="last1")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--image-chunk", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260926)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    import timm
    bundle = torch.load(args.weights, map_location="cpu", weights_only=True)
    model = nn.Module()
    model.backbone = timm.create_model("deit_small_patch16_224", pretrained=False, num_classes=0)
    model.temporal = TemporalUNet(384)
    model.backbone.load_state_dict(bundle, strict=True)
    model.backbone.to(device); model.temporal.to(device)
    for p in model.backbone.parameters(): p.requires_grad = False
    if args.unfreeze == "last1":
        for p in model.backbone.blocks[-1].parameters(): p.requires_grad = True
        for p in model.backbone.norm.parameters(): p.requires_grad = True
    for p in model.temporal.parameters(): p.requires_grad = True
    groups = [{"params": model.temporal.parameters(), "lr": 1e-3},
              {"params": [p for p in model.backbone.parameters() if p.requires_grad], "lr": 1e-5}]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    train_rows, val_rows = load_rows(args.train), load_rows(args.val)
    history = []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        train_rows_out = []
        for row in train_rows:
            train_rows_out.append(video_step(model, row, device, args.sample_fps, args.image_chunk, True, optimizer))
        model.eval()
        with torch.no_grad():
            val_rows_out = [video_step(model, row, device, args.sample_fps, args.image_chunk, False) for row in val_rows]
        history.append({"epoch": epoch, "train_loss": float(np.mean([x[0] for x in train_rows_out])),
                        "train_f1": float(np.mean([x[1] for x in train_rows_out])),
                        "val_loss": float(np.mean([x[0] for x in val_rows_out])),
                        "val_f1": float(np.mean([x[1] for x in val_rows_out])),
                        "val_score_mean": float(np.mean([x[2] for x in val_rows_out])),
                        "val_frames": int(sum(x[3] for x in val_rows_out))})
        print(json.dumps(history[-1]), flush=True)
    out = {"run_id": "QVH_FINETUNE_PROBE_20260926", "unfreeze": args.unfreeze,
           "train_count": len(train_rows), "val_count": len(val_rows),
           "sample_fps": args.sample_fps, "epochs": args.epochs, "seed": args.seed,
           # Labels come from the importer’s frame-aligned interpolation of
           # teacher_signals.timeline; keep this exact protocol in every
           # result so it cannot be confused with segment-only targets.
           "label_protocol": "qvh_seed_timeline_linear_v1", "history": history,
           "elapsed_seconds": time.time() - start, "official_f_video": None,
           "competition_score": None}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
