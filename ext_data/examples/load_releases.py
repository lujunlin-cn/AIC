#!/usr/bin/env python3
"""Minimal loading examples for the frozen releases (real batches, CPU only).

  source /home/supie/AIC/ext_data/scripts/env.sh
  cd /home/supie/AIC/ext_data && $AIC_EXT_PY examples/load_releases.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.adapters.base import collate_pad  # noqa: E402
from aicext.release import (Release, SpatialCropUnits, TemporalEvidenceUnits, preference_pairs,  # noqa: E402
                            score_unit, score_unit_frame)
from aicext import window_scorer as ws  # noqa: E402


def spatial():
    rel = Release("spatial_crop_v1")                       # verifies RELEASE.json + manifests against SHA256SUMS
    train = SpatialCropUnits(rel, "train", frames=2.0)     # RetargetVid: frames sampled at 2 fps on real PTS
    dl = DataLoader(train, batch_size=4, shuffle=True, num_workers=2,
                    collate_fn=lambda b: collate_pad(b, ("boxes_xywh", "valid", "frame_idx", "time_s")))
    batch = next(iter(dl))
    print("spatial batch", batch["dataset"], batch["boxes_xywh"].shape, batch["valid"].shape, batch["gt_is_max_window"])
    # window-scorer target for one annotated frame: per-annotator IoU, mean and valid mask
    item = train[0]
    s = score_unit_frame(item, 0, n=65, iou="halfopen")
    best = int(np.nanargmax(s["iou_mean"]))
    print("scorer", item["unit_id"], s["iou"].shape, "best cand", best, float(s["iou_mean"][best]), s["conventions"])
    # a whole unit and an explicit candidate set (e.g. a model's proposals)
    su = score_unit(item, n=65)
    print("unit targets", su["iou_mean"].shape, su["axis"])
    W, H, r = item["width"], item["height"], item["target_ratio_wh"]
    own = ws.legal_candidates(W, H, r, n=9)["boxes_xywh"]
    print("own candidates", ws.score_candidates(W, H, r, own, item["boxes_xywh"][:, 0], item["valid"][:, 0],
                                                gt_source="human", gt_is_max_window=item["gt_is_max_window"])["iou_mean"])
    live = SpatialCropUnits(rel, "train", datasets=["LIVE_YT_VC"], include_derived=True)
    it = live[0]
    s = score_unit_frame(it, 0)
    print("LIVE", it["boxes_xywh"].shape, "gt_is_max_window", it["gt_is_max_window"],
          "ceiling", float(s["iou_ceiling"][0]), "derived", it.get("derived_boxes_xywh", np.zeros(0)).shape)


def temporal():
    rel = Release("temporal_evidence_v1")
    train = TemporalEvidenceUnits(rel, "train", fps=2.0, require_mturk=True)
    dl = DataLoader(train, batch_size=4, shuffle=True, num_workers=2,
                    collate_fn=lambda b: collate_pad(b, ("votes_max", "human_code", "human_valid", "auto_code",
                                                         "auto_valid", "frame_idx", "time_s")))
    batch = next(iter(dl))
    print("temporal batch", batch["votes_max"].shape, batch["human_valid"].sum(1), batch["auto_code"].shape)
    it = train[0]
    pairs = preference_pairs(it, margin=1.0)               # within-video relative preferences from raw votes
    print("pairs", pairs.shape, "codes", np.unique(it["human_code"], return_counts=True))


if __name__ == "__main__":
    torch.manual_seed(0)
    spatial()
    temporal()
