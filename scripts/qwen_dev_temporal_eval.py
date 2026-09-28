#!/usr/bin/env python3
"""Dev-set temporal evaluation for the Qwen teacher (no crop GT involved).

Uses TVSum proxy-v2 labels: per-frame importance expanded from 2s shot scores,
binary GT = labels >= 0.5 (protocol tvsum_summary_mean_norm_ge_0.5_v1). This
evaluates ONLY the temporal-selection component -- the same thing Qwen does in
the competition pipeline -- so no spatial crop is run and no AIC test video is
touched.

For each video: uniform anchor grid -> Qwen select_segments -> expand to
original frame indices -> frame-level P/R/F1 vs binary GT. Also emits
baselines computed from GT alone (all-frames, oracle-budget uniform stride)
to check whether gains come merely from output volume (doc section 5.2).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from aic.contract import load_jsonl
from aic.qwen_teacher import QwenTeacher, build_anchors, extract_anchor_frames, \
    segments_to_frames
from aic.video import frame_timeline


def load_gt(npz_path):
    d = np.load(npz_path, allow_pickle=True)
    labels = d["labels"].astype(np.float64)
    frames = d["frame_indices"].astype(np.int64)
    gt = set(frames[labels >= 0.5].tolist())
    return gt, dict(zip(frames.tolist(), labels.tolist()))


def rank_metrics(selected: set[int], labels_map: dict, n_frames: int):
    """Temporal-selection quality without crop GT (route doc 5.2).

    lift = mean GT importance over selected frames / mean over all frames.
    Coverage = fraction of GT-positive frames that fall inside a selected
    anchor-cell region. These measure whether the teacher *picked the right
    moments*, independent of frame-expansion density.
    """
    if not labels_map:
        return {}
    all_mean = sum(labels_map.values()) / len(labels_map)
    sel_scores = [labels_map[f] for f in selected if f in labels_map]
    sel_mean = (sum(sel_scores) / len(sel_scores)) if sel_scores else 0.0
    gt = {f for f, s in labels_map.items() if s >= 0.5}
    covered = len(gt & selected) / len(gt) if gt else 0.0
    return {"importance_lift": sel_mean / all_mean if all_mean else 0.0,
            "gt_coverage": covered, "sel_importance": sel_mean}


def prf1(selected: set[int], gt: set[int]):
    tp = len(selected & gt)
    p = tp / len(selected) if selected else 0.0
    r = tp / len(gt) if gt else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", required=True,
                    help="tvsum_manifest_proxy_v2.jsonl (needs path + annotation_path)")
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True, help="JSONL diagnostics")
    ap.add_argument("--sample-step", type=float, default=2.0,
                    help="anchor grid step in seconds (1/sample_fps)")
    ap.add_argument("--image-size", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", nargs="*", default=None,
                    help="restrict to these video_ids")
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--block-size", type=int, default=32)
    ap.add_argument("--cache-dir", default=None,
                    help="reuse per-video selection JSON for repeat metric runs")
    args = ap.parse_args(argv)

    records = load_jsonl(args.manifest)
    if args.only:
        keep = set(args.only)
        records = [r for r in records if r["video_id"] in keep]
    if args.limit:
        records = records[: args.limit]

    teacher = QwenTeacher(model_path=args.model, backend="hf",
                          device_map="auto", max_new_tokens=args.max_new_tokens,
                          image_size=args.image_size,
                          attn_implementation=args.attn,
                          block_size=args.block_size)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    out_f = open(args.output, "a", encoding="utf-8")
    for rec in records:
        vid = rec["video_id"]
        path = rec["path"]
        cache_file = cache_dir / f"{vid}.json" if cache_dir else None
        t0 = time.time()
        if cache_file and cache_file.is_file():
            cached = json.loads(cache_file.read_text())
            selected = set(cached["selected_frames"])
            status, n_anchors, n_seg, raw = (cached["status"],
                                           cached["n_anchors"],
                                           cached["n_segments"],
                                           cached.get("raw_reply", ""))
        else:
            from aic.qwen_teacher import anchors_and_frames
            anchors, frames, timeline = anchors_and_frames(
                path, 1.0 / args.sample_step, size=args.image_size)
            segs, status, raw = teacher.select_segments(
                frames, anchors, [16.0, 9.0],
                1.0 / args.sample_step,
                duration=timeline[-1].time_seconds)
            selected = set(segments_to_frames(segs, anchors, timeline))
            n_anchors, n_seg = len(anchors), len(segs)
            if cache_file:
                cache_file.write_text(json.dumps({
                    "video_id": vid, "status": status, "n_anchors": n_anchors,
                    "n_segments": n_seg, "selected_frames": sorted(selected),
                    "raw_reply": raw}))
        gt, labels_map = load_gt(rec["annotation_path"])
        n_frames = rec["frame_count"]
        p, r, f = prf1(selected, gt)
        # baselines from GT alone
        p_all, r_all, f_all = prf1(set(range(n_frames)), gt)
        # oracle-budget uniform stride: same count, evenly spaced
        k = min(len(selected), n_frames)
        stride_sel = set(int(round(i * (n_frames - 1) / max(k - 1, 1)))
                         for i in range(k)) if k else set()
        p_u, r_u, f_u = prf1(stride_sel, gt)
        row = rank_metrics(selected, labels_map, n_frames)
        row.update({
            "video_id": vid, "status": status, "n_frames": n_frames,
            "n_anchors": n_anchors, "n_segments": n_seg,
            "selected": len(selected), "select_rate": len(selected) / n_frames,
            "gt_frames": len(gt), "gt_rate": len(gt) / n_frames,
            "precision": p, "recall": r, "f1": f,
            "f1_all_frames": f_all, "f1_uniform_budget": f_u,
            "seconds": round(time.time() - t0, 2),
        })
        out_f.write(json.dumps(row) + "\n")
        out_f.flush()
        print(json.dumps(row), flush=True)
    out_f.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
