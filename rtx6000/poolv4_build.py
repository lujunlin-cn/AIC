#!/usr/bin/env python3
"""Build grpo_pool_v4: whole-timeline 12-point segments (official-inference
shape, 2026-10-13 line-2 granularity alignment).

Source: the 203 PHD2 youtube videos behind sft_data_v2 (mp4s now local).
Per video: uniform n=max(4,min(12,round(dur))) timeline points (same
formula as the 426 scorer); frames cv2-seeked to
poolv4_frames/phd2_{ytid}_FULL/f{i:02d}.jpg.
target  = VPB-head score interpolated at those timeline points (RL learns
          calibration/correction of the perceptual prior; requires --vpb-inject
          and the pool 'vpb' field re-scored on the same 12 points).
gif_hard= nearest-neighbour vote from sft_data_v2 window segments (each
          window's 12 frames carry absolute timestamps + gif_hard); a point
          is positive if any window gif frame lands within +/-1.5s.
teacher/gif_soft are left null (windows-level only). split=pos/neg.
Usage: poolv4_build.py <sft_data_v2.jsonl> <frag_labels.jsonl> <vpb_scores.jsonl> <video_dir> <out.jsonl> <frames_root>
Run on CPU (AutoDL no-GPU mode is fine, 208 cores); ~minutes.
"""
import json
import os
import sys
from collections import defaultdict

import cv2

sft_p, frag_p, vpb_p, video_dir, out_p, frames_root = sys.argv[1:7]

vpb = {}
for l in open(vpb_p):
    d = json.loads(l)
    vpb[d["vid"]] = d["scores"]

# window segments grouped by youtube id: (abs_time, gif_hard)
wins = defaultdict(list)
meta = {}
for l in open(sft_p):
    r = json.loads(l)
    if not r["vid"].startswith("phd2_"):
        continue
    ytid = r["vid"].replace("phd2_", "").rsplit("_", 1)[0]
    # sft_data_v2 rows carry no seg_t0/video_id fields: parse both from vid
    meta[ytid] = ytid
    t0 = float(r["vid"].rsplit("_", 1)[1])
    wins[ytid].append((r["vid"], r, t0))

frags = {}
for l in open(frag_p):
    r = json.loads(l)
    frags[(r["video_id"], round(float(r["t0"]), 1))] = r["frames"]

out = []
for ytid, segs in sorted(wins.items()):
    mp4 = os.path.join(video_dir, f"{ytid}.mp4")
    if not os.path.exists(mp4):
        print(f"skip {ytid}: no mp4")
        continue
    cap = cv2.VideoCapture(mp4)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if total <= 0 or fps <= 0:
        cap.release()
        print(f"skip {ytid}: bad header")
        continue
    dur = total / fps
    n = max(4, min(12, round(dur)))
    idxs = [int(round(i * (total - 1) / max(n - 1, 1))) for i in range(n)]
    vid = f"phd2_{ytid}_FULL"
    fdir = os.path.join(frames_root, vid)
    os.makedirs(fdir, exist_ok=True)
    frames = []
    ok_n = 0
    for i, fi in enumerate(idxs):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fi)
        ok, fr = cap.read()
        if ok:
            h, w = fr.shape[:2]
            if max(h, w) > 1024:
                s = 1024.0 / max(h, w)
                fr = cv2.resize(fr, (int(round(w * s)), int(round(h * s))))
            cv2.imwrite(os.path.join(fdir, f"f{i:02d}.jpg"), fr)
            ok_n += 1
            frames.append(fi / fps)
    cap.release()
    if ok_n < 4:
        print(f"skip {ytid}: only {ok_n} frames")
        continue
    n = ok_n
    frames = frames[:n]
    # gif_hard: any window gif frame within +/-1.5 s of the timeline point
    gh = []
    for t in frames:
        pos = 0
        for _, r, _t0 in segs:
            fl = (frags.get((ytid, round(t0, 1)))
                  or frags.get((ytid, t0)))
            if not fl:
                continue
            for j, fmeta in enumerate(fl["frames"][:len(r["gif_hard"])]):
                if abs(float(fmeta["t"]) - t) <= 1.5 and r["gif_hard"][j]:
                    pos = 1
        gh.append(pos)
    # target: mean window VPB score nearest to each timeline point, else
    # linear ramp over the video (segment-level VPB -> whole-axis)
    seg_vp = []
    for v, _r, _t0 in segs:
        if v in vpb:
            seg_vp.append(float(sum(vpb[v]) / len(vpb[v])))
    if seg_vp:
        tgt = [round(sum(seg_vp) / len(seg_vp), 4)] * n
    else:
        tgt = [round(i / max(n - 1, 1), 4) for i in range(n)]
    out.append({"vid": vid, "files": [f"f{i:02d}.jpg" for i in range(n)],
                "frames_dir": fdir, "gif_hard": gh, "target": tgt,
                "teacher": None, "gif_soft": None,
                "split": "pos" if sum(gh) > 0 else "neg",
                "source": "phd2_full",
                "meta": {"video_id": meta[ytid], "duration": round(dur, 2),
                         "frame_indices": idxs}})

with open(out_p, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
pos = sum(1 for r in out if r["split"] == "pos")
print(f"pool_v4: {len(out)} videos, pos={pos} -> {out_p}")
