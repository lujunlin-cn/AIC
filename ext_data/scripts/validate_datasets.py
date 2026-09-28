#!/usr/bin/env python3
"""Per-dataset validation: integrity, stratified decode, adapter batches, visual samples.

For each dataset with a processed index:
  1. index integrity: required fields, file existence/size for sampled media;
  2. stratified sample (by official split / variant) -> decode the exact frames
     the adapter returns and compare shapes with the probed metadata;
  3. adapter -> torch DataLoader for a few batches (the path training code uses);
  4. contact sheets with frame index + real timestamp, boxes / maps / masks /
     label strips drawn on the decoded frames.
Results: ``<ds>/logs/validation.json`` and ``<ds>/processed/viz/*.jpg``.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext import viz  # noqa: E402
from aicext.adapters import (CropCandidatesAdapter, DenseCropAdapter, GridScoreAdapter,  # noqa: E402
                             MaskletAdapter, MaskSequenceAdapter, ObjectTrackAdapter, SaliencyMapAdapter,
                             SegmentLabelAdapter, ShotBoundaryAdapter, SparseCropAdapter, UserSelectionAdapter,
                             collate_pad)
from aicext.adapters.base import Index, load_video_frames, media_times  # noqa: E402
from aicext.common import EXT_ROOT, now, write_json  # noqa: E402


def loader_check(ds, pad_keys, batch_size=4, batches=3, workers=2):
    import torch
    from torch.utils.data import DataLoader

    class W(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            return ds[i]

    dl = DataLoader(W(), batch_size=batch_size, shuffle=True, num_workers=workers,
                    collate_fn=lambda b: collate_pad(b, pad_keys), generator=torch.Generator().manual_seed(0))
    shapes, t0 = [], time.time()
    for k, b in enumerate(dl):
        shapes.append({key: list(np.asarray(b[key]).shape) for key in pad_keys})
        if k + 1 >= batches:
            break
    return {"batches": len(shapes), "shapes": shapes, "seconds": round(time.time() - t0, 2)}


def stratified(items, key, n, seed=0):
    rng = random.Random(seed)
    groups = {}
    for i, it in enumerate(items):
        groups.setdefault(key(it), []).append(i)
    out = []
    per = max(1, n // max(1, len(groups)))
    for g in sorted(groups, key=str):
        out += rng.sample(groups[g], min(per, len(groups[g])))
    return out[:max(n, len(groups))]


def v_retargetvid(out_dir):
    ds = DenseCropAdapter("RetargetVid", fps=2.0, decode=False)
    res = {"samples": len(ds)}
    picks = stratified(ds.items, lambda it: (it[0][1], ds.ix.media[it[0][0]]["official_split"]), 6)
    checks = []
    for j, i in enumerate(picks):
        s = ds[i]
        m = ds.ix.media[s["uid"]]
        k = np.linspace(0, len(s["frame_idx"]) - 1, 4).astype(int)
        idx = s["frame_idx"][k]
        frames = load_video_frames(m, idx, None)
        assert frames.shape[1:3] == (m["height"], m["width"]), "decoded size != probed size"
        b = s["boxes_xywh"][:, k]
        inb = ((b[..., 0] >= 0) & (b[..., 1] >= 0) & (b[..., 0] + b[..., 2] <= m["width"] + 0.5) &
               (b[..., 1] + b[..., 3] <= m["height"] + 0.5))[s["valid"][:, k]]
        checks.append({"uid": s["uid"], "ratio": s["ratio"], "annotators": len(s["annotators"]),
                       "frames": int(len(s["frame_idx"])), "in_bounds_frac": float(inb.mean())})
        tiles = [viz.label(viz.draw_boxes(fr, s["boxes_xywh_norm"][:, kk]),
                           f"{s['uid']} {s['ratio']} f={int(fi)} t={s['time_s'][kk]:.2f}s")
                 for fr, fi, kk in zip(frames, idx, k)]
        viz.save(viz.grid(tiles, 4), out_dir / f"retargetvid_{j}_{s['uid'].replace(':', '_')}_{s['ratio']}.jpg")
    res["decode_checks"] = checks
    res["loader"] = loader_check(ds, ["boxes_xywh", "valid", "frame_idx", "time_s"])
    return res


def v_dhf1k(out_dir):
    res = {}
    for kind in ("saliency_map", "fixation_points"):
        ds = SaliencyMapAdapter("DHF1K", kind=kind, fps=2.0, max_frames=8, map_size=128)
        res[kind] = {"samples": len(ds)}
        picks = stratified(ds.items, lambda it: it[0]["official_split"], 4)
        checks = []
        for j, i in enumerate(picks):
            s = ds[i]
            m = ds.ix.media[s["uid"]]
            k = np.arange(min(4, len(s["frame_idx"])))
            frames = load_video_frames(m, s["frame_idx"][k], None)
            checks.append({"uid": s["uid"], "valid_frac": float(s["valid"].mean()), "map_shape": list(s["maps"].shape),
                           "map_max": float(s["maps"].max())})
            tiles = []
            for fr, kk in zip(frames, k):
                mp = s["maps"][kk]
                if kind == "fixation_points":
                    import cv2
                    mp = cv2.dilate(mp, np.ones((3, 3), np.uint8))
                tiles.append(viz.label(viz.overlay_map(fr, mp), f"{s['uid']} {kind[:4]} f={int(s['frame_idx'][kk])} "
                                                                 f"t={s['time_s'][kk]:.2f}s"))
            viz.save(viz.grid(tiles, 4), out_dir / f"dhf1k_{kind}_{j}_{s['uid'].replace(':', '_')}.jpg")
        res[kind]["checks"] = checks
        res[kind]["loader"] = loader_check(ds, ["maps", "valid", "frame_idx"])
    return res


def v_live(out_dir):
    ds = SparseCropAdapter("LIVE_YT_VC", decode=False)
    res = {"samples": len(ds)}
    if not len(ds):
        return res
    picks = random.Random(0).sample(range(len(ds)), min(6, len(ds)))
    checks = []
    for j, i in enumerate(picks):
        s = ds[i]
        m = ds.ix.media[s["uid"]]
        k = np.linspace(0, len(s["frame_idx"]) - 1, 4).astype(int)
        idx = s["frame_idx"][k]
        ok = idx < m["frame_count"]
        frames = load_video_frames(m, idx[ok], None)
        W, H = m["width"], m["height"]
        b = s["boxes_xywh"]
        inb = (b[:, 0] >= 0) & (b[:, 1] >= 0) & (b[:, 0] + b[:, 2] <= W + 1) & (b[:, 1] + b[:, 3] <= H + 1)
        checks.append({"uid": s["uid"], "W": W, "H": H, "frames": len(s["frame_idx"]), "in_bounds_frac": float(inb.mean()),
                       "box_w_over_W_median": float(np.median(b[:, 2] / W))})
        tiles = [viz.label(viz.draw_boxes(fr, s["boxes_xywh_norm"][kk][None]),
                           f"{s['uid']} f={int(fi)} t={s['time_s'][kk]:.2f}s")
                 for fr, fi, kk in zip(frames, idx[ok], k[ok])]
        if tiles:
            viz.save(viz.grid(tiles, 4), out_dir / f"livevc_{j}_{s['uid'].replace(':', '_')}.jpg")
    res["decode_checks"] = checks
    res["loader"] = loader_check(ds, ["boxes_xywh", "valid", "frame_idx"])
    return res


def v_yth(out_dir):
    res = {}
    for variant, field in (("mturk_votes", "mturk_votes"), ("match_label", "match_label")):
        ds = SegmentLabelAdapter("YouTubeHighlights", field=field, variant=variant, fps=2.0)
        res[variant] = {"samples": len(ds)}
        picks = stratified(ds.items, lambda it: it[0]["official_split"], 4)
        checks = []
        for j, i in enumerate(picks):
            s = ds[i]
            m = ds.ix.media[s["uid"]]
            checks.append({"uid": s["uid"], "valid_frac": float(s["valid"].mean()),
                           "n_segments": len(s["segments"]), "value_range": [float(s["value"].min()), float(s["value"].max())]})
            if variant == "mturk_votes":
                k = np.linspace(0, len(s["frame_idx"]) - 1, 4).astype(int)
                frames = load_video_frames(m, s["frame_idx"][k], None)
                tiles = [viz.label(fr, f"{s['uid'][4:]} f={int(s['frame_idx'][kk])} t={s['time_s'][kk]:.1f}s "
                                       f"v={s['value'][kk]:.2f}{'' if s['valid'][kk] else ' (unlab)'}")
                         for fr, kk in zip(frames, k)]
                g = viz.grid(tiles, 4)
                strip = viz.timeline_strip(s["time_s"], s["value"], s["valid"], width=g.shape[1],
                                           marks=s["time_s"][k])
                viz.save(np.concatenate([g, strip], 0), out_dir / f"yth_{j}_{s['uid'].replace(':', '_')}.jpg")
        res[variant]["checks"] = checks
        res[variant]["loader"] = loader_check(ds, ["value", "valid", "frame_idx", "time_s"])
    return res


def v_mrhisum(out_dir):
    ds = GridScoreAdapter("MrHiSum", mode="features")
    res = {"feature_samples": len(ds)}
    if len(ds):
        picks = random.Random(0).sample(range(len(ds)), min(6, len(ds)))
        checks = []
        for i in picks:
            s = ds[i]
            checks.append({"uid": s["uid"], "steps": int(len(s["gtscore"])), "feat_shape": list(s["features"].shape),
                           "valid_frac": float(s["valid"].mean()), "gtscore_range": [float(s["gtscore"].min()),
                                                                                     float(s["gtscore"].max())]})
        res["checks"] = checks
        res["loader"] = loader_check(ds, ["features", "gtscore", "valid"], batch_size=8)
    return res


def v_gaicd(out_dir):
    ds = CropCandidatesAdapter("GAICD", decode=False)
    res = {"samples": len(ds)}
    picks = stratified(ds.items, lambda it: it[0]["official_split"], 6)
    checks = []
    for j, i in enumerate(picks):
        s = ds[i]
        m = ds.ix.media[s["uid"]]
        img = load_video_frames(m, np.array([0]), None)[0]
        assert img.shape[:2] == (s["height"], s["width"]), "decoded size != recorded size"
        v = s["valid"]
        b = s["boxes_xywh"][v]
        inb = (b[:, 0] >= 0) & (b[:, 1] >= 0) & (b[:, 0] + b[:, 2] <= s["width"]) & (b[:, 1] + b[:, 3] <= s["height"])
        top = np.argsort(-np.where(v, s["mos"], -np.inf))[:3]
        checks.append({"uid": s["uid"], "split": m["official_split"], "candidates": int(v.sum()),
                       "in_bounds_frac": float(inb.mean()), "mos_range": [float(s["mos"][v].min()),
                                                                          float(s["mos"][v].max())]})
        tiles = [viz.label(viz.draw_boxes(img, s["boxes_xywh_norm"][[k]]),
                           f"{s['uid']} #{int(k)} MOS={s['mos'][k]:.2f}") for k in top]
        worst = int(np.argmin(np.where(v, s["mos"], np.inf)))
        tiles.append(viz.label(viz.draw_boxes(img, s["boxes_xywh_norm"][[worst]]),
                               f"{s['uid']} worst MOS={s['mos'][worst]:.2f}"))
        viz.save(viz.grid(tiles, 4), out_dir / f"gaicd_{j}_{s['uid'].replace(':', '_')}.jpg")
    res["checks"] = checks
    res["loader"] = loader_check(ds, ["boxes_xywh", "mos", "valid"], batch_size=8)
    return res


def v_clipshots(out_dir):
    res = {}
    for subset in ("train", "only_gradual", "test"):
        # labels are checked even before the videos arrive (ready_only=False); frames only when present
        ds = ShotBoundaryAdapter("ClipShots", subset=subset, window=64, ready_only=False)
        res[subset] = {"samples": len(ds), "with_media": sum(bool(m.get("media_path")) for m, _ in ds.items)}
        if not len(ds):
            continue
        picks = random.Random(0).sample(range(len(ds)), min(4, len(ds)))
        checks = []
        for j, i in enumerate(picks):
            s = ds[i]
            m = ds.ix.media[s["uid"]]
            checks.append({"uid": s["uid"], "frames": m.get("frame_count"), "label_frames": s["label_frames"],
                           "valid_frac": float(s["valid"].mean()), "cuts": int((s["label"] == 1).sum()),
                           "gradual": int((s["label"] == 2).sum())})
            if m.get("media_path") and j < 2:
                tr = [t for t in s["transitions"] if t[0] < (m.get("frame_count") or 0) - 2]
                if tr:
                    s0, e0 = tr[len(tr) // 2]
                    idx = np.clip(np.array([s0 - 2, s0, e0, e0 + 2]), 0, m["frame_count"] - 1)
                    frames = load_video_frames(m, idx, None)
                    t = media_times(m)
                    tiles = [viz.label(fr, f"{s['uid'][10:]} f={int(k)} t={t[k]:.2f}s tr=[{s0},{e0}]")
                             for fr, k in zip(frames, idx)]
                    viz.save(viz.grid(tiles, 4), out_dir / f"clipshots_{subset}_{j}.jpg")
        res[subset]["checks"] = checks
        res[subset]["loader"] = loader_check(ds, ["label", "valid", "frame_idx"])
    return res


def v_davsod(out_dir):
    ds = MaskSequenceAdapter("DAVSOD", max_frames=6, size=None, decode=True)
    res = {"samples": len(ds)}
    picks = stratified(ds.items, lambda it: it[0]["official_split"], 6)
    checks = []
    for j, i in enumerate(picks):
        s = ds[i]
        assert s["frames"].shape[1:3] == s["masks"].shape[1:3], "frame / mask size mismatch"
        checks.append({"uid": s["uid"], "annotated_sampled": int(len(s["frame_idx"])),
                       "fg_frac": float(s["masks"].mean())})
        k = np.arange(min(4, len(s["frame_idx"])))
        tiles = [viz.label(viz.overlay_mask(s["frames"][kk], s["masks"][kk]),
                           f"{s['uid']} f={int(s['frame_idx'][kk])} t={s['time_s'][kk]:.2f}s") for kk in k]
        viz.save(viz.grid(tiles, 4), out_dir / f"davsod_{j}_{s['uid'].replace(':', '_').replace('/', '_')}.jpg")
    res["checks"] = checks
    ds_small = MaskSequenceAdapter("DAVSOD", max_frames=4, size=128, decode=False)
    res["loader"] = loader_check(ds_small, ["masks", "frame_idx"])
    return res


def v_phd2(out_dir):
    ds = UserSelectionAdapter("PHD2", fps=2.0)
    res = {"samples_user_video": len(ds)}
    if not len(ds):
        res["note"] = "no downloaded sample videos yet"
        return res
    picks = random.Random(0).sample(range(len(ds)), min(4, len(ds)))
    checks = []
    for j, i in enumerate(picks):
        s = ds[i]
        m = ds.ix.media[s["uid"]]
        checks.append({"uid": s["uid"], "user": s["user_id"], "selected_frac": float(s["selected"].mean()),
                       "valid_frac": float(s["valid"].mean()), "others_max": int(s["others_selected"].max())})
        if s["selected"].any():
            k = np.where(s["selected"])[0][:2].tolist() + np.where(~s["selected"])[0][:2].tolist()
            k = np.asarray(sorted(k))
            frames = load_video_frames(m, s["frame_idx"][k], None)
            tiles = [viz.label(fr, f"{s['uid'][5:]} u{s['user_id']} t={s['time_s'][kk]:.1f}s "
                                   f"{'SEL' if s['selected'][kk] else 'unlab'}") for fr, kk in zip(frames, k)]
            g = viz.grid(tiles, 4)
            strip = viz.timeline_strip(s["time_s"], s["selected"].astype(np.float32), s["valid"], width=g.shape[1],
                                       marks=s["time_s"][k])
            viz.save(np.concatenate([g, strip], 0), out_dir / f"phd2_{j}_{s['uid'].replace(':', '_')}.jpg")
    res["checks"] = checks
    res["loader"] = loader_check(ds, ["selected", "valid", "frame_idx"])
    return res


def v_lasot(out_dir):
    ds = ObjectTrackAdapter("LaSOT", fps=None, max_frames=32)
    res = {"samples": len(ds)}
    picks = random.Random(0).sample(range(len(ds)), min(4, len(ds)))
    checks = []
    for j, i in enumerate(picks):
        s = ds[i]
        m = ds.ix.media[s["uid"]]
        v = s["visible"]
        b = s["boxes_xywh"][v]
        inb = (b[:, 0] >= -0.5) & (b[:, 1] >= -0.5) & (b[:, 0] + b[:, 2] <= s["width"] + 0.5) & \
              (b[:, 1] + b[:, 3] <= s["height"] + 0.5)
        checks.append({"uid": s["uid"], "frames": int(len(s["frame_idx"])), "visible_frac": float(v.mean()),
                       "in_bounds_frac": float(inb.mean()) if len(b) else None, "target": s["target"]})
        k = np.linspace(0, len(s["frame_idx"]) - 1, 4).astype(int)
        frames = load_video_frames(m, s["frame_idx"][k], None)
        tiles = [viz.label(viz.draw_boxes(fr, s["boxes_xywh_norm"][[kk]]) if s["visible"][kk] else fr,
                           f"{s['uid']} f={int(s['frame_idx'][kk])} {'vis' if s['visible'][kk] else 'INVISIBLE'}")
                 for fr, kk in zip(frames, k)]
        viz.save(viz.grid(tiles, 4), out_dir / f"lasot_{j}_{s['uid'].replace(':', '_')}.jpg")
    res["checks"] = checks
    res["loader"] = loader_check(ds, ["boxes_xywh", "visible", "valid", "frame_idx"])
    return res


def v_sav(out_dir):
    res = {}
    for src in ("manual", "auto"):
        ds = MaskletAdapter("SA_V", source=src, max_objects=3, max_frames=4, size=None, decode=True)
        res[src] = {"samples": len(ds)}
        if not len(ds):
            continue
        s = ds[0]
        res[src]["check"] = {"uid": s["uid"], "masks": list(s["masks"].shape), "visible_frac": float(s["visible"].mean()),
                             "frame_idx": s["frame_idx"].tolist()}
        tiles = []
        for jj in range(len(s["frame_idx"])):
            fr = s["frames"][jj]
            for ii in range(s["masks"].shape[0]):
                fr = viz.overlay_mask(fr, s["masks"][ii, jj], color=[(255, 0, 0), (0, 255, 0), (0, 0, 255)][ii % 3])
            tiles.append(viz.label(fr, f"{s['uid']} {src} f={int(s['frame_idx'][jj])} t={s['time_s'][jj]:.2f}s"))
        viz.save(viz.grid(tiles, 4), out_dir / f"sav_{src}.jpg")
        small = MaskletAdapter("SA_V", source=src, max_objects=3, max_frames=4, size=64)
        res[src]["loader"] = loader_check(small, ["masks", "visible", "frame_idx"], batch_size=1, batches=1, workers=0)
    return res


VALIDATORS = {"RetargetVid": v_retargetvid, "DHF1K": v_dhf1k, "LIVE_YT_VC": v_live,
              "YouTubeHighlights": v_yth, "MrHiSum": v_mrhisum, "GAICD": v_gaicd,
              "ClipShots": v_clipshots, "DAVSOD": v_davsod, "PHD2": v_phd2, "LaSOT": v_lasot, "SA_V": v_sav}


def integrity(dataset: str, n: int = 40) -> dict:
    ix = Index.load(dataset)
    media = list(ix.media.values())
    rng = random.Random(1)
    sample = rng.sample(media, min(n, len(media)))
    bad = []
    for m in sample:
        p = m.get("media_path")
        if not p:
            continue
        if not Path(p).exists():
            bad.append({"uid": m["uid"], "problem": "path missing"})
        elif m.get("file_size") and Path(p).is_file() and Path(p).stat().st_size != m["file_size"]:
            bad.append({"uid": m["uid"], "problem": "size changed"})
    ann_missing = [a["uid"] for a in rng.sample(ix.annotations, min(n, len(ix.annotations)))
                   if a.get("annotation_path") and not Path(a["annotation_path"]).exists()]
    return {"media": len(media), "annotations": len(ix.annotations), "sampled": len(sample),
            "media_problems": bad, "annotation_paths_missing": ann_missing}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=list(VALIDATORS))
    a = ap.parse_args()
    summary_path = EXT_ROOT / "_registry" / "validation_summary.json"
    summary = json.loads(summary_path.read_text()).get("results", {}) if summary_path.exists() else {}
    for name in a.datasets:
        out_dir = EXT_ROOT / name / "processed" / "viz"
        rep = {"dataset": name, "time": now()}
        try:
            rep["integrity"] = integrity(name)
            rep.update(VALIDATORS[name](out_dir))
            rep["status"] = "pass"
        except Exception as exc:
            rep["status"] = "fail"
            rep["error"] = repr(exc)
            rep["traceback"] = traceback.format_exc()[-2000:]
        write_json(EXT_ROOT / name / "logs" / "validation.json", rep)
        summary[name] = rep["status"]
        print(json.dumps({"dataset": name, "status": rep["status"], "error": rep.get("error")}), flush=True)
    write_json(summary_path, {"time": now(), "results": summary})


if __name__ == "__main__":
    main()
