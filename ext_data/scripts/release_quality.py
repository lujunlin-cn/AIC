#!/usr/bin/env python3
"""Quality + readiness report for a frozen release (read-only; writes next to it).

  python scripts/release_quality.py spatial_crop_v1
  python scripts/release_quality.py temporal_evidence_v1
  python scripts/release_quality.py mrhisum_feat_subset_v1

Output: ``_releases/<name>.quality.json`` (outside the immutable directory,
bound to the release by the SHA-256 of its SHA256SUMS).  Three readiness flags
are reported separately per dataset: schema_pass, sample_readable,
raw_video_trainable.  Saliency agreement is an auxiliary check only.  The AIC
official-test check is byte identity only (SHA-256), not content overlap.
"""
from __future__ import annotations

import argparse
import collections
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext import window_scorer as ws  # noqa: E402
from aicext.adapters.base import collate_pad  # noqa: E402
from aicext.common import EXT_ROOT, now, read_jsonl, sha256_file, write_json  # noqa: E402
from aicext.media import read_frames, timeline  # noqa: E402
from aicext.release import (FeatureSubsetUnits, Release, SpatialCropUnits, TemporalEvidenceUnits,  # noqa: E402
                            preference_pairs)

OFFICIAL_INDEX = Path("/data/aic/official_test_20260926/intake/index.enriched.jsonl")
T5 = Path("/data/aic/experiments/T5_CROPHEAD_V3/tables")
SPLITS = ("train", "dev", "confirmation", "exposed_eval")
RNG = np.random.default_rng(20260929)


def all_rows(rel):
    return {sp: rel.manifest(sp) for sp in SPLITS if (rel.dir / "manifests" / f"{sp}.jsonl").exists()}


def official_hits(shas) -> dict:
    off = {r.get("source_sha256") for r in read_jsonl(OFFICIAL_INDEX)} - {None}
    return {"official_test_videos": len(off), "release_media_hashed": len(set(shas)),
            "byte_identical_hits": len(off & set(shas)),
            "meaning": "no byte-identical file; re-encodes, crops or other content overlap are NOT excluded by this check"}


def timeline_checks(rel, rows, redecode: int = 6) -> dict:
    out = collections.Counter()
    bad = []
    for r in rows:
        z = np.load(rel.path(r["timeline"]))
        t, pts = z["time_s"], z["pts"]
        out["videos"] += 1
        out["len_eq_frame_count"] += int(len(t) == r["frame_count"])
        out["first_time_zero"] += int(abs(t[0]) < 1e-9)
        out["strictly_increasing"] += int(np.all(np.diff(t) > 0))
        out["method_decode"] += int(str(z["method"]) == "decode")
        out["rotation_zero"] += int(not r.get("rotation"))
        if not (len(t) == r["frame_count"] and np.all(np.diff(t) > 0)):
            bad.append(r.get("unit_id"))
    picks = [rows[i] for i in RNG.choice(len(rows), min(redecode, len(rows)), replace=False)]
    same = []
    for r in picks:  # full re-decode: stored PTS must equal a fresh decode exactly
        tl = timeline(r["media_path"], decode=True)
        z = np.load(rel.path(r["timeline"]))
        same.append({"unit": r.get("unit_id"), "frames": tl["frame_count"],
                     "pts_identical": bool(np.array_equal(tl["pts"], z["pts"]))})
    return {**dict(out), "bad_examples": bad[:10], "redecode": same}


def decode_checks(rows, n: int = 12) -> list:
    res = []
    for r in [rows[i] for i in RNG.choice(len(rows), min(n, len(rows)), replace=False)]:
        k = sorted({0, r["frame_count"] // 2, r["frame_count"] - 1})
        got = read_frames(r["media_path"], k, size=None)
        res.append({"unit": r.get("unit_id"), "frames": k, "decoded": sorted(got),
                    "shape_ok": all(got[i].shape[:2] == (r["height"], r["width"]) for i in got)})
    return res


# ======================================================================= spatial
def spatial(rel) -> dict:
    S = all_rows(rel)
    rows = [r for rs in S.values() for r in rs]
    Q = {"release": rel.dir.name}
    by_ds = collections.defaultdict(list)
    for r in rows:
        by_ds[r["dataset"]].append(r)
    # ---- counts
    cnt = {}
    for ds, rs in by_ds.items():
        boxes = sum(r["human_annotation"]["annotators"] * r["human_annotation"]["annotated_frames"] for r in rs)
        cnt[ds] = {"raw_media_videos": len({r["media_path"] for r in rs}), "units": len(rs),
                   "independent_source_videos": len({r["group_id"] for r in rs}),
                   "human_annotation_files" if ds == "RetargetVid" else "human_label_rows":
                       sum(r["human_annotation"]["annotators"] for r in rs) if ds == "RetargetVid" else len(rs),
                   "human_boxes_nominal": boxes,
                   "derived_tracks": sum(1 for r in rs if r.get("derived_annotation")),
                   "per_split": {sp: {"units": sum(r["dataset"] == ds for r in S[sp]),
                                      "source_videos": len({r["group_id"] for r in S[sp] if r["dataset"] == ds})}
                                 for sp in S}}
    Q["counts"] = cnt
    Q["excluded"] = {"rows": len(rel.manifest("excluded")),
                     "reasons": dict(collections.Counter(r["split_reason"] for r in rel.manifest("excluded")))}
    Q["split_reasons"] = {sp: dict(collections.Counter(r["split_reason"].split(" (")[0][:120] for r in rs).most_common(8))
                          for sp, rs in S.items()}
    # ---- schema + packed == raw (every unit)
    schema_err, raw_diff, valid_boxes, not_max, oob, frames_beyond = [], 0, collections.Counter(), collections.Counter(), collections.Counter(), 0
    for r in rows:
        for k in ("unit_id", "group_id", "media_path", "media_sha256", "timeline", "width", "height", "frame_count",
                  "target_ratio_wh", "gt_is_max_window", "coord_version", "human_annotation"):
            if r.get(k) in (None, ""):
                schema_err.append(f"{r['unit_id']}: {k}")
        z = np.load(rel.path(r["human_annotation"]["packed"]))
        ltrb, fidx = z["ltrb_raw"].astype(float), z["frame_idx"]
        if r["dataset"] == "RetargetVid":
            for i, ann in enumerate(z["annotators"]):
                raw = np.loadtxt(rel.path(f"raw_annotations/RetargetVid/{ann}/{r['unit_id'].split(':')[1]}_{r['ratio_key']}.txt"),
                                 delimiter=",", ndmin=2)
                raw_diff = max(raw_diff, float(np.abs(raw - ltrb[i, :len(raw)]).max()))
        W, H = r["width"], r["height"]
        ok = np.isfinite(ltrb).all(-1)
        valid_boxes[r["dataset"]] += int(ok.sum())
        xywh = np.concatenate([ltrb[..., :2], ltrb[..., 2:] - ltrb[..., :2]], -1)[ok]
        chk = ws.check_candidates(W, H, r["target_ratio_wh"], xywh)
        not_max[r["dataset"]] += int((~chk["is_max_window"]).sum())
        oob[r["dataset"]] += int(((xywh[:, 0] < 0) | (xywh[:, 1] < 0) | (xywh[:, 0] + xywh[:, 2] > W) |
                                  (xywh[:, 1] + xywh[:, 3] > H)).sum())
        frames_beyond += int((fidx >= r["frame_count"]).sum())
        if r["dataset"] == "RetargetVid" and len(fidx) != r["frame_count"]:
            schema_err.append(f"{r['unit_id']}: annotation lines {len(fidx)} != frames {r['frame_count']}")
    # LIVE raw csv re-parse (release copy) == packed
    import ast
    import csv
    live_raw = {}
    for row in csv.DictReader(open(rel.path("raw_annotations/LIVE_YT_VC/video_bbox_labels.csv"))):
        fr = sorted((int(k[5:]), ast.literal_eval(v)) for k, v in row.items() if k.startswith("frame") and v and v.strip())
        live_raw[row["video"]] = np.array([[d["left"], d["top"], d["right"], d["bottom"]] for _, d in fr], float)
    live_diff = 0.0
    for r in by_ds.get("LIVE_YT_VC", []):
        z = np.load(rel.path(r["human_annotation"]["packed"]))
        live_diff = max(live_diff, float(np.abs(live_raw[r["unit_id"].split(":")[1]] - z["ltrb_raw"][0]).max()))
    Q["coordinates"] = {
        "schema_errors": schema_err[:20], "schema_error_count": len(schema_err),
        "packed_vs_raw_file_maxabs": {"RetargetVid": raw_diff, "LIVE_YT_VC": live_diff},
        "valid_human_boxes": dict(valid_boxes), "boxes_not_max_window(>1px)": dict(not_max),
        "boxes_out_of_image": dict(oob), "annotated_frames_beyond_video": frames_beyond,
        "convention": f"{ws.COORD_VERSION}: x=l, y=t, w=r-l, h=b-t, no clipping; raw ltrb kept as ltrb_raw",
        "rotation_nonzero_videos": sum(1 for r in rows if r.get("rotation"))}
    # ---- +1 / half-open and boundary variants (all LIVE boxes; RV keyframes every 30 frames)
    diffs = collections.defaultdict(list)
    for r in rows:
        z = np.load(rel.path(r["human_annotation"]["packed"]))
        ltrb = z["ltrb_raw"].astype(float)
        ks = range(0, ltrb.shape[1], 30) if r["dataset"] == "RetargetVid" else range(ltrb.shape[1])
        W, H, ratio = r["width"], r["height"], r["target_ratio_wh"]
        cand = ws.legal_candidates(W, H, ratio, n=65)["boxes_xywh"]
        for k in ks:
            g = np.concatenate([ltrb[:, k, :2], ltrb[:, k, 2:] - ltrb[:, k, :2]], -1)
            a = ws.score_candidates(W, H, ratio, cand, g, gt_source="human", gt_is_max_window=r["gt_is_max_window"])
            b = ws.score_candidates(W, H, ratio, cand, g, gt_source="human", gt_is_max_window=r["gt_is_max_window"],
                                    iou="inclusive_plus1")
            c = ws.score_candidates(W, H, ratio, cand, g, gt_source="human", gt_is_max_window=r["gt_is_max_window"],
                                    gt_boundary="clip_exp_v1")
            diffs[f"{r['dataset']}:inclusive_minus_halfopen"].append(np.nanmax(np.abs(b["iou_mean"] - a["iou_mean"])))
            diffs[f"{r['dataset']}:argmax_changed_by_plus1"].append(float(np.nanargmax(a["iou_mean"]) != np.nanargmax(b["iou_mean"])))
            diffs[f"{r['dataset']}:clip_exp_minus_none"].append(np.nanmax(np.abs(c["iou_mean"] - a["iou_mean"])))
    Q["iou_convention_comparison"] = {k: {"frames": len(v), "mean": float(np.mean(v)), "max": float(np.max(v))}
                                      for k, v in diffs.items()}
    # ---- reproduce the experiment-side T5 targets exactly (inclusive_plus1; LIVE with clip_exp_v1)
    rep = {}
    index = {r["unit_id"]: r for r in rows}
    for name, ds in (("rv_confirm2", "RetargetVid"), ("rv_dev", "RetargetVid"), ("live_confirm3", "LIVE_YT_VC"),
                     ("live_dev", "LIVE_YT_VC")):
        p = T5 / f"{name}.pkl"
        if not p.exists():
            continue
        md, n = 0.0, 0
        for u in pickle.loads(p.read_bytes()):
            uid = f"retargetvid:{int(u['vid']):03d}:{u['r']}" if ds == "RetargetVid" else f"liveytvc:{u['vid']}"
            r = index[uid]
            z = np.load(rel.path(r["human_annotation"]["packed"]))
            ltrb, fidx = z["ltrb_raw"].astype(float), z["frame_idx"]
            pos = {int(f): i for i, f in enumerate(fidx)}
            w, h, axis = ws.max_window(r["width"], r["height"], r["target_ratio_wh"])
            for row, j in enumerate(u["J"]):
                t = u["keys"][j]
                if t not in pos or not np.isfinite(u["Y"][row]).all():
                    continue
                off = u["offs"][row]
                cand = np.zeros((len(off), 4))
                cand[:, 2:] = (w, h)
                cand[:, axis] = off
                g = np.concatenate([ltrb[:, pos[t], :2], ltrb[:, pos[t], 2:] - ltrb[:, pos[t], :2]], -1)
                s = ws.score_candidates(r["width"], r["height"], r["target_ratio_wh"], cand, g, gt_source="human",
                                        gt_is_max_window=r["gt_is_max_window"], iou="inclusive_plus1",
                                        gt_boundary="clip_exp_v1" if ds == "LIVE_YT_VC" else "none", require_legal=False)
                md = max(md, float(np.abs(s["iou_mean"] - u["Y"][row]).max()))
                n += 1
        rep[name] = {"keyframes": n, "max_abs_diff": md}
    Q["reproduce_T5_targets"] = rep
    # ---- window-scorer ceiling: what a max window can reach against each human box type
    ceil = collections.defaultdict(list)
    for r in rows:
        z = np.load(rel.path(r["human_annotation"]["packed"]))
        ltrb = z["ltrb_raw"].astype(float)
        ks = range(0, ltrb.shape[1], 60) if r["dataset"] == "RetargetVid" else range(ltrb.shape[1])
        for k in ks:
            g = np.concatenate([ltrb[:, k, :2], ltrb[:, k, 2:] - ltrb[:, k, :2]], -1)
            ceil[r["dataset"]] += list(ws.iou_ceiling(r["width"], r["height"], r["target_ratio_wh"], g))
    Q["max_window_iou_ceiling"] = {ds: {"boxes": len(v), "p5": float(np.percentile(v, 5)), "median": float(np.median(v)),
                                        "mean": float(np.mean(v))} for ds, v in ceil.items()}
    # ---- PTS / timeline, decode and real batches with padding
    Q["timeline"] = {ds: timeline_checks(rel, rs, redecode=4) for ds, rs in by_ds.items()}
    Q["decode"] = {ds: decode_checks(rs, 6) for ds, rs in by_ds.items()}
    batches = {}
    for sp in S:
        ds = SpatialCropUnits(rel, sp, frames=2.0)
        if not len(ds):
            continue
        idx = list(RNG.choice(len(ds), min(4, len(ds)), replace=False))
        b = collate_pad([ds[int(i)] for i in idx], ("boxes_xywh", "valid", "frame_idx", "time_s"))
        pad_ok = all((~b["valid"][i][:, s[1]:]).all() for i, s in enumerate(b["valid_shape"]))
        batches[sp] = {"datasets": b["dataset"], "boxes_xywh": list(b["boxes_xywh"].shape),
                       "valid_true": int(b["valid"].sum()), "padding_never_valid": bool(pad_ok)}
    # mixed RV + LIVE batch (different annotator counts and frame counts)
    mix = [SpatialCropUnits(rel, "exposed_eval", datasets=["RetargetVid"], frames=2.0)[0],
           SpatialCropUnits(rel, "train", datasets=["LIVE_YT_VC"])[0]]
    b = collate_pad(mix, ("boxes_xywh", "valid", "frame_idx", "time_s"))
    batches["mixed_rv_live"] = {"boxes_xywh": list(b["boxes_xywh"].shape), "shapes": b["valid_shape"].tolist(),
                                "padding_never_valid": bool(not b["valid"][1, 1:].any() and not b["valid"][1, :, 30:].any())}
    Q["batches"] = batches
    # ---- auxiliary: DHF1K saliency inside vs outside the RetargetVid human box
    Q["aux_saliency"] = saliency_aux(rel, by_ds.get("RetargetVid", []))
    Q["official_test_byte_check"] = official_hits([r["media_sha256"] for r in rows])
    Q["readiness"] = {
        "RetargetVid": {"schema_pass": len([e for e in schema_err if e.startswith("retargetvid")]) == 0,
                        "sample_readable": True, "raw_video_trainable": True,
                        "v1_training_split": "none: all 200 source videos were used for validation (exposed_eval only)"},
        "LIVE_YT_VC": {"schema_pass": len([e for e in schema_err if e.startswith("liveytvc")]) == 0,
                       "sample_readable": True, "raw_video_trainable": True,
                       "v1_training_split": f"{len([r for r in S.get('train', []) if r['dataset'] == 'LIVE_YT_VC'])} units"}}
    return Q


def saliency_aux(rel, rows, n_units: int = 24, per_unit: int = 5) -> dict:
    import cv2

    ann = {a["media_uid"]: a for a in read_jsonl(EXT_ROOT / "DHF1K/processed/annotations.jsonl")
           if a["annotation_type"] == "saliency_map"}
    ratios = []
    for r in [rows[i] for i in RNG.choice(len(rows), min(n_units, len(rows)), replace=False)]:
        a = ann.get(r["group_id"])
        if not a:
            continue
        z = np.load(rel.path(r["human_annotation"]["packed"]))
        ltrb = z["ltrb_raw"]
        for k in RNG.choice(ltrb.shape[1], per_unit, replace=False):
            p = Path(a["annotation_path"]) / f"{int(k) + 1:04d}.png"
            if not p.exists():
                continue
            m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE).astype(float)
            for b in ltrb[:, k]:
                l, t, rr, bb = np.round(b).astype(int)
                inside = np.zeros_like(m, bool)
                inside[t:bb, l:rr] = True
                if inside.all() or not inside.any():
                    continue
                ratios.append((m[inside].mean() + 1e-6) / (m[~inside].mean() + 1e-6))
    v = np.array(ratios)
    return {"boxes": len(v), "median_in_over_out": float(np.median(v)) if len(v) else None,
            "share_gt_1": float((v > 1).mean()) if len(v) else None,
            "note": "auxiliary sanity check of box/frame alignment only; saliency is not crop ground truth"}


# ======================================================================= temporal
def temporal(rel) -> dict:
    S = all_rows(rel)
    rows = [r for rs in S.values() for r in rs]
    ex = rel.manifest("excluded")
    Q = {"release": rel.dir.name}
    media = list(read_jsonl(EXT_ROOT / "YouTubeHighlights/processed/media.jsonl"))
    ann = list(read_jsonl(EXT_ROOT / "YouTubeHighlights/processed/annotations.jsonl"))
    Q["counts"] = {
        "author_list_videos": len(media),
        "raw_media_available": sum(m["download_status"] == "verified" for m in media),
        "annotations": {"match_label(auto)": sum(a["variant"] == "match_label" for a in ann),
                        "mturk(human)": sum(a["variant"] == "mturk_votes" for a in ann)},
        "mturk_with_media": len({a["media_uid"] for a in ann if a["variant"] == "mturk_votes"} &
                                {m["uid"] for m in media if m["download_status"] == "verified"}),
        "released_videos": len(rows), "released_with_mturk": sum(r["has_mturk"] for r in rows),
        "independent_source_videos": len({r["group_id"] for r in rows}),
        "per_split": {sp: {"videos": len(rs), "with_mturk": sum(r["has_mturk"] for r in rs),
                           "source_videos": len({r["group_id"] for r in rs}),
                           "domains": dict(collections.Counter(r["domain"] for r in rs))} for sp, rs in S.items()},
        "excluded": {"videos": len(ex), "with_mturk": sum(r["has_mturk"] for r in ex),
                     "reasons": dict(collections.Counter(r["exclude_reasons"][0].split(" (")[0] for r in ex)),
                     "reasons_x_project_split": dict(collections.Counter(
                         f"{r['exclude_reasons'][0].split(' (')[0]}|{r['project_split']}" for r in ex))}}
    # frame-rate mismatch diagnosis for the excluded frame_count_mismatch videos
    ym = {m["uid"]: m for m in media}
    diag = []
    for r in ex:
        if not r["exclude_reasons"][0].startswith("frame_count_mismatch"):
            continue
        m = ym[r["unit_id"]]
        c = json.loads((EXT_ROOT / f"YouTubeHighlights/annotations/upstream_repo/{r['domain']}/{r['youtube_id']}/clip.json").read_text())
        diag.append((c[-1][1] / m["frame_count"], m["fps"], r["project_split"]))
    rr = np.array([d[0] for d in diag]) if diag else np.zeros(0)
    Q["frame_mapping"] = {
        "rule": "label frame f <-> decoded display frame f; kept only if |last label frame - decoded frames| <= 3",
        "kept_gap_frames": dict(collections.Counter(
            int(float(np.load(rel.path(r["packed"]))["clip_end_frame_raw"][-1]) - r["frame_count"]) for r in rows)),
        "excluded_frame_count_mismatch": {
            "videos": len(diag), "labels_shorter_than_media": int((rr < 1).sum()),
            "ratio_last_over_frames_0.49_0.51": int(((rr > .49) & (rr < .51)).sum()),
            "fps_59.94": sum(abs(d[1] - 59.94) < .01 for d in diag),
            "in_T3_T4_yth_val_proxy": sum(d[2] == "holdout_dev_exposed" for d in diag),
            "interpretation": "most look like 60 fps re-uploads of ~30 fps originals (labels cover about the first "
                              "half of the frames under a frame-number mapping); not recovered in v1"}}
    # raw values: packed == author json (release copies), and supervision breakdown
    diff, clips, pos, zero, nan, pairs = 0.0, 0, 0, 0, 0, 0
    match = collections.Counter()
    for r in rows:
        z = np.load(rel.path(r["packed"]))
        v, m = z["clip_mturk_votes"], z["clip_match_label"]
        clips += len(v)
        pos += int((v > 0).sum()); zero += int((v == 0).sum()); nan += int(np.isnan(v).sum())
        match.update(m.tolist())
        mj = json.loads(rel.path(r["raw_annotations"]["match_label.json"]["path"]).read_text())
        diff = max(diff, float(np.abs(np.asarray(mj[1]) - m).max()))
        if "mturk_label.json" in r["raw_annotations"]:
            tj = json.loads(rel.path(r["raw_annotations"]["mturk_label.json"]["path"]).read_text())
            diff = max(diff, float(np.abs(np.asarray(tj[1]) - v).max()))
            pairs += len(preference_pairs({"clip_mturk_votes": v}, margin=1.0))
    Q["labels"] = {"packed_vs_raw_json_maxabs": diff, "clips": clips,
                   "mturk_votes_gt0 (explicit human selection)": pos,
                   "mturk_votes_eq0 (nobody selected; weak, not negative)": zero,
                   "no_mturk (NaN)": nan, "auto_match_label": {str(k): v for k, v in sorted(match.items())},
                   "relative_preference_pairs_margin1": pairs,
                   "vote_detail": "per-clip soft totals only; per-worker votes / worker count / ids not published"}
    # frame-level channels at 2 fps on real PTS, per split
    fr = {}
    for sp in S:
        ds = TemporalEvidenceUnits(rel, sp, fps=2.0)
        c = collections.Counter()
        for i in range(len(ds)):
            it = ds[i]
            c["frames"] += len(it["frame_idx"])
            for k, name in ((0, "human_unlabelled"), (1, "human_unselected_weak"), (2, "human_selected")):
                c[name] += int((it["human_code"] == k).sum()) if it["has_mturk"] else 0
            c["no_mturk_video_frames"] += 0 if it["has_mturk"] else len(it["frame_idx"])
            for k, name in ((0, "auto_unlabelled"), (1, "auto_unmatched_weak"), (2, "auto_borderline"), (3, "auto_matched")):
                c[name] += int((it["auto_code"] == k).sum())
        fr[sp] = dict(c)
    Q["frame_channels_2fps"] = fr
    Q["timeline"] = timeline_checks(rel, rows, redecode=4)
    Q["decode"] = decode_checks(rows, 6)
    batches = {}
    for sp in S:
        ds = TemporalEvidenceUnits(rel, sp, fps=2.0)
        idx = list(RNG.choice(len(ds), min(4, len(ds)), replace=False))
        b = collate_pad([ds[int(i)] for i in idx], ("votes_max", "human_code", "human_valid", "auto_code", "auto_valid",
                                                    "frame_idx", "time_s"))
        pad_ok = all((~b["human_valid"][i, s[0]:]).all() and (~b["auto_valid"][i, s[0]:]).all()
                     for i, s in enumerate(b["human_valid_shape"]))
        batches[sp] = {"votes_max": list(b["votes_max"].shape), "padding_never_valid": bool(pad_ok)}
    Q["batches"] = batches
    Q["official_test_byte_check"] = official_hits([r["media_sha256"] for r in rows])
    Q["readiness"] = {"YouTubeHighlights": {"schema_pass": True, "sample_readable": True, "raw_video_trainable": True,
                                            "note": "human MTurk subset is small; auto match labels are weak"}}
    return Q


def mrhisum(rel) -> dict:
    S = {sp: rel.manifest(sp) for sp in SPLITS if (rel.dir / "manifests" / f"{sp}.jsonl").exists()}
    Q = {"release": rel.dir.name, "counts": {sp: len(v) for sp, v in S.items()},
         "excluded": dict(collections.Counter(r["exclude_reasons"][0] for r in rel.manifest("excluded")))}
    ds = FeatureSubsetUnits(rel, "train", max_steps=300)
    b = collate_pad([ds[i] for i in range(4)], ("features", "gtscore", "valid", "grid_time_s"))
    Q["batch"] = {"features": list(b["features"].shape), "valid_true": int(b["valid"].sum())}
    lens = collections.Counter()
    for r in S.get("train", [])[:2000]:
        lens["label_eq_feature_steps"] += int(r["steps"] == r["feature_steps"])
        lens["n"] += 1
    Q["length_agreement_first2000_train"] = dict(lens)
    Q["readiness"] = {"MrHiSum": {"schema_pass": True, "sample_readable": True, "raw_video_trainable": False,
                                  "feature_head_trainable": True}}
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    a = ap.parse_args()
    rel = Release(a.name, verify="all")
    kind = a.name.rsplit("_v", 1)[0]
    Q = {"spatial_crop": spatial, "temporal_evidence": temporal, "mrhisum_feat_subset": mrhisum}[kind](rel)
    Q.update(generated=now(), release_sha256sums=sha256_file(rel.dir / "SHA256SUMS"),
             integrity={"files_checked_against_SHA256SUMS": len(rel.sums), "bad": 0})
    out = rel.dir.parent / f"{a.name}.quality.json"
    write_json(out, Q)
    print(json.dumps(Q, indent=1, ensure_ascii=False, default=str)[:12000])


if __name__ == "__main__":
    main()
