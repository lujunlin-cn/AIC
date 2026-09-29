#!/usr/bin/env python3
"""Build a frozen, immutable data release.

  python scripts/build_release.py spatial_crop_v1
  python scripts/build_release.py temporal_evidence_v1
  python scripts/build_release.py mrhisum_feat_subset_v1

Every release copies the (small) labels it uses, packs them per unit, keeps
the raw annotation files byte-identical under ``raw_annotations/``, references
media by path + SHA-256 (not copied), snapshots the code and the holdout
ledger, and ends with SHA256SUMS; see aicext/release.py.  Splits come from
``project_split`` of the last refresh; the per-row reason is kept.

Release splits (both temporal and spatial): ``train`` / ``dev`` / ``confirmation``
plus ``exposed_eval`` (already used for validation; evaluation/diagnostics only)
and ``excluded`` (with reasons).  ``dev`` and ``confirmation`` never overlap a
training group.
"""
from __future__ import annotations

import argparse
import collections
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext import window_scorer as ws  # noqa: E402
from aicext.common import EXT_ROOT, REGISTRY_DIR, read_jsonl, sha256_file  # noqa: E402
from aicext.holdout import current_ledger  # noqa: E402
from aicext.ids import split_bucket  # noqa: E402
from aicext.release import ReleaseWriter, sha256_many  # noqa: E402

DEV_SALT = "release_dev_v1"
DEV_FRACTION = 0.10
MAX_END_GAP = 3  # YTH: last clip end must match the decoded frame count within 3 frames
TRAIN_LIKE = {"train"}
EXPOSED = {"holdout_dev_exposed", "holdout_confirm_exposed"}


def rj(p):
    return list(read_jsonl(p))


def dev_bucket(g: str) -> float:
    return split_bucket(g, salt=DEV_SALT)


def release_split(m: dict) -> tuple[str, str]:
    """Map the project split of one media record to a release split."""
    ps, why = m.get("project_split"), m.get("project_split_reason") or ""
    if ps is None:
        raise SystemExit(f"{m['uid']}: project_split is null; run scripts/refresh_all.sh first")
    if ps == "holdout_confirm_reserved":
        return "confirmation", why
    if ps in EXPOSED:
        return "exposed_eval", why
    if ps == "quarantine":
        return "excluded", f"quarantine: {why}"
    if ps == "train":
        if dev_bucket(m["group_id"]) < DEV_FRACTION:
            return "dev", f"{DEV_SALT} bucket < {DEV_FRACTION} of unexposed train groups; {why}".strip("; ")
        return "train", why
    if ps == "val":
        # official eval split (or hash val) that no experiment has used yet
        return "confirmation", f"unexposed evaluation split: {why}"
    return "excluded", f"project_split={ps}: {why}"


def git_state() -> dict:
    repo = Path(__file__).resolve().parents[2]
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        head = None
    return {"repo_head": head or None, "note": "ext_data is not committed; code identity = code_tree_sha256"}


def base_meta(kind: str, extra: dict) -> dict:
    led = current_ledger()
    return {"kind": kind, "ext_root": str(EXT_ROOT), "holdout_ledger_version": (led or {}).get("version"),
            "holdout_ledger_sha256": sha256_file(REGISTRY_DIR / f"holdout_exposure_{led['version']}.json") if led else None,
            "registry_source_ledger_sha256": sha256_file(REGISTRY_DIR / "source_ledger.jsonl"),
            "split_rule": {"train/dev": f"project_split=train; dev = sha256('{DEV_SALT}:'+group) bucket < {DEV_FRACTION}",
                           "confirmation": "project_split=holdout_confirm_reserved (fresh reserve frozen in the ledger)",
                           "exposed_eval": "project_split in holdout_dev_exposed / holdout_confirm_exposed: already used "
                                           "for validation, never train, not a fresh confirmation",
                           "excluded": "quarantine or other; reason per row"},
            "git": git_state(), **extra}


# ======================================================================= spatial
def build_spatial(name: str) -> Path:
    rw = ReleaseWriter(name)
    manifests = collections.defaultdict(list)
    excluded, checks = [], collections.Counter()
    # ------------------------------------------------ RetargetVid (dense, 6 annotators, 1:3 / 3:1)
    rvm = {m["uid"]: m for m in rj(EXT_ROOT / "RetargetVid/processed/media.jsonl")}
    rva = rj(EXT_ROOT / "RetargetVid/processed/annotations.jsonl")
    media_sha = sha256_many([m["media_path"] for m in rvm.values()])
    by_unit = collections.defaultdict(list)
    for a in rva:
        by_unit[(a["media_uid"], a["variant"])].append(a)
    for (muid, ratio), anns in sorted(by_unit.items()):
        m = rvm[muid]
        vid = m["item_id"]
        anns = sorted(anns, key=lambda a: a["annotator"])
        raws, hashes = [], {}
        for a in anns:
            rel = f"raw_annotations/RetargetVid/{a['annotator']}/{vid}_{ratio}.txt"
            hashes[a["annotator"]] = rw.copy(a["annotation_path"], rel)
            raws.append(np.loadtxt(a["annotation_path"], delimiter=",", ndmin=2))
        K = max(len(r) for r in raws)
        ltrb = np.full((len(anns), K, 4), np.nan)
        for i, r in enumerate(raws):
            ltrb[i, :len(r)] = r
        W, H, N = m["width"], m["height"], m["frame_count"]
        checks["rv_units"] += 1
        checks["rv_boxes"] += int(np.isfinite(ltrb).all(-1).sum())
        checks["rv_len_mismatch"] += int(any(len(r) != N for r in raws))
        w, h = ltrb[..., 2] - ltrb[..., 0], ltrb[..., 3] - ltrb[..., 1]
        mw, mh, axis = ws.max_window(W, H, ratio)
        checks["rv_not_max_window_px>1"] += int(np.nansum((np.abs(w - mw) > 1) | (np.abs(h - mh) > 1)))
        checks["rv_out_of_bounds"] += int(np.nansum((ltrb[..., 0] < 0) | (ltrb[..., 1] < 0) | (ltrb[..., 2] > W) | (ltrb[..., 3] > H)))
        packed = f"packed/RetargetVid/{vid}_{ratio}.npz"
        rw.npz(packed, ltrb_raw=ltrb.astype(np.float32), frame_idx=np.arange(K, dtype=np.int32),
               annotators=np.array([a["annotator"] for a in anns]))
        tl = f"timelines/RetargetVid/{vid}.npz"
        if not rw.p(tl).exists():
            rw.copy(m["timeline_path"], tl)
        split, why = release_split(m)
        row = {"unit_id": f"retargetvid:{vid}:{ratio}", "dataset": "RetargetVid", "group_id": m["group_id"],
               "source_video_id": m["source_video_id"], "media_path": m["media_path"],
               "media_sha256": media_sha[m["media_path"]], "timeline": tl, "width": W, "height": H,
               "frame_count": N, "fps": m["fps"], "rotation": m["rotation"],
               "target_ratio_wh": [int(x) for x in ratio.split("-")], "ratio_key": ratio,
               "free_axis": {0: "x", 1: "y", None: None}[axis], "max_window_wh": [mw, mh],
               "gt_is_max_window": True, "coord_version": ws.COORD_VERSION,
               "human_annotation": {"packed": packed, "density": "dense", "annotators": len(anns),
                                    "annotated_frames": K, "raw_files_sha256": hashes,
                                    "raw_format": "txt 'left,top,right,bottom' per line; line i <-> frame i-1",
                                    "label_version": "RetargetVid@43673dd (annotations_all)"},
               "derived_annotation": None, "official_split": m["official_split"], "aic_split": m.get("aic_split"),
               "project_split": m.get("project_split"), "exposure_roles": m.get("exposure_roles"),
               "split_reason": why}
        (excluded.append(row | {"release_split": split}) if split == "excluded" else manifests[split].append(row))
    # ------------------------------------------------ LIVE-YT-VC (sparse, 1 annotator per frame, 9:16)
    lm = {m["uid"]: m for m in rj(EXT_ROOT / "LIVE_YT_VC/processed/media.jsonl")}
    la = rj(EXT_ROOT / "LIVE_YT_VC/processed/annotations.jsonl")
    sparse = {a["media_uid"]: a for a in la if a["annotation_type"] == "crop_box_sparse"}
    derived = {a["media_uid"]: a for a in la if a["annotation_type"] == "crop_box_derived"}
    csvp = next((EXT_ROOT / "LIVE_YT_VC/annotations/box").rglob("video_bbox_labels.csv"))
    csv_sha = rw.copy(csvp, "raw_annotations/LIVE_YT_VC/video_bbox_labels.csv")
    lm_sha = sha256_many([m["media_path"] for m in lm.values() if m.get("media_path")])
    for muid, a in sorted(sparse.items(), key=lambda kv: kv[0]):
        m = lm[muid]
        d = json.loads(Path(a["annotation_path"]).read_text())
        raw = d["raw_ltrb"]
        ltrb = np.array([[[x["left"], x["top"], x["right"], x["bottom"]] for x in raw]], float)
        fidx = np.asarray(d["frames"], np.int32)
        W, H, N = m["width"], m["height"], m["frame_count"]
        # cross-check: ingest xywh == raw ltrb converted without clipping
        xywh = np.asarray(d["boxes_xywh"], float)
        conv = np.column_stack([ltrb[0, :, 0], ltrb[0, :, 1], ltrb[0, :, 2] - ltrb[0, :, 0], ltrb[0, :, 3] - ltrb[0, :, 1]])
        checks["live_ingest_vs_raw_maxabs"] = max(checks["live_ingest_vs_raw_maxabs"], float(np.abs(conv - xywh).max()))
        checks["live_units"] += 1
        checks["live_boxes"] += len(raw)
        checks["live_frames_beyond_video"] += int((fidx >= N).sum())
        checks["live_out_of_bounds"] += int(((ltrb[0, :, 0] < 0) | (ltrb[0, :, 1] < 0) | (ltrb[0, :, 2] > W) | (ltrb[0, :, 3] > H)).sum())
        vid = m["item_id"]
        packed = f"packed/LIVE_YT_VC/{vid}.npz"
        rw.npz(packed, ltrb_raw=ltrb.astype(np.float32), frame_idx=fidx, annotators=np.array(["subject_per_frame"]))
        tl = f"timelines/LIVE_YT_VC/{vid}.npz"
        rw.copy(m["timeline_path"], tl)
        der = None
        if muid in derived:
            z = np.load(derived[muid]["annotation_path"])
            dp = f"derived/LIVE_YT_VC/{vid}_interp_v1.npz"
            rw.npz(dp, boxes_xywh=z["boxes_xywh"].astype(np.float32), valid=z["valid"], anchor_frames=z["anchor_frames"])
            der = {"packed": dp, "version": "interp_v1", "source": "derived",
                   "derivation": derived[muid]["derivation"], "derived_from": derived[muid]["derived_from"]}
        mw, mh, axis = ws.max_window(W, H, [9, 16])
        split, why = release_split(m)
        row = {"unit_id": f"liveytvc:{vid}", "dataset": "LIVE_YT_VC", "group_id": m["group_id"],
               "source_video_id": m["source_video_id"], "media_path": m["media_path"],
               "media_sha256": lm_sha.get(m["media_path"]), "timeline": tl, "width": W, "height": H,
               "frame_count": N, "fps": m["fps"], "rotation": m["rotation"], "target_ratio_wh": [9, 16],
               "ratio_key": "9-16", "free_axis": {0: "x", 1: "y", None: None}[axis], "max_window_wh": [mw, mh],
               "gt_is_max_window": False, "num_scenes": d["num_scenes"], "coord_version": ws.COORD_VERSION,
               "human_annotation": {"packed": packed, "density": "sparse", "annotators": 1,
                                    "annotated_frames": int(len(fidx)), "raw_csv_sha256": csv_sha,
                                    "raw_format": "csv column frameN = {'left','top','right','bottom'} (int px)",
                                    "label_version": "LIVE-YT-VC base (Box), repo@2a91e94"},
               "derived_annotation": der, "official_split": m["official_split"], "aic_split": m.get("aic_split"),
               "project_split": m.get("project_split"), "exposure_roles": m.get("exposure_roles"),
               "split_reason": why}
        (excluded.append(row | {"release_split": split}) if split == "excluded" else manifests[split].append(row))
    for sp, rows in manifests.items():
        rw.jsonl(f"manifests/{sp}.jsonl", rows)
    rw.jsonl("manifests/excluded.jsonl", excluded)
    counts = {sp: dict(collections.Counter(r["dataset"] for r in rows)) for sp, rows in manifests.items()}
    groups = {sp: len({r["group_id"] for r in rows}) for sp, rows in manifests.items()}
    leak = _cross_split_groups(manifests)
    if leak:
        raise SystemExit(f"group leakage across release splits: {leak[:5]}")
    code = rw.snapshot_code()
    rw.copy(REGISTRY_DIR / f"holdout_exposure_{current_ledger()['version']}.json", "holdout_exposure.json")
    rw.json("quality/build_checks.json", dict(checks))
    meta = base_meta("spatial_crop", {
        "datasets": {"RetargetVid": {"supervision": "crop_box_dense (human, 6 annotators, every frame)",
                                     "ratios": ["1:3", "3:1"], "gt_is_max_window": True},
                     "LIVE_YT_VC": {"supervision": "crop_box_sparse (human, 30 frames, 1 subject per frame)",
                                    "ratio": "9:16 (variable size)", "gt_is_max_window": False,
                                    "derived": "interp_v1 under derived/ (single-scene videos only), never mixed "
                                               "with human boxes"}},
        "not_included": {"GAICD": "image crop candidates + MOS, separate auxiliary release (not a video crop target)"},
        "coord_version": ws.COORD_VERSION, "scorer_version": ws.SCORER_VERSION,
        "iou_conventions": {"default": "halfopen", "teacher_compat": "inclusive_plus1 (RetargetVid upstream)"},
        "counts": counts, "groups": groups, "excluded": len(excluded),
        "media": "referenced by path + sha256, not copied", "code_tree_sha256": code["tree_sha256"]})
    return rw.finalize(meta)


def _cross_split_groups(manifests: dict) -> list:
    seen = collections.defaultdict(set)
    for sp, rows in manifests.items():
        for r in rows:
            seen[r["group_id"]].add(sp)
    return [g for g, s in seen.items() if len(s) > 1]


# ======================================================================= temporal
def build_temporal(name: str) -> Path:
    rw = ReleaseWriter(name)
    X = EXT_ROOT / "YouTubeHighlights"
    ym = {m["uid"]: m for m in rj(X / "processed/media.jsonl")}
    ya = rj(X / "processed/annotations.jsonl")
    has_mt = {a["media_uid"] for a in ya if a["variant"] == "mturk_votes"}
    has_match = {a["media_uid"] for a in ya if a["variant"] == "match_label"}
    up = X / "annotations/upstream_repo"
    mir = X / "raw/mirror/youtube_highlights_full"
    warn = {(w["tag"], w["video_id"]): w for w in json.loads((mir / "annotation_alignment_warnings.json").read_text())}
    media_sha = sha256_many([m["media_path"] for m in ym.values() if m.get("media_path") and m["download_status"] == "verified"])
    manifests, excluded = collections.defaultdict(list), []
    inter = collections.Counter()
    for uid, m in sorted(ym.items()):
        dom, vid = m["item_id"].split("/")
        reasons = []
        if m["download_status"] != "verified" or not m.get("media_path"):
            reasons.append("no_media (not in mirror; yt-dlp removed/private)")
        if uid not in has_match:
            reasons.append("no_annotation")
        mis = "misaligned_suspect" in (m.get("anomalies") or [])
        if mis:
            reasons.append("misaligned_suspect (label frames beyond decoded media)")
        if (dom, vid) in warn and not mis:
            reasons.append("mirror_alignment_warning (annotation_frame_count_mismatch)")
        split, why = release_split(m)
        inter[(m["official_split"], m.get("project_split"), uid in has_mt, "ok" if not reasons else reasons[0].split(" ")[0])] += 1
        vdir = up / dom / vid
        clips = json.loads((vdir / "clip.json").read_text()) if (vdir / "clip.json").exists() else None
        if clips is None:
            reasons.append("no_clip_json")
        base = {"unit_id": f"yth:{dom}:{vid}", "dataset": "YouTubeHighlights", "group_id": m["group_id"],
                "domain": dom, "youtube_id": vid, "official_split": m["official_split"],
                "aic_split": m.get("aic_split"), "project_split": m.get("project_split"),
                "exposure_roles": m.get("exposure_roles"), "split_reason": why, "has_mturk": uid in has_mt,
                "domain_relevant_manual_selection": json.loads(m.get("notes") or "{}").get("domain_relevant_manual_selection")}
        if reasons or split == "excluded":
            excluded.append(base | {"release_split": split, "exclude_reasons": reasons or [why]})
            continue
        raw_rel = {}
        for fn in ("clip.json", "match_label.json", "mturk_label.json", "info.json"):
            if (vdir / fn).exists():
                raw_rel[fn] = {"path": f"raw_annotations/YouTubeHighlights/{dom}/{vid}/{fn}",
                               "sha256": rw.copy(vdir / fn, f"raw_annotations/YouTubeHighlights/{dom}/{vid}/{fn}")}
        match = json.loads((vdir / "match_label.json").read_text())
        mt = json.loads((vdir / "mturk_label.json").read_text()) if (vdir / "mturk_label.json").exists() else None
        c = np.asarray(clips, float)
        if [list(map(float, x)) for x in match[0]] != c.tolist():
            excluded.append(base | {"release_split": split, "exclude_reasons": ["match_clip_list_differs_from_clip_json"]})
            continue
        votes = np.full(len(c), np.nan)
        if mt is not None:
            if [list(map(float, x)) for x in mt[0]] != c.tolist():
                excluded.append(base | {"release_split": split, "exclude_reasons": ["mturk_clip_list_differs"]})
                continue
            votes = np.asarray(mt[1], float)
        N = int(m["frame_count"])
        gap = float(c[-1, 1]) - N
        if abs(gap) > MAX_END_GAP:
            excluded.append(base | {"release_split": split,
                                    "exclude_reasons": [f"frame_count_mismatch (last label frame {c[-1, 1]:.0f} vs "
                                                        f"{N} decoded; |gap| > {MAX_END_GAP})"]})
            continue
        s = c[:, 0].astype(np.int64)
        e = np.minimum(c[:, 1].astype(np.int64), N)
        # the last clip of a video ends at the published end_frame, which may be N or N-1 (mirror == author)
        packed = f"packed/YouTubeHighlights/{dom}__{vid}.npz"
        rw.npz(packed, clip_start_idx=s, clip_end_idx=e, clip_end_frame_raw=c[:, 1], clip_mturk_votes=votes,
               clip_match_label=np.asarray(match[1], np.int8))
        tl = f"timelines/YouTubeHighlights/{dom}__{vid}.npz"
        rw.copy(m["timeline_path"], tl)
        row = base | {"media_path": m["media_path"], "media_sha256": media_sha[m["media_path"]], "timeline": tl,
                      "width": m["width"], "height": m["height"], "frame_count": N, "fps": m["fps"],
                      "rotation": m["rotation"], "duration_s": m["duration_s"], "packed": packed, "clips": int(len(c)),
                      "raw_annotations": raw_rel,
                      "label_version": "DomainSpecificHighlight@29083d2 (clip/match_label/mturk_label json)",
                      "media_version": "hf:jhanglee/youtube-highlights-full@b28d020",
                      "frame_mapping": "label frame f <-> decoded display frame f (0-based); clip = [start, end) "
                                       "clipped to frame_count; times from the real-PTS timeline"}
        manifests[split].append(row)
    for sp, rows in manifests.items():
        rw.jsonl(f"manifests/{sp}.jsonl", rows)
    rw.jsonl("manifests/excluded.jsonl", excluded)
    leak = _cross_split_groups(manifests)
    if leak:
        raise SystemExit(f"group leakage across release splits: {leak[:5]}")
    code = rw.snapshot_code()
    rw.copy(REGISTRY_DIR / f"holdout_exposure_{current_ledger()['version']}.json", "holdout_exposure.json")
    rw.json("quality/intersection_table.json", [{"official_split": k[0], "project_split": k[1], "has_mturk": k[2],
                                                 "status": k[3], "videos": v} for k, v in sorted(inter.items(), key=str)])
    counts = {sp: {"videos": len(rows), "with_mturk": sum(r["has_mturk"] for r in rows)} for sp, rows in manifests.items()}
    meta = base_meta("temporal_evidence", {
        "datasets": {"YouTubeHighlights": {
            "supervision_channels": {
                "human_mturk_votes": "raw soft vote count per ~2 s clip (MTurk, coverage-weighted, relative within "
                                     "video); explicit human selection only where votes > 0",
                "human_unselected": "votes == 0: nobody selected it; weak, NOT a reliable negative",
                "relative_preference": "pairs of clips of the same video ordered by votes (aicext.release.preference_pairs)",
                "auto_match": "match_label 1/0/-1 from edited-video matching; -1 = not reused by the uploader, NOT a "
                              "human negative",
                "unlabelled": "frames covered by no clip"},
            "vote_detail_available": "only per-clip soft totals are published; per-worker votes, number of workers "
                                     "and worker ids are not available"}},
        "not_included": {"MrHiSum": "separate feature-only release mrhisum_feat_subset_v1 (no raw video)",
                         "PHD2": "media sample only (40 videos)"},
        "counts": counts, "excluded": len(excluded), "code_tree_sha256": code["tree_sha256"]})
    return rw.finalize(meta)


# ======================================================================= Mr.HiSum feature subset
def build_mrhisum(name: str) -> Path:
    rw = ReleaseWriter(name)
    X = EXT_ROOT / "MrHiSum"
    mm = {m["uid"]: m for m in rj(X / "processed/media.jsonl")}
    ann = [a for a in rj(X / "processed/annotations.jsonl") if a["annotation_type"] == "temporal_score_1d"]
    shards = {r["shard"]: r for r in read_jsonl(X / "logs/yt8m_shards.jsonl") if r.get("status") == "done"}
    used_shards = collections.Counter()
    manifests, excluded = collections.defaultdict(list), []
    import h5py
    for a in ann:
        m = mm[a["media_uid"]]
        fp = (a.get("coverage") or {}).get("feature_path")
        split, why = release_split(m)
        base = {"video_id": a["coverage"]["h5_key"], "group_id": m["group_id"], "official_split": m["official_split"],
                "project_split": m.get("project_split"), "split_reason": why}
        if not fp:
            excluded.append(base | {"exclude_reasons": ["features_not_extracted_yet"]})
            continue
        shard = Path(fp).stem
        if shard not in shards:
            excluded.append(base | {"exclude_reasons": ["shard_not_in_done_ledger"]})
            continue
        if split == "excluded":
            excluded.append(base | {"exclude_reasons": [why]})
            continue
        used_shards[shard] += 1
        manifests[split].append(base | {"feature_h5": f"features/{shard}.h5", "labels_h5": "labels/mr_hisum.h5",
                                        "steps": a["coverage"]["steps"], "feature_steps": a["coverage"]["feature_steps"]})
    for shard in sorted(used_shards):
        src = X / "processed/yt8m" / f"{shard}.h5"
        with h5py.File(src, "r"):
            pass
        rw.copy(src, f"features/{shard}.h5")
    rw.copy(X / "annotations/mr_hisum.h5", "labels/mr_hisum.h5")
    rw.copy(X / "annotations/metadata.csv", "labels/metadata.csv")
    for sp, rows in manifests.items():
        rw.jsonl(f"manifests/{sp}.jsonl", rows)
    rw.jsonl("manifests/excluded.jsonl", excluded)
    rw.jsonl("shards.jsonl", [shards[s] for s in sorted(used_shards)])
    code = rw.snapshot_code()
    counts = {sp: len(rows) for sp, rows in manifests.items()}
    meta = base_meta("feature_only", {
        "use": "FEATURE HEAD ONLY: YouTube-8M v2 frame features (1 fps, 1024-d rgb PCA + 128-d audio, quantized) + "
               "Mr.HiSum gtscore. No raw video; not raw-video / VideoMAE / Qwen fine-tuning data and not counted as "
               "video training data.",
        "subset": f"fixed subset: the {len(used_shards)} shards completed at build time; later shards go into a new version",
        "labels": "gtscore = Most-Replayed aggregate (viewer behaviour, mixed/weak), gt_summary algorithmic",
        "counts": counts, "shards": len(used_shards), "excluded": len(excluded), "code_tree_sha256": code["tree_sha256"]})
    return rw.finalize(meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name")
    a = ap.parse_args()
    kind = a.name.rsplit("_v", 1)[0]
    fn = {"spatial_crop": build_spatial, "temporal_evidence": build_temporal, "mrhisum_feat_subset": build_mrhisum}[kind]
    out = fn(a.name)
    print(json.dumps({"release": str(out), "meta": json.loads((out / "RELEASE.json").read_text())}, indent=1,
                     ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
