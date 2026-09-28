#!/usr/bin/env python3
"""DAVSOD (Fan et al. CVPR'19) — video salient-object masks, validated against the released package.

Package (Google Drive, README "Updated link: 2021-02-18"): five zips
``DAVSOD-TrainingSet / DAVSOD-ValidationSet / Easy-35 / Normal-25 / Difficult-20``.
Each sequence ``select_<vid>[_k]/`` holds same-named PNGs in
``Imgs`` (RGB), ``GT_object_level`` (binary), ``GT_instance_level`` (palette ids),
``Fixation_maps`` (binary points), ``Fixation_maps_smoothed``, ``Fixation_records`` (.mat)
and ``Imgs_fused`` (visualisation).  File names are *source-video* frame numbers.

The paper (Sec. 3.1) says the stimuli come from DHF1K, and the README says the
released training set merges short sequences (61 vs 90 in the paper; package
187 sequences vs "226 videos, 23,938 frames" in the paper).  Counts below come
from the package itself.  ``select_<vid>`` -> DHF1K video ``<vid>`` is checked by
comparing pixels against the DHF1K AVI at frame ``name-1`` (DHF1K frame PNGs
are 1-based); only verified sequences join the ``dhf1k:<vid>`` leakage group and
take real timestamps from the DHF1K timeline, the rest keep a native group and a
nominal clock (labelled as such).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.common import (EXT_ROOT, dataset_dir, ensure_space, log, now, read_jsonl,  # noqa: E402
                           sha256_file, write_json, write_jsonl)
from aicext.ids import dhf1k_id, native_id  # noqa: E402
from aicext.media import load_timeline, read_frames  # noqa: E402
from aicext.schema import anomaly_record, annotation_record, media_record  # noqa: E402

DS = "DAVSOD"
REPO_COMMIT = "769c91e06c092df5127776758be1a017c2e31dc1"
PKGS = {  # zip -> (official split, Drive id, bytes)
    "DAVSOD-TrainingSet.zip": ("train", "1DnkirrC5R16DSrqEPkMMwY516GudVnYX", 3275931542),
    "DAVSOD-ValidationSet.zip": ("val", "1loE9_f947tpkrmM3MkRR04VFi7plJFUN", 2476015874),
    "Easy-35.zip": ("test_easy35", "1hwlb1t7S_Zahp6GSKW9qlRMQh7lMUhph", 2236362741),
    "Normal-25.zip": ("test_normal25", "1bclC888a-3yXc-yTVB5PPnwyPiPi2hNJ", 1535830992),
    "Difficult-20.zip": ("test_difficult20", "1ZHf3y96tF7m92L08DnctI3uRh7h-gZo9", 1488447205),
}
SUBDIRS = ("Imgs", "GT_object_level", "GT_instance_level", "Fixation_maps", "Fixation_maps_smoothed",
           "Fixation_records", "Imgs_fused")
SEQ_RE = re.compile(r"^select_(\d{4})(?:_(\d+))?$")
MATCH_MAD = 12.0  # mean |diff| (0-255) below which a DAVSOD frame equals the DHF1K frame


def extract(root: Path, zname: str) -> Path:
    out = root / "raw" / "extracted" / zname[:-4]
    marker = out / ".extracted"
    if not marker.exists():
        z = root / "raw" / "archives" / zname
        ensure_space(int(1.1 * z.stat().st_size))
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
        marker.write_text(now() + "\n")
        log(DS, "extracted", zip=zname)
    return out


def dhf1k_media() -> dict[str, dict]:
    return {m["uid"]: m for m in read_jsonl(EXT_ROOT / "DHF1K" / "processed" / "media.jsonl")}


def check_source(seq_imgs: list[Path], m: dict) -> dict:
    """Compare 2 DAVSOD frames with the DHF1K AVI frame (name - 1, 0-based)."""
    import cv2

    picks = [seq_imgs[0], seq_imgs[len(seq_imgs) // 2]]
    nums = [int(p.stem) for p in picks]
    got = read_frames(m["media_path"], [n - 1 for n in nums])
    mads = []
    for p, n in zip(picks, nums):
        a = cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
        b = got.get(n - 1)
        if b is None or a.shape != b.shape:
            mads.append(None)
            continue
        mads.append(float(np.abs(a - b.astype(np.float32)).mean()))
    ok = all(x is not None and x < MATCH_MAD for x in mads)
    return {"frames": nums, "mad": mads, "match": ok}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true", help="index whichever zips are present")
    a = ap.parse_args()
    root = dataset_dir(DS)
    arch = root / "raw" / "archives"
    present = {z: v for z, v in PKGS.items() if (arch / z).exists() and (arch / z).stat().st_size == v[2]}
    if len(present) < len(PKGS) and not a.allow_partial:
        raise SystemExit(f"only {sorted(present)} complete; rerun later or pass --allow-partial")
    dh = dhf1k_media()
    media, annots, anomalies = [], [], []
    tl_dir = root / "processed" / "timelines"
    tl_dir.mkdir(parents=True, exist_ok=True)
    pkg_stats = {}
    for zname, (osplit, gid_drive, size) in present.items():
        base = extract(root, zname)
        seqs = sorted(p.parent for p in base.rglob("Imgs") if p.is_dir())
        pkg_stats[zname] = {"sequences": len(seqs), "frames": 0, "sha256": sha256_file(arch / zname)}
        for sd in seqs:
            mm = SEQ_RE.match(sd.name)
            imgs = sorted((sd / "Imgs").glob("*.png"))
            if not imgs:
                anomalies.append(anomaly_record(DS, str(sd), "empty_sequence", "no Imgs/*.png", "skipped"))
                continue
            pkg_stats[zname]["frames"] += len(imgs)
            counts = {s: len(list((sd / s).glob("*"))) if (sd / s).exists() else 0 for s in SUBDIRS}
            names = {s: {p.stem for p in (sd / s).glob("*")} for s in ("GT_object_level", "GT_instance_level",
                                                                       "Fixation_maps")}
            uid = f"davsod:{osplit}:{sd.name}"
            src = None
            if mm and dhf1k_id(mm.group(1)) in dh and dh[dhf1k_id(mm.group(1))].get("media_path"):
                src = check_source(imgs, dh[dhf1k_id(mm.group(1))])
            verified = bool(src and src["match"])
            gid = dhf1k_id(mm.group(1)) if verified else native_id(DS, sd.name)
            if mm and not verified:
                anomalies.append(anomaly_record(DS, uid, "source_not_verified", json.dumps(src),
                                                "native group id; nominal 30 fps clock"))
            # timeline: real DHF1K times at the source frames when verified, else nominal
            src_frames = np.asarray([int(p.stem) for p in imgs])
            if verified:
                dm = dh[gid]
                t_all = load_timeline(dm["timeline_path"])
                k = np.clip(src_frames - 1, 0, len(t_all["time_s"]) - 1)
                time_s, pts, method = t_all["time_s"][k], t_all["pts"][k], "dhf1k_source_timeline"
                fps = dm.get("fps")
            else:
                time_s = (src_frames - src_frames[0]) / 30.0
                pts, method, fps = np.arange(len(imgs)), "nominal_30fps_from_frame_names", 30.0
            tlp = tl_dir / f"{uid.replace(':', '_')}.npz"
            np.savez_compressed(tlp, pts=pts, time_s=time_s.astype(np.float64), time_base=np.array("source"),
                                method=np.array(method), source_frame=src_frames)
            if np.any(np.diff(src_frames) != 1):
                anomalies.append(anomaly_record(DS, uid, "non_contiguous_frames",
                                                f"gaps in source frame numbers {src_frames[:3]}...",
                                                "kept; timeline follows real source frame numbers"))
            import cv2
            h, w = cv2.imread(str(imgs[0]), cv2.IMREAD_UNCHANGED).shape[:2]
            m = media_record(
                uid=uid, dataset=DS, release=f"Drive package 2021-02-18 ({zname}); repo @{REPO_COMMIT[:7]}",
                item_id=f"{osplit}/{sd.name}", source_video_id=gid, source_platform="youtube (via DHF1K)",
                group_id=gid, official_split=osplit, media_kind="frame_folder", media_path=str(sd / "Imgs"),
                media_reused_from=None, download_url=f"gdrive:{gid_drive}",
                download_source="author Google Drive (DAVSOD README)", download_status="verified",
                preprocess_status="timeline_ok", width=w, height=h, fps=fps, frame_count=len(imgs),
                duration_s=float(time_s[-1] - time_s[0]) + (1.0 / fps if fps else 0.0),
                timeline_path=str(tlp), timeline_method=method, frame_file_pattern="*.png",
                annotation_types=["salient_object_mask", "fixation_points", "saliency_map"],
                notes=json.dumps({"source_check": src, "subdir_counts": counts,
                                  "source_frames": [int(src_frames[0]), int(src_frames[-1])]}))
            media.append(m)
            idx_of = {p.stem: i for i, p in enumerate(imgs)}

            def cover(sub):
                files = sorted((sd / sub).glob("*.png"))
                keep = [(idx_of[p.stem], p.name) for p in files if p.stem in idx_of]
                extra = [p.name for p in files if p.stem not in idx_of]
                if extra:
                    anomalies.append(anomaly_record(DS, uid, "annotation_without_frame", f"{sub}: {extra[:3]}",
                                                    "ignored"))
                return keep

            for sub, var, atype, sem in (
                    ("GT_object_level", "object_level", "salient_object_mask",
                     "binary mask of the salient object(s) chosen from human fixations on this frame"),
                    ("GT_instance_level", "instance_level", "salient_object_mask",
                     "palette PNG: pixel value = salient instance id (0 = background)"),
                    ("Fixation_maps", "fixation_binary", "fixation_points",
                     "binary eye-fixation points (DHF1K eye-tracking records)"),
                    ("Fixation_maps_smoothed", "fixation_smoothed", "saliency_map",
                     "smoothed fixation density (continuous)")):
                keep = cover(sub)
                if not keep:
                    continue
                annots.append(annotation_record(
                    uid=f"{uid}:{var}", dataset=DS, media_uid=uid, annotation_type=atype,
                    annotation_source="human" if atype == "salient_object_mask" else "mixed",
                    annotator="DAVSOD annotators (fixation-guided)" if atype == "salient_object_mask"
                    else "DHF1K eye tracking, rendered by DAVSOD", variant=var,
                    annotation_path=str(sd / sub), annotation_format="png per frame, same name as Imgs",
                    coverage={"annotated_frame_idx": [k for k, _ in keep], "mask_files": [f for _, f in keep],
                              "map_files": [f for _, f in keep], "annotated_frames": len(keep),
                              "sequence_frames": len(imgs)},
                    label_semantics=sem,
                    coord_format="pixel masks aligned with Imgs (same size)", coord_target="same",
                    time_reference="file name = source (DHF1K) frame number, 1-based; index = position in Imgs",
                    valid_mask="only frames listed in coverage.annotated_frame_idx; others unlabelled"))
                if len(keep) != len(imgs):
                    anomalies.append(anomaly_record(DS, uid, "partial_annotation", f"{sub}: {len(keep)}/{len(imgs)}",
                                                    "valid mask = listed frames only"))
    write_jsonl(root / "processed" / "media.jsonl", media)
    write_jsonl(root / "processed" / "annotations.jsonl", annots)
    write_jsonl(root / "processed" / "anomalies.jsonl", anomalies)
    ver = sum(1 for m in media if m["group_id"].startswith("dhf1k:"))
    dh_ids = sorted({m["group_id"] for m in media if m["group_id"].startswith("dhf1k:")})
    dh_split = {}
    for g in dh_ids:
        n = int(g.split(":")[1])
        dh_split[g] = "train" if n <= 600 else ("val" if n <= 700 else "test")
    cross = {}
    for m in media:
        if m["group_id"] in dh_split:
            key = f"{m['official_split']} <- DHF1K {dh_split[m['group_id']]}"
            cross[key] = cross.get(key, 0) + 1
    write_json(root / "processed" / "dataset_card.json", {
        "dataset": DS, "release": "Drive packages (updated 2021-02-18)", "repo_commit": REPO_COMMIT,
        "source": "https://github.com/DengPingFan/DAVSOD", "packages": pkg_stats,
        "packages_missing": sorted(set(PKGS) - set(present)),
        "media_expected_from_packages": 187, "media_indexed": len(media),
        "frames_indexed": int(sum(m["frame_count"] for m in media)),
        "paper_counts": {"videos": 226, "frames": 23938, "train_sequences": 90},
        "package_vs_paper": "README: short training sequences merged -> 61 train sequences; package counts used",
        "dhf1k_source_verified": ver, "dhf1k_split_of_sources": cross,
        "official_split": "train / val / test_easy35 / test_normal25 / test_difficult20 (package folders)",
        "readiness": "frame_folder_trainable" if len(present) == len(PKGS) else "partial",
        "trainable_as": ["salient_object_mask", "fixation_points", "saliency_map"],
        "blockers": [] if len(present) == len(PKGS) else [f"missing {sorted(set(PKGS) - set(present))}"],
        "notes": "Frames shipped as PNG folders (kept as-is). Masks mark fixation-guided salient objects; "
                 "they are not generic 'main subject' labels. Same source videos as DHF1K -> shared groups.",
        "updated": now()})
    log(DS, "ingest_done", media=len(media), annotations=len(annots), dhf1k_verified=ver)
    print(json.dumps({"media": len(media), "annotations": len(annots), "dhf1k_verified": ver,
                      "packages": pkg_stats, "cross": cross}, ensure_ascii=False))


if __name__ == "__main__":
    main()
