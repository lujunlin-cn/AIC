#!/usr/bin/env python3
"""LIVE-YT Video Cropping (base release) acquisition + index.

Source: author Box share (github.com/steven413d/LIVE-YT-VideoCropping).
VC++ is "Coming soon" upstream and is not waited for.

Labels: ``video_bbox_labels.csv`` — per video, 30 sampled frames
(``frame0, frame6, ..., frame174``), each labelled by one human subject with a
portrait crop ``{'left','right','top','bottom'}`` in source pixels.  These are
*sparse* boxes, not a smooth trajectory; they are kept exactly as published
(``crop_box_sparse``).  A linear interpolation between annotated frames is
written separately as ``crop_box_derived`` and only for single-scene videos.
The boxes have variable size; nothing here turns them into a default
smaller-window target.

Stages (all resumable): list -> labels -> media -> index.
"""
from __future__ import annotations

import argparse
import ast
import csv
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.box import file_url, list_folder  # noqa: E402
from aicext.common import dataset_dir, log, now, read_jsonl, write_json, write_jsonl  # noqa: E402
from aicext.download import fetch, session  # noqa: E402
from aicext.probe_worker import probe_all  # noqa: E402
from aicext.schema import anomaly_record, annotation_record, media_record  # noqa: E402

HOST, SHARED, FOLDER = "utexas.app.box.com", "hylumfu8akjhdgdd4teynsyc6ickwv1j", 407721578256
DS = "LIVE_YT_VC"
REPO_COMMIT = "2a91e9492c092c966b10fdc9b369b057d92e8ffb"


def stage_list(root: Path) -> list[dict]:
    out = root / "annotations" / "box_listing.json"
    if out.exists():
        return json.loads(out.read_text())["items"]
    items = list_folder(HOST, SHARED, FOLDER)
    write_json(out, {"host": HOST, "shared_name": SHARED, "folder_id": FOLDER, "listed": now(), "items": items})
    log(DS, "listed", files=sum(i["type"] == "file" for i in items))
    return items


def stage_files(root: Path, items: list[dict], workers: int, videos: bool) -> None:
    sess = session()
    jobs = []
    for it in items:
        if it["type"] != "file" or it["path"].endswith(".DS_Store"):
            continue
        is_video = it["path"].lower().endswith(".mp4")
        if is_video != videos:
            continue
        dest = (root / "raw" / it["path"]) if is_video else (root / "annotations" / "box" / it["path"])
        jobs.append((it, dest))
    todo = [(it, d) for it, d in jobs if not (d.exists() and d.stat().st_size == it["size"])]
    log(DS, "files_stage", videos=videos, total=len(jobs), todo=len(todo))
    failed = []

    def one(it, dest):
        return fetch(file_url(HOST, SHARED, it["id"]), dest, dataset=DS, expected_size=it["size"],
                     retries=8, sess=sess, quiet=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(one, it, d): it for it, d in todo}
        for k, f in enumerate(as_completed(futs), 1):
            it = futs[f]
            try:
                f.result()
            except Exception as exc:
                failed.append({"path": it["path"], "id": it["id"], "error": repr(exc)[:300]})
            if k % 50 == 0:
                log(DS, "files_progress", done=k, of=len(todo), failed=len(failed))
    write_json(root / "logs" / f"failed_{'videos' if videos else 'labels'}.json", failed)


def parse_labels(csv_path: Path) -> dict[str, dict]:
    out = {}
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            frames, boxes, raw = [], [], []
            for k, v in row.items():
                if not k.startswith("frame"):
                    continue
                n = int(k[5:])
                if v is None or not v.strip():
                    continue
                d = ast.literal_eval(v)
                frames.append(n)
                raw.append(d)
                boxes.append([d["left"], d["top"], d["right"] - d["left"], d["bottom"] - d["top"]])
            order = np.argsort(frames)
            out[row["video"]] = {"num_scenes": int(row["num_scenes"]),
                                 "frames": [frames[i] for i in order],
                                 "boxes_xywh": [boxes[i] for i in order],
                                 "raw_ltrb": [raw[i] for i in order]}
    return out


def interpolate(frames: list[int], boxes: list[list[float]], n_frames: int) -> tuple[np.ndarray, np.ndarray]:
    f = np.asarray(frames, float)
    b = np.asarray(boxes, float)
    t = np.arange(n_frames, dtype=float)
    out = np.stack([np.interp(t, f, b[:, j]) for j in range(4)], 1)
    valid = (t >= f.min()) & (t <= f.max())
    return out, valid


def stage_index(root: Path, workers: int) -> None:
    labels_csv = next((root / "annotations" / "box").rglob("video_bbox_labels.csv"), None)
    if labels_csv is None:
        raise FileNotFoundError("video_bbox_labels.csv not downloaded")
    labels = parse_labels(labels_csv)
    listing = {Path(i["path"]).stem: i for i in json.loads((root / "annotations" / "box_listing.json").read_text())["items"]
               if i["type"] == "file" and i["path"].endswith(".mp4")}
    media, annots, anomalies = [], [], []
    for name in sorted(set(labels) | set(listing), key=lambda s: (len(s), s)):
        it = listing.get(name)
        path = root / "raw" / it["path"] if it else None
        ok = bool(path and path.exists() and path.stat().st_size == it["size"])
        media.append(media_record(
            uid=f"liveytvc:{name}", dataset=DS, release=f"LIVE-YT-VC base (Box share, repo@{REPO_COMMIT[:7]})",
            item_id=name, source_video_id=f"liveytvc:{name}", source_platform="LSVQ/YT-UGC (anonymised name)",
            group_id=f"liveytvc:{name}", official_split="none_published", media_kind="video_file",
            media_path=str(path) if ok else None, download_url=file_url(HOST, SHARED, it["id"]) if it else None,
            download_source="author Box share", file_size=it["size"] if it else None,
            download_status="downloaded" if ok else ("missing" if it else "failed"),
            annotation_types=["crop_box_sparse"]))
        if not it:
            anomalies.append(anomaly_record(DS, name, "missing_media", "label row without video in Box listing", "excluded"))
        elif not ok:
            anomalies.append(anomaly_record(DS, name, "missing_media", "not downloaded yet", "excluded until present"))
        if name not in labels:
            anomalies.append(anomaly_record(DS, name, "missing_annotation", "video without label row", "media only"))
    res = probe_all([(m["uid"], m["media_path"]) for m in media if m["media_path"]], root, workers)
    ann_dir = root / "annotations" / "sparse"
    der_dir = root / "processed" / "derived_interp_v1"
    ann_dir.mkdir(parents=True, exist_ok=True)
    der_dir.mkdir(parents=True, exist_ok=True)
    for m in media:
        r = res.get(m["uid"])
        name = m["item_id"]
        if r and r.get("ok"):
            m.update(width=r["width"], height=r["height"], fps=r.get("fps_avg"), frame_count=r["frame_count"],
                     duration_s=r["duration_s"], rotation=r["rotation"], codec=r["codec"], has_audio=r["has_audio"],
                     timeline_path=r["timeline_path"], timeline_method=r["method"],
                     preprocess_status="timeline_ok", download_status="verified")
            if r.get("rotation"):
                anomalies.append(anomaly_record(DS, m["uid"], "rotation_metadata", f"rotation={r['rotation']}",
                                                "coordinates kept in coded pixels"))
        elif r:
            m.update(download_status="failed", preprocess_status="failed")
            anomalies.append(anomaly_record(DS, m["uid"], "decode_failed", r.get("error", ""), "excluded"))
        lab = labels.get(name)
        if not lab:
            continue
        W, H, N = m.get("width"), m.get("height"), m.get("frame_count")
        b = np.asarray(lab["boxes_xywh"], float)
        issues = {}
        if W and H:
            oob = (b[:, 0] < 0) | (b[:, 1] < 0) | (b[:, 0] + b[:, 2] > W) | (b[:, 1] + b[:, 3] > H)
            if oob.any():
                issues["out_of_bounds_boxes"] = int(oob.sum())
        if N and max(lab["frames"]) >= N:
            issues["frames_beyond_video"] = int(sum(f >= N for f in lab["frames"]))
        ar = b[:, 2] / np.maximum(b[:, 3], 1)
        for k, v in issues.items():
            anomalies.append(anomaly_record(DS, m["uid"], k, f"{v} of {len(b)} boxes",
                                            "kept as published; adapter valid mask excludes frames beyond video"))
        ann_path = ann_dir / f"{name}.json"
        write_json(ann_path, {"video": name, **lab, "coord_format": "source pixels; left/right/top/bottom as published",
                              "frame_index_note": "frameN = 0-based decoded frame index N (verified against decode)"})
        annots.append(annotation_record(
            uid=f"liveytvc:{name}:sparse", dataset=DS, media_uid=m["uid"], annotation_type="crop_box_sparse",
            annotation_source="human", annotator="one subject per frame (identity not released)",
            variant="portrait", annotation_path=str(ann_path), annotation_format="json frames[] + boxes_xywh[] + raw_ltrb[]",
            coverage={"annotated_frames": len(lab["frames"]), "frame_ids": [lab["frames"][0], lab["frames"][-1]],
                      "step": 6, "video_frames": N, "num_scenes": lab["num_scenes"],
                      "aspect_w_over_h_median": float(np.median(ar))},
            label_semantics="human-drawn portrait crop region the subject would keep; variable size (not max window)",
            coord_format="{'left','right','top','bottom'} source pixels",
            coord_target="xywh pixels (x=left, y=top, w=right-left, h=bottom-top); normalized by (W,H)",
            time_reference="frameN -> decoded display frame N (0-based); time from timeline npz",
            valid_mask="only the 30 annotated frames are valid", notes=json.dumps(issues) if issues else None))
        if lab["num_scenes"] == 1 and N:
            interp, valid = interpolate(lab["frames"], lab["boxes_xywh"], N)
            dpath = der_dir / f"{name}.npz"
            np.savez_compressed(dpath, boxes_xywh=interp.astype(np.float32), valid=valid,
                                anchor_frames=np.asarray(lab["frames"]))
            annots.append(annotation_record(
                uid=f"liveytvc:{name}:interp_v1", dataset=DS, media_uid=m["uid"], annotation_type="crop_box_derived",
                annotation_source="derived", variant="portrait", annotation_path=str(dpath),
                annotation_format="npz boxes_xywh[T,4], valid[T], anchor_frames",
                coverage={"valid_frames": int(valid.sum()), "video_frames": N},
                label_semantics="per-frame linear interpolation of the 30 sparse human boxes (x,y,w,h independently)",
                coord_format="xywh pixels", coord_target="xywh pixels",
                time_reference="index = decoded display frame", valid_mask="valid between first and last anchor",
                derived_from=f"liveytvc:{name}:sparse", derivation="interp_v1: numpy.interp per coordinate; "
                "single-scene videos only; no smoothing; not human ground truth"))
    write_jsonl(root / "processed" / "media.jsonl", media)
    write_jsonl(root / "processed" / "annotations.jsonl", annots)
    write_jsonl(root / "processed" / "anomalies.jsonl", anomalies)
    ver = sum(m["download_status"] == "verified" for m in media)
    write_json(root / "processed" / "dataset_card.json", {
        "dataset": DS, "release": f"LIVE-YT-VC base release via Box; repo {REPO_COMMIT}",
        "source": "https://github.com/steven413d/LIVE-YT-VideoCropping",
        "media_expected": len(listing), "media_verified": ver, "label_rows": len(labels),
        "official_split": "none published; aic_split assigned by source-video hash",
        "readiness": "raw_video_trainable" if ver == len(listing) and labels else "partial",
        "trainable_as": ["crop_box_sparse"] + (["crop_box_derived"] if labels else []),
        "blockers": [] if ver == len(listing) else [f"{len(listing) - ver} videos not yet verified"],
        "notes": "VC++ upstream 'Coming soon' (not waited for). Boxes are sparse (30 frames, step 6), variable size.",
        "updated": now()})
    log(DS, "index_done", media=len(media), verified=ver, annotations=len(annots))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["list", "labels", "media", "index", "all"], default="all")
    ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    root = dataset_dir(DS)
    items = stage_list(root)
    if a.stage in ("labels", "all"):
        stage_files(root, items, a.workers, videos=False)
    if a.stage in ("media", "all"):
        stage_files(root, items, a.workers, videos=True)
    if a.stage in ("index", "all"):
        stage_index(root, 4)


if __name__ == "__main__":
    main()
