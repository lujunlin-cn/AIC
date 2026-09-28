#!/usr/bin/env python3
"""DHF1K + RetargetVid: shared media, unified source IDs, separate supervision.

Media
  * DHF1K ``video.rar`` (author Google Drive) is extracted once into
    ``DHF1K/raw/video/NNN.AVI``; existing ``/data/aic/datasets/DHF1K`` subsets are
    only compared by hash (no re-download of what the full archive replaces).
  * Both datasets reference the same ``media_uid = dhf1k:NNN``.

Annotations
  * DHF1K ``annotation.rar``: per video ``fixation/`` (binary fixation-point
    maps) and ``maps/`` (continuous saliency density maps) — kept as two
    annotation types. Test videos 701-1000 have no public annotation.
  * RetargetVid: 6 annotators x {1-3, 3-1} dense per-frame human crop windows,
    ``left,top,right,bottom`` zero-based, referenced from the existing clone.

Re-running is safe: every stage checks for its outputs first.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.common import (EXT_ROOT, dataset_dir, ensure_space, log, now, read_jsonl,  # noqa: E402
                           sha256_file, write_json, write_jsonl)
from aicext.ids import dhf1k_id  # noqa: E402
from aicext.probe_worker import probe_all  # noqa: E402
from aicext.schema import anomaly_record, annotation_record, media_record  # noqa: E402

EXISTING_DHF1K = Path("/data/aic/datasets/DHF1K")
EXISTING_RV = Path("/data/aic/datasets/RetargetVid")
RV_COMMIT = "43673dd83b279c4aedeeea22f32d03582ac45194"
DHF1K_VIDEO_GDRIVE = "1UEFQmRdDbtVT-ePjMZVrv9oVV0ra631s"


def official_split(n: int) -> str:
    # DHF1K README: 600 train, 100 val, 300 test (test annotations withheld).
    return "train" if n <= 600 else ("val" if n <= 700 else "test")


def extract_rar(archive: Path, dest: Path, members: str = "*") -> None:
    dest.mkdir(parents=True, exist_ok=True)
    size = archive.stat().st_size
    ensure_space(int(size * 1.15))
    subprocess.run(["nice", "-n", "15", "ionice", "-c2", "-n7", "unrar", "x", "-o-", "-idq", str(archive),
                    members, str(dest) + "/"], check=True)


def stage_extract(root: Path) -> None:
    arc = root / "raw" / "archives" / "video.rar"
    out = root / "raw"
    marker = out / ".video_extracted"
    if marker.exists():
        return
    if not arc.exists():
        log("DHF1K", "video_archive_missing", path=str(arc))
        return
    log("DHF1K", "extract_start", archive=str(arc))
    extract_rar(arc, out)
    marker.write_text(now() + "\n")
    log("DHF1K", "extract_done", files=len(list((out / "video").glob("*.AVI"))) if (out / "video").exists() else 0)


def stage_extract_annotation(root: Path) -> None:
    arc = root / "raw" / "archives" / "annotation.rar"
    out = root / "annotations"
    marker = out / ".annotation_extracted"
    if marker.exists() or not arc.exists():
        return
    log("DHF1K", "extract_annotation_start", archive=str(arc))
    extract_rar(arc, out)
    marker.write_text(now() + "\n")


def find_video_dir(root: Path) -> Path | None:
    for cand in (root / "raw" / "video", root / "raw"):
        if cand.exists() and any(cand.glob("*.AVI")):
            return cand
    return None


def find_annotation_root(root: Path) -> Path | None:
    base = root / "annotations"
    for cand in [base / "annotation", base]:
        if cand.exists() and any(p.is_dir() and re.fullmatch(r"\d{4}|\d{3}", p.name) for p in cand.iterdir()):
            return cand
    return None


def rv_files() -> dict[tuple[int, str, int], Path]:
    out = {}
    for p in (EXISTING_RV / "annotations_all").glob("annotator_*/*.txt"):
        m = re.fullmatch(r"(\d{3})_(1-3|3-1)", p.stem)
        if m:
            out[(int(m.group(1)), m.group(2), int(p.parent.name.split("_")[1]))] = p
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-probe", action="store_true")
    a = ap.parse_args()

    droot = dataset_dir("DHF1K")
    rroot = dataset_dir("RetargetVid")
    stage_extract(droot)
    stage_extract_annotation(droot)
    vdir = find_video_dir(droot)
    ann_root = find_annotation_root(droot)
    anomalies_d, anomalies_r = [], []

    # ---- media (shared) ----
    media = []
    ids = range(1, 1001)
    for n in ids:
        path = vdir / f"{n:03d}.AVI" if vdir else None
        reused = None
        if path is None or not path.exists():
            # fall back to previously extracted subsets so partial state is still usable
            for sub in ("subset20", "subset21_30"):
                p = EXISTING_DHF1K / sub / f"{n:03d}.AVI"
                if p.exists():
                    path, reused = p, str(p)
                    break
            else:
                path = None
        m = media_record(
            uid=dhf1k_id(n), dataset="DHF1K", release="DHF1K author release (video.rar, Google Drive)",
            item_id=f"{n:03d}", source_video_id=dhf1k_id(n), source_platform="dataset_native",
            group_id=dhf1k_id(n), official_split=official_split(n), media_kind="video_file",
            media_path=str(path) if path else None, media_reused_from=reused,
            download_url=f"https://drive.google.com/file/d/{DHF1K_VIDEO_GDRIVE}",
            download_source="author Google Drive (github.com/wenguanwang/DHF1K)",
            download_status="downloaded" if path else "missing",
        )
        if path:
            m["file_size"] = path.stat().st_size
        media.append(m)
        if not path:
            anomalies_d.append(anomaly_record("DHF1K", dhf1k_id(n), "missing_media",
                                              "video not extracted yet", "excluded until present"))

    # ---- probe + timeline ----
    if not a.skip_probe:
        res = probe_all([(m["uid"], m["media_path"]) for m in media if m["media_path"]], droot, a.workers)
        for m in media:
            r = res.get(m["uid"])
            if not r:
                continue
            if r.get("ok"):
                m.update(width=r["width"], height=r["height"], fps=r.get("fps_avg"), frame_count=r["frame_count"],
                         duration_s=r["duration_s"], rotation=r["rotation"], codec=r["codec"],
                         has_audio=r["has_audio"], timeline_path=r["timeline_path"],
                         timeline_method=r["method"], preprocess_status="timeline_ok", download_status="verified")
                if r.get("container_nb_frames") and r["container_nb_frames"] != r["frame_count"]:
                    anomalies_d.append(anomaly_record("DHF1K", m["uid"], "frame_count_mismatch",
                                                      f"container {r['container_nb_frames']} vs decoded {r['frame_count']}",
                                                      "decoded count used as truth"))
                if r.get("non_increasing"):
                    anomalies_d.append(anomaly_record("DHF1K", m["uid"], "non_monotonic_pts",
                                                      f"{r['non_increasing']} steps", "kept; timeline sorted by decode order"))
            else:
                m.update(preprocess_status="failed", download_status="failed")
                anomalies_d.append(anomaly_record("DHF1K", m["uid"], "decode_failed", r.get("error", ""), "excluded"))
    by_uid = {m["uid"]: m for m in media}

    # ---- hash check against previously extracted subsets (dedupe evidence) ----
    for sub in ("subset20", "subset21_30"):
        for p in sorted((EXISTING_DHF1K / sub).glob("*.AVI")):
            uid = dhf1k_id(int(p.stem))
            m = by_uid[uid]
            if m["media_reused_from"] or not m["media_path"]:
                continue
            if m.get("sha256") is None:
                m["sha256"] = sha256_file(m["media_path"])
            old = sha256_file(p)
            if old != m["sha256"]:
                anomalies_d.append(anomaly_record("DHF1K", uid, "hash_mismatch_existing",
                                                  f"{p} differs from full-archive copy", "full-archive copy used"))

    # ---- DHF1K annotations: fixation vs saliency maps ----
    annots_d = []
    for n in ids:
        uid = dhf1k_id(n)
        m = by_uid[uid]
        if n > 700:
            continue  # official test: labels withheld by authors
        vdir_a = None
        if ann_root:
            for name in (f"{n:04d}", f"{n:03d}"):
                if (ann_root / name).exists():
                    vdir_a = ann_root / name
                    break
        if vdir_a is None:
            if ann_root:
                anomalies_d.append(anomaly_record("DHF1K", uid, "missing_annotation",
                                                  "no annotation folder", "saliency/fixation unavailable"))
            continue
        for sub, atype, sem in (("fixation", "fixation_points",
                                 "binary map, 1 at recorded eye fixation locations (17 observers pooled)"),
                                ("maps", "saliency_map",
                                 "continuous saliency density (fixations blurred with Gaussian), 8-bit PNG")):
            d = vdir_a / sub
            files = sorted(d.glob("*.png")) if d.exists() else []
            fc = m.get("frame_count")
            cov = {"annotated_frames": len(files), "video_frames": fc,
                   "fraction": (len(files) / fc) if fc else None,
                   "file_pattern": f"{sub}/%04d.png (1-based frame number)"}
            if fc and len(files) != fc:
                anomalies_d.append(anomaly_record("DHF1K", uid, "annotation_frame_count_mismatch",
                                                  f"{sub}: {len(files)} png vs {fc} decoded frames",
                                                  "adapter masks frames without a png"))
            annots_d.append(annotation_record(
                uid=f"{uid}:{atype}", dataset="DHF1K", media_uid=uid, annotation_type=atype,
                annotation_source="human", annotator="pooled_17_observers", annotation_path=str(d),
                annotation_format="png per frame", coverage=cov, label_semantics=sem,
                coord_format="pixel grid of the source frame (same W x H as video)",
                coord_target="HxW float32 in [0,1] (maps /255; fixation >0 -> 1)",
                time_reference="png k (1-based) <-> decoded frame k-1 (0-based)",
                valid_mask="frame valid iff png exists"))
        m["annotation_types"] = sorted({*m["annotation_types"], "fixation_points", "saliency_map"})

    # ---- RetargetVid annotations over the same media ----
    annots_r = []
    rv = rv_files()
    rv_media = []
    rv_ids = sorted({k[0] for k in rv})
    for n in rv_ids:
        uid = dhf1k_id(n)
        m = by_uid[uid]
        rm = dict(m)
        rm.update(uid=f"retargetvid:{n:03d}", dataset="RetargetVid",
                  release=f"RetargetVid github bmezaris/RetargetVid@{RV_COMMIT[:7]}",
                  media_reused_from=m["uid"], annotation_types=["crop_box_dense"],
                  notes="media shared with DHF1K; group_id identical to DHF1K")
        rv_media.append(rm)
        for ratio in ("1-3", "3-1"):
            for ann in range(1, 7):
                p = rv.get((n, ratio, ann))
                if p is None:
                    anomalies_r.append(anomaly_record("RetargetVid", uid, "missing_annotation",
                                                      f"annotator {ann} ratio {ratio}", "not used"))
                    continue
                lines = sum(1 for _ in p.open())
                fc = m.get("frame_count")
                if fc and lines != fc:
                    anomalies_r.append(anomaly_record("RetargetVid", uid, "annotation_frame_count_mismatch",
                                                      f"{p.name} annotator {ann}: {lines} lines vs {fc} frames",
                                                      "adapter masks frames beyond annotation length"))
                annots_r.append(annotation_record(
                    uid=f"retargetvid:{n:03d}:{ratio}:a{ann}", dataset="RetargetVid",
                    media_uid=f"retargetvid:{n:03d}", annotation_type="crop_box_dense",
                    annotation_source="human", annotator=f"annotator_{ann}", variant=ratio,
                    annotation_path=str(p), annotation_format="txt: one 'left,top,right,bottom' line per frame",
                    coverage={"annotated_frames": lines, "video_frames": fc,
                              "fraction": lines / fc if fc else None},
                    label_semantics=f"human-chosen crop window for target aspect {ratio.replace('-', ':')} (W:H)",
                    coord_format="zero-based pixel left,top,right,bottom; upstream IoU treats right/bottom "
                                 "as inclusive (+1 in area); annotation widths match right-left exactly "
                                 "(e.g. 120x360 for 1:3), so right/bottom behave as exclusive in practice",
                    coord_target="xywh float pixels, x=left, y=top, w=right-left, h=bottom-top; "
                                 "also normalized by (W,H)",
                    time_reference="line i (1-based) <-> decoded frame i-1",
                    valid_mask="frame valid iff line exists"))
    # ---- write ----
    write_jsonl(droot / "processed" / "media.jsonl", media)
    write_jsonl(droot / "processed" / "annotations.jsonl", annots_d)
    write_jsonl(droot / "processed" / "anomalies.jsonl", anomalies_d)
    write_jsonl(rroot / "processed" / "media.jsonl", rv_media)
    write_jsonl(rroot / "processed" / "annotations.jsonl", annots_r)
    write_jsonl(rroot / "processed" / "anomalies.jsonl", anomalies_r)
    ready = sum(1 for m in media if m["download_status"] == "verified")
    have_ann = len({x["media_uid"] for x in annots_d})
    write_json(droot / "processed" / "dataset_card.json", {
        "dataset": "DHF1K", "release": "author release: video.rar + annotation.rar (Google Drive)",
        "source": "https://github.com/wenguanwang/DHF1K",
        "media_expected": 1000, "media_verified": ready, "videos_with_annotation": have_ann,
        "official_split": {"train": "001-600", "val": "601-700", "test": "701-1000 (no public labels)"},
        "readiness": "raw_video_trainable" if ready >= 700 and have_ann >= 690 else
                     ("media_ready_annotation_pending" if ready >= 700 else "partial"),
        "trainable_as": ["saliency_map", "fixation_points"] if have_ann else [],
        "blockers": [] if have_ann >= 690 else ["annotation.rar not yet extracted"],
        "updated": now()})
    rv_ready = sum(1 for m in rv_media if m["download_status"] == "verified")
    write_json(rroot / "processed" / "dataset_card.json", {
        "dataset": "RetargetVid", "release": f"github bmezaris/RetargetVid@{RV_COMMIT}",
        "source": "https://github.com/bmezaris/RetargetVid",
        "media_expected": 200, "media_verified": rv_ready, "annotation_files": len(annots_r),
        "official_split": {"note": "videos are DHF1K 001-100 (DHF1K train) and 601-700 (DHF1K val); "
                                   "RetargetVid itself defines no train/test split"},
        "readiness": "raw_video_trainable" if rv_ready == 200 and len(annots_r) == 2400 else "partial",
        "trainable_as": ["crop_box_dense"], "blockers": [], "updated": now()})
    log("DHF1K", "ingest_done", media=len(media), verified=ready, annotations=len(annots_d),
        rv_media=len(rv_media), rv_verified=rv_ready, rv_annotations=len(annots_r))


if __name__ == "__main__":
    main()
