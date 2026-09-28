#!/usr/bin/env python3
"""ClipShots (Tang et al. 2018) — shot-boundary labels + videos.

Labels: author repo Tangshitao/ClipShots @8d033e2 ``annotations/{train,test,only_gradual}.json``:
``{video: {"transitions": [[s, e], ...], "frame_num": N}}``.  Upstream
``tools/evaluate.py`` treats ``e - s == 1`` as a hard cut (last frame of shot A,
first frame of shot B) and ``e - s > 1`` as a gradual transition spanning frames
``s..e``.  Rows with ``e == s`` belong to neither class upstream and rows with
``e < s`` or beyond ``frame_num`` are malformed; both are kept in the raw file
and masked (``valid=False``) by the adapter, never used as negatives.

Subsets: ``train`` and ``test`` list every transition (non-transition frames are
reliable negatives).  ``only_gradual`` was added to boost gradual transitions;
its labels are not exhaustive, so everything outside a labelled transition is
unlabelled (``negatives_reliable=False``).

Videos: Google Drive folder 1AAhTbNroSFsygHBXa88emCU7f50MxI8t holds one
``data.tar.gz`` split into ``ClipShots-a/b/c`` (the upstream ``videos/README``
still says "not ready for release"; the Drive link is in the top-level README).
``--stage media`` streams ``cat a b c | tar -xz`` once all parts are verified.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.common import dataset_dir, ensure_space, log, now, write_json, write_jsonl  # noqa: E402
from aicext.ids import YT_RE, native_id, youtube_id  # noqa: E402
from aicext.probe_worker import probe_all  # noqa: E402
from aicext.schema import anomaly_record, annotation_record, media_record  # noqa: E402

DS = "ClipShots"
REPO_COMMIT = "8d033e2591c23b00806179676e0a0b7461468281"
SUBSETS = ("train", "test", "only_gradual")
PARTS = {"ClipShots-a": 18874368000, "ClipShots-b": 18874368000, "ClipShots-c": 10555027769}
DRIVE_FOLDER = "https://drive.google.com/drive/folders/1AAhTbNroSFsygHBXa88emCU7f50MxI8t"


def parts_ready(root: Path) -> bool:
    arch = root / "raw" / "archives"
    return all((arch / n).exists() and (arch / n).stat().st_size == s for n, s in PARTS.items())


def stage_media(root: Path) -> Path | None:
    out = root / "raw" / "data"
    marker = out / ".extracted"
    if marker.exists():
        return out
    if not parts_ready(root):
        log(DS, "media_not_ready", parts={n: (root / "raw" / "archives" / n).exists() for n in PARTS})
        return None
    ensure_space(sum(PARTS.values()))
    out.mkdir(parents=True, exist_ok=True)
    arch = root / "raw" / "archives"
    cat = subprocess.Popen(["cat", *[str(arch / n) for n in PARTS]], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-xz", "-C", str(out)], stdin=cat.stdout, check=True)
    if cat.wait() != 0:
        raise RuntimeError("cat of split parts failed")
    marker.write_text(now() + "\n")
    log(DS, "extracted", files=sum(1 for _ in out.rglob("*.mp4")))
    return out


def find_videos(data: Path | None) -> dict[str, Path]:
    if not data:
        return {}
    out = {}
    for p in data.rglob("*.mp4"):
        out.setdefault(p.name, p)
    return out


def classify(s: int, e: int, n: int) -> str:
    if e < s or s < 0 or e >= n:
        return "malformed"
    if e == s:
        return "degenerate"
    return "cut" if e - s == 1 else "gradual"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["labels", "media", "all"], default="all",
                    help="labels: index labels only; media: extract parts if complete, then probe")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    root = dataset_dir(DS)
    repo = root / "annotations" / "upstream_repo"
    data = stage_media(root) if a.stage in ("media", "all") else (root / "raw" / "data"
                                                               if (root / "raw" / "data" / ".extracted").exists()
                                                               else None)
    vids = find_videos(data)
    media, annots, anomalies = [], [], []
    ann_dir = root / "annotations" / "transitions"
    ann_dir.mkdir(parents=True, exist_ok=True)
    counts = {s: {"videos": 0, "cut": 0, "gradual": 0, "degenerate": 0, "malformed": 0} for s in SUBSETS}
    for subset in SUBSETS:
        labels = json.loads((repo / "annotations" / f"{subset}.json").read_text())
        listed = [x.strip() for x in (repo / "video_lists" / f"{subset}.txt").read_text().splitlines() if x.strip()]
        for name in sorted(set(listed) - set(labels)):
            anomalies.append(anomaly_record(DS, name, "missing_annotation", f"{subset} list entry without labels",
                                            "media only"))
        for name, v in sorted(labels.items()):
            stem = name.rsplit(".", 1)[0]
            yt = bool(YT_RE.match(stem))
            gid = youtube_id(stem) if yt else native_id(DS, stem)
            uid = f"clipshots:{stem}"
            p = vids.get(name)
            n = int(v["frame_num"])
            m = media_record(
                uid=uid, dataset=DS, release=f"labels @{REPO_COMMIT[:7]}; videos Drive data.tar.gz (a/b/c)",
                item_id=name, source_video_id=gid, source_platform="youtube" if yt else "weibo_or_unknown",
                group_id=gid, official_split=subset, media_kind="video_file",
                media_path=str(p) if p else None, download_url=DRIVE_FOLDER,
                download_source="author Google Drive (README link)",
                download_status="downloaded" if p else "pending", file_size=p.stat().st_size if p else None,
                annotation_types=["shot_boundary"],
                notes=json.dumps({"group_id_rule": "11-char [A-Za-z0-9_-] stem treated as YouTube id"
                                  if yt else "native name"}))
            media.append(m)
            tr = [[int(s), int(e)] for s, e in v["transitions"]]
            kinds = [classify(s, e, n) for s, e in tr]
            c = counts[subset]
            c["videos"] += 1
            for k in kinds:
                c[k] += 1
            bad = [t for t, k in zip(tr, kinds) if k in ("degenerate", "malformed")]
            if bad:
                anomalies.append(anomaly_record(DS, uid, "odd_transitions", f"{len(bad)} rows e<=s or beyond frame_num: "
                                                f"{bad[:5]}", "kept raw; adapter masks them (valid=False)"))
            if n <= 0:
                anomalies.append(anomaly_record(DS, uid, "zero_frame_num", "frame_num=0", "labels unusable"))
            out = ann_dir / f"{subset}__{stem}.json"
            write_json(out, {"video": name, "subset": subset, "frame_num": n, "transitions": tr, "kinds": kinds,
                             "negatives_reliable": subset != "only_gradual",
                             "semantics": "[s,e] frame indices (0-based) as in upstream json; cut: e-s==1, "
                                          "gradual: e-s>1 (frames s..e)"})
            annots.append(annotation_record(
                uid=f"{uid}:transitions", dataset=DS, media_uid=uid, annotation_type="shot_boundary",
                annotation_source="human", annotator="ClipShots authors", variant=subset,
                annotation_path=str(out), annotation_format="json transitions[[s,e]], kinds[], frame_num",
                coverage={"transitions": len(tr), "cut": kinds.count("cut"), "gradual": kinds.count("gradual"),
                          "label_frame_num": n, "negatives_reliable": subset != "only_gradual"},
                label_semantics="cut = boundary between frames s and s+1; gradual = dissolve/fade/slide over "
                                "frames s..e; degenerate (e==s) and malformed rows are ignored regions",
                time_reference="0-based frame index of the released mp4 (checked against decoded frame count); "
                               "seconds from the timeline npz",
                valid_mask="train/test: all frames valid except ignored rows; only_gradual: only frames inside "
                           "labelled transitions are valid (unlabelled != negative)"))
    if vids:
        res = probe_all([(m["uid"], m["media_path"]) for m in media if m["media_path"]], root, a.workers)
        for m in media:
            r = res.get(m["uid"])
            if not r:
                continue
            if not r.get("ok"):
                m.update(download_status="failed", preprocess_status="failed")
                anomalies.append(anomaly_record(DS, m["uid"], "decode_failed", r.get("error", ""), "excluded"))
                continue
            m.update(width=r["width"], height=r["height"], fps=r.get("fps_avg"), frame_count=r["frame_count"],
                     duration_s=r["duration_s"], rotation=r["rotation"], codec=r["codec"], has_audio=r["has_audio"],
                     timeline_path=r["timeline_path"], timeline_method=r["method"], preprocess_status="timeline_ok",
                     download_status="verified")
        lab_n = {x["media_uid"]: x["coverage"]["label_frame_num"] for x in annots}
        for m in media:
            if m.get("frame_count") and abs(m["frame_count"] - lab_n[m["uid"]]) > 2:
                m["anomalies"] = ["frame_count_mismatch"]
                anomalies.append(anomaly_record(DS, m["uid"], "frame_count_mismatch",
                                                f"label frame_num {lab_n[m['uid']]} vs decoded {m['frame_count']}",
                                                "kept; adapter clips labels to min(frame_num, decoded)"))
    write_jsonl(root / "processed" / "media.jsonl", media)
    write_jsonl(root / "processed" / "annotations.jsonl", annots)
    write_jsonl(root / "processed" / "anomalies.jsonl", anomalies)
    ver = sum(m["download_status"] == "verified" for m in media)
    write_json(root / "processed" / "dataset_card.json", {
        "dataset": DS, "release": f"labels {REPO_COMMIT}; videos {DRIVE_FOLDER}",
        "source": "https://github.com/Tangshitao/ClipShots", "media_expected": len(media), "media_verified": ver,
        "label_counts": counts, "official_split": "train / only_gradual (training), test (evaluation)",
        "readiness": "raw_video_trainable" if ver == len(media) else ("partial" if ver else "labels_only"),
        "trainable_as": ["shot_boundary"],
        "blockers": [] if ver == len(media) else
        ["videos: Drive parts not yet complete/extracted" if not vids else f"{len(media) - ver} videos not verified"],
        "updated": now()})
    log(DS, "ingest_done", media=len(media), verified=ver, annotations=len(annots))
    print(json.dumps({"media": len(media), "verified": ver, "counts": counts}))


if __name__ == "__main__":
    main()
