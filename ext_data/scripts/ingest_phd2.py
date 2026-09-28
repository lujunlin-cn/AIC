#!/usr/bin/env python3
"""PHD² / PHD-GIFs (García del Molino & Gygli, MM'18) — personalized highlight selections.

Source: gifs/personalized-highlights-dataset @01d5240, ``training.csv`` / ``testing.csv`` with
columns youtubeId, start (s), duration (s), user_id, video_duration (s), is_last.
Each row = one GIF a user made from a YouTube video: interval [start, start+duration).
The official split is by *user* (train users 1-12973, test users 12999-13849);
895 videos appear in both files.

Semantics kept as published:
* a selection is a positive *for that user*; parts of the video the user did not
  select are unlabelled (not negatives), so the adapter's valid mask covers the
  selections only (optionally other users' selections, labelled separately);
* ``is_last`` marks the user's last video (the official test target is
  ``is_last == True`` in testing.csv); history rows are kept for personalization.

Media: YouTube only.  This round queues a small sample (``--sample N`` test videos
with is_last) through the shared yt-dlp queue; everything else is metadata_only.
Leakage group = ``yt:<id>``, so a video shared by train and test users stays in one
aic split (eval-like wins).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext import ytqueue  # noqa: E402
from aicext.common import dataset_dir, log, now, write_json, write_jsonl  # noqa: E402
from aicext.ids import youtube_id  # noqa: E402
from aicext.probe_worker import probe_all  # noqa: E402
from aicext.schema import anomaly_record, annotation_record, media_record  # noqa: E402

DS = "PHD2"
REPO_COMMIT = "01d5240a876fd5b2d20899bea066d24dbdb89ee2"
FILES = {"train": "training.csv", "test": "testing.csv"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40, help="test is_last videos to queue for yt-dlp")
    ap.add_argument("--download", type=int, default=0, help="attempt up to N queued ids now")
    ap.add_argument("--proxy", default=os.environ.get("AIC_EXT_PROXY") or None)
    a = ap.parse_args()
    root = dataset_dir(DS)
    repo = root / "annotations" / "upstream_repo"
    rows = {}
    for split, fn in FILES.items():
        with (repo / fn).open() as f:
            rows[split] = list(csv.DictReader(f))
    anomalies = []
    # grouped selections: split -> video -> user -> [ {t0,t1,is_last} ]
    grouped = {s: collections.defaultdict(lambda: collections.defaultdict(list)) for s in FILES}
    vdur, splits_of, is_last_test = {}, collections.defaultdict(set), set()
    for split, rs in rows.items():
        for r in rs:
            y = r["youtubeId"]
            t0, d = float(r["start"]), float(r["duration"])
            vd = float(r["video_duration"]) if r["video_duration"] else None
            if vd is not None:
                vdur.setdefault(y, vd)
            flag = None
            if vd is not None and t0 + d > vd + 0.5:
                flag = "beyond_video_duration"
            grouped[split][y][r["user_id"]].append({"t0": t0, "t1": t0 + d, "is_last": r["is_last"] == "True",
                                                    "flag": flag})
            splits_of[y].add(split)
            if split == "test" and r["is_last"] == "True":
                is_last_test.add(y)
    ann_dir = root / "annotations" / "selections"
    ann_dir.mkdir(parents=True, exist_ok=True)
    for split in FILES:
        write_json(ann_dir / f"{split}.json", {v: dict(u) for v, u in grouped[split].items()})
    beyond = sum(1 for s in grouped.values() for u in s.values() for segs in u.values() for x in segs
                 if x["flag"])
    if beyond:
        anomalies.append(anomaly_record(DS, "*", "selection_beyond_video_duration",
                                        f"{beyond} selections end > video_duration + 0.5 s",
                                        "kept; adapter clips to media end when video is present"))
    # sample queue (small-scale onboarding)
    queue = root / "logs" / "yt_queue.jsonl"
    sample = sorted(is_last_test)
    random.Random(0).shuffle(sample)
    sample = sorted(sample[:a.sample])
    if queue.exists():
        ytqueue.reclassify(queue)
        ytqueue.reset_transient(queue)
    ytqueue.enqueue(queue, DS, sample, root / "raw" / "youtube")
    if a.download:
        log(DS, "yt_queue_run", **ytqueue.run(queue, limit=a.download, proxy=a.proxy))
    qstate = ytqueue.latest(queue)
    media, annots = [], []
    for y in sorted(splits_of):
        sp = splits_of[y]
        osplit = "test" if "test" in sp else "train"
        q = qstate.get(y)
        path = q["path"] if q and q["status"] == "done" else None
        status = "downloaded" if path else ("queued" if q and q["status"] in ("queued", "failed") else
                                            "failed" if q else "metadata_only")
        uid = f"phd2:{y}"
        m = media_record(
            uid=uid, dataset=DS, release=f"repo @{REPO_COMMIT[:7]}", item_id=y, source_video_id=youtube_id(y),
            source_platform="youtube", group_id=youtube_id(y), official_split=osplit,
            media_kind="video_file" if path else "remote_only", media_path=path,
            download_url=f"https://www.youtube.com/watch?v={y}",
            download_source="youtube via yt-dlp (current upload)" if path else "not downloaded this round",
            download_status=status, duration_s=vdur.get(y),
            annotation_types=["temporal_user_selection"],
            notes=json.dumps({"in_train_csv": "train" in sp, "in_test_csv": "test" in sp,
                              "test_is_last_target": y in is_last_test,
                              "queue_error": (q or {}).get("error", "")[-160:] if q and not path else None}))
        if q and q["status"] == "gave_up":
            anomalies.append(anomaly_record(DS, uid, "youtube_unavailable", (q.get("error") or "")[-200:],
                                            "labels kept; media absent"))
        media.append(m)
        for split in sorted(sp):
            users = grouped[split][y]
            annots.append(annotation_record(
                    uid=f"{uid}:{split}", dataset=DS, media_uid=uid, annotation_type="temporal_user_selection",
                    annotation_source="human", annotator=f"{len(users)} gifs.com users (ids in file)", variant=split,
                    annotation_path=str(ann_dir / f"{split}.json"),
                    annotation_format="json {video: {user: [{t0,t1,is_last,flag}]}}; one record per video x csv",
                    coverage={"key": y, "users": len(users), "selections": sum(len(v) for v in users.values()),
                              "is_last_users": sum(any(x["is_last"] for x in v) for v in users.values()),
                              "video_duration_s": vdur.get(y)},
                    label_semantics="interval this user turned into a GIF (personal highlight). Unselected time is "
                                    "unlabelled for this user, not a negative.",
                    time_reference="seconds from video start as logged by gifs.com (no frame alignment needed; "
                                   "current YouTube upload may differ from the one the user saw)",
                    valid_mask="selected intervals only (positives); everything else valid=False"))
    probed = probe_all([(m["uid"], m["media_path"]) for m in media if m["media_path"]], root, 2)
    for m in media:
        r = probed.get(m["uid"])
        if r and r.get("ok"):
            m.update(width=r["width"], height=r["height"], fps=r.get("fps_avg"), frame_count=r["frame_count"],
                     duration_s=r["duration_s"], codec=r["codec"], has_audio=r["has_audio"], rotation=r["rotation"],
                     timeline_path=r["timeline_path"], timeline_method=r["method"], preprocess_status="timeline_ok",
                     download_status="verified")
            vd = vdur.get(m["item_id"])
            if vd and abs(vd - r["duration_s"]) > 2:
                m["anomalies"] = ["duration_mismatch"]
                anomalies.append(anomaly_record(DS, m["uid"], "duration_mismatch",
                                                f"csv {vd}s vs media {r['duration_s']:.1f}s",
                                                "flagged; excluded from default training"))
    write_jsonl(root / "processed" / "media.jsonl", media)
    write_jsonl(root / "processed" / "annotations.jsonl", annots)
    write_jsonl(root / "processed" / "anomalies.jsonl", anomalies)
    ver = sum(m["download_status"] == "verified" for m in media)
    card = {
        "dataset": DS, "release": f"repo {REPO_COMMIT}", "source": "https://github.com/gifs/personalized-highlights-dataset",
        "rows": {s: len(r) for s, r in rows.items()},
        "users": {s: len({r["user_id"] for r in rs}) for s, rs in rows.items()},
        "videos": len(splits_of), "videos_in_both_csv": sum(len(v) == 2 for v in splits_of.values()),
        "test_is_last_videos": len(is_last_test), "sample_queued": len(sample), "media_verified": ver,
        "official_split": "by user (training.csv / testing.csv); aic_split by video group, eval-like wins",
        "readiness": "labels_ready_media_sample_only",
        "trainable_as": ["temporal_user_selection (sample videos only)"],
        "blockers": ["media: YouTube only; full download not attempted this round (~121k videos, ~30% "
                     "unavailable by oEmbed sample)"],
        "updated": now()}
    write_json(root / "processed" / "dataset_card.json", card)
    log(DS, "ingest_done", media=len(media), annotations=len(annots), verified=ver)
    print(json.dumps(card, ensure_ascii=False))


if __name__ == "__main__":
    main()
