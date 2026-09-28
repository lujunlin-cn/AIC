#!/usr/bin/env python3
"""Mr.HiSum: labels, metadata, YouTube-8M features and raw-video readiness, separately.

* Labels  ``mr_hisum.h5`` (author Google Drive): per ``video_N`` —
  ``gtscore`` (Most-Replayed intensity, 1 value per second, 0..1),
  ``change_points`` (KTS shots, algorithmic), ``gt_summary`` (0/1 knapsack
  over shots — an algorithm-derived summary, registered as
  ``temporal_summary_derived``, never as human GT).
* Metadata ``metadata.csv``: youtube_id, yt8m shard, yt8m random_id, duration.
* Split ``dataset/mr_hisum_split.json`` (27,892 / 2,000 / 2,000).
* Features: YouTube-8M v2 frame-level (1 fps, rgb 1024 + audio 128, quantized)
  streamed per shard; only Mr.HiSum examples are kept (``processed/yt8m/``).
* Raw videos are not distributed; a YouTube queue is built, nothing more.

Readiness is reported per component. Without raw video the dataset is
"feature-head trainable" only.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.common import append_jsonl, dataset_dir, log, now, read_jsonl, write_json, write_jsonl  # noqa: E402
from aicext.download import session  # noqa: E402
from aicext.ids import youtube_id  # noqa: E402
from aicext.schema import anomaly_record, annotation_record, media_record  # noqa: E402
from aicext import yt8m, ytqueue  # noqa: E402

DS = "MrHiSum"
REPO_COMMIT = "aeb667f71239887034c2974062a341d312762867"
H5_GDRIVE, META_GDRIVE = "1ahLq7h-VE410cVTsRl1Kwno4mIQeQdkr", "1GhUSEzPif5h2sUtHsSK9zn4qlEqeKcgY"
PLAN_URL = "http://data.yt8m.org/2/download_plans/frame_train.json"
MIRRORS = ("asia", "eu", "us")


def load_meta(root: Path) -> dict[str, dict]:
    return {r["video_id"]: r for r in csv.DictReader((root / "annotations" / "metadata.csv").open())}


def stage_features(root: Path, workers: int, limit: int | None) -> None:
    meta = load_meta(root)
    out_dir = root / "processed" / "yt8m"
    out_dir.mkdir(parents=True, exist_ok=True)
    ledger = root / "logs" / "yt8m_shards.jsonl"
    plan_p = root / "raw" / "yt8m_frame_train_plan.json"
    sess = session()
    if not plan_p.exists():
        plan_p.write_text(sess.get(PLAN_URL, timeout=60).text)
    plan = json.loads(plan_p.read_text())["files"]
    by_shard: dict[str, dict[str, str]] = {}
    for vid, r in meta.items():
        by_shard.setdefault(r["yt8m_file"], {})[r["random_id"]] = vid
    done = {r["shard"] for r in read_jsonl(ledger) if r.get("status") == "done"}
    todo = [s for s in sorted(by_shard) if s not in done or not (out_dir / f"{s}.h5").exists()]
    if limit:
        todo = todo[:limit]
    log(DS, "yt8m_stage", shards_total=len(by_shard), done=len(done), todo=len(todo))

    def one(k, shard):
        remote = yt8m.shard_remote_name(shard) + ".tfrecord"
        md5 = plan.get(remote)
        last = None
        for attempt in range(4):
            mirror = MIRRORS[(k + attempt) % len(MIRRORS)]
            url = f"http://{mirror}.data.yt8m.org/2/frame/train/{remote}"
            t0 = time.time()
            try:
                res = yt8m.extract_shard(url, by_shard[shard], out_dir / f"{shard}.h5", md5, session())
                missing = sorted(set(by_shard[shard].values()) - set(res["found"]))
                row = {"shard": shard, "remote": remote, "url": url, "status": "done", "md5": res["md5"],
                       "records": res["records"], "found": len(res["found"]), "wanted": len(by_shard[shard]),
                       "missing": missing, "seconds": round(time.time() - t0, 1), "time": now()}
                append_jsonl(ledger, row)
                return row
            except Exception as exc:
                last = repr(exc)[:300]
                append_jsonl(ledger, {"shard": shard, "url": url, "status": "retry", "error": last, "time": now()})
                time.sleep(10 * (attempt + 1))
        append_jsonl(ledger, {"shard": shard, "status": "failed", "error": last, "time": now()})
        return {"shard": shard, "status": "failed"}

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(one, k, s) for k, s in enumerate(todo)]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result()
            if i % 10 == 0 or r["status"] != "done":
                log(DS, "yt8m_progress", done=i, of=len(todo), last=r.get("shard"), status=r["status"])


def stage_index(root: Path) -> None:
    import h5py
    import numpy as np

    meta = load_meta(root)
    split = json.loads((root / "annotations" / "upstream_repo" / "dataset" / "mr_hisum_split.json").read_text())
    split_of = {k: s.replace("_keys", "") for s, ks in split.items() for k in ks}
    h5p = root / "annotations" / "mr_hisum.h5"
    feat_rows = {}
    for r in read_jsonl(root / "logs" / "yt8m_shards.jsonl"):
        if r.get("status") == "done":
            feat_rows[r["shard"]] = r
    ytq = {k: v for k, v in ytqueue.latest(root / "logs" / "yt_queue.jsonl").items() if v["status"] == "done"}
    media, annots, anomalies = [], [], []
    feat_len = {}
    for shard in feat_rows:
        p = root / "processed" / "yt8m" / f"{shard}.h5"
        if p.exists():
            with h5py.File(p, "r") as h:
                for k in h:
                    feat_len[k] = (str(p), int(h[k]["rgb"].shape[0]))
    with h5py.File(h5p, "r") as h:
        for vid in sorted(h.keys(), key=lambda s: int(s.split("_")[1])):
            r = meta[vid]
            g = h[vid]
            n = int(g["gtscore"].shape[0])
            yt = r["youtube_id"]
            fpath, flen = feat_len.get(vid, (None, None))
            raw = ytq.get(yt)
            m = media_record(
                uid=f"mrhisum:{vid}", dataset=DS, release=f"Mr.HiSum h5+metadata (Drive), repo@{REPO_COMMIT[:7]}",
                item_id=vid, source_video_id=youtube_id(yt), source_platform="youtube", group_id=youtube_id(yt),
                official_split=split_of.get(vid, "unknown"),
                media_kind="video_file" if raw else "feature_only",
                media_path=raw["path"] if raw else None,
                download_url=f"https://www.youtube.com/watch?v={yt}",
                download_source="youtube (not distributed by authors)",
                download_status="downloaded" if raw else ("metadata_only" if not fpath else "metadata_only"),
                duration_s=float(r["duration"]), annotation_types=["temporal_score_1d", "temporal_summary_derived"],
                notes=json.dumps({"yt8m_file": r["yt8m_file"], "yt8m_random_id": r["random_id"],
                                  "views": int(r["views"]), "yt8m_labels": r["labels"],
                                  "features_ready": bool(fpath), "raw_video_ready": bool(raw)}))
            m["preprocess_status"] = "done" if fpath else "pending"
            media.append(m)
            cov = {"h5_key": vid, "steps": n, "step_s": 1.0, "grid_anchor_s": 0.0,
                   "feature_path": fpath, "feature_steps": flen, "feature_source": "YouTube-8M v2 frame-level "
                   "(1 fps, rgb 1024 PCA+quantized, audio 128)" if fpath else None}
            if flen is not None and flen != n:
                anomalies.append(anomaly_record(DS, m["uid"], "feature_label_length_mismatch",
                                                f"features {flen} vs gtscore {n}",
                                                "adapter marks steps lacking either as invalid"))
            annots.append(annotation_record(
                uid=f"mrhisum:{vid}:gtscore", dataset=DS, media_uid=m["uid"], annotation_type="temporal_score_1d",
                annotation_source="mixed", annotator="YouTube Most-Replayed aggregate (50k+ viewers)",
                annotation_path=str(h5p), annotation_format="h5 <video_id>/gtscore float64[duration]",
                coverage=cov,
                label_semantics="normalized Most-Replayed intensity of the 100-bin heat map, resampled to 1 value "
                                "per whole second (bin containing instant k s); weak viewer-behaviour signal, "
                                "not a human highlight judgement",
                time_reference="gtscore[k] <-> instant k s of the YouTube video; YT-8M features: row k <-> "
                               "frame decoded at ~k s (1 fps, first 360 s max)",
                valid_mask="steps < min(len(features), len(gtscore)) when features used"))
            annots.append(annotation_record(
                uid=f"mrhisum:{vid}:gt_summary", dataset=DS, media_uid=m["uid"],
                annotation_type="temporal_summary_derived", annotation_source="derived",
                annotation_path=str(h5p), annotation_format="h5 <video_id>/gt_summary float32[duration] 0/1",
                coverage={"h5_key": vid, "steps": n, "step_s": 1.0},
                label_semantics="0/1 knapsack summary over KTS shots maximising gtscore (15% budget); "
                                "algorithmic, not human",
                derived_from=f"mrhisum:{vid}:gtscore",
                derivation="upstream: KTS change_points + 0/1 knapsack (not re-run here)",
                time_reference="same grid as gtscore", valid_mask="all steps"))
            if flen is None and r["yt8m_file"] in feat_rows:
                anomalies.append(anomaly_record(DS, m["uid"], "missing_features",
                                                f"random_id {r['random_id']} not found in {r['yt8m_file']}",
                                                "labels only"))
    # duplicated youtube ids inside Mr.HiSum
    seen = {}
    for m in media:
        seen.setdefault(m["source_video_id"], []).append(m["uid"])
    for yid, uids in seen.items():
        if len(uids) > 1:
            anomalies.append(anomaly_record(DS, yid, "duplicate_source_video", ",".join(uids),
                                            "same group_id; forced into one split"))
    write_jsonl(root / "processed" / "media.jsonl", media)
    write_jsonl(root / "processed" / "annotations.jsonl", annots)
    write_jsonl(root / "processed" / "anomalies.jsonl", anomalies)
    nfeat = sum(1 for v in feat_len)
    nraw = sum(1 for m in media if m["media_kind"] == "video_file")
    write_json(root / "processed" / "dataset_card.json", {
        "dataset": DS, "release": f"mr_hisum.h5 + metadata.csv (author Drive), repo {REPO_COMMIT}",
        "source": "https://github.com/MRHiSum/MR.HiSum",
        "components": {"labels_gtscore": {"ready": True, "videos": len(media)},
                       "metadata": {"ready": True, "rows": len(meta)},
                       "split": {"ready": True, "train": len(split["train_keys"]), "val": len(split["val_keys"]),
                                 "test": len(split["test_keys"])},
                       "yt8m_features": {"ready": nfeat == len(media), "videos": nfeat,
                                         "shards_done": len(feat_rows), "shards_needed": len({r['yt8m_file'] for r in meta.values()})},
                       "raw_video": {"ready": False, "videos": nraw, "note": "YouTube re-download only; queue"}},
        "readiness": ("feature_head_trainable" if nfeat else "labels_only") if nraw == 0 else "partial_raw_video",
        "trainable_as": ["temporal_score_1d (feature head on YT-8M features)"] if nfeat else [],
        "not_trainable_as": ["VideoMAE / raw-video fine-tuning (no raw media)"],
        "blockers": [] if nfeat == len(media) else [f"YT-8M features for {len(media) - nfeat} videos pending"],
        "alternatives_not_used": {"hf:hminjeong/TripleSumm-Mr.HiSum@4f9cd1b": "third-party re-crawl (30,452 "
                                  "videos) with re-extracted InceptionV3/AST/RoBERTa features; different media "
                                  "and features from the official release"},
        "updated": now()})
    log(DS, "index_done", media=len(media), features=nfeat, raw=nraw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["features", "index", "queue"], required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    root = dataset_dir(DS)
    if a.stage == "features":
        stage_features(root, a.workers, a.limit)
    elif a.stage == "queue":
        meta = load_meta(root)
        n = ytqueue.enqueue(root / "logs" / "yt_queue.jsonl", DS, sorted({r["youtube_id"] for r in meta.values()}),
                            root / "raw" / "youtube")
        log(DS, "queue_built", new=n)
    stage_index(root)


if __name__ == "__main__":
    main()
