#!/usr/bin/env python3
"""YouTube Highlights (Sun et al. ECCV'14) — author labels + mirrored media.

Labels come from the author repo (aliensunmin/DomainSpecificHighlight, pinned
commit): per video ``clip.json`` ([start_frame, end_frame] ~2 s clips, 50%
overlap), ``match_label.json`` (auto-harvested: 1 matched / -1 unmatched /
0 borderline) and, for a subset, ``mturk_label.json`` (soft vote count of
turkers selecting the clip).  Both kept separately; ``match_label=-1`` is an
automatic "not used in the edited video" signal, not a human negative.

Media: HF dataset ``jhanglee/youtube-highlights-full`` @b28d020 (tar SHA-256
verified), MP4s re-downloaded from YouTube by the mirror owner.  Frame numbers
in the labels refer to the *original* upload; when the mirror's decoded frame
count differs, clip times are computed as frame / source_fps with source_fps
taken from the mirror file and the video is flagged ``misaligned_suspect``.

Videos missing from the mirror are placed on a yt-dlp queue (resumable,
bounded retries, failure list).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.common import dataset_dir, ensure_space, log, now, read_jsonl, write_json, write_jsonl  # noqa: E402
from aicext.ids import youtube_id  # noqa: E402
from aicext.probe_worker import probe_all  # noqa: E402
from aicext.schema import anomaly_record, annotation_record, media_record  # noqa: E402
from aicext import ytqueue  # noqa: E402

DS = "YouTubeHighlights"
DOMAINS = ("dog", "gymnastics", "parkour", "skating", "skiing", "surfing")
REPO_COMMIT = "29083d2dc8ee951986f6fb436f1be8f92c8937fd"
MIRROR = "hf:jhanglee/youtube-highlights-full@b28d0207471f8baf445dccb10c5e074a717f3b1d"
TAR_SHA = "af5c025a03071b1a3e064da3bb96546fc2ae5de76050973bb3153d354870efa2"


def stage_extract(root: Path) -> Path:
    out = root / "raw" / "mirror"
    marker = out / ".extracted"
    if not marker.exists():
        tar = root / "raw" / "archives" / "youtube_highlights_full.tar"
        ensure_space(tar.stat().st_size)
        out.mkdir(parents=True, exist_ok=True)
        subprocess.run(["tar", "-xf", str(tar), "-C", str(out)], check=True)
        marker.write_text(now() + "\n")
        log(DS, "extracted", files=sum(1 for _ in out.rglob("*.mp4")))
    return out / "youtube_highlights_full"


def load_json(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-missing", type=int, default=0,
                    help="attempt up to N queued YouTube ids with yt-dlp (0 = only build the queue)")
    ap.add_argument("--proxy", default=None)
    a = ap.parse_args()
    root = dataset_dir(DS)
    repo = root / "annotations" / "upstream_repo"
    mirror = stage_extract(root)
    # mirror's own per-video metadata (for provenance + alignment warnings)
    mirror_meta, mirror_warn = {}, {}
    for dom in DOMAINS:
        for kind in ("mturk", "weak_match"):
            d = load_json(mirror / dom / f"annotations_{kind}.json") or {"videos": []}
            for v in d["videos"]:
                mirror_meta[(dom, v["video_id"])] = v
    for w in load_json(mirror / "annotation_alignment_warnings.json") or []:
        mirror_warn[(w["tag"], w["video_id"])] = w
    mirror_fail = {(f["tag"], f["video_id"]): f for f in load_json(mirror / "download_failures.json") or []}

    # author backup ids for missing videos
    backups = {}
    for p in (repo / "missingVideos").glob("*"):
        if p.is_file():
            backups[p.name] = p.read_text().strip()

    media, annots, anomalies, missing = [], [], [], []
    seg_dir = root / "annotations" / "segments"
    seg_dir.mkdir(parents=True, exist_ok=True)
    for dom in DOMAINS:
        vlist = load_json(repo / dom / "vlist.json")
        vsel = load_json(repo / dom / "vlist_sel.json")
        split_of = {i: s for ids, s in vlist for i in ids}
        selected = {i for ids, _ in vsel for i in ids}
        for vid, osplit in sorted(split_of.items()):
            vdir = repo / dom / vid
            clips = load_json(vdir / "clip.json")
            match = load_json(vdir / "match_label.json")
            mturk = load_json(vdir / "mturk_label.json")
            info = load_json(vdir / "info.json")
            mp4 = mirror / dom / "videos" / f"{vid}.mp4"
            uid = f"yth:{dom}:{vid}"
            m = media_record(
                uid=uid, dataset=DS, release=f"author labels @{REPO_COMMIT[:7]}; media {MIRROR}",
                item_id=f"{dom}/{vid}", source_video_id=youtube_id(vid), source_platform="youtube",
                group_id=youtube_id(vid), official_split=osplit, media_kind="video_file",
                media_path=str(mp4) if mp4.exists() else None,
                download_url=f"https://www.youtube.com/watch?v={vid}",
                download_source=f"{MIRROR} (tar sha256 {TAR_SHA[:12]}…)" if mp4.exists() else "youtube (queued)",
                download_status="downloaded" if mp4.exists() else "missing",
                file_size=mp4.stat().st_size if mp4.exists() else None,
                notes=json.dumps({"domain": dom, "domain_relevant_manual_selection": vid in selected,
                                  "title": info[5] if info and len(info) > 5 else None}, ensure_ascii=False))
            m["aic_split"] = None
            media.append(m)
            if not mp4.exists():
                why = mirror_fail.get((dom, vid), {}).get("reason", "not in mirror")
                missing.append({"yt_id": vid, "domain": dom, "reason": why, "author_backup": backups.get(vid)})
                anomalies.append(anomaly_record(DS, uid, "missing_media", why, "queued for yt-dlp"))
            if not clips or not match:
                anomalies.append(anomaly_record(DS, uid, "missing_annotation", "no clip/match_label json", "no labels"))
                continue
            segs = []
            m_clips, m_lab = match
            t_clips, t_lab = (mturk if mturk else (None, None))
            if t_clips is not None and [list(map(float, c)) for c in t_clips] != [list(map(float, c)) for c in m_clips]:
                anomalies.append(anomaly_record(DS, uid, "clip_list_mismatch", "mturk vs match clip lists differ",
                                                "each label keeps its own clip list"))
            for k, (s, e) in enumerate(m_clips):
                segs.append({"start_frame": float(s), "end_frame": float(e), "match_label": int(m_lab[k]),
                             "mturk_votes": (float(t_lab[k]) if t_lab is not None and k < len(t_lab) else None)})
            payload = {"video_id": vid, "domain": dom, "official_split": osplit, "segments_frames": segs,
                       "frame_semantics": "original-upload frame numbers from author clip.json",
                       "fields": {"match_label": "auto: 1 matched in user-edited video, -1 not matched, 0 borderline",
                                  "mturk_votes": "human: soft count of turkers selecting the clip (coverage-weighted)"}}
            ap_ = seg_dir / f"{dom}__{vid}.json"
            write_json(ap_, payload)
            m["annotation_types"] = ["temporal_segment_label"]
            annots.append(annotation_record(
                uid=f"{uid}:match", dataset=DS, media_uid=uid, annotation_type="temporal_segment_label",
                annotation_source="auto", annotator="edited-video matching (Sun et al.)", variant="match_label",
                annotation_path=str(ap_), annotation_format="json segments (frames) -> adapter adds t0/t1 seconds",
                coverage={"clips": len(segs), "last_frame": max(s["end_frame"] for s in segs)},
                label_semantics="1 = clip reused by the uploader's edited video; -1 = not reused (NOT a human "
                                "negative); 0 = borderline",
                time_reference="frames of the original upload; seconds = frame / fps of available media",
                valid_mask="only frames covered by a clip; borderline (0) kept as its own value"))
            if mturk:
                annots.append(annotation_record(
                    uid=f"{uid}:mturk", dataset=DS, media_uid=uid, annotation_type="temporal_segment_label",
                    annotation_source="human", annotator="AMT workers (soft votes)", variant="mturk_votes",
                    annotation_path=str(ap_), annotation_format="json segments (frames)",
                    coverage={"clips": len(segs)},
                    label_semantics="number of turkers (soft, coverage-weighted) selecting this clip as highlight; "
                                    "0 = no one selected it within the video (relative, per-video)",
                    time_reference="frames of the original upload; seconds = frame / fps of available media",
                    valid_mask="only frames covered by a clip"))
    # YouTube queue: attach yt-dlp downloads *before* probing so they get timelines + alignment checks
    queue = root / "logs" / "yt_queue.jsonl"
    n_new = ytqueue.enqueue(queue, DS, [x["yt_id"] for x in missing], root / "raw" / "youtube")
    if a.download_missing:
        stats = ytqueue.run(queue, limit=a.download_missing, proxy=a.proxy)
        log(DS, "yt_queue_run", **stats)
    ytdone = {k: v for k, v in ytqueue.latest(queue).items() if v["status"] == "done"}
    for m in media:
        y = m["source_video_id"].split(":", 1)[1]
        if m["media_path"] is None and y in ytdone:
            m.update(media_path=ytdone[y]["path"], download_status="downloaded",
                     file_size=Path(ytdone[y]["path"]).stat().st_size,
                     download_source="youtube via yt-dlp (current upload; frame numbers may differ)")
    # probe media and attach seconds to segments
    res = probe_all([(m["uid"], m["media_path"]) for m in media if m["media_path"]], root, 4)
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
    by = {m["uid"]: m for m in media}
    for a_ in annots:
        if a_["variant"] != "match_label":
            continue
        m = by[a_["media_uid"]]
        p = Path(a_["annotation_path"])
        d = json.loads(p.read_text())
        fps = m.get("fps")
        dom, vid = m["item_id"].split("/")
        last = max(s["end_frame"] for s in d["segments_frames"])
        mis = None
        if m.get("frame_count"):
            gap = last - m["frame_count"]
            if gap > max(3, 0.02 * m["frame_count"]):
                mis = {"annotation_last_frame": last, "media_frames": m["frame_count"], "gap_frames": gap,
                       "mirror_warning": (dom, vid) in mirror_warn}
                anomalies.append(anomaly_record(DS, m["uid"], "misaligned_suspect",
                                                f"labels reach frame {last}, media has {m['frame_count']}",
                                                "kept; segments beyond media end are dropped by the adapter; "
                                                "flag excludes it from default training"))
        if fps:
            d["segments"] = [{"t0": s["start_frame"] / fps, "t1": s["end_frame"] / fps,
                              "match_label": s["match_label"], "mturk_votes": s["mturk_votes"]}
                             for s in d["segments_frames"]]
            d["seconds_from"] = f"frame / {fps:.6f} (fps of mirror media)"
        d["alignment"] = mis
        write_json(p, d)
        if mis:
            m["anomalies"] = ["misaligned_suspect"]
    # queue + failure list (latest queue state is folded into missing_media.json)
    qstate = ytqueue.latest(queue)
    for x in missing:
        q = qstate.get(x["yt_id"]) or {}
        x.update(queue_status=q.get("status"), queue_error=(q.get("error") or "")[-200:] or None)
    write_json(root / "logs" / "missing_media.json", missing)
    write_jsonl(root / "processed" / "media.jsonl", media)
    write_jsonl(root / "processed" / "annotations.jsonl", annots)
    write_jsonl(root / "processed" / "anomalies.jsonl", anomalies)
    ver = sum(m["download_status"] == "verified" for m in media)
    mis = sum(1 for m in media if m.get("anomalies"))
    write_json(root / "processed" / "dataset_card.json", {
        "dataset": DS, "release": f"author repo {REPO_COMMIT}; media {MIRROR}",
        "source": "https://github.com/aliensunmin/DomainSpecificHighlight",
        "media_expected": len(media), "media_verified": ver,
        "media_missing": sum(1 for m in media if not m["media_path"]),
        "media_recovered_by_ytdlp": sum(1 for m in media if "yt-dlp" in (m["download_source"] or "") and m["media_path"]),
        "misaligned_suspect": mis,
        "annotations": {"match_label(auto)": sum(a["variant"] == "match_label" for a in annots),
                        "mturk(human)": sum(a["variant"] == "mturk_votes" for a in annots)},
        "official_split": "TraiTightL/TraiLooseL (train), TestTightL/TestLooseL (test) per domain",
        "readiness": "raw_video_trainable" if ver else "blocked",
        "trainable_as": ["temporal_segment_label"],
        "blockers": [f"{sum(1 for m in media if not m['media_path'])} author-list videos unavailable (not in mirror; "
                     f"yt-dlp: removed/private, see logs/missing_media.json)"] if any(not m["media_path"] for m in media)
        else [],
        "updated": now(), "queue_new": n_new})
    log(DS, "ingest_done", media=len(media), verified=ver, missing=len(missing), annotations=len(annots))


if __name__ == "__main__":
    main()
