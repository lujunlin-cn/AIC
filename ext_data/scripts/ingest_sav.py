#!/usr/bin/env python3
"""SA-V (Meta, SAM 2) — onboarding from the official in-repo example only.

Full SA-V (train sav_000..055.tar ~8 GB each, val/test ~16 GB) is served from
https://ai.meta.com/datasets/segment-anything-video-downloads/ with
``requiresConsent: true``: the link list is generated after clicking through the
SA-V license in a browser and the links expire.  That is a manual step, so the
full set is recorded as ``blocked`` here.  Unofficial mirrors are not used:
``Voxel51/segment_anything_video_subset51`` re-encodes to 6 fps and converts the
annotations (not the original format); ``pengqinhe/savval`` has no provenance.

The official example ``sav_dataset/example/sav_000001.{mp4,_manual.json,_auto.json}``
(facebookresearch/sam2 @2b90b9f) is indexed with manual and auto masklets as
separate annotation records.  Masklets are COCO RLE at 6 fps (every 4th frame of
the 24 fps video); ``masklet_first_appeared_frame`` is an index into the 6 fps
annotation sequence.  Object masks are not "most important subject" labels.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.common import dataset_dir, log, now, sha256_file, write_json, write_jsonl  # noqa: E402
from aicext.ids import native_id  # noqa: E402
from aicext.probe_worker import probe_all  # noqa: E402
from aicext.schema import anomaly_record, annotation_record, media_record  # noqa: E402

DS = "SA_V"
REPO_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
DL_PAGE = "https://ai.meta.com/datasets/segment-anything-video-downloads/"


def main():
    root = dataset_dir(DS)
    ex = root / "annotations" / "upstream_repo" / "sav_dataset" / "example"
    media, annots, anomalies = [], [], []
    for mp4 in sorted(ex.glob("sav_*.mp4")):
        vid = mp4.stem
        uid = f"sav:{vid}"
        m = media_record(
            uid=uid, dataset=DS, release=f"official example in facebookresearch/sam2 @{REPO_COMMIT[:7]}",
            item_id=vid, source_video_id=native_id(DS, vid), source_platform="sa-v (Meta)",
            group_id=native_id(DS, vid), official_split="train", media_kind="video_file", media_path=str(mp4),
            download_url="https://github.com/facebookresearch/sam2/tree/main/sav_dataset/example",
            download_source="git clone (official repo)", file_size=mp4.stat().st_size, sha256=sha256_file(mp4),
            download_status="sample_only", annotation_types=["object_masklet"])
        media.append(m)
        res = probe_all([(uid, str(mp4))], root, 1)[uid]
        if res.get("ok"):
            m.update(width=res["width"], height=res["height"], fps=res.get("fps_avg"), frame_count=res["frame_count"],
                     duration_s=res["duration_s"], codec=res["codec"], has_audio=res["has_audio"],
                     rotation=res["rotation"], timeline_path=res["timeline_path"], timeline_method=res["method"],
                     preprocess_status="timeline_ok")
        for src, kind in (("manual", "human"), ("auto", "auto")):
            p = ex / f"{vid}_{src}.json"
            if not p.exists():
                anomalies.append(anomaly_record(DS, uid, "missing_annotation", p.name, "skipped"))
                continue
            d = json.loads(p.read_text())
            T = len(d["masklet"])
            step = round(float(d["video_frame_count"]) / T) if T else None
            if m.get("frame_count") and T and abs((T - 1) * 4 - (m["frame_count"] - 1)) > 4:
                anomalies.append(anomaly_record(DS, uid, "annotation_rate_unexpected",
                                                f"{T} annotated frames vs {m['frame_count']} video frames",
                                                "frame_step recorded as measured"))
            annots.append(annotation_record(
                uid=f"{uid}:{src}", dataset=DS, media_uid=uid, annotation_type="object_masklet",
                annotation_source=kind, annotator="SA-V annotators (SAM 2 in the loop)" if src == "manual"
                else "SAM 2 automatic pipeline", variant=src, annotation_path=str(p),
                annotation_format="json masklet[T][K] COCO RLE + per-masklet metadata",
                coverage={"annotated_frames": T, "masklets": int(d["masklet_num"]), "frame_step": 4,
                          "measured_step": step, "video_frame_count": float(d["video_frame_count"])},
                label_semantics="spatio-temporal mask of one object/part per masklet id; ids are per file "
                                "(manual and auto both start at 0); empty mask = not visible",
                coord_format="COCO RLE, size [H,W] of the video", coord_target="binary masks (optionally resized)",
                time_reference="annotated frame j = video frame 4*j (6 fps annotations on 24 fps video)",
                valid_mask="only annotated frames; visibility = non-empty mask",
                notes=json.dumps({"stability_score": "auto only" if src == "auto" else None,
                                  "edited_frame_count": d.get("masklet_edited_frame_count")})))
    write_jsonl(root / "processed" / "media.jsonl", media)
    write_jsonl(root / "processed" / "annotations.jsonl", annots)
    write_jsonl(root / "processed" / "anomalies.jsonl", anomalies)
    write_json(root / "processed" / "dataset_card.json", {
        "dataset": DS, "release": "SA-V (CC BY 4.0); indexed: official example only",
        "source": DL_PAGE, "media_indexed": len(media),
        "full_release": {"train_videos": 50583, "val_videos": 155, "test_videos": 150,
                         "train_tars": "sav_000.tar - sav_055.tar (~8 GB each)", "val_test": "~16 GB each"},
        "readiness": "blocked",
        "blockers": ["full download requires clicking through the SA-V license on the Meta download page, which "
                     "issues an expiring link list (manual browser step); once links.txt is provided, "
                     "scripts/fetch_many.py can take a manifest built from it"],
        "not_used": {"Voxel51/segment_anything_video_subset51": "re-encoded 6 fps + converted labels",
                     "pengqinhe/savval": "no provenance/README"},
        "trainable_as": ["object_masklet (1 example video only)"],
        "updated": now()})
    log(DS, "ingest_done", media=len(media), annotations=len(annots))
    print(json.dumps({"media": len(media), "annotations": len(annots)}))


if __name__ == "__main__":
    main()
