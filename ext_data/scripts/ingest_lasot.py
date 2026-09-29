#!/usr/bin/env python3
"""LaSOT (Fan et al. CVPR'19) — small-scale onboarding by category.

Author Google Drive links are dead (404) and OneDrive/Baidu need a browser/login;
the author server only offers the two monolithic zips (50 GB / 63 GB).  Category
zips come from HF ``l-lt/LaSOT`` @a974646 (data card names the LaSOT authors;
account ownership not verifiable) via hf-mirror.com.  They are *repacked*: the
whole-zip md5 differs from the official ``md5.txt``.  Content is therefore checked
member by member (name, size, CRC-32) against the official ``LaSOTTesting.zip``
central directory on the author server (read with Range requests); the
category's test sequences must be byte-identical, train sequences cannot be
checked against any official per-category archive and are marked as such.
The HF split files equal the official ones line by line.

Per sequence ``<cat>-<k>/``: ``img/%08d.jpg`` (1-based), ``groundtruth.txt``
(``x,y,w,h`` per frame), ``full_occlusion.txt`` / ``out_of_view.txt`` (0/1 per
frame), ``nlp.txt`` (one sentence).  Every frame is annotated.  The coordinate
origin is not documented; the ingest measures it (min x/y and max x+w vs W over
visible boxes) and records the evidence; the adapter converts to 0-based pixels
only when the evidence says 1-based.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.common import dataset_dir, ensure_space, log, now, write_json, write_jsonl  # noqa: E402
from aicext.download import fetch, session  # noqa: E402
from aicext.ids import native_id  # noqa: E402
from aicext.schema import anomaly_record, annotation_record, media_record  # noqa: E402

DS = "LaSOT"
HF_REPO, HF_REV = "l-lt/LaSOT", "a97464600e8c1ab91eb9"
MIRROR = "https://hf-mirror.com/datasets/l-lt/LaSOT/resolve/main"
OFFICIAL_MD5 = "http://vision.cs.stonybrook.edu/~lasot/data/md5.txt"
OFFICIAL_TESTING = "http://vision.cs.stonybrook.edu/~lasot/data/LaSOTTesting.zip"
SIZES = {"mouse": 1077414658, "electricfan": 1640207000, "coin": 1768425058, "gecko": 1982252937,
         "cup": 2075002049, "chameleon": 2107065152, "dog": 2135835403}
NOMINAL_FPS = 30.0


def official_md5(root: Path) -> dict[str, str]:
    p = root / "annotations" / "official_md5.txt"
    if not p.exists():
        r = session().get(OFFICIAL_MD5, timeout=60)
        r.raise_for_status()
        p.write_bytes(r.content)
    out, cur = {}, None
    for line in p.read_text(errors="ignore").splitlines():
        m = re.match(r"File:\s*category\\(\w+)\.zip", line.strip())
        if m:
            cur = m.group(1)
        m = re.match(r"MD5:\s*([0-9a-fA-F]{32})", line.strip())
        if m and cur:
            out[cur] = m.group(1).lower()
    return out


def split_lists(root: Path) -> dict[str, str]:
    out = {}
    for name, split in (("training_set.txt", "train"), ("testing_set.txt", "test")):
        p = root / "annotations" / name
        if not p.exists():
            r = session().get(f"{MIRROR}/{name}", timeout=60)
            r.raise_for_status()
            p.write_bytes(r.content)
        for s in p.read_text().split():
            out[s] = split
    return out


def official_testing_cd(root: Path) -> dict[str, tuple[int, int]]:
    p = root / "annotations" / "official_LaSOTTesting_cd.json"
    if not p.exists():
        from aicext.remote_zip import central_directory

        write_json(p, {k: list(v) for k, v in central_directory(OFFICIAL_TESTING).items()})
    return {k: tuple(v) for k, v in json.loads(p.read_text()).items()}


def member_check(root: Path, z: Path, cat: str) -> dict:
    off = official_testing_cd(root)
    with zipfile.ZipFile(z) as f:
        hf = {i.filename: (i.file_size, i.CRC) for i in f.infolist() if not i.is_dir()}
    offc = {k.split("/", 1)[1]: v for k, v in off.items()
            if len(k.split("/")) > 2 and k.split("/")[1].startswith(cat + "-") and not k.endswith("/")}
    common = set(hf) & set(offc)
    return {"official_files": len(offc), "identical": sum(hf[k] == offc[k] for k in common),
            "missing_in_mirror": len(set(offc) - set(hf)), "seqs": sorted({k.split("/")[0] for k in offc}),
            "method": "zip central directory (name, size, CRC-32) vs official LaSOTTesting.zip"}


def md5_file(p: Path) -> str:
    h = hashlib.md5()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_flags(p: Path) -> np.ndarray:
    return np.array([int(x) for x in re.split(r"[,\s]+", p.read_text().strip()) if x != ""], np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--categories", nargs="+", default=["mouse", "electricfan"])
    ap.add_argument("--fetch-only", action="store_true",
                    help="download + member-check + extract only; do not rewrite processed/*.jsonl. "
                         "Used by lasot_full.sh so per-category fetches never clobber the shared index.")
    a = ap.parse_args()
    root = dataset_dir(DS)
    (root / "annotations").mkdir(parents=True, exist_ok=True)
    md5s, splits = official_md5(root), split_lists(root)
    media, annots, anomalies, pkg = [], [], [], {}
    origin_stats = {"min_x": [], "min_y": [], "max_x2_minus_W": [], "max_y2_minus_H": []}
    for cat in a.categories:
        z = root / "raw" / "archives" / f"{cat}.zip"
        fetch(f"{MIRROR}/{cat}.zip", z, dataset=DS, expected_size=SIZES.get(cat), retries=20, quiet=True)
        got = md5_file(z)
        pkg[cat] = {"size": z.stat().st_size, "md5": got, "official_md5": md5s.get(cat),
                    "md5_match": got == md5s.get(cat), "source": f"hf:{HF_REPO}@{HF_REV} via hf-mirror.com"}
        test_verified: set[str] = set()
        if got != md5s.get(cat):
            # HF zips are repacked (whole-file md5 differs); verify members against the official
            # LaSOTTesting.zip central directory (name, size, CRC-32) for this category's test sequences.
            chk = member_check(root, z, cat)
            pkg[cat]["member_check"] = {k: v for k, v in chk.items() if k != "seqs"}
            if chk["official_files"] == 0 or chk["identical"] != chk["official_files"]:
                anomalies.append(anomaly_record(DS, cat, "md5_mismatch", f"{got} vs official {md5s.get(cat)}; "
                                                f"member check {chk}", "category not indexed"))
                continue
            test_verified = set(chk["seqs"])
            anomalies.append(anomaly_record(
                DS, cat, "repacked_archive", f"whole-zip md5 {got} != official {md5s.get(cat)}; "
                f"{chk['identical']}/{chk['official_files']} test-sequence members CRC-identical to LaSOTTesting.zip",
                "indexed; train-sequence members not verifiable (no official per-category archive online)"))
        out = root / "raw" / cat
        if not (out / ".extracted").exists():
            ensure_space(int(1.05 * z.stat().st_size))
            with zipfile.ZipFile(z) as f:
                f.extractall(out)
            (out / ".extracted").write_text(now() + "\n")
        if a.fetch_only:
            continue   # extracted; indexing deferred to the single full pass
        for sd in sorted(p for p in out.iterdir() if p.is_dir()):
            name = sd.name
            imgs = sorted((sd / "img").glob("*.jpg"))
            gt = np.loadtxt(sd / "groundtruth.txt", delimiter=",", ndmin=2).astype(np.float32)
            occ, oov = read_flags(sd / "full_occlusion.txt"), read_flags(sd / "out_of_view.txt")
            nlp = (sd / "nlp.txt").read_text().strip() if (sd / "nlp.txt").exists() else None
            from PIL import Image
            with Image.open(imgs[0]) as im:
                W, H = im.size
            n = len(imgs)
            issues = {}
            if not (len(gt) == len(occ) == len(oov) == n):
                issues["length_mismatch"] = {"frames": n, "gt": len(gt), "occ": len(occ), "oov": len(oov)}
            L = min(n, len(gt), len(occ), len(oov))
            vis = (occ[:L] == 0) & (oov[:L] == 0) & (gt[:L, 2] > 0) & (gt[:L, 3] > 0)
            zero_box = (gt[:L, 2] <= 0) | (gt[:L, 3] <= 0)
            if (zero_box & ((occ[:L] == 0) & (oov[:L] == 0))).any():
                issues["zero_box_but_visible_flag"] = int((zero_box & ((occ[:L] == 0) & (oov[:L] == 0))).sum())
            if vis.any():
                b = gt[:L][vis]
                origin_stats["min_x"].append(float(b[:, 0].min()))
                origin_stats["min_y"].append(float(b[:, 1].min()))
                origin_stats["max_x2_minus_W"].append(float((b[:, 0] + b[:, 2]).max() - W))
                origin_stats["max_y2_minus_H"].append(float((b[:, 1] + b[:, 3]).max() - H))
            uid = f"lasot:{name}"
            m = media_record(
                uid=uid, dataset=DS, release=f"LaSOT (CVPR19) category zip via hf:{HF_REPO}; md5 = official",
                item_id=name, source_video_id=native_id(DS, name), source_platform="youtube (LaSOT)",
                group_id=native_id(DS, name), official_split=splits.get(name, "unlisted"),
                media_kind="frame_folder", media_path=str(sd / "img"), download_url=f"{MIRROR}/{cat}.zip",
                download_source=f"hf:{HF_REPO} via hf-mirror.com (md5 matches {OFFICIAL_MD5})",
                download_status="sample_only", preprocess_status="done", width=W, height=H, fps=NOMINAL_FPS,
                frame_count=n, duration_s=n / NOMINAL_FPS, timeline_method="nominal_30fps_unverified",
                frame_index_base=0, frame_file_pattern="*.jpg", annotation_types=["object_track_box"],
                notes=json.dumps({"category": cat, "image_names": "00000001.jpg = frame index 0",
                                  "content_verification": "whole-zip md5 = official" if pkg[cat]["md5_match"] else
                                  ("members CRC-identical to official LaSOTTesting.zip" if name in test_verified
                                   else "not verifiable against an official archive (repacked mirror)")}))
            if name not in splits:
                anomalies.append(anomaly_record(DS, uid, "not_in_split_lists", name, "hash split"))
            media.append(m)
            ann = root / "annotations" / "tracks" / f"{name}.json"
            write_json(ann, {"sequence": name, "category": cat, "nlp": nlp, "W": W, "H": H,
                             "raw_xywh": gt.tolist(), "full_occlusion": occ.tolist(), "out_of_view": oov.tolist()})
            for k, v in issues.items():
                anomalies.append(anomaly_record(DS, uid, k, json.dumps(v), "kept; adapter uses min length"))
            annots.append(annotation_record(
                uid=f"{uid}:track", dataset=DS, media_uid=uid, annotation_type="object_track_box",
                annotation_source="human", annotator="LaSOT annotators (per-frame, verified)",
                variant="single_target", annotation_path=str(ann),
                annotation_format="json raw_xywh[T,4], full_occlusion[T], out_of_view[T], nlp",
                coverage={"annotated_frames": L, "frames": n, "sampling": "every frame",
                          "visible_frames": int(vis.sum())},
                label_semantics=f"box of the single tracked target ({cat}; '{nlp}'); an object track, not a "
                                "'most important subject' label; invisible frames flagged",
                coord_format="x,y,w,h pixels per line; origin undocumented (see dataset_card.coord_origin_evidence)",
                coord_target="xywh 0-based pixels (x-1, y-1 when evidence says 1-based), clipped for the "
                             "normalized copy only",
                time_reference="line k = img/%08d.jpg with number k+1; nominal 30 fps clock (unverified)",
                valid_mask="valid = frame annotated; visible = not full_occlusion, not out_of_view, w>0, h>0"))
    ev = {k: (float(np.median(v)) if v else None) for k, v in origin_stats.items()}
    mnx = np.array(origin_stats["min_x"] + origin_stats["min_y"])
    ovr = np.array(origin_stats["max_x2_minus_W"] + origin_stats["max_y2_minus_H"])
    ev.update(edges_at_0=int((mnx == 0).sum()), edges_at_1=int((mnx == 1).sum()),
              far_edges_at_size=int((ovr == 0).sum()), far_edges_beyond_size=int((ovr > 0).sum()),
              reading="near edges touch 1 (never 0) while far edges touch W/H (never W+1): neither pure "
                      "0-based-exclusive nor 1-based-inclusive; <=1 px ambiguity, raw values kept")
    ev.update(n_sequences=len(origin_stats["min_x"]),
              frac_min_x_eq_1=float(np.mean(np.array(origin_stats["min_x"]) == 1)) if origin_stats["min_x"] else None,
              frac_x2_eq_W_plus_1=float(np.mean(np.array(origin_stats["max_x2_minus_W"]) == 1))
              if origin_stats["min_x"] else None)
    if a.fetch_only:
        log(DS, "fetch_only_done", categories=len(a.categories), anomalies=len(anomalies),
            extracted=[c for c in a.categories if (root / "raw" / c / ".extracted").exists()])
        print(json.dumps({"fetch_only": True, "categories": a.categories, "packages": pkg,
                          "anomalies": len(anomalies)}))
        return
    one_based = bool(ev["n_sequences"]) and min(origin_stats["min_x"] + origin_stats["min_y"]) >= 0 and \
        max(origin_stats["max_x2_minus_W"] + origin_stats["max_y2_minus_H"]) >= 1 and \
        (ev["frac_min_x_eq_1"] or 0) > 0.3
    for x in annots:
        x["notes"] = json.dumps({"one_based": one_based})
    write_jsonl(root / "processed" / "media.jsonl", media)
    write_jsonl(root / "processed" / "annotations.jsonl", annots)
    write_jsonl(root / "processed" / "anomalies.jsonl", anomalies)
    write_json(root / "processed" / "dataset_card.json", {
        "dataset": DS, "release": "LaSOT (1,400 seq, 70 categories)", "packages": pkg,
        "source": "http://vision.cs.stonybrook.edu/~lasot/", "categories_indexed": a.categories,
        "media_indexed": len(media), "full_release_sequences": 1400,
        "coord_origin_evidence": ev, "coord_origin_decision": "1-based -> subtract 1" if one_based else "0-based",
        "official_split": "training_set.txt (1120) / testing_set.txt (280)",
        "readiness": "raw_video_trainable" if len(media) else "sample_only",
        "trainable_as": ["object_track_box"],
        "blockers": [] if len(media) >= 1390 else
                    [f"{1400 - len(media)} sequences missing/failed this round"],
        "updated": now()})
    log(DS, "ingest_done", media=len(media), annotations=len(annots), one_based=one_based)
    print(json.dumps({"media": len(media), "packages": pkg, "origin": ev, "one_based": one_based}))


if __name__ == "__main__":
    main()
