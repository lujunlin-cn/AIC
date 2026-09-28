#!/usr/bin/env python3
"""GAICD (Zeng et al., CVPR'19 conference + TPAMI'20 journal) — grid-anchor crop candidates with MOS.

Packages (Google Drive, author READMEs):
  conference  GAIC.zip 122,582,995 B  -> images/{train 1036,test 200}, annotations/*.txt (flat)
  journal     GAIC.zip 322,484,500 B  -> images/{train 2636,val 200,test 500}, annotations/<split>/*.txt
The journal release is a superset of the conference images; its ``val`` was carved
out of ``test`` (``GAIC/untitled.m``: rng(1024), 200 random test images -> val), so
50 conference-test images are journal-val.

Annotation line: ``y1 x1 y2 x2 MOS`` in source pixels (verified against image size:
columns 0/2 stay < H, 1/3 stay < W; the author loader reads xmin=col1, ymin=col0,
xmax=col3, ymax=col2).  MOS = mean of >=7 raters on a 1-5 scale; ``-2`` marks
candidates that were not rated (the author loader drops them) -> valid=False here.

One media record per image (group ``gaicd:<image id>``) so all candidates of an
image stay in one split.  Journal and conference annotations are kept as separate
annotation variants; identical files are recorded as such.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.common import dataset_dir, ensure_space, log, now, sha256_file, write_json, write_jsonl  # noqa: E402
from aicext.schema import anomaly_record, annotation_record, media_record  # noqa: E402

DS = "GAICD"
PKGS = {
    "journal": {"zip": "GAIC_journal.zip", "gdrive": "1tDdQqDe8dMoMIVi9Z0WWI5vtRViy01nR", "size": 322484500,
                "sha256": "b895a3f9e03c8f70c37194370441dc2f17bb60e0ab437447ee240b003cd6550b",
                "readme": "https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch"},
    "conference": {"zip": "GAIC_conference.zip", "gdrive": "1KhmyjoimsQVXqPnLjKZiU4iXNKNyyxqW", "size": 122582995,
                   "sha256": "7735e69e7a608703a341b1a084c9c470678390e133121b3fe589604d605a0fa0",
                   "readme": "https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping"},
}
INVALID_MOS = (-2.0, -1.0)


def extract(root: Path, rel: str) -> Path:
    out = root / "raw" / rel
    marker = out / ".extracted"
    if not marker.exists():
        z = root / "raw" / "archives" / PKGS[rel]["zip"]
        ensure_space(2 * z.stat().st_size)
        with zipfile.ZipFile(z) as f:
            f.extractall(out)
        marker.write_text(now() + "\n")
        log(DS, "extracted", release=rel, files=sum(1 for _ in out.rglob("*") if _.is_file()))
    return out / "GAIC"


def read_ann(p: Path) -> np.ndarray:
    return np.loadtxt(p, ndmin=2).astype(np.float32)


def file_md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    root = dataset_dir(DS)
    for rel, meta in PKGS.items():
        z = root / "raw" / "archives" / meta["zip"]
        if z.stat().st_size != meta["size"] or sha256_file(z) != meta["sha256"]:
            raise SystemExit(f"{z} does not match the recorded size/sha256")
    jroot, croot = extract(root, "journal"), extract(root, "conference")

    conf_split = {p.stem: p.parent.name for p in (croot / "images").glob("*/*.jpg")}
    media, annots, anomalies = [], [], []
    ann_dir = root / "annotations" / "candidates"
    ann_dir.mkdir(parents=True, exist_ok=True)
    stats = {"candidates": 0, "unrated": 0, "conf_identical": 0, "conf_differs": 0, "image_differs": 0}
    for img in sorted((jroot / "images").glob("*/*.jpg")):
        split, iid = img.parent.name, img.stem
        uid = f"gaicd:{iid}"
        with Image.open(img) as im:
            W, H = im.size
        m = media_record(
            uid=uid, dataset=DS, release="GAICD journal (TPAMI) package; conference (CVPR19) labels as variant",
            item_id=iid, source_video_id=uid, source_platform="flickr (via GAICD)", group_id=uid,
            official_split=split, media_kind="image", media_path=str(img),
            download_url=f"gdrive:{PKGS['journal']['gdrive']}", download_source=PKGS["journal"]["readme"],
            file_size=img.stat().st_size, sha256=sha256_file(img), download_status="verified",
            preprocess_status="done", width=W, height=H, frame_count=1,
            annotation_types=["image_crop_candidates"],
            notes=json.dumps({"conference_split": conf_split.get(iid)}))
        media.append(m)
        variants = [("journal", jroot / "annotations" / split / f"{iid}.txt")]
        if iid in conf_split:
            variants.append(("conference", croot / "annotations" / f"{iid}.txt"))
            cimg = croot / "images" / conf_split[iid] / f"{iid}.jpg"
            if file_md5(cimg) != file_md5(img):
                stats["image_differs"] += 1
                anomalies.append(anomaly_record(DS, uid, "image_bytes_differ", "conference vs journal jpg differ",
                                                "journal image used as media; conference boxes refer to its own jpg"))
        jarr = None
        for var, p in variants:
            if not p.exists():
                anomalies.append(anomaly_record(DS, uid, "missing_annotation", f"{var}: {p.name}", "variant skipped"))
                continue
            arr = read_ann(p)
            if arr.shape[1] != 5:
                anomalies.append(anomaly_record(DS, uid, "bad_annotation", f"{var}: shape {arr.shape}", "skipped"))
                continue
            if var == "journal":
                jarr = arr
            elif jarr is not None:
                same = arr.shape == jarr.shape and np.allclose(arr, jarr)
                stats["conf_identical" if same else "conf_differs"] += 1
            y1, x1, y2, x2, mos = arr.T
            rated = ~np.isin(mos, INVALID_MOS)
            oob = (x1 < 0) | (y1 < 0) | (x2 > W) | (y2 > H) | (x2 <= x1) | (y2 <= y1)
            if oob.any():
                anomalies.append(anomaly_record(DS, uid, "out_of_bounds_boxes", f"{var}: {int(oob.sum())} boxes",
                                                "kept; adapter marks them invalid"))
            if var == "journal":
                stats["candidates"] += len(arr)
                stats["unrated"] += int((~rated).sum())
            out = ann_dir / f"{iid}__{var}.json"
            write_json(out, {"image": iid, "release": var, "W": W, "H": H,
                             "raw_y1x1y2x2_mos": arr.round(4).tolist(),
                             "boxes_xywh": np.stack([x1, y1, x2 - x1, y2 - y1], 1).round(3).tolist(),
                             "mos": mos.round(4).tolist(), "valid": (rated & ~oob).tolist(),
                             "source_file": str(p)})
            annots.append(annotation_record(
                uid=f"{uid}:{var}", dataset=DS, media_uid=uid, annotation_type="image_crop_candidates",
                annotation_source="human", annotator=">=7 trained raters per crop (MOS)", variant=var,
                annotation_path=str(out), annotation_format="json boxes_xywh[K,4], mos[K], valid[K], raw rows",
                coverage={"candidates": int(len(arr)), "rated": int(rated.sum())},
                label_semantics="mean opinion score 1-5 of each grid-anchor candidate crop (relative within image); "
                                "-2 = not rated (dropped)",
                coord_format="txt rows 'y1 x1 y2 x2 MOS' in source pixels (author loader: xmin=col1, ymin=col0, "
                             "xmax=col3, ymax=col2)",
                coord_target="xywh pixels with w=x2-x1, h=y2-y1 (x2/y2 treated as exclusive; the author code does "
                             "not define it, <=1 px ambiguity); normalized by (W,H)",
                time_reference="single image", valid_mask="valid = rated (MOS not -2/-1) and box inside image",
                notes="official split from the journal package" if var == "journal" else
                      "conference labels for the same image; journal is the default variant"))
    write_jsonl(root / "processed" / "media.jsonl", media)
    write_jsonl(root / "processed" / "annotations.jsonl", annots)
    write_jsonl(root / "processed" / "anomalies.jsonl", anomalies)
    splits = {}
    for m in media:
        splits[m["official_split"]] = splits.get(m["official_split"], 0) + 1
    write_json(root / "processed" / "dataset_card.json", {
        "dataset": DS, "release": "journal + conference packages (Google Drive, author READMEs)",
        "packages": PKGS, "source": "https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch",
        "media_expected": 3336, "media_verified": len(media), "official_split_counts": splits,
        "annotation_stats": stats,
        "official_split": "journal train/val/test (val carved from test by GAIC/untitled.m)",
        "readiness": "image_trainable" if len(media) == 3336 else "partial",
        "trainable_as": ["image_crop_candidates"], "blockers": [],
        "notes": "Still images (not video). Candidate crops + MOS only; no single 'ground-truth' box.",
        "updated": now()})
    log(DS, "ingest_done", media=len(media), annotations=len(annots), **stats)
    print(json.dumps({"media": len(media), "annotations": len(annots), **stats, "splits": splits}))


if __name__ == "__main__":
    main()
