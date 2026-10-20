#!/usr/bin/env python3
"""Build a source-disjoint Valen crop-choice SFT JSONL from public native GT.

Only public TRAIN rows are emitted by default.  Candidate images are materialized
once and the target is a soft distribution over the same legal candidates used at
inference.  No official media or labels are read.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PIL import Image


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--media-out", required=True)
    ap.add_argument("--split", default="train", choices=["train", "dev", "all"])
    ap.add_argument("--temperature", type=float, default=0.12)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    records = [json.loads(x) for x in Path(args.records).read_text().splitlines()]
    out = Path(args.out); media = Path(args.media_out); out.parent.mkdir(parents=True, exist_ok=True); media.mkdir(parents=True, exist_ok=True)
    selected = [r for r in records if args.split == "all" or r.get("split") == args.split]
    # Keep all frames of a source together; optional limit is a deterministic row prefix.
    if args.limit: selected = selected[:args.limit]
    lines = []
    for n, r in enumerate(selected):
        src = Path(r["image"]); assert src.is_file(), src
        with Image.open(src) as im:
            im = im.convert("RGB")
            sx, sy = im.width / float(r["W"]), im.height / float(r["H"])
            folder = media / hashlib.sha256((r["key"] + str(r["ratio"]) + ":sft-v1").encode()).hexdigest()[:20]
            folder.mkdir(parents=True, exist_ok=True)
            candidates = list(r["candidates4"])
            paths = []
            for j in candidates:
                b = r["boxes"][j]
                box = (round(b[0] * sx), round(b[1] * sy), round(b[2] * sx), round(b[3] * sy))
                p = folder / f"{j}.jpg"
                if not p.exists(): im.crop(box).save(p, quality=90, optimize=False)
                paths.append(p.resolve())
        keys = [chr(65 + i) for i in range(len(paths))]
        # IoU is the public native crop utility.  Soft targets preserve legitimate
        # near-ties without allowing a low-IoU candidate to dominate.
        u = np.asarray([float(r["iou"][j]) for j in candidates], dtype=np.float64)
        z = np.exp((u - u.max()) / max(args.temperature, 1e-6)); z /= z.sum()
        content = [
            {"type": "text", "text":
             "This is the complete source frame. The images that follow are exact candidate crops from this same frame. "
             "Each candidate already has the required output aspect ratio."},
            {"type": "image_url", "image_url": {"url": str(src.resolve())}},
        ]
        criteria = {}
        for label, p in zip(keys, paths):
            content.append({"type": "text", "text":
                            f"Candidate crop {label}. The image immediately after this description is the exact candidate crop. "
                            "The option name carries no preference; judge crop quality from visible pixels."})
            content.append({"type": "image_url", "image_url": {"url": str(p)}})
            criteria[label] = f"Candidate crop {label}; compare the visible pixels only."
        record = {
            "group_id": f"valen-crop-{r['source']}",
            "request": {"state": {"messages": [{"role": "user", "content": content}]},
                        "questions": {"crop": {"type": "choice",
                            "instructions": "Choose exactly one crop that keeps the important subject, action, interactions and useful context while avoiding harmful clipping. All candidates have the required aspect ratio. Compare visible pixels; option names are not quality labels.",
                            "criteria": criteria}}},
            "targets": {"crop": {"probabilities": {k: float(v) for k, v in zip(keys, z)}}},
            "meta": {"source": r["source"], "split": r.get("split"), "ratio": r["ratio"], "row_key": r["key"], "label_source": "public_native_gt_iou"},
        }
        lines.append(json.dumps(record, ensure_ascii=False))
    out.write_text("\n".join(lines) + "\n")
    print(json.dumps({"records": len(lines), "split": args.split, "out": str(out), "media": str(media), "temperature": args.temperature}, indent=2))


if __name__ == "__main__":
    main()
