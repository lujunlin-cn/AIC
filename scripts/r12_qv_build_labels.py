#!/usr/bin/env python3
"""R12-QV P1: build per-clip soft saliency labels from QVHighlights releases.

QV saliency is annotated ONLY inside relevant_windows (3 annotators, 0-4,
2 s clips).  Label = union across queries of the same vid:
  y[c] = max over queries of mean(annotator scores)/4  for clip c inside any
  relevant window; 0 outside.  75 clips per 150 s segment.
Output: /data/aic/experiments_910a/LFM_V11/r12_qv/qv_labels.json
        {labels: {vid: [75]}, meta: {vid: {split,n_q,n_pos}}, keep_rate}
"""
import json
from collections import defaultdict
from pathlib import Path

ANN = Path("/data/aic/external_datasets/QVHighlights/annotations")
OUT = Path("/data/aic/experiments_910a/LFM_V11/r12_qv")
SPLIT = {"train": "highlight_train_release.jsonl",
         "val": "highlight_val_release.jsonl"}
NC = 75


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    labels, meta = {}, {}
    for sp, fn in SPLIT.items():
        per_vid = defaultdict(list)
        for line in open(ANN / fn):
            r = json.loads(line)
            per_vid[r["vid"]].append(r)
        for vid, rows in per_vid.items():
            y = [0.0] * NC
            for r in rows:
                for cid, scs in zip(r["relevant_clip_ids"], r["saliency_scores"]):
                    if 0 <= cid < NC:
                        y[cid] = max(y[cid], sum(scs) / len(scs) / 4.0)
            labels[vid] = y
            meta[vid] = {"split": sp, "n_q": len(rows),
                         "n_pos": sum(1 for v in y if v > 0)}
        print(f"[qlab] {sp}: vids={len(per_vid)} queries={sum(len(v) for v in per_vid.values())}",
              flush=True)
    keep = sum(m["n_pos"] for m in meta.values()) / (NC * len(meta))
    json.dump({"labels": labels, "meta": meta, "keep_rate": round(keep, 4)},
              open(OUT / "qv_labels.json", "w"))
    print(f"QLAB_DONE vids={len(labels)} keep_rate={keep:.4f}", flush=True)


if __name__ == "__main__":
    main()
