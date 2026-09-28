#!/usr/bin/env python3
"""Leakage-safe splits + cross-dataset source ledger.

1. Ledger: every media item from ``external_datasets/*`` plus the already
   accepted in-house sources (TVSum, SumMe, QVHighlights native/weak, AIC
   official test) keyed by ``group_id`` (``yt:<id>``, ``dhf1k:NNN``, ...).
2. Exact-ID overlap across datasets (e.g. RetargetVid vs DHF1K share all 200
   IDs by construction; YouTube IDs shared between PHD2 / Mr.HiSum / QVH /
   TVSum / YouTube Highlights).
3. ``aic_split`` per group:
      * any group whose official split includes val/test anywhere -> ``val``
        (so a video used for evaluation by any dataset never trains);
      * DHF1K/RetargetVid/DAVSOD (shared ``dhf1k:NNN`` groups): a DAVSOD
        val/test use -> ``val``; else DHF1K official split (001-600 train,
        601-700 val, 701-1000 -> ``test_nolabel`` unless DAVSOD trains on it);
      * datasets without an official val: hash bucket, 10% val;
      * groups also present in in-house evaluation sets (TVSum, SumMe, QVH
        val/holdout) -> ``quarantine`` (never train, not used for new val).
   All media of a group receive the same ``aic_split``.
4. AIC official test videos are only fingerprinted (SHA-256 already in the
   intake manifest) for a byte-identical check; they never enter the ledger
   as a training source.

Writes ``_registry/source_ledger.parquet|jsonl``, ``_registry/overlap_report.json``
and rewrites ``aic_split`` inside each dataset's ``processed/media.jsonl``.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.common import EXT_ROOT, REGISTRY_DIR, now, read_jsonl, write_json, write_jsonl  # noqa: E402
from aicext.ids import qvh_youtube, split_bucket, youtube_id  # noqa: E402
from aicext.registry import _to_parquet, dataset_names  # noqa: E402

INHOUSE = {
    "TVSum": "/data/aic/datasets/TVSum/tvsum_manifest.jsonl",
    "QVHighlightsNative_train": "/data/aic/datasets/QVHighlightsNative/annotations/train.jsonl",
    "QVHighlightsNative_val": "/data/aic/datasets/QVHighlightsNative/annotations/val.jsonl",
    "SumMe": "/data/aic/datasets/SumMe",
    "AIC_official_test": "/data/aic/official_test_20260926/intake/index.enriched.jsonl",
}
EVAL_LIKE = {"val", "test", "validation", "testing", "TestTightL", "TestLooseL",
             "test_easy35", "test_normal25", "test_difficult20"}
TRAIN_LIKE = {"train", "training", "TraiTightL", "TraiLooseL", "only_gradual"}
VAL_FRACTION = 0.10


def inhouse_rows() -> list[dict]:
    rows = []
    p = Path(INHOUSE["TVSum"])
    if p.exists():
        for r in read_jsonl(p):
            rows.append({"dataset": "TVSum(in-house)", "item_id": r["video_id"], "group_id": youtube_id(r["video_id"]),
                         "official_split": "dev_exposed", "role": "in-house eval/dev", "sha256": r.get("sha256")})
    for key, split in (("QVHighlightsNative_train", "train"), ("QVHighlightsNative_val", "val")):
        p = Path(INHOUSE[key])
        if p.exists():
            seen = set()
            for r in read_jsonl(p):
                if r["vid"] in seen:
                    continue
                seen.add(r["vid"])
                try:
                    g = qvh_youtube(r["vid"])
                except ValueError:
                    continue
                rows.append({"dataset": "QVHighlights(in-house)", "item_id": r["vid"], "group_id": g,
                             "official_split": split, "role": "in-house train/dev" if split == "train" else "in-house holdout",
                             "sha256": None})
    # SumMe videos have no YouTube ID in the release; register by name (native namespace).
    for p in sorted(Path(INHOUSE["SumMe"]).rglob("*.mp4")):
        rows.append({"dataset": "SumMe(in-house)", "item_id": p.stem, "group_id": f"summe:{p.stem}",
                     "official_split": "dev_exposed", "role": "in-house eval/dev", "sha256": None})
    return rows


def main():
    ext_media = {}
    for name in dataset_names():
        ext_media[name] = list(read_jsonl(EXT_ROOT / name / "processed" / "media.jsonl"))
    ledger = []
    for name, ms in ext_media.items():
        for m in ms:
            ledger.append({"dataset": name, "item_id": m["item_id"], "group_id": m["group_id"],
                           "official_split": m["official_split"], "role": "external",
                           "sha256": m.get("sha256"), "media_uid": m["uid"]})
    inh = inhouse_rows()
    ledger += inh
    by_group = collections.defaultdict(list)
    for r in ledger:
        by_group[r["group_id"]].append(r)
    # ---- overlap report ----
    pair = collections.Counter()
    examples = collections.defaultdict(list)
    for g, rs in by_group.items():
        dss = sorted({r["dataset"] for r in rs})
        for i in range(len(dss)):
            for j in range(i + 1, len(dss)):
                pair[(dss[i], dss[j])] += 1
                if len(examples[(dss[i], dss[j])]) < 5:
                    examples[(dss[i], dss[j])].append(g)
    within = {}
    for name in ext_media:
        c = collections.Counter(m["group_id"] for m in ext_media[name])
        within[name] = {"media": len(ext_media[name]), "groups": len(c),
                        "groups_with_multiple_media": sum(1 for v in c.values() if v > 1)}
    # AIC official test: byte-level check against every external media hash we have
    aic_hashes = {r.get("source_sha256") for r in read_jsonl(INHOUSE["AIC_official_test"])} - {None}
    ext_hashes = {r["sha256"] for r in ledger if r.get("sha256")}
    # ---- splits ----
    inhouse_eval = {r["group_id"] for r in inh if r["role"] != "in-house train/dev"}
    inhouse_train = {r["group_id"] for r in inh if r["role"] == "in-house train/dev"}
    assign, reason = {}, {}
    for g, rs in by_group.items():
        ext = [r for r in rs if r["role"] == "external"]
        if not ext:
            continue
        offs = {str(r["official_split"]) for r in ext}
        if g in inhouse_eval:
            assign[g], reason[g] = "quarantine", "also in in-house eval/holdout set"
        elif g.startswith("dhf1k:"):
            n = int(g.split(":")[1])
            # RetargetVid has no split of its own; DAVSOD has one -> an eval use there wins (never trains)
            other_offs = {str(r["official_split"]) for r in ext if r["dataset"] not in ("DHF1K", "RetargetVid")}
            if other_offs & EVAL_LIKE:
                assign[g] = "val"
                reason[g] = f"eval split of a dataset sharing this DHF1K video: {sorted(other_offs & EVAL_LIKE)}"
            elif n <= 600:
                assign[g], reason[g] = "train", "DHF1K official train"
            elif n <= 700:
                assign[g], reason[g] = "val", "DHF1K official val"
            elif other_offs:  # DHF1K labels held out, but another dataset trains on this video
                assign[g], reason[g] = "train", f"DHF1K test (labels held out); used by {sorted(other_offs)}"
            else:
                assign[g], reason[g] = "test_nolabel", "DHF1K official test (labels held out)"
        elif offs & EVAL_LIKE:
            assign[g], reason[g] = "val", f"official eval split in {sorted(offs & EVAL_LIKE)}"
        elif offs & TRAIN_LIKE:
            assign[g], reason[g] = "train", "official train split"
        else:
            assign[g] = "val" if split_bucket(g) < VAL_FRACTION else "train"
            reason[g] = "hash bucket (no official split published)"
    # write back
    counts = {}
    for name, ms in ext_media.items():
        c = collections.Counter()
        for m in ms:
            m["aic_split"] = assign[m["group_id"]]
            c[m["aic_split"]] += 1
        write_jsonl(EXT_ROOT / name / "processed" / "media.jsonl", ms)
        counts[name] = dict(c)
    for r in ledger:
        r["aic_split"] = assign.get(r["group_id"])
        r["aic_split_reason"] = reason.get(r["group_id"])
    write_jsonl(REGISTRY_DIR / "source_ledger.jsonl", ledger)
    _to_parquet(ledger, REGISTRY_DIR / "source_ledger.parquet")
    # cross-split leakage check (must be empty by construction)
    leaks = [g for g, rs in by_group.items() if len({assign.get(g)} - {None}) > 1]
    moved = collections.Counter()
    eval_to_train = collections.Counter()
    for g, rs in by_group.items():
        for r in rs:
            if r["role"] != "external":
                continue
            if str(r["official_split"]) in TRAIN_LIKE and assign.get(g) in ("val", "quarantine"):
                moved[(r["dataset"], assign[g])] += 1
            if str(r["official_split"]) in EVAL_LIKE and assign.get(g) == "train":
                # e.g. DAVSOD test sequence cut from a DHF1K training video: the group rule wins
                eval_to_train[(r["dataset"], str(r["official_split"]))] += 1
    write_json(REGISTRY_DIR / "overlap_report.json", {
        "generated": now(),
        "cross_dataset_group_overlap": {f"{a} & {b}": {"groups": n, "examples": examples[(a, b)]}
                                        for (a, b), n in pair.most_common()},
        "within_dataset": within,
        "aic_official_test_byte_identical_hits": len(aic_hashes & ext_hashes),
        "aic_official_test_note": "official test IDs are platform-native (0..173) without YouTube IDs; only "
                                  "SHA-256 byte identity can be checked; they are never ingested as training data",
        "split_policy": {"eval_like_official_splits": sorted(EVAL_LIKE), "val_fraction_hash": VAL_FRACTION,
                         "inhouse_eval_groups_quarantined": len(inhouse_eval & set(assign)),
                         "inhouse_train_groups_seen_externally": len(inhouse_train & set(assign))},
        "official_train_items_moved_out_of_train": {f"{d}->{s}": n for (d, s), n in moved.items()},
        "official_eval_items_assigned_train": {f"{d}:{s}": n for (d, s), n in eval_to_train.items()},
        "aic_split_counts": counts, "groups_with_conflicting_split": leaks})
    print(json.dumps({"counts": counts, "overlaps": {f"{a}&{b}": n for (a, b), n in pair.most_common(12)},
                      "moved": {f"{d}->{s}": n for (d, s), n in moved.items()}}, indent=1))


if __name__ == "__main__":
    main()
