"""Dataset registry: per-dataset JSONL -> merged Parquet/JSONL + status table.

Each dataset directory owns ``processed/media.jsonl``, ``processed/annotations.jsonl``
and ``processed/anomalies.jsonl`` plus ``processed/dataset_card.json``.
``build_registry`` merges them into ``_registry/`` without touching raw data.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from .common import EXT_ROOT, REGISTRY_DIR, now, read_jsonl, write_json, write_jsonl
from .schema import ANNOT_REQUIRED, MEDIA_REQUIRED, validate

LIST_FIELDS = ("annotation_types", "anomalies")


def dataset_names() -> list[str]:
    return sorted(p.parent.parent.name for p in EXT_ROOT.glob("*/processed/dataset_card.json"))


def _to_parquet(rows: list[dict], path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        return
    # Nested dict/list values are stored as JSON strings so the schema stays flat.
    keys = sorted({k for r in rows for k in r})
    cols = {}
    for k in keys:
        vals = [r.get(k) for r in rows]
        if any(isinstance(v, (dict, list)) for v in vals):
            vals = [None if v is None else json.dumps(v, ensure_ascii=False) for v in vals]
        elif any(isinstance(v, float) for v in vals) and all(isinstance(v, (int, float)) or v is None
                                                             for v in vals):
            vals = [None if v is None else float(v) for v in vals]
        elif len({type(v) for v in vals if v is not None}) > 1:
            vals = [None if v is None else str(v) for v in vals]
        cols[k] = vals
    tmp = path.with_suffix(".tmp.parquet")
    pq.write_table(pa.table(cols), tmp, compression="zstd")
    tmp.replace(path)


def build_registry() -> dict:
    REGISTRY_DIR.mkdir(parents=True, exist_ok=True)
    media, annots, anomalies, cards = [], [], [], []
    for name in dataset_names():
        d = EXT_ROOT / name / "processed"
        cards.append(json.loads((d / "dataset_card.json").read_text()))
        media += list(read_jsonl(d / "media.jsonl"))
        annots += list(read_jsonl(d / "annotations.jsonl"))
        anomalies += list(read_jsonl(d / "anomalies.jsonl"))
    errors = validate(media, MEDIA_REQUIRED) + validate(annots, ANNOT_REQUIRED)
    media_uids = {m["uid"] for m in media}
    dangling = [a["uid"] for a in annots if a["media_uid"] not in media_uids]
    errors += [f"annotation {u} references unknown media" for u in dangling[:50]]
    write_jsonl(REGISTRY_DIR / "media.jsonl", media)
    write_jsonl(REGISTRY_DIR / "annotations.jsonl", annots)
    write_jsonl(REGISTRY_DIR / "anomalies.jsonl", anomalies)
    _to_parquet(media, REGISTRY_DIR / "media.parquet")
    _to_parquet(annots, REGISTRY_DIR / "annotations.parquet")
    _to_parquet(anomalies, REGISTRY_DIR / "anomalies.parquet")
    status = summarize(media, annots, anomalies, cards)
    status["schema_errors"] = errors[:200]
    status["schema_error_count"] = len(errors)
    write_json(REGISTRY_DIR / "status.json", status)
    return status


def summarize(media, annots, anomalies, cards) -> dict:
    by_ds = defaultdict(lambda: {"media": 0, "media_ready": 0, "bytes": 0, "download_status": Counter(),
                                 "official_split": Counter(), "aic_split": Counter(),
                                 "annotation_types": Counter(), "annotations": 0, "anomalies": Counter()})
    ready = {"verified", "downloaded", "referenced_existing", "sample_only"}
    for m in media:
        s = by_ds[m["dataset"]]
        s["media"] += 1
        s["media_ready"] += m["download_status"] in ready and m["media_path"] is not None
        s["bytes"] += m.get("file_size") or 0
        s["download_status"][m["download_status"]] += 1
        s["official_split"][str(m.get("official_split"))] += 1
        s["aic_split"][str(m.get("aic_split"))] += 1
    for a in annots:
        s = by_ds[a["dataset"]]
        s["annotations"] += 1
        s["annotation_types"][a["annotation_type"]] += 1
    for x in anomalies:
        by_ds[x["dataset"]]["anomalies"][x["kind"]] += 1
    card_by = {c["dataset"]: c for c in cards}
    out = {"generated": now(), "root": str(EXT_ROOT), "datasets": {}}
    for name in sorted(set(by_ds) | set(card_by)):
        s = by_ds[name]
        c = card_by.get(name, {})
        out["datasets"][name] = {
            "readiness": c.get("readiness"), "trainable_as": c.get("trainable_as"),
            "blockers": c.get("blockers"), "release": c.get("release"),
            **{k: (dict(v) if isinstance(v, Counter) else v) for k, v in s.items()},
        }
    return out
