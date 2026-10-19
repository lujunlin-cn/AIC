#!/usr/bin/env python3
"""Merge the fixed TimeLens2 shard JSON files with coverage checks.

The shard runner writes one JSON document per range.  This utility only
concatenates responses; it never changes a response or accesses competition
labels.  A missing, duplicate, or out-of-range video id is a hard error.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=Path, nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--expected", type=int, default=426)
    args = ap.parse_args()

    docs = [json.loads(p.read_text(encoding="utf-8")) for p in args.shard]
    rows = []
    for d in docs:
        if not isinstance(d.get("results"), list):
            raise ValueError("each shard must contain a results list")
        rows.extend(d["results"])
    rows.sort(key=lambda x: int(str(x["video_id"])))
    ids = [str(x["video_id"]) for x in rows]
    expected = [str(i) for i in range(args.expected)]
    if ids != expected:
        raise ValueError(
            f"coverage mismatch: rows={len(rows)} unique={len(set(ids))}; "
            f"first/last={ids[:2]}/{ids[-2:]}"
        )
    base = dict(docs[0])
    base["results"] = rows
    args.output.write_text(json.dumps(base, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "videos": len(rows), "nonempty": sum(bool(x.get("response")) for x in rows)}))


if __name__ == "__main__":
    main()
