#!/usr/bin/env python3
"""Merge fixed TimeLens2 shard outputs without changing their responses."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", nargs="+", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    by = {}
    prompt = None
    for path in args.shards:
        doc = json.loads(path.read_text())
        prompt = prompt or {k: doc.get(k) for k in ("model", "prompt", "fps", "max_pixels")}
        for row in doc.get("results", []):
            vid = str(row["video_id"])
            if vid in by and by[vid] != row:
                raise SystemExit(f"duplicate conflicting video_id={vid}")
            by[vid] = row
    rows = [by[k] for k in sorted(by, key=lambda x: int(x) if x.isdigit() else x)]
    if len(rows) != 426:
        raise SystemExit(f"expected 426 shard rows, got {len(rows)}")
    if any("error" in row for row in rows):
        bad = [row for row in rows if "error" in row]
        raise SystemExit(f"missing/error rows: {bad[:3]}")
    out = dict(prompt or {})
    out["results"] = rows
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"videos": len(rows), "output": str(args.output)}, ensure_ascii=False))

if __name__ == "__main__":
    main()
