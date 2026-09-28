#!/usr/bin/env python3
"""Sequential, resumable download of a manifest of files (one stream at a time).

The manifest is a JSON list of objects with keys:
  dataset, dest (relative to the external root), url | gdrive, expected_size (optional),
  sha256 (optional), note (optional).
Finished files are skipped on re-run; partial files resume from ``<dest>.part``.

Example:
  python scripts/fetch_many.py --manifest configs/batch2_downloads.json
  python scripts/fetch_many.py --manifest configs/batch2_downloads.json --only GAICD DAVSOD
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.common import EXT_ROOT, log, write_json  # noqa: E402
from aicext.download import fetch, gdrive_url  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--only", nargs="*", help="restrict to these datasets")
    ap.add_argument("--retries", type=int, default=30)
    a = ap.parse_args()
    items = json.loads(Path(a.manifest).read_text())
    status = []
    for it in items:
        if a.only and it["dataset"] not in a.only:
            continue
        url = it.get("url") or gdrive_url(it["gdrive"])
        dest = EXT_ROOT / it["dest"]
        log(it["dataset"], "fetch_start", url=url, dest=str(dest), note=it.get("note"))
        try:
            row = fetch(url, dest, dataset=it["dataset"], expected_size=it.get("expected_size"),
                        expected_sha256=it.get("sha256"), retries=a.retries, quiet=True)
            status.append({**it, "status": row.get("status"), "sha256": row.get("sha256"),
                           "size": row.get("size")})
            log(it["dataset"], "fetch_done", path=str(dest), size=row.get("size"),
                sha256=row.get("sha256"), status=row.get("status"))
        except Exception as exc:  # record and continue with the next file
            status.append({**it, "status": "failed", "error": repr(exc)[:400]})
            log(it["dataset"], "fetch_failed", path=str(dest), error=repr(exc)[:400])
        write_json(EXT_ROOT / "_logs" / f"fetch_many_{Path(a.manifest).stem}.json", status)
    bad = [s for s in status if s["status"] == "failed"]
    print(json.dumps({"total": len(status), "failed": len(bad)}, ensure_ascii=False))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
