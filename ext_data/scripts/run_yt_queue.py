#!/usr/bin/env python3
"""Work through a dataset's YouTube queue (resumable, bounded retries).

  python scripts/run_yt_queue.py --dataset YouTubeHighlights --limit 200
Writes ``logs/yt_failures.json`` (latest status of every non-done id).
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext import ytqueue  # noqa: E402
from aicext.common import EXT_ROOT, log, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--proxy", default=os.environ.get("AIC_EXT_PROXY") or None)
    a = ap.parse_args()
    q = EXT_ROOT / a.dataset / "logs" / "yt_queue.jsonl"
    ytqueue.reset_transient(q)
    n_re = ytqueue.reclassify(q)
    if n_re:
        log(a.dataset, "yt_queue_reclassified", gave_up=n_re)
    stats = ytqueue.run(q, max_attempts=a.max_attempts, limit=a.limit, proxy=a.proxy)
    cur = ytqueue.latest(q)
    fails = [v for v in cur.values() if v["status"] != "done"]
    write_json(EXT_ROOT / a.dataset / "logs" / "yt_failures.json", fails)
    log(a.dataset, "yt_queue_done", **stats, remaining=len(fails),
        done_total=sum(v["status"] == "done" for v in cur.values()))


if __name__ == "__main__":
    main()
