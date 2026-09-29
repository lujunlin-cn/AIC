#!/usr/bin/env python3
"""Download the PHD2 curated subset with a hard byte budget (resumable).

  python scripts/run_phd2_subset.py                # uses subset_download_v1.json
Needs yt-dlp + deno on PATH (see env.sh note) and AIC_EXT_PROXY for YouTube.
Stops once media bytes on disk reach the subset's budget_bytes; re-running
continues where it stopped (done files are skipped, budget counts all media).
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext import ytqueue  # noqa: E402
from aicext.common import EXT_ROOT, dataset_dir, log, write_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=None,
                    help="subset json (default: PHD2/splits/subset_download_v1.json)")
    ap.add_argument("--max-attempts", type=int, default=3)
    ap.add_argument("--proxy", default=os.environ.get("AIC_EXT_PROXY") or None)
    a = ap.parse_args()
    sub_path = Path(a.subset or dataset_dir("PHD2") / "splits" / "subset_download_v1.json")
    sub = json.loads(sub_path.read_text())
    q = EXT_ROOT / "PHD2" / "logs" / "yt_queue.jsonl"
    ytqueue.reset_transient(q)
    n_re = ytqueue.reclassify(q)
    if n_re:
        log("PHD2", "yt_queue_reclassified", gave_up=n_re)
    stats = ytqueue.run(q, max_attempts=a.max_attempts, proxy=a.proxy,
                        order=sub["order"], budget_bytes=sub["budget_bytes"])
    cur = ytqueue.latest(q)
    fails = [v for v in cur.values() if v["status"] != "done"]
    write_json(EXT_ROOT / "PHD2" / "logs" / "yt_failures.json", fails)
    done_total = sum(v["status"] == "done" for v in cur.values())
    log("PHD2", "phd2_subset_run_done", **{k: v for k, v in stats.items()},
        subset=str(sub_path), remaining=len(fails), done_total=done_total)
    print(json.dumps({"stats": stats, "done_total": done_total, "remaining": len(fails)}))


if __name__ == "__main__":
    main()
