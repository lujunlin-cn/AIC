#!/usr/bin/env python3
"""Fetch and summarize the public AIC semifinal leaderboard.

This uses only the public ranking endpoint. It never accesses test labels,
videos, submissions, or platform evaluation internals.
"""
from __future__ import annotations

import argparse
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen


API = "https://jluat-smart-app-api.yuntu.cn/third/jsphb"
DEFAULT_PAYLOAD = {
    "pageNo": 0,
    "pageSize": 200,
    "type": "JSDF",
    "rwId": "4863942083800600605",
    "stbh": "4863942083800600619",
    "jd": "复赛",
}


def fetch(payload: dict) -> dict:
    body = json.dumps(payload, ensure_ascii=False).encode()
    req = Request(API, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=30) as response:  # noqa: S310 - fixed public URL
        result = json.load(response)
    if not result.get("success"):
        raise RuntimeError(result)
    return result


def summarize(result: dict) -> dict:
    rows = [r for r in result.get("data", []) if r.get("FS_") not in (None, "")]
    scores = sorted((float(r["FS_"]) for r in rows), reverse=True)
    if not scores:
        raise RuntimeError("public endpoint returned no scores")
    def count(lo: float, hi: float | None = None) -> int:
        return sum(s >= lo and (hi is None or s < hi) for s in scores)
    nearest = min(range(len(scores)), key=lambda i: abs(scores[i] - 36.43)) + 1
    return {
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "api": API,
        "stage": "复赛",
        "rows": len(scores),
        "min": scores[-1],
        "max": scores[0],
        "median": statistics.median(scores),
        "count_ge_45": count(45),
        "count_ge_40": count(40),
        "count_38_40": count(38, 40),
        "count_34_38": count(34, 38),
        "count_lt_30": count(float("-inf"), 30),
        "nearest_rank_to_36_43": nearest,
        "nearest_score_to_36_43": scores[nearest - 1],
        "scores_desc": scores,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    result = fetch(DEFAULT_PAYLOAD)
    report = {"payload": DEFAULT_PAYLOAD, "summary": summarize(result), "data": result["data"]}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
