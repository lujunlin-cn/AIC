#!/usr/bin/env python3
"""Pick the PHD2 download subset within a byte budget (user cap: 500 GB).

Composition, in priority order:
  tier0  every testing.csv video hosting an is_last GIF — the official test
         objective is defined on those (845 videos, ~55 h, ~33 GB expected);
  tier1  training videos ranked by supervision density (GIF intervals per video
         hour), duration within [--min-dur, --max-dur] to bound per-file cost;
  tier2  remaining testing.csv videos by interval count — eval-only spares,
         only reached if the budget survives tier1.

The list is a *preference order*, not a fixed selection: the runner stops on
ACTUAL downloaded bytes (~30% of ids are dead), so it walks deeper than the
nominal budget whenever ids fail.  Rank ties break on the id for determinism.

  python scripts/select_phd2_subset.py --budget-gb 420
Writes ``PHD2/splits/subset_download_v1.json`` (+ ``.ids.txt``, one id per line).
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext import ytqueue  # noqa: E402
from aicext.common import EXT_ROOT, dataset_dir, log, now, write_json  # noqa: E402

RATE_MB_PER_H = 595.0   # measured on the 25 sample downloads (height<=720), 2026-09-29


def load(name: str) -> dict[str, dict]:
    vids: dict[str, dict] = {}
    with open(dataset_dir("PHD2") / "annotations" / "upstream_repo" / name) as f:
        for r in csv.DictReader(f):
            v = vids.setdefault(r["youtubeId"], {"dur": 0.0, "iv": 0, "users": set(), "last": 0, "reach": 0.0})
            v["iv"] += 1
            v["users"].add(r["user_id"])
            if r["is_last"] == "True":
                v["last"] += 1
            if r["video_duration"]:
                v["dur"] = max(v["dur"], float(r["video_duration"]))
            v["reach"] = max(v["reach"], float(r["start"]) + float(r["duration"]))
    for v in vids.values():   # rows with empty video_duration: bound the video by its last GIF
        v["dur"] = v["dur"] or v["reach"]
        v["users"] = len(v["users"])
    return vids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget-gb", type=float, default=420.0,
                    help="stop-the-runner budget for media bytes (subset of the user's 500 GB cap)")
    ap.add_argument("--min-dur", type=float, default=90.0)
    ap.add_argument("--max-dur", type=float, default=7200.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    test, train = load("testing.csv"), load("training.csv")
    queue = EXT_ROOT / "PHD2" / "logs" / "yt_queue.jsonl"
    dead = {y for y, r in ytqueue.latest(queue).items() if r["status"] == "gave_up"}

    tier0 = sorted((y for y, v in test.items() if v["last"] > 0 and y not in dead),
                   key=lambda y: (-test[y]["last"], y))
    used = set(tier0)
    tier1 = sorted((y for y, v in train.items()
                    if y not in dead and y not in used and a.min_dur <= v["dur"] <= a.max_dur),
                   key=lambda y: (-(train[y]["iv"] / (train[y]["dur"] / 3600)), y))
    used |= set(tier1)
    tier2 = sorted((y for y, v in test.items() if y not in dead and y not in used),
                   key=lambda y: (-test[y]["iv"], y))

    order = tier0 + tier1 + tier2
    h0 = sum(test[y]["dur"] for y in tier0) / 3600
    h1, i1, n1 = 0.0, 0, 0
    for y in tier1:
        if h1 * RATE_MB_PER_H / 1000 >= a.budget_gb - h0 * RATE_MB_PER_H / 1000:
            break
        h1 += train[y]["dur"] / 3600
        i1 += train[y]["iv"]
        n1 += 1
    out = a.out or str(dataset_dir("PHD2") / "splits" / "subset_download_v1.json")
    write_json(Path(out), {
        "name": "phd2_subset_download_v1", "created": now(),
        "budget_bytes": int(a.budget_gb * 1e9), "rate_mb_per_h": RATE_MB_PER_H,
        "rate_note": "median 548 / mean 595 MB per video-hour measured on 25 samples, height<=720 cap",
        "rules": {"tier0": "testing.csv videos hosting is_last GIFs (official test objective)",
                  "tier1": "training videos, 90s<=dur<=2h, ranked by intervals/hour desc",
                  "tier2": "remaining testing.csv videos by interval count (eval-only spares)",
                  "excluded": f"ids with ytqueue status gave_up ({len(dead)} known dead)"},
        "nominal_reach": {"tier0_videos": len(tier0), "tier0_hours": round(h0, 1),
                          "tier1_videos_in_budget": n1, "tier1_intervals_in_budget": i1,
                          "tier1_hours_in_budget": round(h1, 1)},
        "tiers": {"tier0": tier0, "tier1": tier1, "tier2": tier2},
        "order": order})
    ids_txt = Path(out).with_suffix(".ids.txt")
    ids_txt.write_text("\n".join(order) + "\n")
    log("PHD2", "subset_selected", tier0=len(tier0), tier1=len(tier1), tier2=len(tier2),
        dead_excluded=len(dead), budget_gb=a.budget_gb)
    print(json.dumps({"tier0": len(tier0), "tier0_hours": round(h0, 1),
                      "tier1": len(tier1), "tier2": len(tier2),
                      "tier1_in_budget": {"videos": n1, "intervals": i1, "hours": round(h1, 1)},
                      "expected_gb_tier0": round(h0 * RATE_MB_PER_H / 1000, 1),
                      "dead_excluded": len(dead), "order_len": len(order),
                      "out": out, "ids": str(ids_txt)}, indent=1))


if __name__ == "__main__":
    main()
