#!/usr/bin/env python3
"""Freeze the holdout / exposure ledger (and the fresh confirmation reserve).

  python scripts/freeze_holdout_ledger.py --version v1 \
      --qvh-bounded configs/inhouse/qvh_native_bounded_v1.json \
      --reserve LIVE_YT_VC:0.15 --reserve YouTubeHighlights:0.15

Exposures are read from experiment outputs (aicext.holdout.collect).  The
reserve takes, per listed dataset, the groups that are working-split ``train``
and were never used for selection / diagnostic / confirmation, and keeps those
with sha256("confirm_reserve_v1:<group>") bucket < fraction.  Reserved groups
become ``holdout_confirm_reserved``: never train, not for model selection.

Writes ``_registry/holdout_exposure_<version>.json`` (refuses to overwrite),
points ``holdout_exposure_current.json`` at it and copies it into
``configs/`` so the code snapshot of every release carries it.
"""
import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.common import REGISTRY_DIR, now, read_jsonl, write_json  # noqa: E402
from aicext.holdout import CURRENT, RESERVE_SALT, collect, reserve_bucket, worst_role  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", required=True)
    ap.add_argument("--qvh-bounded", type=Path, default=None)
    ap.add_argument("--reserve", action="append", default=[], help="DATASET:FRACTION")
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    out = REGISTRY_DIR / f"holdout_exposure_{a.version}.json"
    if out.exists():
        raise SystemExit(f"{out} exists; ledgers are frozen, use a new --version")
    led = collect(qvh_bounded=a.qvh_bounded)
    by = defaultdict(list)
    for e in led["exposures"]:
        by[e["group_id"]].append(e)
    # working split per (dataset, group) from the last refresh
    work = {}
    for r in read_jsonl(REGISTRY_DIR / "source_ledger.jsonl"):
        if r.get("role") == "external":
            work[(r["dataset"], r["group_id"])] = r.get("aic_split")
    reservations = {}
    for spec in a.reserve:
        ds, frac = spec.split(":")
        frac = float(frac)
        for (d, g), sp in sorted(work.items()):
            if d != ds or sp != "train" or g in reservations:
                continue
            roles = {e["role"] for e in by.get(g, [])}
            if roles - {"fit"}:
                continue
            if reserve_bucket(g) < frac:
                fit = sorted({f"{e['experiment']}:{e['split_name']}" for e in by.get(g, [])})
                reservations[g] = {"dataset": ds, "fraction": frac, "salt": RESERVE_SALT,
                                   "fit_exposure": fit,
                                   "reason": f"fresh confirmation reserve ({RESERVE_SALT}, {ds} train bucket < {frac})"
                                             + (f"; fit-only exposure {', '.join(fit)}" if fit else "")}
    led.update(version=a.version, note=a.note, reservations=reservations)
    summ = {"groups_exposed": len(by),
            "reserved": dict(Counter(r["dataset"] for r in reservations.values())),
            "by_experiment": dict(Counter(f"{e['experiment']}:{e['split_name']}:{e['role']}" for e in led["exposures"])),
            "by_dataset_worst_role": {}}
    for ds in sorted({e["dataset"] for e in led["exposures"]}):
        summ["by_dataset_worst_role"][ds] = dict(Counter(
            worst_role([e for e in es if e["dataset"] == ds]) for es in by.values() if any(e["dataset"] == ds for e in es)))
    led["summary"] = summ
    write_json(out, led)
    write_json(CURRENT, {"version": a.version, "path": str(out), "set": now()})
    shutil.copyfile(out, Path(__file__).resolve().parents[1] / "configs" / out.name)
    print(json.dumps(summ, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
