#!/usr/bin/env python3
"""Compute reproducible repeatability diagnostics for the V1.3 CSV exports.

This measures the same account's repeated labels on matched source windows.  It
does not estimate human accuracy or official-label agreement.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


GRADE = {"0普通": 0, "1有信息": 1, "2值得保留": 2, "3高光核心": 3}
PRIMARY = {
    "GtJXHRFhm0E_210.0_360.0",
    "bGU97Zw6Jfg_60.0_210.0",
    "eLlUVxVLiLU_60.0_210.0",
    "m6U-5Hck5z0_60.0_210.0",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict[tuple[str, str], dict[str, str]]:
    with path.open(newline="") as f:
        rows = csv.DictReader(f)
        return {(r["video_id"], r["time_sec"]): r for r in rows}


def kappa(x: list, y: list, levels: list, power: int | None = None) -> float:
    n = len(x)
    if n == 0:
        return None
    ix = {v: i for i, v in enumerate(levels)}
    cm = [[0 for _ in levels] for _ in levels]
    for a, b in zip(x, y):
        cm[ix[a]][ix[b]] += 1
    px = [sum(row) / n for row in cm]
    py = [sum(cm[i][j] for i in range(len(levels))) / n for j in range(len(levels))]
    def w(i, j):
        if power is None:
            return 1.0 if i == j else 0.0
        return 1.0 - (abs(i - j) / max(len(levels) - 1, 1)) ** power
    po = sum(cm[i][j] * w(i, j) for i in range(len(levels)) for j in range(len(levels))) / n
    pe = sum(px[i] * py[j] * w(i, j) for i in range(len(levels)) for j in range(len(levels)))
    return None if abs(1.0 - pe) < 1e-12 else (po - pe) / (1.0 - pe)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project1", type=Path, required=True)
    ap.add_argument("--project2", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    a, b = load(args.project1), load(args.project2)
    keys = sorted((set(a) & set(b)) & {(v, t) for v, t in set(a) if v in PRIMARY})
    if len(keys) != 120:
        raise SystemExit(f"expected 120 primary V1.3 matches, found {len(keys)}")
    out = {
        "purpose": "same-account repeatability diagnostic; not inter-rater accuracy or official GT",
        "primary_v13_video_ids": sorted(PRIMARY),
        "n_frames": len(keys),
        "input_sha256": {"project1": sha256(args.project1), "project2": sha256(args.project2)},
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "fields": {},
        "per_source": {},
    }
    for field in ["generic_highlight", "event_phase", "event_type", "confidence", "needs_context"]:
        pairs = [(a[k].get(field, ""), b[k].get(field, "")) for k in keys]
        pairs = [(x, y) for x, y in pairs if x and y]
        x, y = zip(*pairs) if pairs else ([], [])
        item = {"n": len(pairs), "exact_rate": sum(i == j for i, j in pairs) / len(pairs) if pairs else None}
        if field == "generic_highlight":
            gx, gy = [GRADE[v] for v in x], [GRADE[v] for v in y]
            item.update({
                "within_one_rate": sum(abs(i - j) <= 1 for i, j in zip(gx, gy)) / len(gx),
                "mean_absolute_grade_error": sum(abs(i - j) for i, j in zip(gx, gy)) / len(gx),
                "kappa": kappa(list(x), list(y), list(GRADE), None),
                "linear_weighted_kappa": kappa(gx, gy, list(range(4)), 1),
                "quadratic_weighted_kappa": kappa(gx, gy, list(range(4)), 2),
                "confusion": {v: dict(Counter(yy for xx, yy in pairs if xx == v)) for v in GRADE},
            })
        else:
            item["kappa"] = kappa(list(x), list(y), sorted(set(x) | set(y)), None)
        out["fields"][field] = item
    for vid in sorted(PRIMARY):
        ks = [k for k in keys if k[0] == vid]
        g = [(GRADE[a[k]["generic_highlight"]], GRADE[b[k]["generic_highlight"]]) for k in ks]
        out["per_source"][vid] = {
            "n": len(ks),
            "project1_grade_distribution": dict(Counter(a[k]["generic_highlight"] for k in ks)),
            "project2_grade_distribution": dict(Counter(b[k]["generic_highlight"] for k in ks)),
            "exact_grade_rate": sum(x == y for x, y in g) / len(g),
            "mean_absolute_grade_error": sum(abs(x - y) for x, y in g) / len(g),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"n_frames": len(keys), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
