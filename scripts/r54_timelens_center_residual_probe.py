#!/usr/bin/env python3
"""Public source-held-out probe for the R54 TimeLens2 residual rule."""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.r44_local_state_residual import GRADE, proxy_f, select

SPAN_RE = re.compile(r"\[\s*\[.*?\]\s*\]")


def parse(text: str):
    m = SPAN_RE.search(text or "")
    if not m:
        return []
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return []
    return [(float(x[0]), float(x[1])) for x in arr if len(x) >= 2]


def rank01(x):
    return np.argsort(np.argsort(x, kind="mergesort"), kind="mergesort").astype(float) / max(len(x) - 1, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", nargs=2, type=Path, required=True)
    ap.add_argument("--responses", type=Path, required=True,
                    help="public proxy JSON containing source spans, not the official 0-425 response file")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    resp = json.loads(args.responses.read_text())
    if "source" in resp:
        # The public raw responses are retained remotely only as a compact
        # proxy report; use its source/stream-aligned spans.
        spans = {(str(x["video_id"]), int(x["stream"])): [tuple(z) for z in x.get("span_seconds", [])]
                 for x in resp["source"]}
    else:
        spans = {(str(x["video_id"]), 0): parse(x.get("response", "")) for x in resp.get("results", [])}
    rows = []
    for stream, path in enumerate(args.labels):
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                q = dict(row); q["_stream"] = stream; rows.append(q)
    by = {}
    for row in rows:
        by.setdefault(row["video_id"], []).append(row)

    entries = []
    for vid, source_rows in sorted(by.items()):
        for stream in sorted(set(int(x["_stream"]) for x in source_rows)):
            rs = [x for x in source_rows if int(x["_stream"]) == stream]
            times = np.asarray(sorted({int(float(x["time_sec"])) for x in rs}), dtype=int)
            y = np.asarray([GRADE[next(x["generic_highlight"] for x in rs if int(float(x["time_sec"])) == t)] for t in times], float)
            n, k = len(times), int(round(.85 * len(times)))
            center = 1.0 - np.abs(times - 14.5) / 14.5
            inside = np.asarray([any(a <= t <= b for a, b in spans.get((vid, stream), [])) for t in times], float)
            center_rank = rank01(center)
            inside_rank = rank01(inside)
            base = proxy_f(select(center, k), y, True)
            direct = proxy_f(select(2.0 * inside + 1e-3 * center, k), y, True)
            residual = proxy_f(select(center_rank + .20 * (inside_rank - .5), k), y, True)
            base_b = proxy_f(select(center, k), y, False)
            residual_b = proxy_f(select(center_rank + .20 * (inside_rank - .5), k), y, False)
            entries.append({"video_id": vid, "stream": stream, "base": base,
                            "direct": direct, "residual": residual,
                            "residual_delta": residual - base,
                            "base_binary": base_b, "residual_binary": residual_b,
                            "residual_binary_delta": residual_b - base_b,
                            "intervals": spans.get((vid, stream), [])})
    out = {
        "experiment_id": "R54_TIMELENS_CENTER_RESIDUAL_PROBE",
        "status": "PUBLIC_SOURCE_HELD_OUT_COMPLETED",
        "hypothesis": "TimeLens2 interval evidence is useful as a small correction around center ranking, while hard replacement is too aggressive",
        "eta": .20, "rows": len(rows), "sources": len(by), "views": len(entries),
        "macro": {k: float(np.mean([x[k] for x in entries])) for k in ("base", "direct", "residual", "residual_delta", "base_binary", "residual_binary", "residual_binary_delta")},
        "positive_views": {"direct": int(sum(x["direct"] > x["base"] for x in entries)),
                           "residual": int(sum(x["residual_delta"] > 0 for x in entries))},
        "responses_sha256": __import__("hashlib").sha256(args.responses.read_bytes()).hexdigest(),
        "per_view": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: out[k] for k in ("experiment_id", "macro", "positive_views")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
