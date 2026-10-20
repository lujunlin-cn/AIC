#!/usr/bin/env python3
"""Evaluate a frozen Valen choice checkpoint on prepared JSONL requests.

The request JSONL is kept verbatim so the evaluation uses the same prompt,
candidate order, media paths and question contract as the corresponding SFT
run.  Output is compatible with ``valen_eval_report.py``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="prepared request JSONL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    if args.batch < 1:
        raise ValueError("batch must be positive")

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from aic.valen_adapter import ValenDecisionAdapter

    data = Path(args.data)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = [json.loads(line) for line in data.read_text().splitlines() if line.strip()]
    pred_path = out.with_suffix(".predictions.jsonl")
    predictions = []
    if pred_path.exists():
        predictions = [json.loads(line) for line in pred_path.read_text().splitlines() if line.strip()]
    done = {int(p["record_index"]) for p in predictions}
    jobs = [(i, rows[i]) for i in range(len(rows)) if i not in done]

    adapter = ValenDecisionAdapter(args.model, checkpoint=args.checkpoint)
    started = time.perf_counter()
    with pred_path.open("a") as stream:
        for start in range(0, len(jobs), args.batch):
            batch = jobs[start:start + args.batch]
            results, timing = adapter.decide_batch([row["request"] for _, row in batch])
            for (index, row), result in zip(batch, results):
                answer = result["raw"]["answers"]["crop"]
                prediction = {
                    "record_index": index,
                    "group_id": row.get("group_id"),
                    "source": row.get("meta", {}).get("source"),
                    "answer": answer,
                    "raw": result["raw"],
                    "tokens": result["tokens"],
                }
                predictions.append(prediction)
                stream.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            stream.flush()
            print(json.dumps({"done": min(start + len(batch), len(jobs)),
                              "total": len(jobs), "batch": timing,
                              "elapsed_s": time.perf_counter() - started}), flush=True)

    predictions.sort(key=lambda p: int(p["record_index"]))
    report = {
        "data": str(data),
        "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "model": str(args.model),
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "records": len(rows),
        "predictions": len(predictions),
        "batch": args.batch,
        "parameters": adapter.parameters,
        "checkpoint_tensors": getattr(adapter, "checkpoint_parameters", 0),
        "seconds": time.perf_counter() - started,
        "official_test_used": False,
        "prediction_file": str(pred_path),
    }
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
