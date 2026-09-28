#!/usr/bin/env python3
"""Merge every dataset's processed index into _registry/ (JSONL + Parquet + status.json)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aicext.registry import build_registry  # noqa: E402

if __name__ == "__main__":
    s = build_registry()
    print(json.dumps({k: {x: v[x] for x in ("readiness", "media", "media_ready", "annotations", "aic_split")}
                      for k, v in s["datasets"].items()}, indent=1, ensure_ascii=False))
    print("schema errors:", s["schema_error_count"])
