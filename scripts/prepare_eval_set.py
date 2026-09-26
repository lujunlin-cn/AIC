#!/usr/bin/env python3
"""Prepare a newly released AIC evaluation set for frozen inference."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.eval_intake import prepare_eval_set


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-index", required=True,
                        help="official compact JSON/JSONL index")
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--output-index", required=True)
    parser.add_argument("--output-manifest", required=True)
    parser.add_argument("--extension", default=".mp4")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    result = prepare_eval_set(args.compact_index, args.video_root,
                              args.output_index, args.output_manifest,
                              extension=args.extension, force=args.force)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
