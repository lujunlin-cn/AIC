#!/usr/bin/env python3
"""Probe all videos referenced by the public compact AIC index."""
from __future__ import annotations

import argparse
import json

from aic.index import enrich_index


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact-index", required=True)
    parser.add_argument("--video-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--extension", default=".mp4")
    args = parser.parse_args(argv)
    print(json.dumps(enrich_index(args.compact_index, args.video_dir, args.output,
                                  extension=args.extension), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
