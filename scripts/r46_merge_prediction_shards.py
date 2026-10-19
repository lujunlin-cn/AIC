#!/usr/bin/env python3
"""Merge disjoint R46 prediction shards and enforce complete 426 coverage."""
from __future__ import annotations
import argparse, json
from pathlib import Path

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--shards', nargs='+', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    by = {}
    for path in args.shards:
        rows = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        for row in rows:
            vid = str(row['video_id'])
            if vid in by:
                raise SystemExit(f'duplicate video_id={vid}')
            by[vid] = row
    if len(by) != 426 or set(by) != {str(i) for i in range(426)}:
        raise SystemExit(f'expected exact 0..425 coverage, got {len(by)} rows')
    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('w') as f:
        for vid in sorted(by, key=int):
            f.write(json.dumps(by[vid], ensure_ascii=False, separators=(',', ':')) + '\n')
    print(json.dumps({'videos': len(by), 'predictions': sum(len(r['predictions']) for r in by.values()), 'output': str(out)}))

if __name__ == '__main__':
    main()
