"""Integrity gate before any multi-hour teacher run on the PHD2 fragment pool.

Verifies, for every row of index.jsonl, that each skeleton keyframe has a
readable jpg on disk.  A single missing file aborts a 19-hour NPU job at the
first video, so this runs first.  (Heredocs get swallowed over ssh+docker exec;
this file is shipped with scp instead.)
"""
import json
import sys
from pathlib import Path

P = Path(sys.argv[1] if len(sys.argv) > 1 else '/data/aic/experiments_910a/PHD2_FRAG_V1')
rows = [json.loads(l) for l in (P / 'index.jsonl').read_text().splitlines() if l.strip()]
miss, tot, bad, empty = 0, 0, [], []
for r in rows:
    sk = json.loads((P / 'skel' / f"{r['video_id']}.json").read_text())
    d = P / 'frames' / r['video_id']
    n = 0
    for k in sk['keyframes']:
        tot += 1
        f = d / f'{k}.jpg'
        if f.exists():
            n += 1
        else:
            miss += 1
            if len(bad) < 5:
                bad.append(str(f))
    if n == 0:
        empty.append(r['video_id'])
    elif n < len(sk['keyframes']):
        empty.append(f'{r["video_id"]}({n}/{len(sk["keyframes"])})')
print(f'FRAMES total={tot} missing={miss} incomplete_fragments={len(empty)}')
if bad:
    print('  examples:', bad)
if empty[:5]:
    print('  incomplete:', empty[:5])
sys.exit(1 if miss else 0)