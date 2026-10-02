"""Move teacher points whose ratio does not match the current fragment index.

The teacher skips a video whose points file already exists, so any point file
written under a previous target ratio would silently survive a re-target and be
paired with a fragment that now asks for a different ratio.  The teacher prompt
embeds the ratio, so such a point answers a different question.

Scans every index row against its points file and quarantines the mismatches
into points_stale/.  Idempotent; safe to run while the teacher is running (it
only ever removes files the running job would have skipped anyway).
"""
import json, shutil
from pathlib import Path

P = Path('/data/aic/experiments_910a/PHD2_FRAG_V1')
rows = [json.loads(l) for l in (P / 'index_v2.jsonl').read_text().splitlines() if l.strip()]
pd, sd = P / 'teacher' / 'points', P / 'teacher' / 'points_stale'
sd.mkdir(parents=True, exist_ok=True)
bad, ok = [], 0
for r in rows:
    f = pd / f"{r['video_id']}.json"
    if not f.exists():
        continue
    try:
        rec = json.loads(f.read_text())
        got = rec['ratios']['t']['ratio']
    except Exception:
        bad.append((r['video_id'], 'unreadable'))
        continue
    if list(got) != list(r['targetRatioWH']):
        bad.append((r['video_id'], f'{got} != {r["targetRatioWH"]}'))
        shutil.move(str(f), str(sd / f.name))
    else:
        ok += 1
print(json.dumps({'matched': ok, 'quarantined': len(bad),
                  'examples': bad[:5]}, indent=1))