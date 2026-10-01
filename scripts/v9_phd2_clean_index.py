"""Drop fragments whose source video also appears in the PHD2 test annotations.

selections/test.json is the upstream PHD2 test split - the same GIF-highlight
objective the AIC official drop is drawn from (tier0 = its 831 is_last GIF
hosts).  667 tier-1 videos appear in both train.json and test.json, and 45 of
them survived the tier filter into the fragment pool, contributing 115
fragments.  Fragment anchors may have been placed on those test intervals, so
they are removed outright rather than trusted.

Only rows are dropped; points already computed stay valid because the teacher
writes one file per video_id and the consumer matches on that name.
"""
import argparse, json
from pathlib import Path

D = Path('/data/aic/external_datasets/PHD2')
ap = argparse.ArgumentParser()
ap.add_argument('--pool', type=Path, required=True)
ap.add_argument('--out-name', default='index_clean.jsonl')
ap.add_argument('--report', type=Path, default=None)
a = ap.parse_args()

test_srcs = set(json.loads((D / 'annotations' / 'selections' / 'test.json').read_text()))
rows = [json.loads(l) for l in (a.pool / 'index.jsonl').read_text().splitlines() if l.strip()]
keep = [r for r in rows if r['src'] not in test_srcs]
drop = [r for r in rows if r['src'] in test_srcs]
(a.pool / a.out_name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in keep))
rep = {'before': len(rows), 'after': len(keep), 'dropped': len(drop),
       'dropped_sources': len({r['src'] for r in drop}),
       'rule': 'source video absent from PHD2 selections/test.json'}
print(json.dumps(rep, indent=1))
if a.report:
    a.report.write_text(json.dumps(rep, indent=1))