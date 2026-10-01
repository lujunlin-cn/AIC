import json
from pathlib import Path
rows = [json.loads(l) for l in open('/data/aic/semifinal_20261001/intake/index.enriched.jsonl')]
sub = rows[4::5]                      # the shard that failed on NPU
d = Path('/data/aic/semifinal_20261001/intake_parts/s4')
d.mkdir(parents=True, exist_ok=True)
for p in range(8):
    with open(d / f'p{p}.jsonl', 'w') as f:
        for r in sub[p::8]:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
print('shard4 videos:', len(sub), '-> 8 parts of', len(sub) // 8)
