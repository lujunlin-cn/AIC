"""Split per-source S2c npz into per-frag files (R10 readout contract)."""
import json
from pathlib import Path

import numpy as np

R = Path('/data/aic/experiments_910a/LFM_V11/r11_s2c_asd')
man = []
for split in ('train_public', 'eval_public'):
    f = Path('/data/aic/experiments_910a/LFM_V11/r8_npu/r9_audit') / f'{split}.jsonl'
    if f.exists():
        man += [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
n = 0
for r in man:
    fid = r['fragment_id']
    vid = fid.rsplit('_f', 1)[0]
    src = R / f'{vid}.npz'
    dst = R / f'{fid}.npz'
    if dst.exists() or not src.exists():
        continue
    d = np.load(src)
    np.savez(dst, s2c_asd=d['s2c_asd'], avalid=d['avalid'], t=d['t'])
    n += 1
print('split written:', n)
