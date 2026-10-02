"""Add the field names the V8 inference script expects to a V9 head checkpoint.

v8_s_official_points.py reads ck['config']['d'] and ck['config']['nc']; the V9
trainer stored the feature width as a top-level 'D' and never mirrored it into
config.  Patching the checkpoint avoids a retrain just to rename two keys.
"""
import sys
from pathlib import Path

import torch

for p in sys.argv[1:]:
    f = Path(p)
    ck = torch.load(f, map_location='cpu', weights_only=False)
    cfg = dict(ck.get('config') or {})
    cfg.setdefault('d', int(ck.get('D', 2305)))
    cfg.setdefault('nc', int(ck.get('nc', 129)))
    cfg.setdefault('seed', 0)
    ck['config'] = cfg
    torch.save(ck, f)
    print(f'{f}: config.d={cfg["d"]} config.nc={cfg["nc"]} seed={cfg["seed"]}')