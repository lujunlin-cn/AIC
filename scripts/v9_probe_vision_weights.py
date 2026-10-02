import json
from collections import Counter
from pathlib import Path

import safetensors

d = Path('/data/aic/pretrained/lfm2_5_vl_450m')
files = sorted(d.glob('*.safetensors'))
print('shards:', [f.name for f in files])
tot = 0
vt_bytes = 0
prefix = Counter()
vt_names = []
for f in files:
    with safetensors.safe_open(str(f), framework='pt') as fh:
        for k in fh.keys():
            sl = fh.get_slice(k)
            shape = sl.get_shape()
            n = 1
            for s in shape:
                n *= s
            tot += n
            prefix[k.split('.')[0] + '.' + (k.split('.')[1] if len(k.split('.')) > 1 else '')] += 1
            if 'vision' in k:
                vt_bytes += n * 2
                vt_names.append((k, tuple(shape)))
print(f'total params {tot:,}  vision fp16 bytes {vt_bytes/1e6:.1f} MB')
print('prefix counts:', dict(prefix))
print('vision tensor sample:')
for k, s in vt_names[:5]:
    print('  ', k, s)
print('n vision tensors', len(vt_names))
cfg = json.loads((d / 'config.json').read_text())
print('vision_config:', json.dumps(cfg.get('vision_config') or cfg.get('vision_config', {}))[:600])