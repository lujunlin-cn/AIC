import collections
import glob
import random
import sys

from PIL import Image

for root, pat, n in (('/data/aic/experiments_910a/PHD2_FRAG_V1/frames', '*.jpg', 400),
                     ('/data/aic/semifinal_20261001/keyframes', '*.png', 200)):
    fs = glob.glob(f'{root}/*/{pat}')
    random.seed(0)
    s = random.sample(fs, min(n, len(fs)))
    c = collections.Counter()
    for f in s:
        try:
            c[Image.open(f).size] += 1
        except Exception:
            pass
    print(root, 'n_files=', len(fs), 'top_sizes=', c.most_common(8), flush=True)