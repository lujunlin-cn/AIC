"""Crop overlays for LFM450 success/failure cases.

For each chosen (vid, ratio) draws, on 4 evenly spaced keyframes:
  green  = mean 6-annotator GT crop
  yellow = LFM subject box (GROUND, raw model output, before geometry)
  red    = LFM final crop (LFM_GROUND policy, after hold+EMA+max window)
  blue   = B0 crop
  white dashed = CENTER crop
Title row carries video-level IoUs.  Output: one PNG per case + index JSON.
"""
import argparse, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument('--run-dir', type=Path, required=True)
ap.add_argument('--analysis', type=Path, required=True)
ap.add_argument('--frames-dir', type=Path, default=Path('/data/aic/experiments/QWEN_RV200_KEYFRAMES/d1'))
ap.add_argument('--annotations', default='/data/aic/experiments_910a/LFM450_EVAL_V1/annotations')
ap.add_argument('--output-dir', type=Path, required=True)
a = ap.parse_args()

import sys
sys.path.insert(0, '/root/AIC')
from scripts.max_window_path_eval import RATIOS, boxes, load_gt  # noqa: E402

an = json.loads(a.analysis.read_text())
cases = [('fail', c) for c in an['worst_vs_b0'][:4]] + [('success', c) for c in an['best_vs_b0'][:4]]
a.output_dir.mkdir(parents=True, exist_ok=True)
index = []


def rect(d, b, col, w=3, dash=False):
    x1, y1, x2, y2 = [float(v) for v in b]
    if not dash:
        d.rectangle([x1, y1, x2, y2], outline=col, width=w)
        return
    for (p, q) in (((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)), ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))):
        n = int(max(abs(q[0] - p[0]), abs(q[1] - p[1])) // 10) + 1
        for i in range(0, n, 2):
            s, e = i / n, min((i + 1) / n, 1)
            d.line([p[0] + (q[0] - p[0]) * s, p[1] + (q[1] - p[1]) * s, p[0] + (q[0] - p[0]) * e, p[1] + (q[1] - p[1]) * e], fill=col, width=2)


for tag, c in cases:
    vid, rk = c['vid'], c['ratio']
    ratio = RATIOS[rk]
    z = np.load(a.run_dir / 'crops' / f'{vid}_{rk}.npz')
    kfs = z['kfs'].tolist()
    gt = load_gt(a.annotations, vid, rk)
    pick = [kfs[i] for i in np.round(np.linspace(0, len(kfs) - 1, 4)).astype(int)]
    tiles = []
    for kf in pick:
        im = Image.open(a.frames_dir / vid / f'{kf}.png').convert('RGB')
        W, H = im.size
        d = ImageDraw.Draw(im)
        g = gt[:, kf].mean(0)
        rect(d, g, (0, 220, 0), 4)
        rect(d, boxes(z['CENTER'][kf:kf + 1], ratio)[0], (255, 255, 255), dash=True)
        rect(d, boxes(z['B0'][kf:kf + 1], ratio)[0], (60, 120, 255), 3)
        rect(d, boxes(z['LFM_GROUND'][kf:kf + 1], ratio)[0], (255, 40, 40), 3)
        gb = z['ground_boxes'][kfs.index(kf)]
        if np.isfinite(gb).all():
            rect(d, [gb[0] * W, gb[1] * H, gb[2] * W, gb[3] * H], (255, 220, 0), 2)
        d.text((6, 6), f'kf {kf}', fill=(255, 255, 255))
        tiles.append(im)
    W, H = tiles[0].size
    sheet = Image.new('RGB', (W * 2 + 6, H * 2 + 6 + 34), (20, 20, 20))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % 2) * (W + 6), 34 + (i // 2) * (H + 6)))
    ImageDraw.Draw(sheet).text((8, 8), f'{tag.upper()} {vid} {rk} ({c["axis"]}, {c["layer"]})  IoU  LFM {c["lfm"]:.3f} | B0 {c["b0"]:.3f} | CENTER {c["center"]:.3f} | QWEN_T(ref) {c["qwen_t"]:.3f}   green=GT yellow=LFM box red=LFM crop blue=B0 white=center', fill=(255, 255, 255))
    f = a.output_dir / f'{tag}_{vid}_{rk}.png'
    sheet.save(f)
    index.append({'file': str(f), 'tag': tag, **c})
(a.output_dir / 'index.json').write_text(json.dumps(index, indent=1) + '\n')
print('\n'.join(x['file'] for x in index))
