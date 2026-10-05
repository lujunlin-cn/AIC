"""R8: generate 128 fixed real crops from cached dev80 shortlist rows.

Preregistered input for NPU_PARITY_PROBE0 (reports/r8/preregistration.yaml).
Rows: first 128 lines of r6_spatial_shortlist_oracle_v2_g33_crops.csv
(vid,frame,ratio,cand,x1,y1,x2,y2 - pixel coords).  Video paths from the
t5 live_train_index.  Output: JPEG 448x448 + a manifest json with sha256.

CPU only; no NPU; public data only (dev80 sources).
"""
import csv, hashlib, json, os
from pathlib import Path
import cv2
import numpy as np

CROPS_CSV = Path('/data/aic/experiments_910a/LFM_V11/r6_spatial_shortlist_oracle_v2_g33_crops.csv')
T5_IDX = Path('/data/aic/experiments/T5_CROPHEAD_V3/live_train_index.jsonl')
OUT = Path('/data/aic/experiments_910a/LFM_V11/r8_crops')
N = 128


def main():
    t5 = {}
    for l in T5_IDX.read_text().splitlines():
        r = json.loads(l)
        t5[r['video_id']] = r
    rows = list(csv.DictReader(CROPS_CSV.open()))[:N]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = []
    cap_cache = {}
    for i, r in enumerate(rows):
        vid, frame = r['vid'], int(r['frame'])
        t5r = t5.get(vid)
        if t5r is None:
            continue
        vpath = t5r['video_path']
        cap = cap_cache.get(vpath)
        if cap is None:
            cap = cv2.VideoCapture(vpath)
            cap_cache.clear()
            cap_cache[vpath] = cap
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, img = cap.read()
        if not ok:
            print(f'WARN decode fail {vid} f{frame}', flush=True)
            continue
        H, W = img.shape[:2]
        x1, y1, x2, y2 = (float(r[k]) for k in ('x1', 'y1', 'x2', 'y2'))
        # stored coords are for the ratio-space frame; rescale to source
        ratio = r['ratio']
        if ratio == '9-16':          # stored on a 360x640 canvas? scale x by W/360
            pass
        # crops csv was written against the SOURCE resolution for 9-16
        # (x2 up to 202.5 on a 1080p frame would be tiny; verify and scale)
        if x2 <= 360.0 and W > 720:  # coordinates look like 360-wide space
            sx, sy = W / 360.0, H / 640.0
            x1, x2, y1, y2 = x1 * sx, x2 * sx, y1 * sy, y2 * sy
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(W, int(round(x2))), min(H, int(round(y2)))
        if x2 - x1 < 8 or y2 - y1 < 8:
            print(f'WARN degenerate box {vid} f{frame}', flush=True)
            continue
        crop = cv2.resize(img[y1:y2, x1:x2], (448, 448), interpolation=cv2.INTER_CUBIC)
        name = f'{i:03d}_{vid}_f{frame}_c{r["cand"]}.jpg'
        p = OUT / name
        cv2.imwrite(str(p), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
        manifest.append({'idx': i, 'file': name, 'vid': vid, 'frame': frame,
                         'cand': int(r['cand']), 'box_src': [x1, y1, x2, y2],
                         'sha256': hashlib.sha256(p.read_bytes()).hexdigest()})
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print(f'crops written: {len(manifest)} / {N}')


if __name__ == '__main__':
    main()
