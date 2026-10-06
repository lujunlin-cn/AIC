"""R8 VLM_SFT frame pre-extraction.

Runs ONCE on CPU before the two parallel lr configs start; both configs share
--frames-dir and the training script aborts if any frame is missing.  Covers
train240 + dev80 rows of the freeze CSV (frame identity = vid|frame, ratio
does not change the frame file).
"""
import argparse, csv, json
from pathlib import Path
import cv2

ap = argparse.ArgumentParser()
ap.add_argument('--freeze-csv', default='/data/aic/experiments_910a/LFM_V11/r6_s_pool_freeze_rows.csv')
ap.add_argument('--t5-index', default='/data/aic/experiments/T5_CROPHEAD_V3/live_train_index.jsonl')
ap.add_argument('--frames-dir', default='/data/aic/experiments_910a/LFM_V11/r8_sft_frames')
ap.add_argument('--out', default='/data/aic/experiments_910a/LFM_V11/r8_npu/sft_prep_done.json')
args = ap.parse_args()

t5 = {}
for l in open(args.t5_index):
    r = json.loads(l)
    t5[r['video_id']] = r
FD = Path(args.frames_dir)
FD.mkdir(parents=True, exist_ok=True)

need = {}
n_rows = 0
for r in csv.DictReader(open(args.freeze_csv)):
    n_rows += 1
    fp = FD / f"{r['vid']}_f{r['frame']}.jpg"
    if not fp.exists():
        need.setdefault(r['vid'], set()).add(int(r['frame']))
print(f'rows {n_rows}; vids needing extraction {len(need)}; '
      f'frames {sum(len(v) for v in need.values())}', flush=True)

n, n_fail = 0, 0
for vid, frames in sorted(need.items()):
    cap = cv2.VideoCapture(t5[vid]['video_path'])
    for fr in sorted(frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
        ok, img = cap.read()
        if ok:
            cv2.imwrite(str(FD / f'{vid}_f{fr}.jpg'), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
            n += 1
        else:
            n_fail += 1
    cap.release()
    if n % 400 < 8:
        print(f'extracted {n}', flush=True)

missing = sum(1 for r in csv.DictReader(open(args.freeze_csv))
              if not (FD / f"{r['vid']}_f{r['frame']}.jpg").exists())
Path(args.out).write_text(json.dumps(
    {'extracted': n, 'decode_failures': n_fail, 'missing_after': missing}) + '\n')
print(f'DONE extracted {n} decode_failures {n_fail} missing {missing}', flush=True)
