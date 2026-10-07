"""R11 S2a readout prerequisite: RT-DETR person boxes on the LIVE pool.

The day-1 S2a extraction ran on PHD2 (slot-pool keep-mask protocol - no
spatial candidates), while the S2 readouts a/b/c need detections on the SAME
pool as the B3 score table + spatial GT: live_dev / live_confirmation
(LIVE-YT-VC, 9-16 crop, frames 0..174 step 6).  This extracts one npz per
(vid, frame) with the SAME detection contract as the PHD2 run (448x448
square resize, top-1 person >= conf) plus a top-3 box array for an oracle
side-readout.  CPU-only, source-sharded, resumable (skip existing npz).

Usage (one process per shard):
  HF_ENDPOINT=https://hf-mirror.com python3 scripts/r11_s2a_live_extract.py --shard s --nshards 32
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")  # BEFORE transformers import
os.environ.setdefault("OMP_NUM_THREADS", "6")

import numpy as np  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--manifest", type=Path,
                default=Path("/data/aic/experiments_910a/LFM_V8/v8_manifest.jsonl"))
ap.add_argument("--media-root", type=Path,
                default=Path("/data/aic/external_datasets/LIVE_YT_VC/raw/study_videos"))
ap.add_argument("--out", type=Path,
                default=Path("/data/aic/experiments_910a/LFM_V11/r11_s2a_live"))
ap.add_argument("--model-path", default="PekingU/rtdetr_r50vd")
ap.add_argument("--conf", type=float, default=0.3)
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--device", default="cpu", choices=["cpu"])
a = ap.parse_args()

import cv2  # noqa: E402
import av  # noqa: E402
import torch  # noqa: E402
from transformers import AutoImageProcessor, AutoModelForObjectDetection  # noqa: E402

proc = AutoImageProcessor.from_pretrained(a.model_path)
model = AutoModelForObjectDetection.from_pretrained(a.model_path).to(a.device)
model.eval()
ID2LAB = model.config.id2label
PERSON = next((k for k, v in ID2LAB.items() if str(v).lower() == "person"), None)
print(f"{a.model_path} on {a.device}; person id={PERSON}", flush=True)

rows = [json.loads(l) for l in a.manifest.read_text().splitlines() if l.strip()]
rows = [r for r in rows if r["split"] in ("live_dev", "live_confirmation")]
need = {}
for r in rows:
    need.setdefault((r["vid"], int(r["frame"])), set())
keys = sorted(need)
print(f"{len(keys)} (vid,frame) pairs", flush=True)
mine = [k for i, k in enumerate(keys) if i % a.nshards == a.shard]
a.out.mkdir(parents=True, exist_ok=True)

by_vid = {}
for vid, fr in mine:
    by_vid.setdefault(vid, []).append(fr)

n_done, n_skip, n_err, t0 = 0, 0, 0, time.time()
for vid, frs in sorted(by_vid.items()):
    media = a.media_root / f"{vid}.mp4"
    todo = [f for f in frs if not (a.out / f"{vid}_{f:06d}.npz").exists()]
    n_skip += len(frs) - len(todo)
    if not todo:
        continue
    if not media.exists():
        n_err += 1
        print(f"ERR {vid}: media missing", flush=True)
        continue
    try:
        c = av.open(str(media))
        need_fr = {}
        for f in todo:
            need_fr.setdefault(max(int(f), 0), []).append(f)
        # Frame index = decode order counter (0-based), the SAME convention as
        # v8_live_extract.py built the features from - NOT pts*tb*fps, which is
        # offset when the stream's first pts is non-zero (LIVE-YT-VC: +4).
        got = {}
        fi = 0
        for frame in c.decode(video=0):
            if fi in need_fr:
                got[fi] = cv2.resize(frame.to_ndarray(format="rgb24"), (448, 448),
                                     interpolation=cv2.INTER_CUBIC)
            if len(got) == len(need_fr) or fi > max(need_fr):
                break
            fi += 1
        c.close()
        for idx, tags in need_fr.items():
            im = got.get(idx)
            for f in tags:
                outp = a.out / f"{vid}_{f:06d}.npz"
                if im is None:
                    args_npz = dict(det_box=np.zeros((3, 4), np.float32),
                                    det_conf=np.zeros(3, np.float32),
                                    n_det=np.int16(0), ok=np.int8(0))
                else:
                    with torch.inference_mode():
                        inputs = proc(images=[im], return_tensors="pt").to(a.device)
                        out = model(**inputs)
                        prob = out.logits.softmax(-1)
                    # RT-DETR: pure-class logits, no background column (day-1 lesson).
                    has_bg = prob.shape[-1] == len(ID2LAB) + 1
                    row = prob[0]
                    sc, lb = row.max(-1)
                    sc = sc.float().cpu().numpy()
                    lb = (lb.float() + 1).cpu().numpy() if has_bg else lb.cpu().numpy()
                    order = np.argsort(-sc)
                    persons = [j for j in order if int(lb[j]) == PERSON and sc[j] >= a.conf][:3]
                    boxes = np.zeros((3, 4), np.float32)
                    confs = np.zeros(3, np.float32)
                    for k, j in enumerate(persons):
                        boxes[k] = out.pred_boxes[0, j].float().cpu().numpy()
                        confs[k] = float(sc[j])
                    args_npz = dict(det_box=boxes, det_conf=confs,
                                    n_det=np.int16(len(persons)), ok=np.int8(1))
                np.savez(outp, **args_npz)
                n_done += 1
    except Exception as e:  # noqa: BLE001
        n_err += 1
        print(f"ERR {vid}: {str(e)[:180]}", flush=True)
    if n_done % 100 < 32:
        print(f"done={n_done} skip={n_skip} err={n_err} {time.time() - t0:.0f}s", flush=True)

print(json.dumps({"shard": a.shard, "pairs_total": len(mine), "new": n_done,
                  "skipped_existing": n_skip, "errors": n_err,
                  "runtime_s": round(time.time() - t0, 1)}), flush=True)
