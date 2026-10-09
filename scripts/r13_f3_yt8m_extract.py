#!/usr/bin/env python3
"""R13 F3: extract YouTube-8M-contract RGB features (1024-d) for the 426
semifinal videos, using the OFFICIAL yt8m extractor stack:
  classify_image_graph_def.pb (inception-2015-12-05) -> pool_3 2048-d
  -> yt8m PCA (mean/eigenvecs.T[:,:1024]/sqrt(eigenvals+1e-4)) -> 1024-d
Frame sampling: 1 frame per second (yt8m contract), matching the gtscore
grid of Mr.HiSum (T ~= duration in seconds).

Parallel: shard by video_id mod n-shards; per-video incremental npz
(resumable by file existence); TF intra_op threads bounded per worker.

Output: <out>/<vid>.npz {feat (T,1024) f16, frame_idx (T,) i8}
"""
import argparse
import json
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import time

import cv2
import numpy as np

INTAKE = "/data/aic/semifinal_20261001/intake/index.enriched.jsonl"
EXT_DIR = "/data/aic/tools/yt8m_extractor"
OUT_DIR = "/data/aic/experiments_910a/LFM_V13/f3_yt8m_feats"

ap = argparse.ArgumentParser()
ap.add_argument("--shard", type=int, required=True)
ap.add_argument("--n-shards", type=int, default=8)
ap.add_argument("--out-dir", default=OUT_DIR)
ap.add_argument("--intra-threads", type=int, default=8)
ap.add_argument("--limit", type=int, default=0)
args = ap.parse_args()

import tensorflow as tf  # noqa: E402

tf1 = tf.compat.v1
tf1.logging.set_verbosity(tf1.logging.ERROR)

os.makedirs(args.out_dir, exist_ok=True)

vids = []
for line in open(INTAKE):
    r = json.loads(line)
    vids.append((r["video_id"], r["video_path"]))
mine = sorted(v for v in vids if int(v[0]) % args.n_shards == args.shard)
if args.limit:
    mine = mine[: args.limit]
print(f"[shard {args.shard}] {len(mine)} videos", flush=True)

graph = tf1.Graph()
with graph.as_default():
    gd = tf1.GraphDef()
    with open(os.path.join(EXT_DIR, "classify_image_graph_def.pb"), "rb") as f:
        gd.ParseFromString(f.read())
    tf1.import_graph_def(gd, name="")
cfg = tf1.ConfigProto(device_count={"GPU": 0},
                      intra_op_parallelism_threads=args.intra_threads,
                      inter_op_parallelism_threads=2)
sess = tf1.Session(graph=graph, config=cfg)

PCA_MEAN = np.load(os.path.join(EXT_DIR, "mean.npy"))[:, 0]
PCA_VECS = np.load(os.path.join(EXT_DIR, "eigenvecs.npy")).T[:, :1024]
PCA_VALS = np.load(os.path.join(EXT_DIR, "eigenvals.npy"))[:1024, 0]


def pca1024(f2048):
    """pool_3 2048 -> yt8m PCA white-1024, then uint8-quantize round-trip to
    match the tfrecord storage convention exactly (dequant domain [-2, 2])."""
    f = f2048.reshape(-1) - PCA_MEAN
    f = (f.reshape(1, 2048).dot(PCA_VECS).reshape(1024)
         / np.sqrt(PCA_VALS + 1e-4))
    q = np.clip(np.round((f + 2.0) * 255.0 / 4.0), 0, 255)
    return q * 4.0 / 255.0 - 2.0


t0, done = time.time(), 0
for vid, path in mine:
    out_path = os.path.join(args.out_dir, f"{vid}.npz")
    if os.path.exists(out_path):
        done += 1
        continue
    try:
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if total <= 0 or fps <= 0:
            cap.release()
            print(f"[shard {args.shard}] {vid} BAD_META total={total} fps={fps}",
                  flush=True)
            continue
        n_sec = int(total / fps)
        feats = np.zeros((n_sec, 1024), dtype=np.float32)
        ok_n = 0
        cur_target = 0
        fi = 0
        while fi < n_sec:
            target = int(fi * fps)
            if target >= total:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ok, fr = cap.read()
            if not ok:
                fi += 1
                continue
            rgb = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            f2048 = sess.run("pool_3/_reshape:0",
                             feed_dict={"DecodeJpeg:0": rgb})
            feats[fi] = pca1024(f2048)
            ok_n += 1
            fi += 1
        cap.release()
        if ok_n < max(4, n_sec // 4):
            print(f"[shard {args.shard}] {vid} TOO_FEW_FRAMES {ok_n}/{n_sec}",
                  flush=True)
            continue
        np.savez_compressed(out_path, feat=feats.astype(np.float16),
                            frame_idx=np.arange(n_sec, dtype=np.int64))
        done += 1
        if done % 10 == 0:
            el = time.time() - t0
            print(f"[shard {args.shard}] {done}/{len(mine)} "
                  f"({el / done:.1f}s/vid, {feats.shape[0]}s)", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[shard {args.shard}] {vid} ERR {type(e).__name__}: {str(e)[:150]}",
              flush=True)

print(f"[shard {args.shard}] ALL_DONE {done}/{len(mine)} "
      f"({(time.time() - t0) / 60:.1f}m)", flush=True)
