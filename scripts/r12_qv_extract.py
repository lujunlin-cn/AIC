#!/usr/bin/env python3
"""R12-QV P1: LFM2VL pooled (768,) features for QVHighlights 150 s segments.

Same vision() contract as p0c (v8_live_extract port + siglip2 NPU patch):
640 long-side, min_tiles=max_tiles=1, fp16 eager forward, grid mean -> f16.
Frames: mid-frame of each 2 s clip (t = 1, 3, ..., 149 s), sequential
grab/retrieve decode on a producer thread, NPU forward on the main thread.

Output: /data/aic/experiments_910a/LFM_V11/r12_qv/feats/<vid>.npz
        {ts [<=75] f8, pooled [<=75,768] f16}
Resume: existing npz skipped (pilot --limit N is resumable-safe).
"""
import argparse
import json
import os
import queue
import sys
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--label-json", default="/data/aic/experiments_910a/LFM_V11/r12_qv/qv_labels.json")
ap.add_argument("--video-root", default="/data/aic/external_datasets/QVHighlights/raw/videos")
ap.add_argument("--model", default="/data/aic/pretrained/lfm2_5_vl_450m")
ap.add_argument("--out", default="/data/aic/experiments_910a/LFM_V11/r12_qv/feats")
ap.add_argument("--cards", default="0")
ap.add_argument("--shard", type=int, default=0)
ap.add_argument("--nshards", type=int, default=1)
ap.add_argument("--limit", type=int, default=0, help="debug: cap seg count")
ap.add_argument("--only-split", default="", help="train|val: restrict")
args = ap.parse_args()

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["ASCEND_RT_VISIBLE_DEVICES"] = args.cards

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch_npu  # noqa: F401,E402

torch.npu.set_compile_mode(jit_compile=False)
torch.npu.config.allow_internal_format = False
assert torch.npu.device_count() > 0, "no NPU"
cv2.setNumThreads(2)
from PIL import Image  # noqa: E402
from transformers import AutoProcessor, Lfm2VlForConditionalGeneration  # noqa: E402

sys.path.insert(0, "/root/AIC/scripts")
from v11_npu_siglip2_patch import patch_siglip2_npu  # noqa: E402

NC = 75
TAG = f"[qvx s{args.shard}/{args.nshards}]"

proc = AutoProcessor.from_pretrained(args.model, min_tiles=1, max_tiles=1, max_image_tokens=256)
model = Lfm2VlForConditionalGeneration.from_pretrained(
    args.model, dtype=torch.float16, low_cpu_mem_usage=True,
    attn_implementation="eager").eval().to("npu")
assert patch_siglip2_npu(model), "siglip2 npu patch failed"
print(f"{TAG} model ready", flush=True)


def vision(im: Image.Image):
    """Verbatim port of v8_live_extract.vision(): pooled (768,) f16."""
    msgs = [{"role": "user", "content": [{"type": "image", "image": im},
                                         {"type": "text", "text": "describe"}]}]
    x = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True,
                                 return_dict=True, return_tensors="pt")
    x = {k: v.to("npu") for k, v in x.items() if isinstance(v, torch.Tensor)}
    x["pixel_values"] = x["pixel_values"].to(torch.float16)
    with torch.no_grad():
        out = model.model.vision_tower(pixel_values=x["pixel_values"], spatial_shapes=x["spatial_shapes"],
                                       pixel_attention_mask=x["pixel_attention_mask"], return_dict=True)
        valid = int(x["pixel_attention_mask"][0].sum())
        fh, fw = [int(v) for v in x["spatial_shapes"][0]]
        grid = out.last_hidden_state[0, :valid].reshape(fh, fw, -1)[:, : (fw // 2) * 2, :]
        grid = grid[: (fh // 2) * 2]
        pooled = grid.float().mean((0, 1))
    return pooled.cpu().numpy().astype(np.float16)


def decode_video(path):
    """Yield (ts_seconds, PIL) for mid-frames of the 75 2 s clips."""
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    want = {int(round((2 * c + 1) * fps)): float(2 * c + 1) for c in range(NC)}
    i, n_hit = -1, 0
    while True:
        if not cap.grab():
            break
        i += 1
        if i in want:
            ok, fr = cap.retrieve()
            if not ok:
                continue
            im = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
            if max(im.size) > 640:
                im = im.resize((round(im.width * 640 / max(im.size)),
                                round(im.height * 640 / max(im.size))))
            yield want[i], im
            n_hit += 1
            if n_hit >= NC:
                break
    cap.release()


def main():
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lab = json.load(open(args.label_json))
    meta = lab["meta"]
    vid2path = {p.stem: p for p in Path(args.video_root).rglob("*.mp4")}
    mine = []
    n_nof = 0
    for k, (vid, m) in enumerate(sorted(meta.items())):
        if args.only_split and m["split"] != args.only_split:
            continue
        if k % args.nshards != args.shard:
            continue
        if (out / f"{vid}.npz").exists():
            continue
        if vid not in vid2path:
            n_nof += 1
            continue
        mine.append(vid)
    if args.limit:
        mine = mine[: args.limit]
    print(f"{TAG} todo={len(mine)} no_file={n_nof}", flush=True)

    q = queue.Queue(maxsize=2)

    def producer():
        try:
            for vid in mine:
                frames = []
                try:
                    for ts, im in decode_video(vid2path[vid]):
                        frames.append((ts, im))
                    q.put((vid, frames, None))
                except Exception as e:  # noqa: BLE001
                    q.put((vid, [], str(e)[:300]))
        finally:
            q.put(None)

    threading.Thread(target=producer, daemon=True).start()
    t0, n_ok, n_err = time.time(), 0, 0
    while True:
        item = q.get()
        if item is None:
            break
        vid, frames, err = item
        if err or not frames:
            n_err += 1
            (out / f"{vid}.err").write_text(err or "0 frames")
            continue
        try:
            pools = [vision(im) for _, im in frames]
            np.savez(out / f"{vid}.npz",
                     ts=np.array([t for t, _ in frames], np.float64),
                     pooled=np.stack(pools))
            n_ok += 1
        except Exception as e:  # noqa: BLE001
            n_err += 1
            (out / f"{vid}.err").write_text(str(e)[:300])
        done = n_ok + n_err
        if done % 10 == 0:
            rate = done / (time.time() - t0) * 3600
            print(f"{TAG} {done}/{len(mine)} ok={n_ok} err={n_err} seg/h={rate:.0f}",
                  flush=True)
    print(f"{TAG} SHARD_DONE total={len(mine)} ok={n_ok} err={n_err} "
          f"min={(time.time() - t0) / 60:.1f}", flush=True)
    if args.limit:
        print(f"{TAG} PILOT_DONE", flush=True)


if __name__ == "__main__":
    main()
