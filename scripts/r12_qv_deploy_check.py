#!/usr/bin/env python3
"""R12-QV deployment check: score the 426 semifinal videos with a QV-trained
GATT ranker and measure how much the keep-80% set differs from the shipped
R12_GATT_LFM_SEMIFINAL (35.17) prediction set.

Adaptation: training used one token per 2 s clip (75 tokens / 150 s segment);
deployment scores the 1 s keyframes at stride 2 (even seconds) and broadcasts
each score to the 1 s bins it covers, so sequence length matches training.

Usage: r12_qv_deploy_check.py --ckpt <qv best.pt> [--keep-frac 0.8]
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/root/AIC/scripts")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from r12_gatt_vmaef_train import Ranker  # noqa: E402

INTAKE = "/data/aic/semifinal_20261001/intake/index.enriched.jsonl"
FEAT = "/data/aic/semifinal_20261001/temporal_feats"
OLD_PRED = "/data/aic/semifinal_20261001/submissions/R12_GATT_LFM_SEMIFINAL/predictions.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--keep-frac", type=float, default=0.8)
    args = ap.parse_args()

    try:
        import torch_npu  # noqa: F401
        torch.npu.set_compile_mode(jit_compile=False)
        dev = "npu"
    except ImportError:
        dev = "cpu"
    model = Ranker().to(dev)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    print(f"[qvd] ckpt {os.path.basename(args.ckpt)} on {dev}", flush=True)

    intake = {}
    for line in open(INTAKE):
        r = json.loads(line)
        intake[r["video_id"]] = int(r["frame_count"])
    old = {}
    for line in open(OLD_PRED):
        r = json.loads(line)
        old[r["video_id"]] = {p["frame"] for p in r["predictions"]}

    diffs, sizes, r_shift = [], [], []
    n_missing = 0
    for vid in sorted(intake, key=int):
        total = intake[vid]
        vdir = Path(FEAT) / vid
        kfs = sorted(int(p.stem) for p in vdir.glob("*.npy"))
        if not kfs:
            n_missing += 1
            continue
        kfs2 = [k for k in kfs if k % 2 == 0]  # stride-2 tokens (train dist)
        x = np.stack([np.load(vdir / f"{k}.npy").astype(np.float32) for k in kfs2])
        with torch.no_grad():
            s2 = model(torch.from_numpy(x).unsqueeze(0).to(dev))[0].cpu().numpy()
        score = {k: float(s2[i]) for i, k in enumerate(kfs2)}
        # broadcast to every 1 s keyframe: nearest scored token at or before f
        # (keyframe files can be missing on some seconds)
        import bisect
        tk = kfs2
        all_score = {}
        for k in kfs:
            i = bisect.bisect_right(tk, k) - 1
            all_score[k] = score[tk[max(i, 0)]]
        budget = int(round(total * args.keep_frac))
        order = sorted(kfs, key=lambda k: -all_score[k])
        kept, acc = set(), 0
        for k in order:
            kept.add(k)
            acc += 1
            if acc >= budget:
                break
        # kept seconds -> frames: the old set stores frames; compare on seconds
        old_secs = {int(round(float(f))) for f in old.get(vid, set())}
        new_secs = kept
        union = old_secs | new_secs
        if union:
            diffs.append(len(old_secs ^ new_secs) / len(union))
        sizes.append(len(new_secs))
    d = float(np.mean(diffs)) if diffs else float("nan")
    print(f"QVD videos={len(diffs)} missing={n_missing} keep_set_sym_diff={d:.4f} "
          f"mean_kept={np.mean(sizes):.0f}", flush=True)
    print("QVD_DONE", flush=True)


if __name__ == "__main__":
    main()
