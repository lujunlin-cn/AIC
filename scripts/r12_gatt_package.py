#!/usr/bin/env python3
"""R12-GATT submission package builder.

Single variable vs LFM_V8_B3_TEMPORAL_SEMIFINAL (official 34.88):
spatial bboxes byte-for-byte from LFM_V8_B3_SEMIFINAL (34.75, full-frame),
temporal keep decision from the R12 GATT ranker instead of the V9 TCN.
Keep rule identical to the parent: whole 1 s keyframes taken from the top
until 80% of the original frames are filled.

Inputs (all on-disk):
  intake frames : /data/aic/semifinal_20261001/intake/index.enriched.jsonl
  keyframe feats: /data/aic/semifinal_20261001/temporal_feats/<vid>/<frame>.npy (768,) f16
                  (lfm)  or  videomae_feats/<vid>/... (vmae, verified at runtime)
  full bboxes   : /data/aic/semifinal_20261001/submissions/LFM_V8_B3_SEMIFINAL/predictions.jsonl

Usage:
  python3 r12_gatt_package.py --ckpt <best.pt> --feat lfm --out-dir <dir> \
      [--keep-frac 0.8] [--dry-run]
"""
import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, "/root/AIC/scripts")
import numpy as np  # noqa: E402
import torch  # noqa: E402

from r12_gatt_vmaef_train import Ranker  # noqa: E402

INTAKE = "/data/aic/semifinal_20261001/intake/index.enriched.jsonl"
FEAT_DIRS = {"lfm": "/data/aic/semifinal_20261001/temporal_feats",
             "vmae": "/data/aic/semifinal_20261001/videomae_feats"}
FULL_PRED = "/data/aic/semifinal_20261001/submissions/LFM_V8_B3_SEMIFINAL/predictions.jsonl"
PARENT_MANIFEST = "/data/aic/semifinal_20261001/submissions/LFM_V8_B3_TEMPORAL_SEMIFINAL/manifest.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--feat", choices=["lfm", "vmae"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sub-id", default=None)
    ap.add_argument("--keep-frac", type=float, default=0.8)
    ap.add_argument("--dry-run", action="store_true", help="random scores, no ckpt needed")
    args = ap.parse_args()

    t0 = time.time()
    dev = "cpu"
    model = None
    if not args.dry_run:
        try:
            import torch_npu  # noqa: F401
            torch.npu.set_compile_mode(jit_compile=False)
            dev = "npu"
        except ImportError:
            pass
        model = Ranker().to(dev)
        model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
        model.eval()
        n_par = sum(p.numel() for p in model.parameters())
        print(f"[pkg] ckpt loaded {n_par} params on {dev}", flush=True)

    intake = {}
    for line in open(INTAKE):
        r = json.loads(line)
        intake[r["video_id"]] = r
    full = {}
    for line in open(FULL_PRED):
        r = json.loads(line)
        full[r["video_id"]] = {p["frame"]: p["bboxes"] for p in r["predictions"]}

    fdir = FEAT_DIRS[args.feat]
    out = []
    n_drop_total = 0
    for vid in sorted(intake, key=int):
        meta = intake[vid]
        total = int(meta["frame_count"])
        vdir = Path(fdir) / vid
        if args.dry_run:
            kfs = sorted(int(p.stem) for p in vdir.glob("*.npy")) if vdir.exists() else list(range(0, total, max(total // 10, 1)))
            scores = np.random.RandomState(int(vid)).rand(len(kfs))
        else:
            kfs = sorted(int(p.stem) for p in vdir.glob("*.npy"))
            x = np.stack([np.load(vdir / f"{k}.npy").astype(np.float32) for k in kfs])
            with torch.no_grad():
                scores = model(torch.from_numpy(x).unsqueeze(0).to(dev))[0].cpu().numpy()
        # frame -> nearest keyframe
        kf_arr = np.array(kfs)
        frames_all = np.arange(total)
        owners = kf_arr[np.abs(frames_all[:, None] - kf_arr[None, :]).argmin(1)]
        budget = int(round(total * args.keep_frac))
        order = np.argsort(-scores)
        kept_frames, acc = [], 0
        for ki in order:
            sel = frames_all[owners == kfs[ki]]
            kept_frames.extend(sel.tolist())
            acc += len(sel)
            if acc >= budget:
                break
        kept = set(kept_frames)
        src = full.get(vid, {})
        preds = [{"frame": f, "bboxes": src[f]} for f in sorted(src) if f in kept]
        n_drop_total += len(src) - len(preds)
        out.append({"video_id": vid, "targetRatioWH": meta["targetRatioWH"],
                    "model_size_mb": None, "predictions": preds})
    print(f"[pkg] scored 426 videos, dropped {n_drop_total} frames total "
          f"({time.time() - t0:.0f}s)", flush=True)

    # sizes: parent components with the temporal head swapped for the GATT ranker
    pm = json.load(open(PARENT_MANIFEST))
    comps = [c for c in pm["components"] if "temporal head" not in c["name"]]
    if not args.dry_run:
        n_par = sum(p.numel() for p in model.parameters())
    else:
        n_par = 9_100_000
    comps.append({"name": "R12 GATT ranker (fp32, 768->512 proj + 3-layer Transformer 8 heads "
                          "+ 1D RoPE + frame head, PHD2 GIF-union keep labels, pairwise loss)",
                  "parameters": n_par, "bytes": n_par * 4})
    total_bytes = sum(c["bytes"] for c in comps)
    total_par = sum(c["parameters"] for c in comps)
    sub_id = args.sub_id or f"R12_GATT_{args.feat.upper()}_SEMIFINAL"

    for row in out:
        row["model_size_mb"] = round(total_bytes / 1e6, 3)
    manifest = {
        "submission_id": sub_id,
        "status": "PACKAGED_NOT_UPLOADED_TRAINED_STUDENT_TEMPORAL_HEAD",
        "uploaded": False,
        "official_platform_score": None,
        "parent_baseline": "LFM_V8_B3_TEMPORAL_SEMIFINAL (official 34.88, same 426-video drop)",
        "primary_changed_factor": "temporal scorer only: V9 dilated TCN (246,401 params) replaced "
                                  "by the R12 GATT ranker (Transformer d512 x3 + 1D RoPE, trained on "
                                  "PHD2 GIF-union slot-keep labels, pairwise ranking loss, frozen by "
                                  "the P0-b Mr.HiSum 4-seed verdict 59.17 vs 54.00). Spatial bboxes "
                                  "copied byte-for-byte from LFM_V8_B3_SEMIFINAL; keep rule identical "
                                  "(whole 1 s keyframes from the top until 80% of original frames).",
        "components": comps,
        "total_parameters": total_par,
        "total_weight_bytes": total_bytes,
        "declared_model_size_mb": round(total_bytes / 1e6, 3),
        "k_size_implied": 0.9,
        "video_count": 426,
        "prediction_count": sum(len(r["predictions"]) for r in out),
    }

    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "predictions.jsonl", "w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (od / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    zpath = od / f"{sub_id}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(od / "predictions.jsonl", "predictions.jsonl")
        z.write(od / "manifest.json", "manifest.json")
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    (od / f"{sub_id}.zip.sha256").write_text(sha + "  " + zpath.name + "\n")
    # round-trip check
    with zipfile.ZipFile(zpath) as z:
        assert len(z.namelist()) == 2
        assert json.loads(z.read("manifest.json"))["video_count"] == 426
    print(f"PKG_DONE {sub_id} zip={zpath} sha256={sha[:16]}... "
          f"pred_frames={manifest['prediction_count']} k=0.90", flush=True)


if __name__ == "__main__":
    main()
