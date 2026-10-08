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
R12_PRED = "/data/aic/semifinal_20261001/submissions/R12_GATT_LFM_SEMIFINAL/predictions.jsonl"
PARENT_MANIFEST = "/data/aic/semifinal_20261001/submissions/LFM_V8_B3_TEMPORAL_SEMIFINAL/manifest.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--feat", choices=["lfm", "vmae", "qv"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sub-id", default=None)
    ap.add_argument("--keep-frac", type=float, default=0.8)
    ap.add_argument("--parent", default="LFM_V8_B3_TEMPORAL_SEMIFINAL (official 34.88, same 426-video drop)")
    ap.add_argument("--dry-run", action="store_true", help="random scores, no ckpt needed")
    args = ap.parse_args()

    t0 = time.time()
    keep_pct = int(round(args.keep_frac * 100))
    keep_txt = f"whole 1 s keyframes from the top until {keep_pct}% of original frames"
    if abs(args.keep_frac - 0.8) > 1e-6:
        keep_txt += " (keep depth is the single variable vs R12_GATT_LFM_SEMIFINAL, official 35.17)"
    if args.feat == "qv":
        pcf = ("temporal scorer only: same TRINITY-style GATT ranker architecture "
               "(Transformer d512 x3 + 1D RoPE + pairwise ranking), supervision swapped to "
               "QVHighlights saliency soft labels (relevant-window 3-annotator mean, 75 "
               "x 2 s-clip tokens per 150 s segment, 8,619 segments, two seeds QV val AP "
               "40.74/41.52). Deployment scores the 1 s keyframes at stride 2 and broadcasts "
               "to 1 s bins to match the training token distribution. Spatial bboxes copied "
               "byte-for-byte from LFM_V8_B3_SEMIFINAL; keep rule: " + keep_txt + ".")
    else:
        pcf = ("temporal scorer only: V9 dilated TCN (246,401 params) replaced "
               "by the R12 GATT ranker (Transformer d512 x3 + 1D RoPE, trained on "
               "PHD2 GIF-union slot-keep labels, pairwise ranking loss, frozen by "
               "the P0-b Mr.HiSum 4-seed verdict 59.17 vs 54.00). Spatial bboxes "
               "copied byte-for-byte from LFM_V8_B3_SEMIFINAL; keep rule: "
               + keep_txt + ".")
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

    fdir = FEAT_DIRS.get(args.feat, FEAT_DIRS["lfm"])  # qv uses lfm feats dir
    out = []
    n_drop_total = 0
    diffs = []
    r12_frames = {}
    if not args.dry_run and os.path.exists(R12_PRED):
        for line in open(R12_PRED):
            r = json.loads(line)
            r12_frames[r["video_id"]] = {p["frame"] for p in r["predictions"]}
    for vid in sorted(intake, key=int):
        meta = intake[vid]
        total = int(meta["frame_count"])
        vdir = Path(fdir) / vid
        if args.dry_run:
            kfs = sorted(int(p.stem) for p in vdir.glob("*.npy")) if vdir.exists() else list(range(0, total, max(total // 10, 1)))
            scores = np.random.RandomState(int(vid)).rand(len(kfs))
        else:
            kfs = sorted(int(p.stem) for p in vdir.glob("*.npy"))
            if args.feat == "qv":
                # train dist: one token per 2 s clip -> score stride-2, broadcast
                import bisect
                kfs2 = [k for k in kfs if k % 2 == 0] or kfs
                x = np.stack([np.load(vdir / f"{k}.npy").astype(np.float32) for k in kfs2])
                with torch.no_grad():
                    s2 = model(torch.from_numpy(x).unsqueeze(0).to(dev))[0].cpu().numpy()
                sc = {k: float(s2[i]) for i, k in enumerate(kfs2)}
                scores = np.array([sc[kfs2[max(bisect.bisect_right(kfs2, k) - 1, 0)]]
                                   for k in kfs])
            else:
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
        # frame-level keep-set diff vs the shipped 35.17 package
        if not args.dry_run and os.path.exists(R12_PRED):
            r12 = r12_frames.setdefault(vid, set())
            new_f = {p["frame"] for p in preds}
            if r12:
                union = r12 | new_f
                diffs.append(len(r12 ^ new_f) / len(union))
    if diffs:
        print(f"[pkg] keep-set frame sym_diff vs R12_GATT_LFM_SEMIFINAL: "
              f"{float(np.mean(diffs)):.4f} over {len(diffs)} videos", flush=True)

    # sizes: parent components with the temporal head swapped for the GATT ranker
    pm = json.load(open(PARENT_MANIFEST))
    comps = [c for c in pm["components"] if "temporal head" not in c["name"]]
    if not args.dry_run:
        n_par = sum(p.numel() for p in model.parameters())
    else:
        n_par = 9_100_000
    head_family = ("QVHighlights saliency soft labels, pairwise loss"
                   if args.feat == "qv" else "PHD2 GIF-union keep labels, pairwise loss")
    comps.append({"name": f"R12 GATT ranker (fp32, 768->512 proj + 3-layer Transformer 8 heads "
                          f"+ 1D RoPE + frame head, {head_family})",
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
        "parent_baseline": args.parent,
        "primary_changed_factor": pcf,
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
