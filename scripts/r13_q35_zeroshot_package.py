#!/usr/bin/env python3
"""R13 stage-A packager: Qwen3.5-4B zero-shot frame scores -> submission zip.

Single variable vs R12_GATT_LFM_SEMIFINAL (official 35.17): the temporal
keep decision comes from Qwen3.5-4B zero-shot frame scoring (16-64 uniformly
sampled frames per video, one generate call, JSON array parsed); spatial
bboxes are copied byte-for-byte from LFM_V8_B3_SEMIFINAL; keep rule
identical to the parent (whole keyframes from the top until 80% of frames).

Videos whose JSON parse failed fall back to uniform time-ramp scores
(preserves chronological order = no information assumption).

Usage:
  python3 r13_q35_zeroshot_package.py --out-dir <dir> [--keep-frac 0.8] \
      [--sub-id R13_Q35_4B_ZS]
"""
import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, "/root/AIC/scripts")
import numpy as np  # noqa: E402

INTAKE = "/data/aic/semifinal_20261001/intake/index.enriched.jsonl"
SCORES = "/data/aic/experiments_910a/LFM_V13/q35_zs_scores"
FULL_PRED = "/data/aic/semifinal_20261001/submissions/LFM_V8_B3_SEMIFINAL/predictions.jsonl"
R12_PRED = "/data/aic/semifinal_20261001/submissions/R12_GATT_LFM_SEMIFINAL/predictions.jsonl"
PARENT_MANIFEST = "/data/aic/semifinal_20261001/submissions/LFM_V8_B3_TEMPORAL_SEMIFINAL/manifest.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sub-id", default="R13_Q35_4B_ZS")
    ap.add_argument("--keep-frac", type=float, default=0.8)
    args = ap.parse_args()

    t0 = time.time()
    keep_pct = int(round(args.keep_frac * 100))
    intake = {}
    for line in open(INTAKE):
        r = json.loads(line)
        intake[r["video_id"]] = r
    full = {}
    for line in open(FULL_PRED):
        r = json.loads(line)
        full[r["video_id"]] = {p["frame"]: p["bboxes"] for p in r["predictions"]}
    r12_frames = {}
    for line in open(R12_PRED):
        r = json.loads(line)
        r12_frames[r["video_id"]] = {p["frame"] for p in r["predictions"]}

    out, diffs, n_parse_fail = [], [], 0
    for vid in sorted(intake, key=int):
        meta = intake[vid]
        total = int(meta["frame_count"])
        sp = Path(SCORES) / f"{vid}.json"
        if sp.exists():
            d = json.load(open(sp))
            idxs, scores = d["frame_indices"], d["scores"]
            if scores is None or len(scores) != len(idxs):
                n_parse_fail += 1
                scores = list(np.linspace(0, 1, len(idxs)))
        else:
            n_parse_fail += 1
            idxs = list(np.linspace(0, total - 1, 16).astype(int))
            scores = list(np.linspace(0, 1, 16))
        sc = np.asarray(scores, dtype=np.float64)
        idx_arr = np.asarray(idxs)
        # broadcast sampled-frame scores to all original frames
        frames_all = np.arange(total)
        owners = idx_arr[np.abs(frames_all[:, None] - idx_arr[None, :]).argmin(1)]
        frame_score = {idx_arr[i]: float(sc[i]) for i in range(len(idxs))}
        budget = int(round(total * args.keep_frac))
        order = np.argsort(-sc)
        kept_frames, acc = [], 0
        for ki in order:
            sel = frames_all[owners == idx_arr[ki]]
            kept_frames.extend(sel.tolist())
            acc += len(sel)
            if acc >= budget:
                break
        kept = set(kept_frames)
        src = full.get(vid, {})
        preds = [{"frame": f, "bboxes": src[f]} for f in sorted(src) if f in kept]
        out.append({"video_id": vid, "targetRatioWH": meta["targetRatioWH"],
                    "model_size_mb": None, "predictions": preds})
        r12 = r12_frames.get(vid, set())
        new_f = {p["frame"] for p in preds}
        if r12 and new_f:
            diffs.append(len(r12 ^ new_f) / len(r12 | new_f))

    if diffs:
        print(f"[pkg] keep-set sym_diff vs R12_GATT_LFM (35.17): {float(np.mean(diffs)):.4f} "
              f"over {len(diffs)} videos", flush=True)
    pm = json.load(open(PARENT_MANIFEST))
    comps = [c for c in pm["components"] if "temporal head" not in c["name"]]
    comps.append({"name": "Qwen3.5-4B zero-shot frame scorer (natively multimodal, "
                          "GDN hybrid; 4.3B params; no training, prompt-only)",
                  "parameters": 4_300_000_000, "bytes": 8_600_000_000})
    total_bytes = sum(c["bytes"] for c in comps)
    total_par = sum(c["parameters"] for c in comps)
    for row in out:
        row["model_size_mb"] = round(total_bytes / 1e6, 3)
    manifest = {
        "submission_id": args.sub_id,
        "status": "PACKAGED_NOT_UPLOADED_ZEROSHOT_VLM_FRAME_SCORER",
        "uploaded": False,
        "official_platform_score": None,
        "parent_baseline": "R12_GATT_LFM_SEMIFINAL (official 35.17, same 426-video drop)",
        "primary_changed_factor": (
            "temporal scorer only: the R12 GATT ranker (9M params, PHD2-trained) is "
            "replaced by Qwen3.5-4B zero-shot frame scoring — 16-64 uniformly sampled "
            "frames per video, single generate call, per-frame highlight likelihood "
            f"0-1, whole-keyframe keep until {keep_pct}% of frames; spatial bboxes "
            "copied byte-for-byte from LFM_V8_B3_SEMIFINAL; no training of any kind."),
        "components": comps,
        "total_parameters": total_par,
        "total_weight_bytes": total_bytes,
        "declared_model_size_mb": round(total_bytes / 1e6, 3),
        "k_size_implied": 0.9,
        "video_count": 426,
        "prediction_count": sum(len(r["predictions"]) for r in out),
        "score_parse_failures": n_parse_fail,
    }

    od = Path(args.out_dir)
    od.mkdir(parents=True, exist_ok=True)
    with open(od / "predictions.jsonl", "w") as f:
        for row in out:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (od / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))
    zpath = od / f"{args.sub_id}.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(od / "predictions.jsonl", "predictions.jsonl")
        z.write(od / "manifest.json", "manifest.json")
    sha = hashlib.sha256(zpath.read_bytes()).hexdigest()
    (od / f"{args.sub_id}.zip.sha256").write_text(sha + "  " + zpath.name + "\n")
    with zipfile.ZipFile(zpath) as z:
        assert len(z.namelist()) == 2
        assert json.loads(z.read("manifest.json"))["video_count"] == 426
    print(f"PKG_DONE {args.sub_id} zip={zpath} sha256={sha[:16]}... "
          f"pred_frames={manifest['prediction_count']} parse_fail={n_parse_fail} "
          f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
