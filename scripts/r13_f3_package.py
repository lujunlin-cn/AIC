#!/usr/bin/env python3
"""R13 F3 packager: Mr.HiSum-supervised yt8m-feature ranker -> submission zip.

Single variable vs R12_GATT_LFM_SEMIFINAL (official 35.17): the temporal
keep decision comes from the F1-frozen recipe ranker trained on Mr.HiSum
(27,892 videos, user Most-Replayed behavior labels), scoring the 426 videos
through the SAME yt8m feature contract (inception-2015-12-05 pool_3 -> yt8m
PCA -> uint8 round-trip, 1 fps). Spatial bboxes byte-identical to
LFM_V8_B3_SEMIFINAL; keep rule identical (whole seconds from the top until
80% of frames; second-of-scores broadcast to all frames in that second).

Usage:
  python3 r13_f3_package.py --ckpt <f1_*.pt> --feat rgb --out-dir <dir> \
      [--sub-id R13_F3_MRHISUM_RGB]
"""
import argparse
import hashlib
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/AIC/scripts")

INTAKE = "/data/aic/semifinal_20261001/intake/index.enriched.jsonl"
FEATS = "/data/aic/experiments_910a/LFM_V13/f3_yt8m_feats"
FULL_PRED = "/data/aic/semifinal_20261001/submissions/LFM_V8_B3_SEMIFINAL/predictions.jsonl"
R12_PRED = "/data/aic/semifinal_20261001/submissions/R12_GATT_LFM_SEMIFINAL/predictions.jsonl"
PARENT_MANIFEST = "/data/aic/semifinal_20261001/submissions/LFM_V8_B3_TEMPORAL_SEMIFINAL/manifest.json"

D_MODEL = 512


def rope_cache(n, d, device):
    inv = 1.0 / (10000.0 ** (torch.arange(0, d, 2, device=device).float() / d))
    t = torch.arange(n, device=device).float()
    return torch.cos(torch.outer(t, inv)), torch.sin(torch.outer(t, inv))


def apply_rope(x, cos, sin):
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None].to(x.dtype)
    sin = sin[None, None].to(x.dtype)
    o1, o2 = x1 * cos - x2 * sin, x1 * sin + x2 * cos
    out = torch.empty_like(x)
    out[..., 0::2], out[..., 1::2] = o1, o2
    return out


class Block(torch.nn.Module):
    def __init__(self, d, h, nh):
        super().__init__()
        self.nh, self.dh = nh, d // nh
        self.qkv = torch.nn.Linear(d, 3 * d)
        self.proj = torch.nn.Linear(d, d)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(d, 2 * d), torch.nn.GELU(),
                                       torch.nn.Linear(2 * d, d))
        self.n1, self.n2 = torch.nn.LayerNorm(d), torch.nn.LayerNorm(d)

    def forward(self, x, cos, sin):
        b, t, _ = x.shape
        qkv = self.qkv(self.n1(x)).view(b, t, 3, self.nh, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        att = (q @ k.transpose(-2, -1)) / (self.dh ** 0.5)
        o = (att.softmax(-1) @ v).transpose(1, 2).reshape(b, t, -1)
        return x + self.mlp(self.n2(x + self.proj(o)))


class Ranker(torch.nn.Module):
    def __init__(self, d_in, d_model=D_MODEL, heads=8, depth=5):
        super().__init__()
        self.proj = torch.nn.Linear(d_in, d_model)
        self.blocks = torch.nn.ModuleList([Block(d_model, d_model, heads) for _ in range(depth)])
        self.head = torch.nn.Linear(d_model, 1)
        self.norm = torch.nn.LayerNorm(d_model)
        self.dh = d_model // heads

    def forward(self, x):
        h = self.proj(x)
        cos, sin = rope_cache(h.shape[1], self.dh, h.device)
        for blk in self.blocks:
            h = blk(h, cos, sin)
        return self.head(self.norm(h)).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--feat", choices=["rgb", "rgb+audio"], default="rgb")
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--keep-frac", type=float, default=0.8)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sub-id", default="R13_F3_MRHISUM_RGB")
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

    model = Ranker(1024 if args.feat == "rgb" else 1152, depth=args.depth)
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu"))
    model.eval()
    torch.set_num_threads(32)

    out, diffs, n_missing = [], [], 0
    for vid in sorted(intake, key=int):
        meta = intake[vid]
        total = int(meta["frame_count"])
        fp = Path(FEATS) / f"{vid}.npz"
        if not fp.exists():
            n_missing += 1
            scores, n_sec = list(np.linspace(0, 1, 16)), 16
        else:
            d = np.load(fp)
            x = d["feat"].astype(np.float32)
            if args.feat == "rgb+audio":
                x = np.concatenate([x, np.zeros((len(x), 128), np.float32)], 1)  # NO-OP guard
            with torch.no_grad():
                scores = model(torch.from_numpy(x).unsqueeze(0))[0].numpy()
            n_sec = len(scores)
        # second-of-scores -> frame-level owners (second s owns frames [s*fps, (s+1)*fps))
        fps = max(1.0, total / max(n_sec, 1))
        frames_all = np.arange(total)
        owners = np.minimum((frames_all / fps).astype(int), n_sec - 1)
        sc = np.asarray(scores, dtype=np.float64)
        budget = int(round(total * args.keep_frac))
        order = np.argsort(-sc)
        kept_frames, acc = [], 0
        for ki in order:
            sel = frames_all[owners == ki]
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
    comps.append({"name": "Mr.HiSum-supervised yt8m-feature frame ranker "
                          "(d5/h512/RoPE light transformer; ~13M params; trained on "
                          "Mr.HiSum 27,892 videos user Most-Replayed labels)",
                  "parameters": 13_000_000, "bytes": 26_000_000})
    total_bytes = sum(c["bytes"] for c in comps)
    total_par = sum(c["parameters"] for c in comps)
    for row in out:
        row["model_size_mb"] = round(total_bytes / 1e6, 3)
    manifest = {
        "submission_id": args.sub_id,
        "status": "PACKAGED_NOT_UPLOADED_MRHISUM_RANKER",
        "uploaded": False,
        "official_platform_score": None,
        "parent_baseline": "R12_GATT_LFM_SEMIFINAL (official 35.17, same 426-video drop)",
        "primary_changed_factor": (
            "temporal scorer only: the R12 GATT ranker (9M params, PHD2-frag-trained) is "
            "replaced by a light transformer ranker trained on Mr.HiSum 27,892 videos "
            "(YouTube Most-Replayed behavior labels) with the F1-frozen recipe; the 426 "
            f"videos are scored through the same yt8m feature contract (1 fps, "
            f"inception pool_3 -> yt8m PCA), whole-second keep until {keep_pct}% of frames; "
            "spatial bboxes copied byte-for-byte from LFM_V8_B3_SEMIFINAL."),
        "components": comps,
        "total_parameters": total_par,
        "total_weight_bytes": total_bytes,
        "declared_model_size_mb": round(total_bytes / 1e6, 3),
        "video_count": 426,
        "prediction_count": sum(len(r["predictions"]) for r in out),
        "feat_missing_fallback": n_missing,
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
          f"pred_frames={manifest['prediction_count']} feat_missing={n_missing} "
          f"elapsed={(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
