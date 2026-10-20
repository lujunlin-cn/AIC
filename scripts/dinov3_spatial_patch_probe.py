#!/usr/bin/env python3
"""DINOv3 patch-token spatial utility probe on the existing R70 crop tables.

This keeps the 129 legal windows, native crop labels, source split and
letterbox contract fixed.  Only the visual representation is changed:
``patch`` uses DINOv3 patch tokens, ``cls`` broadcasts the global CLS token
over the grid, and ``shuffle`` randomly permutes patch locations.  All three
arms use the same candidate head and decoder, so a patch-vs-CLS gap is a
useful representation diagnostic rather than a post-processing comparison.
No official test asset is read.
"""
from __future__ import annotations

import argparse, json, math, pickle, time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from dinov3_runtime import encode_rgb, load_model

NC = 129


def geometry(W: float, H: float, ratio):
    target = float(ratio[0]) / float(ratio[1])
    if W / H > target:
        h, w = H, H * target
        return w, h, 0
    if W / H < target:
        w, h = W, W / target
        return w, h, 1
    return W, H, None


def candidate_offsets(W, H, ratio):
    w, h, axis = geometry(W, H, ratio)
    if axis == 0:
        return np.linspace(0.0, W - w, NC, dtype=np.float32), w, h, axis
    if axis == 1:
        return np.linspace(0.0, H - h, NC, dtype=np.float32), w, h, axis
    return np.zeros(NC, np.float32), w, h, axis


def candidate_boxes(offsets, W, H, ratio):
    w, h, axis = geometry(W, H, ratio)
    out = np.zeros((len(offsets), 4), np.float32)
    if axis == 0:
        out[:] = np.stack([offsets, np.zeros(len(offsets)), offsets + w,
                           np.full(len(offsets), h)], 1)
    elif axis == 1:
        out[:] = np.stack([np.zeros(len(offsets)), offsets,
                           np.full(len(offsets), w), offsets + h], 1)
    else:
        out[:] = [0, 0, W, H]
    return out


def iou(a, b):
    a = np.asarray(a, np.float32)[:, None, :]
    b = np.asarray(b, np.float32)[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
    aa = np.maximum(a[..., 2] - a[..., 0], 0) * np.maximum(a[..., 3] - a[..., 1], 0)
    bb = np.maximum(b[..., 2] - b[..., 0], 0) * np.maximum(b[..., 3] - b[..., 1], 0)
    return inter / np.maximum(aa + bb - inter, 1e-8)


def grid_masks(grid, transform, W, H, ratio):
    """Candidate and valid patch masks in DINO's 224-pixel letterbox space."""
    fh, fw, _ = grid.shape
    offsets, w, h, axis = candidate_offsets(W, H, ratio)
    sx, sy, px, py = (transform[k] for k in ("scale_x", "scale_y", "pad_x", "pad_y"))
    # Patch centers are used to avoid counting a patch solely because its
    # corner touches a crop.  Letterbox padding is excluded from both means.
    yy, xx = np.mgrid[0:fh, 0:fw]
    cx = (xx.reshape(-1) + .5) * (224.0 / fw)
    cy = (yy.reshape(-1) + .5) * (224.0 / fh)
    valid = ((cx >= px) & (cx < px + W * sx) &
             (cy >= py) & (cy < py + H * sy))
    masks = np.zeros((NC, fh * fw), np.float32)
    for j, o in enumerate(offsets):
        if axis == 0:
            x1, y1, x2, y2 = o, 0., o + w, h
        elif axis == 1:
            x1, y1, x2, y2 = 0., o, w, o + h
        else:
            x1, y1, x2, y2 = 0., 0., W, H
        x1, x2 = x1 * sx + px, x2 * sx + px
        y1, y2 = y1 * sy + py, y2 * sy + py
        masks[j] = ((cx >= x1) & (cx < x2) & (cy >= y1) & (cy < y2) & valid)
        if masks[j].sum() == 0:
            # This can only occur for extreme aspect ratios; retain a legal
            # feature instead of silently producing NaN.
            k = np.flatnonzero(valid)
            masks[j, k[len(k) // 2] if len(k) else 0] = 1.
    return masks, valid.astype(np.float32), offsets


def candidate_features(grid, cls, transform, W, H, ratio, mode, rng):
    if mode == "cls":
        grid = np.broadcast_to(cls.reshape(1, 1, -1), grid.shape).copy()
    elif mode == "shuffle":
        flat = grid.reshape(-1, grid.shape[-1]).copy()
        grid = flat[rng.permutation(len(flat))].reshape(grid.shape)
    masks, valid, offsets = grid_masks(grid, transform, W, H, ratio)
    flat = grid.reshape(-1, grid.shape[-1]).astype(np.float32)
    win = masks @ flat / np.maximum(masks.sum(1, keepdims=True), 1.)
    outside_mask = valid[None, :] - masks
    outside = outside_mask @ flat / np.maximum(outside_mask.sum(1, keepdims=True), 1.)
    pos = (offsets / (offsets[-1] if offsets[-1] > 0 else 1.)).reshape(-1, 1)
    return np.concatenate([win, outside, win - outside, pos], 1).astype(np.float16)


def table_rows(root: Path, names, rv_frames: Path):
    rows, images = [], {}
    for name in names:
        units = pickle.loads((root / "tables" / f"{name}.pkl").read_bytes())
        split = "dev" if "dev" in name else "train"
        for ui, u in enumerate(units):
            ds, vid = str(u["ds"]), str(u["vid"])
            # The public LIVE dev table is source-disjoint but its keyframes
            # remain in the historical train keyframe directory.  Only the
            # old confirm3 split uses keyframes_val (not part of this probe).
            frame_root = root / "keyframes_train"
            if ds == "rv": frame_root = root / "keyframes_train"
            for jj, j in enumerate(np.asarray(u["J"]).tolist()):
                kf = int(u["keys"][int(j)])
                p = (frame_root / vid / f"{kf}.png" if ds == "live"
                     else rv_frames / "d1" / vid / f"{kf}.png")
                if not p.exists():
                    raise FileNotFoundError(p)
                key = f"{ds}:{vid}:{kf}"
                images[key] = p
                gf = {int(f): i for i, f in enumerate(np.asarray(u["gf"]).tolist())}
                if kf not in gf: continue
                gt = np.asarray(u["gt"])[:, gf[kf], :]
                rows.append({"key": key, "source": key.rsplit(":", 1)[0],
                             "split": split, "name": name, "W": float(u["W"]),
                             "H": float(u["H"]), "ratio": list(u["ratio"]),
                             "gt": gt.tolist()})
    return rows, images


def load_cache(args, names):
    root = Path(args.t5_root)
    rows, images = table_rows(root, names)
    keys = list(images)
    model, meta = load_model(args.dinov3_repo, args.weights, args.device, args.dtype)
    cache = {}
    started = time.time()
    for st in range(0, len(keys), args.batch_size):
        kk = keys[st:st + args.batch_size]
        frames = [np.asarray(Image.open(images[k]).convert("RGB"), np.uint8) for k in kk]
        enc = encode_rgb(model, frames, args.device, args.batch_size, 224, "letterbox", True)
        for i, k in enumerate(kk):
            cache[k] = (enc["patch_tokens"][i].astype(np.float16),
                        enc["pooled"][i].astype(np.float16), enc["transforms"][i])
        if st == 0 or st + len(kk) == len(keys) or st // args.batch_size % 10 == 0:
            print(json.dumps({"frames": st + len(kk), "total": len(keys),
                              "elapsed_s": round(time.time() - started, 1)}), flush=True)
    return rows, cache, meta


def build_arrays(rows, cache, mode, seed):
    rng = np.random.default_rng(seed)
    xs, ys, srcs, splits = [], [], [], []
    for r in rows:
        grid, cls, tr = cache[r["key"]]
        x = candidate_features(grid, cls, tr, r["W"], r["H"], r["ratio"], mode, rng)
        offsets = candidate_offsets(r["W"], r["H"], r["ratio"])[0]
        yy = iou(candidate_boxes(offsets, r["W"], r["H"], r["ratio"]), np.asarray(r["gt"])).mean(1)
        xs.append(x); ys.append(yy.astype(np.float32)); srcs.append(r["source"]); splits.append(r["split"])
    return np.stack(xs), np.stack(ys), np.asarray(srcs), np.asarray(splits)


def train_eval(x, y, src, split, args):
    import torch
    torch.manual_seed(args.seed)
    tr, va = split == "train", split == "dev"
    xt, yt = torch.from_numpy(x[tr]).float(), torch.from_numpy(y[tr]).float()
    xv, yv = torch.from_numpy(x[va]).float().to(args.device), torch.from_numpy(y[va]).float()
    class Head(torch.nn.Module):
        def __init__(self, d):
            super().__init__(); self.net = torch.nn.Sequential(torch.nn.Linear(d, 256), torch.nn.GELU(), torch.nn.Linear(256, 1))
        def forward(self, z): return self.net(z).squeeze(-1)
    model = Head(x.shape[-1]).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-2)
    rng = np.random.default_rng(args.seed)
    bs = min(args.train_batch, len(xt))
    for step in range(args.steps):
        ii = rng.integers(0, len(xt), bs)
        pred = model(xt[ii].to(args.device))
        # Utility is a continuous IoU; regression preserves ranking signal.
        loss = torch.nn.functional.smooth_l1_loss(pred, yt[ii].to(args.device))
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.inference_mode(): p = model(xv).cpu().numpy()
    # Candidate top-1 IoU, source macro.  Center is candidate nearest middle.
    top = p.argmax(1); oracle = yv.numpy().max(1); center = yv.numpy()[:, NC // 2]
    vals = defaultdict(list)
    for i, s in enumerate(src[va]): vals[str(s)].append(float(yv.numpy()[i, top[i]]))
    macro = float(np.mean([np.mean(v) for v in vals.values()]))
    return {"params": sum(p.numel() for p in model.parameters()), "dev_rows": int(va.sum()),
            "dev_sources": len(vals), "source_macro_top_iou": macro,
            "row_top_iou": float(np.mean(yv.numpy()[np.arange(len(top)), top])),
            "row_center_iou": float(center.mean()), "row_oracle_iou": float(oracle.mean()),
            "delta_vs_center": macro - float(center.mean()), "steps": args.steps}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t5-root", required=True); ap.add_argument("--rv-frames", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dinov3-repo", required=True); ap.add_argument("--weights", required=True)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--dtype", default="float16")
    ap.add_argument("--batch-size", type=int, default=128); ap.add_argument("--train-batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=800); ap.add_argument("--seed", type=int, default=20261020)
    args = ap.parse_args(); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    names = ["rv_train", "live_train", "rv_dev", "live_dev"]
    # Keep the legacy RetargetVid frame root explicit: it is not inside the
    # T5 table directory on the cloned host.
    rows, images = table_rows(Path(args.t5_root), names, Path(args.rv_frames))
    model, meta = load_model(args.dinov3_repo, args.weights, args.device, args.dtype)
    keys = list(images); cache = {}; started = time.time()
    for st in range(0, len(keys), args.batch_size):
        kk = keys[st:st + args.batch_size]
        frames = [np.asarray(Image.open(images[k]).convert("RGB"), np.uint8) for k in kk]
        enc = encode_rgb(model, frames, args.device, args.batch_size, 224, "letterbox", True)
        for i, k in enumerate(kk):
            cache[k] = (enc["patch_tokens"][i].astype(np.float16), enc["pooled"][i].astype(np.float16), enc["transforms"][i])
        if st == 0 or st + len(kk) == len(keys) or st // args.batch_size % 10 == 0:
            print(json.dumps({"frames": st + len(kk), "total": len(keys), "elapsed_s": round(time.time() - started, 1)}), flush=True)
    results = {"protocol": "DINOv3 ViT-B/16; fixed R70 129 legal windows; native source-disjoint crop GT",
               "model": meta, "rows": len(rows), "unique_frames": len(cache), "modes": {}}
    for mode in ("patch", "cls", "shuffle"):
        x, y, src, split = build_arrays(rows, cache, mode, args.seed)
        results["modes"][mode] = train_eval(x, y, src, split, args)
        np.savez_compressed(out / f"{mode}_dev_predictions.npz", y=y[split == "dev"], source=src[split == "dev"], split=split[split == "dev"])
        del x, y
        print(json.dumps({"mode": mode, **results["modes"][mode]}), flush=True)
    (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__": main()
