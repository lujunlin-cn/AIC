#!/usr/bin/env python3
"""R92: spatially-preserved traditional views for target-domain time ranking.

This script has four resumable actions:

  cache-phd2   RGB/low/high/motion-compensated ResNet18 features for the public
               PHD2 fragment pool;
  train        source-held-out RGB or four-view gated temporal head;
  infer        score the semifinal keyframe sequences and write a score dump;
  package      select exactly the R54 per-video frame counts and copy audited
               V9/B3 boxes into a submission package.

The filtering views are inputs to a frozen semantic encoder, not replacement
labels or hand-written scores. The motion view estimates a translation with
phase correlation before taking the residual. All image forwards are batched.
No official labels are read.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


VIEWS = ("rgb", "low", "high", "motion")
V_DIM = 512
IMG_SIZE = 160
T_STEPS = 8


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def source_split(src: str) -> str:
    # Stable and independent of Python hash randomisation.
    v = int(hashlib.md5(src.encode()).hexdigest(), 16) % 10
    return "train" if v < 8 else "dev"


def load_annotations(path: Path) -> dict[str, list[tuple[float, float]]]:
    raw = json.loads(path.read_text())
    out: dict[str, list[tuple[float, float]]] = {}
    for src, users in raw.items():
        ivs: list[tuple[float, float]] = []
        for selections in users.values():
            for row in selections:
                a, b = float(row["t0"]), float(row["t1"])
                if b > a:
                    ivs.append((a, b))
        if ivs:
            out[str(src)] = sorted(ivs)
    return out


def label_at(times: list[float], intervals: list[tuple[float, float]]) -> np.ndarray:
    y = np.zeros(len(times), np.float32)
    for i, t in enumerate(times):
        # A decoded frame represents a short neighbourhood.  The PHD2 interval
        # labels are second-scale, so centre membership is the least
        # assumption-heavy projection.
        y[i] = float(any(a <= t < b for a, b in intervals))
    return y


def phd2_rows(index: Path, frames_root: Path, selections: Path) -> list[dict]:
    ann = load_annotations(selections)
    rows: list[dict] = []
    for line in index.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        src = str(r["src"])
        if src not in ann:
            # No positive selection in the public training labels. It is not a
            # verified negative; retain a small background control only when
            # the frame directory is complete and mark it weakly labelled.
            intervals: list[tuple[float, float]] = []
            weak_background = True
        else:
            intervals = ann[src]
            weak_background = False
        d = frames_root / str(r["video_id"])
        paths = sorted(d.glob("*.jpg"), key=lambda p: float(p.stem))
        if len(paths) != T_STEPS:
            continue
        times = [float(p.stem) for p in paths]
        y = label_at(times, intervals)
        rows.append({
            "video_id": str(r["video_id"]), "src": src,
            "paths": [str(p) for p in paths], "times": times,
            "y": y.tolist(), "split": source_split(src),
            "weak_background": weak_background,
        })
    return rows


def letterbox(path: str, size: int = IMG_SIZE) -> np.ndarray:
    from PIL import Image
    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = min(size / max(w, 1), size / max(h, 1))
    rw, rh = max(1, round(w * scale)), max(1, round(h * scale))
    im = im.resize((rw, rh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (size, size), (124, 116, 104))
    canvas.paste(im, ((size - rw) // 2, (size - rh) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def _motion_compensated(cur: np.ndarray) -> np.ndarray:
    """Return [T,H,W,3] residual view after translation registration.

    Phase correlation is deliberately only a global translation model. A
    failed or implausibly large estimate falls back to zero translation; this
    avoids turning scene cuts into arbitrary border artefacts.
    """
    import cv2
    out = np.empty_like(cur)
    out[0] = np.full_like(cur[0], 128)
    prev = cur[0]
    for t in range(1, len(cur)):
        g0 = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY).astype(np.float32)
        g1 = cv2.cvtColor(cur[t], cv2.COLOR_RGB2GRAY).astype(np.float32)
        small0 = cv2.resize(g0, (64, 64), interpolation=cv2.INTER_AREA)
        small1 = cv2.resize(g1, (64, 64), interpolation=cv2.INTER_AREA)
        try:
            shift, response = cv2.phaseCorrelate(small0, small1)
            dx, dy = float(shift[0]) * cur.shape[2] / 64.0, float(shift[1]) * cur.shape[1] / 64.0
            if not np.isfinite(dx + dy) or response < 0.02 or abs(dx) > cur.shape[2] * 0.25 or abs(dy) > cur.shape[1] * 0.25:
                dx = dy = 0.0
        except Exception:
            dx = dy = 0.0
        warp = cv2.warpAffine(prev, np.float32([[1, 0, dx], [0, 1, dy]]),
                              (cur.shape[2], cur.shape[1]),
                              flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        # Residual is centred at 0.5 so it can pass through the same RGB
        # normalisation contract as the other views.
        out[t] = np.clip(128.0 + 4.0 * (cur[t].astype(np.float32) - warp.astype(np.float32)), 0, 255).astype(np.uint8)
        prev = cur[t]
    return out


def make_views(batch_paths: list[list[str]]) -> np.ndarray:
    """Return uint8 images [B,T,V,3,H,W] in one vectorised batch."""
    import cv2
    b = len(batch_paths)
    cur = np.empty((b, T_STEPS, IMG_SIZE, IMG_SIZE, 3), np.uint8)
    with ThreadPoolExecutor(max_workers=min(16, max(2, b * 2))) as pool:
        for i, imgs in enumerate(batch_paths):
            cur[i] = np.stack(list(pool.map(letterbox, imgs)))
    views = np.empty((b, T_STEPS, 4, IMG_SIZE, IMG_SIZE, 3), np.uint8)
    views[:, :, 0] = cur
    for i in range(b):
        f = cur[i].astype(np.float32)
        low = np.stack([cv2.GaussianBlur(frame, (0, 0), sigmaX=3.0) for frame in f])
        views[i, :, 1] = np.clip(low, 0, 255).astype(np.uint8)
        views[i, :, 2] = np.clip(128.0 + 4.0 * (f - low), 0, 255).astype(np.uint8)
        views[i, :, 3] = _motion_compensated(cur[i])
    return views.transpose(0, 1, 2, 5, 3, 4)


def load_resnet(weights: Path, device: str):
    import torch
    from torchvision.models import resnet18
    model = resnet18(weights=None)
    model.fc = torch.nn.Identity()
    state = torch.load(weights, map_location="cpu", weights_only=True)
    # The downloaded ImageNet checkpoint includes the 1000-way classifier;
    # the route keeps only the pooled ResNet18 representation.
    state.pop("fc.weight", None); state.pop("fc.bias", None)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def encode_views(views: np.ndarray, backbone, device: str, image_batch: int = 128) -> np.ndarray:
    import torch
    b, t, v, _, h, w = views.shape
    x = views.reshape(b * t * v, 3, h, w).astype(np.float32) / 255.0
    mean = np.asarray([.485, .456, .406], np.float32)[None, :, None, None]
    std = np.asarray([.229, .224, .225], np.float32)[None, :, None, None]
    x = (x - mean) / std
    out = []
    with torch.inference_mode():
        for i in range(0, len(x), image_batch):
            z = torch.from_numpy(np.ascontiguousarray(x[i:i + image_batch])).to(device, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                y = backbone(z)
            out.append(y.float().cpu().numpy())
    return np.concatenate(out, axis=0).reshape(b, t, v, V_DIM).astype(np.float16)


def cache_phd2(a: argparse.Namespace) -> None:
    import torch
    torch.backends.cudnn.benchmark = True
    rows = phd2_rows(a.index, a.frames_root, a.selections)
    if a.limit:
        rows = rows[:a.limit]
    out = a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.device.startswith("cuda"):
        torch.cuda.set_device(a.device)
    backbone = load_resnet(a.weights, a.device)
    feats = np.empty((len(rows), T_STEPS, 4, V_DIM), np.float16)
    ys = np.empty((len(rows), T_STEPS), np.float32)
    split = []
    t0 = time.time()
    for i in range(0, len(rows), a.rows_batch):
        chunk = rows[i:i + a.rows_batch]
        views = make_views([r["paths"] for r in chunk])
        feats[i:i + len(chunk)] = encode_views(views, backbone, a.device, a.image_batch)
        ys[i:i + len(chunk)] = np.asarray([r["y"] for r in chunk], np.float32)
        split.extend(r["split"] for r in chunk)
        if (i // a.rows_batch) % 20 == 0:
            print(json.dumps({"rows": i + len(chunk), "total": len(rows),
                              "elapsed_s": round(time.time() - t0, 1)}, ensure_ascii=False), flush=True)
    np.savez_compressed(out, features=feats, labels=ys,
                        src=np.asarray([r["src"] for r in rows]),
                        video_id=np.asarray([r["video_id"] for r in rows]),
                        split=np.asarray(split))
    meta = {"rows": len(rows), "features_shape": list(feats.shape),
            "weights": str(a.weights), "weights_sha256": sha256(a.weights),
            "frame_contract": "PHD2_FRAG_V1 8 decoded jpg keyframes; phase-correlation translation residual",
            "views": list(VIEWS), "image_size": IMG_SIZE,
            "elapsed_s": round(time.time() - t0, 2),
            "source_counts": {k: int(sum(s == k for s in split)) for k in sorted(set(split))}}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps(meta, indent=2), flush=True)


class MultiViewTCN:
    @staticmethod
    def build(n_views: int, proj: int = 64, ch: int = 128):
        import torch
        import torch.nn as nn
        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.n_views, self.proj_dim, self.ch = n_views, proj, ch
                self.proj = nn.ModuleList([nn.Sequential(nn.LayerNorm(V_DIM), nn.Linear(V_DIM, proj), nn.GELU()) for _ in range(n_views)])
                self.gate = nn.Sequential(nn.Linear(n_views * proj, max(32, proj)), nn.GELU(), nn.Linear(max(32, proj), n_views)) if n_views > 1 else None
                self.temporal = nn.Conv1d(proj, ch, 1)
                self.blocks = nn.ModuleList([nn.Conv1d(ch, ch, 3, padding=d, dilation=d) for d in (1, 2, 4)])
                self.norm = nn.GroupNorm(8, ch)
                self.head = nn.Sequential(nn.Linear(ch + proj, 64), nn.GELU(), nn.Linear(64, 1))
            def forward(self, x):
                # x=[B,T,V,512]
                vs = [self.proj[i](x[:, :, i]) for i in range(self.n_views)]
                q = torch.cat(vs, dim=-1)
                if self.gate is None:
                    fused = vs[0]
                else:
                    g = torch.softmax(self.gate(q), dim=-1)
                    fused = sum(g[:, :, i:i+1] * vs[i] for i in range(self.n_views))
                h = self.temporal(fused.transpose(1, 2))
                for c in self.blocks:
                    h = torch.nn.functional.gelu(self.norm(c(h) + h))
                h = h.transpose(1, 2)
                return self.head(torch.cat([h, fused], dim=-1)).squeeze(-1)
        return Net()


def fixed_f(scores: np.ndarray, y: np.ndarray, keep: float = .85) -> float:
    k = max(1, int(round(keep * len(y))))
    idx = np.argsort(-scores, kind="mergesort")[:k]
    return float(2.0 * y[idx].sum() / (k + y.sum()))


def source_bootstrap(deltas: dict[str, float], seed: int = 20261019, n: int = 5000):
    ids = sorted(deltas)
    d = np.asarray([deltas[k] for k in ids], np.float64)
    rng = np.random.default_rng(seed)
    bs = np.asarray([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return {"sources": len(d), "mean": float(d.mean()),
            "ci95": [float(np.quantile(bs, .025)), float(np.quantile(bs, .975))]}


def eval_scores(scores: np.ndarray, y: np.ndarray, src: np.ndarray) -> dict:
    by = {}
    for i, s in enumerate(src):
        by.setdefault(str(s), []).append(fixed_f(scores[i], y[i]))
    per = {s: float(np.mean(v)) for s, v in by.items()}
    return {"rows": len(scores), "sources": len(per), "macro_f": float(np.mean(list(per.values()))), "per_source": per}


def train_arm(a: argparse.Namespace, mode: str) -> dict:
    import torch
    torch.manual_seed(a.seed)
    z = np.load(a.cache, allow_pickle=False)
    X = z["features"].astype(np.float32)
    Y = z["labels"].astype(np.float32)
    src = z["src"].astype(str)
    train = np.asarray([source_split(s) == "train" for s in src])
    rng = np.random.default_rng(a.seed)
    device = a.device
    n_views = 1 if mode == "rgb" else 4
    model = MultiViewTCN.build(n_views).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=.01)
    Xt, Yt = torch.from_numpy(X[:, :, :n_views]), torch.from_numpy(Y)
    ids = np.flatnonzero(train)
    # Exclude all-background rows from the positive ranking loss; retain them
    # at low BCE weight because they are not verified negatives.
    row_weight = np.where(Yt.sum(1).numpy() > 0, 1.0, .15).astype(np.float32)
    t0 = time.time(); best = (-1.0, None); hist = []
    for step in range(1, a.steps + 1):
        sel = ids[rng.integers(0, len(ids), a.batch)]
        xb = Xt[sel].to(device, non_blocking=True)
        yb = Yt[sel].to(device, non_blocking=True)
        pred = model(xb)
        per = torch.nn.functional.binary_cross_entropy_with_logits(pred, yb, reduction="none").mean(1)
        loss = (per * torch.from_numpy(row_weight[sel]).to(device)).mean()
        mixed = (yb.sum(1) > 0) & (yb.sum(1) < T_STEPS)
        if mixed.any():
            hi, lo = yb.argmax(1), yb.argmin(1)
            ar = torch.arange(len(yb), device=device)[mixed]
            loss = loss + .25 * torch.nn.functional.softplus(-(pred[ar, hi[mixed]] - pred[ar, lo[mixed]])).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            model.eval()
            with torch.inference_mode():
                scores = model(Xt.to(device)[:, :, :n_views]).cpu().numpy()
            dv = ~train
            m = eval_scores(scores[dv], Y[dv], src[dv])
            hist.append({"step": step, "loss": float(loss.detach().cpu()), "dev_macro_f": m["macro_f"], "elapsed_s": time.time()-t0})
            print(json.dumps({"mode": mode, **hist[-1]}, ensure_ascii=False), flush=True)
            if m["macro_f"] > best[0]:
                best = (m["macro_f"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
            model.train()
    model.load_state_dict(best[1])
    model.eval()
    with torch.inference_mode():
        scores = model(Xt.to(device)[:, :, :n_views]).cpu().numpy()
    dv = ~train
    metrics = eval_scores(scores[dv], Y[dv], src[dv])
    train_metrics = eval_scores(scores[train], Y[train], src[train])
    out = a.out_dir / f"r92_{mode}_s{a.seed}.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mode": mode, "n_views": n_views,
                "proj": 64, "ch": 128, "params": sum(p.numel() for p in model.parameters()),
                "dev_macro_f": metrics["macro_f"], "history": hist}, out)
    result = {"mode": mode, "seed": a.seed, "checkpoint": str(out), "params": sum(p.numel() for p in model.parameters()),
              "dev": metrics, "train": train_metrics, "elapsed_s": time.time()-t0}
    (a.out_dir / f"r92_{mode}_s{a.seed}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def train(a: argparse.Namespace) -> None:
    results = []
    for mode in a.modes:
        results.append(train_arm(a, mode))
    # Compare source-level paired deltas for the same rows. The full model's
    # motion-shuffle is evaluated from its checkpoint, not used for selection.
    by = {r["mode"]: r for r in results}
    if "rgb" in by and "multiview" in by:
        per_a, per_e = by["rgb"]["dev"]["per_source"], by["multiview"]["dev"]["per_source"]
        delta = {s: per_e[s] - per_a[s] for s in sorted(set(per_a) & set(per_e))}
        summary = {"arms": results, "paired_multiview_vs_rgb": source_bootstrap(delta)}
    else:
        summary = {"arms": results}
    (a.out_dir / "r92_train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "arms"}, indent=2), flush=True)


def infer(a: argparse.Namespace) -> None:
    import torch
    ck = torch.load(a.checkpoint, map_location="cpu", weights_only=False)
    mode = ck.get("mode", "multiview")
    n_views = int(ck.get("n_views", 4))
    model = MultiViewTCN.build(n_views)
    model.load_state_dict(ck["state_dict"]); model.eval().to(a.device)
    if a.device.startswith("cuda"):
        torch.cuda.set_device(a.device)
    backbone = load_resnet(a.weights, a.device)
    full = {r["video_id"]: r for r in (json.loads(x) for x in a.full_b3.read_text().splitlines() if x.strip())}
    scores_out = []
    vids = sorted(full, key=lambda x: int(x) if str(x).isdigit() else str(x))
    t0 = time.time()
    for vi, vid in enumerate(vids):
        d = a.keyframes / vid
        paths = sorted(d.glob("*.png"), key=lambda p: int(p.stem))
        if not paths:
            continue
        all_scores = []
        # Keep one sequence, but encode in row batches to bound RAM. Motion
        # compensation is computed inside each batch; for boundary continuity
        # prepend the previous frame to each batch is unnecessary because the
        # full sequence is normally under a few hundred keyframes.
        for i in range(0, len(paths), a.rows_batch):
            chunk = paths[i:i + a.rows_batch]
            # Each frame is a one-step row for the feature extractor.  For the
            # motion view, adjacent context within this row is unavailable; a
            # temporal context row is built below from cached per-frame views.
            # We use a sliding 8-frame window for the learned head.
            imgs = [str(p) for p in chunk]
            all_scores.extend(imgs)
        # Build all frame views once, then window features in original order.
        cur = np.stack([letterbox(str(p)) for p in paths])
        bviews = make_views([[str(p)] * T_STEPS for p in paths])
        # Replace repeated-frame motion with true sequence motion and encode all
        # view images in one GPU batch.
        seq_motion = _motion_compensated(cur)
        cv2 = __import__('cv2')
        cur_chw = cur.transpose(0, 3, 1, 2)
        low = np.stack([cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigmaX=3.0) for x in cur]).astype(np.uint8)
        high = np.stack([np.clip(128 + 4 * (x.astype(np.float32) - cv2.GaussianBlur(x.astype(np.float32), (0, 0), sigmaX=3.0)), 0, 255).astype(np.uint8) for x in cur])
        bviews[:, :, 0] = np.repeat(cur_chw[:, None], T_STEPS, axis=1)
        bviews[:, :, 1] = np.repeat(low.transpose(0, 3, 1, 2)[:, None], T_STEPS, axis=1)
        bviews[:, :, 2] = np.repeat(high.transpose(0, 3, 1, 2)[:, None], T_STEPS, axis=1)
        bviews[:, :, 3] = np.repeat(seq_motion.transpose(0, 3, 1, 2)[:, None], T_STEPS, axis=1)
        feat = encode_views(bviews, backbone, a.device, a.image_batch)[:, 0]  # [N,V,512]
        # Approximate local context with the same 8-frame windows used in PHD2.
        pad = np.pad(feat, ((T_STEPS//2, T_STEPS//2), (0,0), (0,0)), mode="edge")
        windows = np.stack([pad[i:i+T_STEPS] for i in range(len(feat))])
        with torch.inference_mode():
            sc = model(torch.from_numpy(windows.astype(np.float32)).to(a.device)).cpu().numpy()
        center = sc[:, T_STEPS//2]
        scores_out.append({"video_id": vid, "frames": [int(p.stem) for p in paths], "scores": center.tolist()})
        if vi % 20 == 0:
            print(json.dumps({"videos": vi + 1, "total": len(vids), "elapsed_s": round(time.time()-t0,1)}, ensure_ascii=False), flush=True)
    a.output.write_text(json.dumps({"checkpoint": str(a.checkpoint), "mode": mode,
                                    "videos": len(scores_out), "elapsed_s": time.time()-t0,
                                    "scores": scores_out}, separators=(",", ":")))
    print(json.dumps({"videos": len(scores_out), "elapsed_s": time.time()-t0, "output": str(a.output)}), flush=True)


def package(a: argparse.Namespace) -> None:
    score_dump = json.loads(a.score_dump.read_text())
    scores = {x["video_id"]: (np.asarray(x["frames"], np.int64), np.asarray(x["scores"], np.float32)) for x in score_dump["scores"]}
    parent = {r["video_id"]: r for r in (json.loads(x) for x in a.parent.read_text().splitlines() if x.strip())}
    full = {r["video_id"]: r for r in (json.loads(x) for x in a.full_b3.read_text().splitlines() if x.strip())}
    out = a.output; out.mkdir(parents=True, exist_ok=False)
    rows = []; changed = 0; total = 0
    for vid in sorted(full, key=lambda x: int(x) if str(x).isdigit() else str(x)):
        frames, sc = scores[vid]
        k = len(parent[vid]["predictions"])
        keep = np.argsort(-sc, kind="mergesort")[:k]
        chosen = set(int(frames[i]) for i in keep)
        old = {int(x["frame"]) for x in parent[vid]["predictions"]}
        if chosen != old: changed += 1
        lookup = {int(x["frame"]): x for x in full[vid]["predictions"]}
        pred = [lookup[f] for f in sorted(chosen)]
        rows.append({"video_id": vid, "targetRatioWH": full[vid]["targetRatioWH"],
                     "model_size_mb": 99.0, "model_params_b": 0.099,
                     "predictions": pred})
        total += len(pred)
    pred_path = out / "predictions.jsonl"
    pred_path.write_text("".join(json.dumps(r, separators=(",", ":"), ensure_ascii=False) + "\n" for r in rows))
    zip_path = out / f"{out.name}.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(pred_path, "predictions.jsonl")
    man = {"submission_id": out.name, "status": "PACKAGED_NOT_UPLOADED",
           "official_platform_score": None, "parent_submission": "R54_TIMELENS_CENTER_RESIDUAL_V9_B3",
           "mechanism": "R92 target-domain PHD2 temporal head with RGB + low-frequency + high-frequency + phase-correlation motion residual gated fusion; exact parent per-video counts; audited V9/B3 boxes",
           "official_test_labels_used": False, "video_count": len(rows), "prediction_count": total,
           "changed_videos": changed, "model_params_b": .099, "model_size_mb": 99.0,
           "prediction_sha256": sha256(pred_path), "zip_sha256": sha256(zip_path), "zip_bytes": zip_path.stat().st_size,
           "score_dump_sha256": sha256(a.score_dump)}
    (out / "manifest.json").write_text(json.dumps(man, indent=2) + "\n")
    (out / f"{zip_path.name}.sha256").write_text(f"{sha256(zip_path)}  {zip_path.name}\n")
    print(json.dumps(man, indent=2), flush=True)


def parse() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="action", required=True)
    p = sub.add_parser("cache-phd2"); p.add_argument("--index", type=Path, required=True); p.add_argument("--frames-root", type=Path, required=True); p.add_argument("--selections", type=Path, required=True); p.add_argument("--weights", type=Path, required=True); p.add_argument("--out", type=Path, required=True); p.add_argument("--device", default="cuda"); p.add_argument("--rows-batch", type=int, default=16); p.add_argument("--image-batch", type=int, default=128); p.add_argument("--limit", type=int, default=0)
    p = sub.add_parser("train"); p.add_argument("--cache", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True); p.add_argument("--device", default="cuda"); p.add_argument("--modes", nargs="*", default=["rgb", "multiview"]); p.add_argument("--steps", type=int, default=1000); p.add_argument("--eval-every", type=int, default=100); p.add_argument("--batch", type=int, default=64); p.add_argument("--lr", type=float, default=2e-3); p.add_argument("--seed", type=int, default=20261019)
    p = sub.add_parser("infer"); p.add_argument("--checkpoint", type=Path, required=True); p.add_argument("--keyframes", type=Path, required=True); p.add_argument("--weights", type=Path, required=True); p.add_argument("--full-b3", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--device", default="cuda"); p.add_argument("--rows-batch", type=int, default=32); p.add_argument("--image-batch", type=int, default=128)
    p = sub.add_parser("package"); p.add_argument("--score-dump", type=Path, required=True); p.add_argument("--parent", type=Path, required=True); p.add_argument("--full-b3", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    return ap.parse_args()


def main():
    a = parse()
    {"cache-phd2": cache_phd2, "train": train, "infer": infer, "package": package}[a.action](a)


if __name__ == "__main__":
    main()
