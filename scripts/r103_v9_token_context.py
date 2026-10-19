#!/usr/bin/env python3
"""R103: spatial utility head with genuine neighboring V9 vision tokens.

The current candidate feature is the existing V9 129-window representation.
For each candidate frame, the context consists of 4x4 pooled spatial tokens
from the adjacent keyframes in the same source/ratio unit.  A source-permuted
context evaluation is included to distinguish usable temporal context from a
parameterized pooled proxy.  This script never reads official test labels.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pickle
import time
from pathlib import Path

import numpy as np


def load_r70(path):
    spec = importlib.util.spec_from_file_location("r70", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def pool_grid(grid, size=4):
    h, w, d = grid.shape
    h2, w2 = (h // size) * size, (w // size) * size
    grid = grid[:h2, :w2].reshape(size, h2 // size, size, w2 // size, d)
    return grid.mean((1, 3)).reshape(size * size, d).astype(np.float16)


def image_path(args, ds, vid, kf, split):
    from pathlib import Path
    if ds == "live":
        root = Path(args.t5_dir) / ("keyframes_val" if split == "live_confirm3" else "keyframes_train")
        return root / str(vid) / f"{kf}.png"
    return Path(args.rv_frames) / "d1" / str(vid) / f"{kf}.png"


def extract(args):
    r70 = load_r70(args.r70_script)
    import torch
    from PIL import Image

    torch.set_grad_enabled(False)
    torch_mod, vision, proc = r70.load_vision(args)
    out = Path(args.out); (out / "shards").mkdir(parents=True, exist_ok=True)
    rows, shard = [], 0
    t0 = time.time()

    def flush():
        nonlocal rows, shard
        if not rows:
            return
        np.savez_compressed(
            out / "shards" / f"shard_{shard:05d}.npz",
            x=np.stack([r["x"] for r in rows]),
            y=np.stack([r["y"] for r in rows]),
            context=np.stack([r["context"] for r in rows]),
            cmask=np.stack([r["cmask"] for r in rows]),
            source=np.array([r["source"] for r in rows], dtype=object),
            name=np.array([r["name"] for r in rows], dtype=object),
        )
        shard += 1; rows = []

    for name in args.names:
        units = pickle.loads((Path(args.t5_dir) / "tables" / f"{name}.pkl").read_bytes())
        if args.limit:
            units = units[: args.limit]
        for ui, unit in enumerate(units):
            ds, vid = unit["ds"], unit["vid"]
            valid_js = [int(j) for j in unit["J"] if int(unit["keys"][int(j)]) in {int(f) for f in unit["gf"]}]
            if not valid_js:
                continue
            # Cache all keyframe grids in this unit; neighboring context is
            # defined on the native ordered keyframe list, not wall-clock N.
            grids = {}
            for j in valid_js:
                kf = int(unit["keys"][j]); p = image_path(args, ds, vid, kf, name)
                if not p.exists():
                    raise FileNotFoundError(p)
                image = Image.open(p).convert("RGB")
                inp = proc(images=image, return_tensors="pt")
                inp = {k: v.to(args.device) for k, v in inp.items() if torch.is_tensor(v)}
                o = vision(pixel_values=inp["pixel_values"].half(),
                           spatial_shapes=inp["spatial_shapes"],
                           pixel_attention_mask=inp["pixel_attention_mask"],
                           return_dict=True)
                valid = int(inp["pixel_attention_mask"][0].sum())
                fh, fw = [int(v) for v in inp["spatial_shapes"][0]]
                g = o.last_hidden_state[0, :valid].float().cpu().numpy().reshape(fh, fw, -1)
                g = g[: (fh // 2) * 2, : (fw // 2) * 2]
                grids[j] = g
            positions = {j: i for i, j in enumerate(valid_js)}
            for j in valid_js:
                feat, offsets = r70.feature_from_grid(grids[j], unit["W"], unit["H"], unit["ratio"])
                # table_gt indexes the original unit[J] position; valid_js
                # may be shorter when a keyframe has no native crop GT.
                y = r70.table_gt(unit, list(map(int, unit["J"])).index(j), offsets)
                if y is None:
                    continue
                pos = positions[j]
                neigh, mask = [], []
                for q in (pos - 1, pos + 1):
                    if 0 <= q < len(valid_js):
                        neigh.append(pool_grid(grids[valid_js[q]], 4)); mask.append(True)
                    else:
                        neigh.append(np.zeros((16, grids[j].shape[-1]), np.float16)); mask.append(False)
                rows.append({"x": feat, "y": y, "context": np.stack(neigh),
                             "cmask": np.array(mask, np.bool_), "source": f"{ds}:{vid}", "name": name})
                if len(rows) >= args.shard_rows:
                    flush()
            if (ui + 1) % 10 == 0:
                print(json.dumps({"name": name, "units": ui + 1, "rows_buffer": len(rows),
                                  "shards": shard, "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    flush()
    (out / "meta.json").write_text(json.dumps({"names": args.names, "shards": shard,
        "context": "two adjacent same-unit V9 vision grids, pooled 4x4 tokens", "official_test_labels_used": False}, indent=2) + "\n")


def load_rows(out):
    parts = [np.load(p, allow_pickle=True) for p in sorted((Path(out) / "shards").glob("shard_*.npz"))]
    if not parts:
        raise RuntimeError("no context shards")
    keys = ["x", "y", "context", "cmask", "source", "name"]
    return {k: np.concatenate([p[k] for p in parts]) for k in keys}


def source_macro(pred, y, src, mask):
    vals = []
    for s in sorted(set(src[mask])):
        q = mask & (src == s)
        vals.append(float(y[q, np.argmax(pred[q], axis=1)].mean()))
    return float(np.mean(vals)) if vals else float("nan")


def train_model(x, y, ctx, cmask, src, names, train_mask, device, steps, seed, use_context, permute_eval=False):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    dctx = int(ctx.shape[-1]); nc = int(x.shape[1])
    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.cur = nn.Sequential(nn.LayerNorm(x.shape[-1]), nn.Linear(x.shape[-1], 128), nn.GELU())
            self.ctx = nn.Sequential(nn.LayerNorm(dctx), nn.Linear(dctx, 128), nn.GELU())
            self.attn = nn.MultiheadAttention(128, 4, batch_first=True)
            self.out = nn.Sequential(nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))
        def forward(self, a, c, m):
            q = self.cur(a)
            if not use_context:
                return self.out(q).squeeze(-1)
            kv = self.ctx(c.reshape(c.shape[0], -1, c.shape[-1]))
            keymask = (~m).unsqueeze(-1).expand(-1, -1, c.shape[2]).reshape(c.shape[0], -1)
            # MultiheadAttention returns NaN when every key is masked.  A
            # one-frame unit has no valid neighbor, so expose its zero token
            # as a neutral context instead.
            all_masked = keymask.all(dim=1)
            if all_masked.any():
                keymask = keymask.clone(); keymask[all_masked, 0] = False
            z, _ = self.attn(q, kv, kv, key_padding_mask=keymask)
            return self.out(q + z).squeeze(-1)
    net = Head().to(device).float(); opt = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-3)
    xt, yt = torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
    ct, mt = torch.from_numpy(ctx.astype(np.float32)), torch.from_numpy(cmask.astype(bool))
    ids = np.flatnonzero(train_mask); best = (-1e9, None)
    def predict(xa, ca, ma):
        chunks=[]
        with torch.inference_mode():
            for st in range(0, len(xa), 256):
                chunks.append(net(xa[st:st+256].to(device), ca[st:st+256].to(device), ma[st:st+256].to(device)).cpu().numpy())
        return np.concatenate(chunks)
    for step in range(1, steps + 1):
        take = rng.choice(ids, min(64, len(ids)), replace=False)
        p = net(xt[take].to(device), ct[take].to(device), mt[take].to(device))
        loss = ((p - yt[take].to(device)) ** 2).mean(); opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 100 == 0 or step == steps:
            net.eval(); pp = predict(xt, ct, mt)
            score = source_macro(pp, y, src, ~train_mask)
            print(json.dumps({"step": step, "loss": float(loss.detach().cpu()), "dev_source_macro": score}), flush=True)
            if score > best[0]: best = (score, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()})
            net.train()
    net.load_state_dict(best[1]); net.eval()
    pp = predict(xt, ct, mt)
    if permute_eval and use_context:
        perm = np.random.default_rng(seed + 99).permutation(len(ct)); pp = predict(xt, ct[perm], mt[perm])
    return {"dev_source_macro": source_macro(pp, y, src, ~train_mask),
            "train_source_macro": source_macro(pp, y, src, train_mask),
            "best_dev": best[0], "params": sum(p.numel() for p in net.parameters())}, net


def train(args):
    import torch
    z = load_rows(args.out); x, y, ctx, cmask = z["x"].astype(np.float32), z["y"].astype(np.float32), z["context"], z["cmask"]
    src, names = z["source"].astype(str), z["name"].astype(str); train_mask = np.char.endswith(names, "train")
    center_vals = []
    for s in sorted(set(src[~train_mask])):
        q = (~train_mask) & (src == s)
        center_vals.append(float(y[q, y.shape[1] // 2].mean()))
    results = {"center_dev_source_macro": float(np.mean(center_vals)) if center_vals else float("nan")}
    for label, use, perm in (("control", False, False), ("token_context", True, False), ("token_context_permuted", True, True)):
        met, net = train_model(x, y, ctx, cmask, src, names, train_mask, args.device, args.steps, args.seed, use, perm)
        results[label] = met
        if label != "token_context_permuted": torch.save(net.state_dict(), Path(args.out) / f"{label}.pt")
    results.update({"rows": int(len(y)), "dev_sources": int(len(set(src[~train_mask]))), "official_test_labels_used": False, "audio": "excluded"})
    (Path(args.out) / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results, indent=2))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--mode", choices=["extract", "train"], required=True)
    ap.add_argument("--r70-script", default="/root/autodl-tmp/r70_spatial_utility_pilot.py"); ap.add_argument("--t5-dir", required=True); ap.add_argument("--rv-frames", required=True)
    ap.add_argument("--vision-dir", required=True); ap.add_argument("--processor-dir", required=True); ap.add_argument("--out", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--shard-rows", type=int, default=128); ap.add_argument("--steps", type=int, default=1800); ap.add_argument("--seed", type=int, default=20261019)
    ap.add_argument("--names", nargs="*", default=["rv_train", "live_train", "rv_dev", "live_dev"]); a = ap.parse_args()
    extract(a) if a.mode == "extract" else train(a)


if __name__ == "__main__":
    main()
