#!/usr/bin/env python3
"""Persistent, batched DINO spatial cache and controlled R70 readout study.

Public T5 native-crop tables only. Features and heads stay remote. The exact
same R70 head, train-only normalization, objective and source split are used
for both encoders; early checkpoint choice uses a source holdout carved only
from the historical TRAIN pool. Historical development sources are diagnostic,
not fresh confirmation. Pooling runs on CUDA, preprocessing uses bounded CPU
workers. Full arrays are float16 on disk and on GPU, without float32 host copies.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from dinov3_runtime import load_model, _resize_letterbox, sha256_file
from dinov3_spatial_patch_probe import table_rows, candidate_offsets, candidate_boxes, iou, grid_masks
from r70_spatial_utility_pilot import Head


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def prepare_image(path):
    with Image.open(path) as im:
        im, transform = _resize_letterbox(im.convert("RGB"), 224)
        a = np.asarray(im, np.float32) / np.float32(255.)
    a = (a - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
    return np.ascontiguousarray(a.transpose(2, 0, 1)), transform


def describe_rows(args):
    return table_rows(Path(args.t5_root), ["rv_train", "live_train", "rv_dev", "live_dev"], Path(args.rv_frames))


def pooling(grid, masks, valid, positions=None):
    inside = masks @ grid / masks.sum(-1, keepdim=True).clamp_min(1.)
    outside_mask = (valid[:, None, :] - masks).clamp_min(0.)
    outside = outside_mask @ grid / outside_mask.sum(-1, keepdim=True).clamp_min(1.)
    pos = (torch.linspace(0, 1, 129, device=grid.device).expand(len(grid), -1)
           if positions is None else positions)[..., None]
    return torch.cat([inside, outside, inside - outside, pos], -1)


def extract(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows, images = describe_rows(args)
    keys = list(images); key_index = {key: i for i, key in enumerate(keys)}
    rows_file = out / "rows.jsonl"
    rows_file.write_text("".join(json.dumps({**r, "frame_index": key_index[r["key"]]}, sort_keys=True) + "\n" for r in rows))
    patch_path = out / "patch.npy"
    if not (out / "patch_complete.json").exists():
        model, meta = load_model(args.dinov3_repo, args.weights, args.device, "float16")
        patches = np.lib.format.open_memmap(patch_path, mode="w+", dtype=np.float16, shape=(len(keys), 196, 768))
        cls = np.lib.format.open_memmap(out / "cls.npy", mode="w+", dtype=np.float16, shape=(len(keys), 768))
        transforms = []
        t0 = time.perf_counter(); decode_s = 0.; forward_s = 0.
        with ThreadPoolExecutor(max_workers=args.workers) as pool, torch.inference_mode():
            def prep(st):
                return list(pool.map(prepare_image, [images[k] for k in keys[st:st + args.batch]]))
            # One prefetched batch, bounded to at most 2*batch decoded images.
            with ThreadPoolExecutor(max_workers=1) as prefetch:
                future = prefetch.submit(prep, 0)
                for st in range(0, len(keys), args.batch):
                    t = time.perf_counter(); batch = future.result(); decode_s += time.perf_counter() - t
                    if st + args.batch < len(keys): future = prefetch.submit(prep, st + args.batch)
                    x = torch.from_numpy(np.stack([q[0] for q in batch])).to(args.device, dtype=torch.float16)
                    torch.cuda.synchronize(); t = time.perf_counter()
                    encoded = model.forward_features(x)
                    p = encoded["x_norm_patchtokens"].cpu().numpy()
                    c = encoded["x_norm_clstoken"].cpu().numpy()
                    forward_s += time.perf_counter() - t
                    patches[st:st + len(batch)] = p; cls[st:st + len(batch)] = c
                    transforms.extend(q[1] for q in batch)
                    if st % (10 * args.batch) == 0 or st + len(batch) == len(keys):
                        print(json.dumps({"phase": "patch_extract", "frames": st + len(batch), "total": len(keys), "elapsed_s": time.perf_counter() - t0}), flush=True)
        patches.flush(); cls.flush()
        write_json(out / "transforms.json", transforms)
        write_json(out / "patch_complete.json", {"frames": len(keys), "keys_sha256": hashlib.sha256("\n".join(keys).encode()).hexdigest(), "model": meta,
                                                "elapsed_s": time.perf_counter() - t0, "wait_for_preprocessing_s": decode_s, "forward_and_transfer_s": forward_s,
                                                "batch": args.batch, "workers": args.workers})
        del model; torch.cuda.empty_cache()
    patches = np.load(patch_path, mmap_mode="r")
    transforms = json.loads((out / "transforms.json").read_text())
    x = np.lib.format.open_memmap(out / "x.npy", mode="w+", dtype=np.float16, shape=(len(rows), 129, 2305))
    y = np.zeros((len(rows), 129), np.float32)
    s0 = time.perf_counter(); distinct = []
    with torch.inference_mode():
        for st in range(0, len(rows), args.batch):
            batch = rows[st:st + args.batch]; masks = []; valid = []; grids = []; positions = []
            for j, r in enumerate(batch):
                g = patches[key_index[r["key"]]].reshape(14, 14, 768)
                mm, vv, offsets = grid_masks(g, transforms[key_index[r["key"]]], r["W"], r["H"], r["ratio"])
                # use identical center-mask contract as the first probe
                masks.append(mm); valid.append(vv); grids.append(g.reshape(196, 768))
                positions.append(offsets / (offsets[-1] if offsets[-1] > 0 else 1.))
                distinct.append(len(np.unique(mm, axis=0)))
                y[st+j] = iou(candidate_boxes(offsets, r["W"], r["H"], r["ratio"]), r["gt"]).mean(1)
            grid = torch.from_numpy(np.stack(grids)).to(args.device).float()
            mm = torch.from_numpy(np.stack(masks)).to(args.device)
            vv = torch.from_numpy(np.stack(valid)).to(args.device)
            feature = pooling(grid, mm, vv, torch.from_numpy(np.stack(positions)).to(args.device))
            if not torch.isfinite(feature).all(): raise ValueError("non-finite pooled feature")
            x[st:st+len(batch)] = feature.cpu().numpy().astype(np.float16)
            if st % (20 * args.batch) == 0 or st + len(batch) == len(rows):
                print(json.dumps({"phase": "gpu_pool", "rows": st + len(batch), "elapsed_s": time.perf_counter() - s0}), flush=True)
    x.flush(); np.save(out / "y.npy", y)
    source = np.array([r["source"] for r in rows]); names = np.array([r["name"] for r in rows])
    np.save(out / "source.npy", source); np.save(out / "name.npy", names)
    write_json(out / "complete.json", {"rows": len(rows), "frames": len(keys), "train_rows": int(np.char.endswith(names, "train").sum()),
                                      "dev_rows": int(np.char.endswith(names, "dev").sum()), "pool_s": time.perf_counter() - s0,
                                      "distinct_masks_mean": float(np.mean(distinct)), "distinct_masks_min": int(min(distinct)), "distinct_masks_max": int(max(distinct)),
                                      "rows_sha256": sha256_file(rows_file), "pooling": "center-in-patch; padding excluded; GPU fp32 accumulation; fp16 persistent features"})


def load_v9(args, reference):
    out = Path(args.out); target = out / "v9_x.npy"
    if (out / "v9_complete.json").exists(): return target
    expected_y = np.load(reference / "y.npy"); expected_source = np.load(reference / "source.npy"); expected_name = np.load(reference / "name.npy")
    x = np.lib.format.open_memmap(target, mode="w+", dtype=np.float16, shape=(len(expected_y),129,2305))
    cursor = 0; evidence = []
    for p in sorted((Path(args.r70_cache) / "shards").glob("*.npz")):
        with np.load(p, allow_pickle=True) as z:
            yy = z["y"]; n = len(yy)
            assert np.array_equal(expected_source[cursor:cursor+n], z["source"].astype(str))
            assert np.array_equal(expected_name[cursor:cursor+n], z["name"].astype(str))
            np.testing.assert_allclose(expected_y[cursor:cursor+n], yy, atol=1e-6)
            x[cursor:cursor+n] = z["x"]
        cursor += n; evidence.append({"name": p.name, "bytes": p.stat().st_size})
    assert cursor == len(expected_y), (cursor, len(expected_y))
    x.flush(); write_json(out / "v9_complete.json", {"rows": cursor, "GT_source_split_alignment": "all rows passed", "shards": evidence})
    return target


def source_means(values, sources, mask):
    return {s: float(np.mean(values[mask & (sources == s)])) for s in sorted(set(sources[mask]))}


def summarize(pred, y, sources, names, mask, seed):
    chosen = pred.argmax(1); values = y[np.arange(len(y)), chosen]
    out = {}
    for domain in ("rv", "live", "all"):
        selected = mask & (np.char.startswith(sources, domain + ":") if domain != "all" else True)
        a = source_means(values, sources, selected); b = source_means(y[:,64], sources, selected)
        o = source_means(y.max(1), sources, selected)
        delta = np.array([a[s]-b[s] for s in a]); rng = np.random.default_rng(seed)
        boot = delta[rng.integers(0,len(delta), size=(4000,len(delta)))].mean(1)
        out[domain] = {"rows": int(selected.sum()), "sources": len(a), "iou": float(np.mean(list(a.values()))),
                       "center": float(np.mean(list(b.values()))), "oracle": float(np.mean(list(o.values()))),
                       "delta": float(delta.mean()), "ci95": np.quantile(boot,[.025,.975]).tolist()}
    return out, values


def train(args):
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True); ref = Path(args.cache)
    y = np.load(ref / "y.npy"); src = np.load(ref / "source.npy"); names = np.load(ref / "name.npy")
    original_train = np.char.endswith(names, "train"); dev = np.char.endswith(names, "dev")
    assert not set(src[dev]) & set(src[original_train])
    tune_src = set()
    for dom in ("rv", "live"):
        ss = sorted(set(src[original_train & np.char.startswith(src, dom + ":")]))
        ss.sort(key=lambda s: hashlib.sha1((s + ":dino_matched_v2").encode()).hexdigest())
        tune_src.update(ss[:max(2,round(.1*len(ss)))])
    tune = original_train & np.isin(src,list(tune_src)); fit = original_train & ~tune
    write_json(out / "split.json", {"fit_sources": sorted(set(src[fit])), "tune_sources": sorted(tune_src), "dev_sources": sorted(set(src[dev])),
                                   "note": "checkpoint chosen by TRAIN-held-out source tune; historical dev used only at final evaluation"})
    feature_path = load_v9(args, ref) if args.arm == "v9" else ref / "x.npy"
    raw = np.load(feature_path, mmap_mode="r")
    xt = torch.empty(raw.shape, dtype=torch.float16, device=args.device)
    for st in range(0,len(raw),128):
        xt[st:st+128].copy_(torch.from_numpy(np.array(raw[st:st+128], copy=True)).to(args.device))
    fit_ids = np.flatnonzero(fit); tune_ids = np.flatnonzero(tune); dev_ids = np.flatnonzero(dev)
    # Streaming train-only moments avoid 15-30 GB float32 host temporaries.
    sm = torch.zeros(raw.shape[-1],device=args.device); sq = sm.clone(); count = 0
    for st in range(0,len(fit_ids),64):
        z = xt[fit_ids[st:st+64]].float(); sm += z.sum((0,1)); sq += (z*z).sum((0,1)); count += z.shape[0]*z.shape[1]
    mu = sm/count; sd = (sq/count-mu.square()).clamp_min(0).sqrt(); sd = torch.where(sd<1e-5,torch.ones_like(sd),sd)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    model = Head.make(torch).to(args.device).float()
    opt = torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    yt = torch.from_numpy(y).to(args.device); rng = np.random.default_rng(args.seed)
    best_score = -np.inf; best = None; history = []; started = time.perf_counter()
    def predict(ids):
        model.eval(); parts=[]
        with torch.inference_mode():
            for st in range(0,len(ids),128): parts.append(model((xt[ids[st:st+128]].float()-mu)/sd).cpu().numpy())
        model.train(); return np.concatenate(parts)
    def tune_score(p):
        vals = y[tune_ids,np.argmax(p,1)]; ss=src[tune_ids]
        # Equal domains, then equal sources; independent of video length.
        result=[]
        for dom in ("rv", "live"):
            result.append(np.mean([vals[ss==s].mean() for s in sorted(set(ss[np.char.startswith(ss,dom+":")]))]))
        return float(np.mean(result))
    for step in range(1,args.steps+1):
        ids = rng.choice(fit_ids, min(args.train_batch,len(fit_ids)), replace=False)
        model.train(); logits = model((xt[ids].float()-mu)/sd)
        loss = ((logits-yt[ids])**2).mean()
        if not torch.isfinite(loss): raise ValueError("nonfinite loss")
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        if step % 200 == 0 or step == args.steps:
            score = tune_score(predict(tune_ids)); hist = {"step": step,"loss": float(loss.detach()),"tune_dataset_equal_iou": score,"elapsed_s":time.perf_counter()-started}
            history.append(hist); print(json.dumps({"arm":args.arm, **hist}),flush=True)
            if score > best_score:
                best_score=score; best={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_step=step
    model.load_state_dict(best)
    pred = np.zeros_like(y); pred[dev_ids]=predict(dev_ids)
    metrics, selected = summarize(pred,y,src,names,dev,args.seed)
    parameter_count=sum(p.numel() for p in model.parameters())
    report={"arm":args.arm,"seed":args.seed,"params":parameter_count,"steps":args.steps,"batch":args.train_batch,"selected_step":best_step,
            "head":"r70_spatial_utility_pilot.Head.make","normalization":"fit-only mean/std","loss":"candidate MSE", "metrics":metrics,
            "train_s":time.perf_counter()-started,"history":history,"peak_cuda_bytes":torch.cuda.max_memory_allocated(),"source_rows_sha256":sha256_file(ref/"rows.jsonl")}
    torch.save({"state_dict":best,"mu":mu.cpu(),"sd":sd.cpu(),"report":report},out/f"{args.arm}_{args.seed}_head.pt")
    np.savez(out/f"{args.arm}_{args.seed}_dev.npz",logits=pred[dev],y=y[dev],source=src[dev],name=names[dev])
    write_json(out/f"{args.arm}_{args.seed}_metrics.json",report); print(json.dumps(report,ensure_ascii=False),flush=True)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("mode",choices=["extract","train"])
    ap.add_argument("--t5-root",default="/data/aic/experiments/T5_CROPHEAD_V3"); ap.add_argument("--rv-frames",default="/root/autodl-tmp/evac/data/aic/experiments/QWEN_RV200_KEYFRAMES")
    ap.add_argument("--dinov3-repo",default="/data/aic/pretrained/dinov3/dinov3-main"); ap.add_argument("--weights",default="/data/aic/pretrained/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth")
    ap.add_argument("--device",default="cuda"); ap.add_argument("--out",required=True); ap.add_argument("--cache")
    ap.add_argument("--r70-cache",default="/root/autodl-tmp/experiments/R70_SPATIAL_UTILITY/full")
    ap.add_argument("--arm",choices=["dino","v9"],default="dino"); ap.add_argument("--seed",type=int,default=20261020)
    ap.add_argument("--batch",type=int,default=64); ap.add_argument("--workers",type=int,default=4); ap.add_argument("--train-batch",type=int,default=32); ap.add_argument("--steps",type=int,default=1200)
    args=ap.parse_args(); torch.set_num_threads(4)
    if not torch.cuda.is_available() or not args.device.startswith("cuda"): raise RuntimeError("This experiment requires GPU")
    (extract if args.mode=="extract" else train)(args)


if __name__=="__main__": main()
