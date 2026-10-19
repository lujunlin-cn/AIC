#!/usr/bin/env python3
"""Paired public-data test of real crop re-encoding versus old-grid ROI reading.

This program only consumes a public T5 manifest made by
r85_build_public_manifest.py. It has no official-index, test-label or upload
argument. Extraction uses the existing frozen R70 vision/processor loader;
training compares B/C with matched capacity, seeds, targets and optimizer.
An identity-verified frozen R70 head supplies competitive control A.

Real-data/GPU execution is required before any performance or package claim.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aic.crop_observation import (
    candidate_boxes, crop_extent, load_manifest_image, mean_iou,
    pool_grid_2x2, validate_manifest_rows,
)
from scripts.r70_spatial_utility_pilot import Head, feature_from_grid, load_vision


ARRAYS = {"old": 2305, "roi": 3072, "crop": 3072}
ARRAY_DTYPES = {"old": np.dtype("float16"), "roi": np.dtype("float16"), "crop": np.dtype("float16"),
                "target": np.dtype("float32"), "done": np.dtype("bool"), "row_hashes": np.dtype("S64")}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temp.replace(path)


def read_config(path):
    cfg = json.loads(Path(path).read_text())
    if cfg["experiment_id"] != "R85_REAL_CROP_OBSERVATION":
        raise ValueError("wrong experiment config")
    if cfg["candidate_count"] != 129 or cfg["old_feature_dim"] != 2305 or cfg["local_feature_dim"] != 3072:
        raise ValueError("unsupported frozen feature contract")
    return cfg


def read_manifest(path, verify_images=True):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    validate_manifest_rows(rows, verify_images=verify_images)
    return rows


def canonical_rows(rows):
    return "".join(json.dumps(r, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for r in rows)


def row_feature_hash(arrays, targets, i):
    h = hashlib.sha256()
    for key in ARRAYS:
        h.update(np.ascontiguousarray(arrays[key][i]).tobytes())
    h.update(np.ascontiguousarray(targets[i]).tobytes())
    return h.hexdigest().encode("ascii")


def model_state_hash(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8"))
        h.update(str(tuple(tensor.shape)).encode("ascii"))
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def validate_resume_cache(output, rows, meta):
    """Read-only validation; never bless reordered or modified cached rows."""
    output = Path(output)
    expected_rows = canonical_rows(rows)
    if (output / "rows.jsonl").read_text() != expected_rows:
        raise ValueError("resume ordered rows differ from the selected source manifest")
    if meta["rows_sha256"] != sha(output / "rows.jsonl"):
        raise ValueError("resume rows SHA changed")
    shapes = {k: (len(rows), 129, depth) for k, depth in ARRAYS.items()}
    shapes.update(target=(len(rows), 129), done=(len(rows),), row_hashes=(len(rows),))
    loaded = {}
    for key in shapes:
        arr = np.load(output / f"{key}.npy", mmap_mode="r", allow_pickle=False)
        if arr.shape != shapes[key] or arr.dtype != ARRAY_DTYPES[key]:
            raise ValueError(f"resume array shape/dtype changed: {key}")
        loaded[key] = arr
    if meta["status"] in {"COMPLETE", "SMOKE_COMPLETE"}:
        expected_names = {f"{k}.npy" for k in shapes}
        if set(meta.get("cache_sha256", {})) != expected_names:
            raise ValueError("completed cache inventory is incomplete")
        for name, digest in meta["cache_sha256"].items():
            if sha(output / name) != digest:
                raise ValueError(f"completed cache SHA changed: {name}")
        if not loaded["done"].all():
            raise ValueError("completed cache has incomplete rows")
    else:
        for i in np.flatnonzero(loaded["done"]):
            actual = row_feature_hash(loaded, loaded["target"], i)
            if actual != loaded["row_hashes"][i]:
                raise ValueError(f"interrupted cache completed row changed: {i}")


def encode_grids(images, torch, vision, processor, device):
    batch = processor(images=images, return_tensors="pt")
    batch = {k: v.to(device) for k, v in batch.items() if torch.is_tensor(v)}
    with torch.inference_mode():
        out = vision(pixel_values=batch["pixel_values"].half(),
                     spatial_shapes=batch["spatial_shapes"],
                     pixel_attention_mask=batch["pixel_attention_mask"], return_dict=True)
    if len(out.last_hidden_state) != len(images):
        raise ValueError("processor unexpectedly tiled or merged images")
    grids = []
    for i in range(len(images)):
        fh, fw = map(int, batch["spatial_shapes"][i])
        valid = int(batch["pixel_attention_mask"][i].sum())
        if fh * fw != valid or valid <= 0:
            raise ValueError("valid-token/spatial-shape mismatch")
        grid = out.last_hidden_state[i, :valid].float().cpu().numpy().reshape(fh, fw, -1)
        if grid.shape[-1] != 768 or not np.isfinite(grid).all():
            raise ValueError("unexpected or nonfinite V9 features")
        grids.append(grid)
    return grids


def batch_parity_probe(image, row, torch, vision, processor, device, cfg):
    boxes = candidate_boxes(row["W"], row["H"], row["ratio"])
    images = [image] + [crop_extent(image, boxes[i]) for i in (0, 64, 128)]
    batch = encode_grids(images, torch, vision, processor, device)
    errors = []
    for image, grouped in zip(images, batch):
        single = encode_grids([image], torch, vision, processor, device)[0]
        if single.shape != grouped.shape:
            raise ValueError("mixed-batch processor changed valid grid dimensions")
        difference = np.abs(single - grouped)
        if not np.allclose(single, grouped, rtol=cfg["batch_parity"]["rtol"], atol=cfg["batch_parity"]["atol"]):
            raise ValueError(f"single/batch vision parity failed; max_abs={difference.max()}")
        errors.append({"shape": list(single.shape), "max_abs": float(difference.max()),
                       "mean_abs": float(difference.mean())})
    return {"passed": True, "errors": errors, "tolerance": cfg["batch_parity"]}


def extract(args):
    import torch
    cfg = read_config(args.config)
    rows = read_manifest(args.manifest)
    is_smoke = args.limit > 0
    if is_smoke:
        rows = [rows[i] for i in np.linspace(0, len(rows) - 1, min(args.limit, len(rows)), dtype=int)]
    if not rows:
        raise ValueError("empty manifest")
    vision_path = Path(args.vision_dir) / "vision_tower.safetensors"
    processor_path = Path(args.processor_dir) / "processor_config.json"
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; no real crop extraction has run")
    identity = {
        "manifest_sha256": sha(args.manifest), "config_sha256": sha(args.config),
        "vision_sha256": sha(vision_path), "processor_sha256": sha(processor_path),
        "model_config_sha256": sha(Path(args.processor_dir) / "config.json"),
        "code_sha256": {name: sha(Path(__file__).resolve().parents[1] / name) for name in (
            "scripts/r85_crop_observation_probe.py", "aic/crop_observation.py", "scripts/r70_spatial_utility_pilot.py")},
        "runtime": {name: version(name) for name in ("torch", "transformers", "numpy", "Pillow", "safetensors")},
        "device": args.device,
        "gpu_name": torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None,
        "crop_batch": args.crop_batch,
        "rows": len(rows), "smoke_only": is_smoke,
    }
    output = Path(args.output)
    meta_path = output / "extraction.json"
    if output.exists():
        if not args.resume or not meta_path.exists():
            raise ValueError("output exists; use --resume only for a matching interrupted extraction")
        meta = json.loads(meta_path.read_text())
        if meta["identity"] != identity:
            raise ValueError("resume identity mismatch")
        validate_resume_cache(output, rows, meta)
        if meta["status"] in {"COMPLETE", "SMOKE_COMPLETE"}:
            print(json.dumps({"status": "ALREADY_COMPLETE_VERIFIED", "rows": len(rows)}), flush=True)
            return
        mode = "r+"
    else:
        output.mkdir(parents=True)
        meta = {"identity": identity, "status": "EXTRACTING", "completed_rows": 0}
        mode = "w+"
        (output / "rows.jsonl").write_text(canonical_rows(rows))
        meta["rows_sha256"] = sha(output / "rows.jsonl")
        write_json(meta_path, meta)
    buffers = {
        k: np.lib.format.open_memmap(output / f"{k}.npy", mode=mode, dtype=np.float16,
                                    shape=(len(rows), 129, d)) for k, d in ARRAYS.items()
    }
    targets = np.lib.format.open_memmap(output / "target.npy", mode=mode, dtype=np.float32,
                                       shape=(len(rows), 129))
    done = np.lib.format.open_memmap(output / "done.npy", mode=mode, dtype=np.bool_, shape=(len(rows),))
    row_hashes = np.lib.format.open_memmap(output / "row_hashes.npy", mode=mode, dtype="S64", shape=(len(rows),))
    if mode == "w+":
        done[:] = False
        row_hashes[:] = b""
        row_hashes.flush()
        done.flush()
    torch, vision, processor = load_vision(args)
    actual_vision = sum(p.numel() for p in vision.parameters())
    if actual_vision != cfg["encoder"]["expected_parameters"]:
        raise ValueError(f"vision parameter count mismatch: {actual_vision}")
    for p in vision.parameters():
        p.requires_grad_(False)
    if args.device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    meta["batch_parity"] = batch_parity_probe(load_manifest_image(rows[0]), rows[0], torch,
                                             vision, processor, args.device, cfg)
    write_json(meta_path, meta)
    started = time.monotonic()
    processed = 0
    for i, row in enumerate(rows):
        if done[i]:
            continue
        image = load_manifest_image(row)
        native_boxes = candidate_boxes(row["W"], row["H"], row["ratio"])
        grid = encode_grids([image], torch, vision, processor, args.device)[0]
        # Reproduce the old full-grid path exactly for both B and C.
        fh, fw = grid.shape[:2]
        old_grid = grid[:fh // 2 * 2, :fw // 2 * 2]
        if min(old_grid.shape[:2]) < 2:
            raise ValueError("full image has insufficient valid tokens")
        old, _ = feature_from_grid(old_grid, row["W"], row["H"], row["ratio"])
        buffers["old"][i] = old
        buffers["roi"][i] = pool_grid_2x2(old_grid, row["W"], row["H"], native_boxes)
        for start in range(0, 129, args.crop_batch):
            crop_images = [crop_extent(image, b) for b in native_boxes[start:start + args.crop_batch]]
            crop_grids = encode_grids(crop_images, torch, vision, processor, args.device)
            for j, cg in enumerate(crop_grids, start):
                ch, cw = cg.shape[:2]
                buffers["crop"][i, j] = pool_grid_2x2(cg, cw, ch, [[0, 0, cw, ch]])[0]
        targets[i] = mean_iou(native_boxes, row["gt_xyxy"])
        if not all(np.isfinite(buffers[k][i]).all() for k in buffers) or not np.isfinite(targets[i]).all():
            raise ValueError("nonfinite stored features/targets")
        # Publish the completion bit only after all data for this row is durable.
        for b in [*buffers.values(), targets]:
            b.flush()
        row_hashes[i] = row_feature_hash(buffers, targets, i)
        row_hashes.flush()
        done[i] = True
        done.flush()
        processed += 1
        if processed % 10 == 0 or done.all():
            elapsed = time.monotonic() - started
            meta.update(completed_rows=int(done.sum()), latest_elapsed_s=elapsed)
            write_json(meta_path, meta)
            print(json.dumps({"rows_done": int(done.sum()), "rows_total": len(rows),
                              "new_rows_per_second": processed / max(elapsed, 1e-9)}), flush=True)
    meta.update(status="SMOKE_COMPLETE" if is_smoke else "COMPLETE", completed_rows=int(done.sum()),
                vision_parameters=actual_vision, torch_version=torch.__version__, python=platform.python_version(),
                device=args.device, cuda_available=torch.cuda.is_available(),
                gpu_name=torch.cuda.get_device_name(torch.device(args.device)) if args.device.startswith("cuda") else None,
                cache_sha256={p.name: sha(p) for p in sorted(output.glob("*.npy"))},
                rows_sha256=sha(output / "rows.jsonl"), elapsed_s=time.monotonic() - started,
                peak_cuda_memory_allocated_bytes=(torch.cuda.max_memory_allocated(torch.device(args.device))
                                                  if args.device.startswith("cuda") else None),
                performance_claim=False, official_test_labels_used=False)
    write_json(meta_path, meta)
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


def train_moments(array, ids):
    total = np.zeros(array.shape[-1], dtype=np.float64)
    squared = np.zeros_like(total)
    count = 0
    for i in ids:
        x = np.asarray(array[i], dtype=np.float64)
        if not np.isfinite(x).all():
            raise ValueError("nonfinite training features")
        total += x.sum(0)
        squared += np.square(x).sum(0)
        count += len(x)
    mu = total / count
    sd = np.sqrt(np.maximum(squared / count - mu * mu, 0))
    sd[sd < 1e-5] = 1
    return mu.astype(np.float32), sd.astype(np.float32)


def batch_features(old, added, ids, mu, sd, local_mu=None, local_sd=None, permutation=None):
    a = (np.asarray(old[ids], dtype=np.float32) - mu) / sd
    if added is None:
        return a
    b = (np.asarray(added[ids], dtype=np.float32) - local_mu) / local_sd
    if permutation is not None:
        b = np.take_along_axis(b, permutation[:, :, None], axis=1)
    return np.concatenate([a, b], axis=-1)


def predict(torch, model, old, added, ids, normalizers, device, batch=32, permute_seed=None):
    picked = []
    model.eval()
    rng = np.random.default_rng(permute_seed) if permute_seed is not None else None
    with torch.inference_mode():
        for start in range(0, len(ids), batch):
            take = ids[start:start + batch]
            perm = np.stack([rng.permutation(129) for _ in take]) if rng is not None else None
            x = batch_features(old, added, take, *normalizers, permutation=perm)
            scores = model(torch.from_numpy(x).to(device))
            if not torch.isfinite(scores).all():
                raise ValueError("nonfinite model output")
            picked.extend(scores.argmax(1).cpu().numpy().tolist())
    return np.asarray(picked, dtype=np.int64)


def source_report(values, rows):
    result = {}
    for table in sorted(set(r["table"] for r in rows)):
        grouped = {}
        for value, row in zip(values, rows):
            if row["table"] == table:
                grouped.setdefault(row["source"], []).append(float(value))
        result[table] = {s: float(np.mean(v)) for s, v in sorted(grouped.items())}
    return result


def paired_comparison(c, other, cfg):
    result = {}
    diag = cfg["diagnostic"]
    gate = cfg["engineering_gate"]
    for table in ("rv_dev", "live_dev"):
        a, b = c.get(table, {}), other.get(table, {})
        if not a or set(a) != set(b):
            raise ValueError(f"paired source coverage missing/mismatched: {table}")
        delta = np.asarray([a[s] - b[s] for s in sorted(a)])
        rng = np.random.default_rng(diag["bootstrap_seed"])
        means = np.asarray([delta[rng.integers(len(delta), size=len(delta))].mean()
                            for _ in range(diag["bootstrap_repeats"])])
        lower, upper = np.quantile(means, [0.025, 0.975])
        positive = float(np.mean(delta > 0))
        result[table] = {"sources": len(delta), "mean_delta": float(delta.mean()),
                         "source_paired_ci95": [float(lower), float(upper)],
                         "strictly_positive_source_fraction": positive,
                         "gate_pass": bool(delta.mean() >= gate["mean_iou_delta"] and lower > 0
                                           and positive >= gate["positive_source_fraction"])}
    return result


def train(args):
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    cfg = read_config(args.config)
    root = Path(args.features)
    meta = json.loads((root / "extraction.json").read_text())
    if meta["status"] != "COMPLETE" or meta["identity"]["smoke_only"]:
        raise ValueError("formal paired training requires complete, non-smoke extraction")
    if meta["identity"]["config_sha256"] != sha(args.config):
        raise ValueError("config changed after feature extraction")
    if meta["rows_sha256"] != sha(root / "rows.jsonl"):
        raise ValueError("manifest cache changed")
    rows = read_manifest(root / "rows.jsonl", verify_images=False)
    validate_resume_cache(root, rows, meta)
    tr = np.asarray([i for i, r in enumerate(rows) if r["split"] == "train"])
    dv = np.asarray([i for i, r in enumerate(rows) if r["split"] == "dev"])
    if len(tr) == 0 or len(dv) == 0 or set(rows[i]["table"] for i in dv) != {"rv_dev", "live_dev"}:
        raise ValueError("both train and the two public DEV domains are required")
    if not np.load(root / "done.npy", mmap_mode="r").all():
        raise ValueError("incomplete feature rows")
    arrays = {k: np.load(root / f"{k}.npy", mmap_mode="r") for k in ARRAYS}
    y = np.load(root / "target.npy", mmap_mode="r")
    for key, depth in ARRAYS.items():
        if arrays[key].shape != (len(rows), 129, depth):
            raise ValueError(f"wrong feature shape for {key}")
    if y.shape != (len(rows), 129) or not np.isfinite(y).all() or (y < 0).any() or (y > 1).any():
        raise ValueError("wrong or invalid IoU target array")
    reference = cfg["baseline_reproduction"]
    if len(tr) != reference["train_rows"] or len(dv) != reference["dev_rows"]:
        raise ValueError("frozen T5 source/table row coverage was not reproduced")
    old_mu, old_sd = train_moments(arrays["old"], tr)
    # This checkpoint is an explicit, trusted existing artifact supplied by the operator.
    if sha(args.baseline_head) != args.baseline_head_sha256:
        raise ValueError("frozen A checkpoint SHA mismatch")
    checkpoint = torch.load(args.baseline_head, map_location="cpu", weights_only=False)
    control = Head.make(torch, d=2305).to(args.device)
    control.load_state_dict(checkpoint["state_dict"], strict=True)
    mu_a = np.asarray(checkpoint["mu"], dtype=np.float32)
    sd_a = np.asarray(checkpoint["sd"], dtype=np.float32)
    if mu_a.shape != (2305,) or sd_a.shape != (2305,) or not np.isfinite(mu_a).all() or not np.isfinite(sd_a).all() or (sd_a <= 0).any():
        raise ValueError("invalid frozen A normalization")
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.use_deterministic_algorithms(True)
    picked_a = predict(torch, control, arrays["old"], None, dv, (mu_a, sd_a), args.device)
    dev_rows = [rows[i] for i in dv]
    a_report = source_report(y[dv, picked_a], dev_rows)
    center_report = source_report(y[dv, 64], dev_rows)
    for table in ("rv_dev", "live_dev"):
        if len(a_report[table]) != reference["dev_sources"][table]:
            raise ValueError(f"frozen DEV source coverage changed: {table}")
        for actual, expected in ((np.mean(list(a_report[table].values())), reference["source_macro_iou"][table]),
                                 (np.mean(list(center_report[table].values())), reference["center_source_macro_iou"][table])):
            if abs(actual - expected) > reference["absolute_tolerance"]:
                raise ValueError(f"frozen A/target metrics not reproduced for {table}: actual={actual}, expected={expected}")
    del control
    reports = {"A": a_report}
    comparisons = {}
    train_cfg = cfg["paired_training"]
    timings = {}
    head_count = None
    for seed in train_cfg["seeds"]:
        for arm, key in (("B", "roi"), ("C", "crop")):
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
            rng = np.random.default_rng(seed)
            local_mu, local_sd = train_moments(arrays[key], tr)
            normalizers = (old_mu, old_sd, local_mu, local_sd)
            model = Head.make(torch, d=5377).to(args.device).float()
            head_count = sum(p.numel() for p in model.parameters())
            initial_state_sha256 = model_state_hash(model)
            batch_sequence = hashlib.sha256()
            opt = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"],
                                    weight_decay=train_cfg["weight_decay"])
            started = time.monotonic()
            model.train()
            for step in range(train_cfg["steps"]):
                take = rng.choice(tr, size=min(train_cfg["batch"], len(tr)), replace=False)
                batch_sequence.update(np.asarray(take, dtype=np.int64).tobytes())
                batch = batch_features(arrays["old"], arrays[key], take, *normalizers)
                prediction = model(torch.from_numpy(batch).to(args.device))
                target = torch.from_numpy(np.asarray(y[take], dtype=np.float32)).to(args.device)
                loss = torch.nn.functional.mse_loss(prediction, target)
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite training loss")
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                if (step + 1) % 100 == 0:
                    print(json.dumps({"arm": arm, "seed": seed, "step": step + 1,
                                      "train_loss": float(loss.detach()), "elapsed_s": time.monotonic() - started}), flush=True)
            tag = f"{arm}_{seed}"
            if arm == "C":
                paired = timings[f"B_{seed}"]
                if initial_state_sha256 != paired["initial_state_sha256"] or batch_sequence.hexdigest() != paired["batch_sequence_sha256"]:
                    raise ValueError("B/C initial weights or batch sequence diverged")
            picked = predict(torch, model, arrays["old"], arrays[key], dv, normalizers, args.device)
            values = y[dv, picked]
            reports[tag] = source_report(values, dev_rows)
            train_picked = predict(torch, model, arrays["old"], arrays[key], tr, normalizers, args.device)
            reports[tag + "_train"] = source_report(y[tr, train_picked], [rows[i] for i in tr])
            if arm == "C":
                shuffled = predict(torch, model, arrays["old"], arrays[key], dv, normalizers, args.device,
                                   permute_seed=cfg["diagnostic"]["permutation_seed"])
                reports[tag + "_permuted"] = source_report(y[dv, shuffled], dev_rows)
                comparisons[tag + "-B"] = paired_comparison(reports[tag], reports[f"B_{seed}"], cfg)
                comparisons[tag + "-A"] = paired_comparison(reports[tag], reports["A"], cfg)
                comparisons[tag + "-permuted"] = paired_comparison(reports[tag], reports[tag + "_permuted"], cfg)
            np.savez_compressed(output / f"{tag}_dev.npz", row_id=np.asarray([r["row_id"] for r in dev_rows]),
                                picked=picked, mean_iou=values, center_iou=y[dv, 64])
            path = output / f"{tag}.pt"
            torch.save({"state_dict": model.state_dict(), "mu": old_mu, "sd": old_sd,
                        "local_mu": local_mu, "local_sd": local_sd, "arm": arm, "seed": seed,
                        "config": cfg, "feature_identity": meta["identity"], "head_parameters": head_count,
                        "baseline_head_sha256": sha(args.baseline_head)}, path)
            timings[tag] = {"elapsed_s": time.monotonic() - started, "checkpoint_sha256": sha(path),
                            "initial_state_sha256": initial_state_sha256,
                            "batch_sequence_sha256": batch_sequence.hexdigest()}
            write_json(output / "partial_results.json", {"source_metrics": reports, "comparisons": comparisons,
                                                          "timings": timings, "complete": False})
            del model, opt
    gates = [v["gate_pass"] for key, doms in comparisons.items() if not key.endswith("-permuted") for v in doms.values()]
    alignment = [v["mean_delta"] > 0 for key, doms in comparisons.items() if key.endswith("-permuted") for v in doms.values()]
    inventory = cfg["parameter_inventory"]
    deployment_count = inventory["TimeLens2_reported_parameters"] + meta["vision_parameters"] + head_count
    result = {"status": "PUBLIC_PAIRED_EXPERIMENT_COMPLETE", "config_sha256": sha(args.config),
              "feature_identity": meta["identity"], "baseline_head_sha256": sha(args.baseline_head),
              "source_metrics": reports, "comparisons": comparisons, "timings": timings,
              "engineering_gate_pass": bool(gates and all(gates)), "head_parameters": head_count,
              "candidate_content_alignment_supported": bool(alignment and all(alignment)),
              "permutation_scope": "Supporting diagnostic only; shuffling also causes distribution shift",
              "provisional_deployment_parameters": deployment_count,
              "below_9b_by_reported_timelens_inventory": deployment_count <= inventory["limit"],
              "TimeLens2_weights_loaded_this_run": False, "official_test_labels_used": False,
              "public_dev_already_exposed": True, "official_score_prediction": None,
              "new_competition_package": None, "torch_version": torch.__version__,
              "device": args.device, "cuda_available": torch.cuda.is_available()}
    write_json(output / "results.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--manifest", type=Path, required=True)
    ex.add_argument("--vision-dir", required=True)
    ex.add_argument("--processor-dir", required=True)
    ex.add_argument("--output", type=Path, required=True)
    ex.add_argument("--crop-batch", type=int, default=32)
    ex.add_argument("--limit", type=int, default=0, help="Positive limit marks cache smoke-only; formal train rejects it")
    ex.add_argument("--resume", action="store_true")
    tr = sub.add_parser("train")
    tr.add_argument("--features", type=Path, required=True)
    tr.add_argument("--baseline-head", type=Path, required=True)
    tr.add_argument("--baseline-head-sha256", required=True, help="SHA from the independently verified R70 topsoft checkpoint inventory")
    tr.add_argument("--output", type=Path, required=True)
    for parser in (ex, tr):
        parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/R85_REAL_CROP_OBSERVATION.json")
        parser.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if args.command == "extract" and (args.crop_batch < 1 or args.limit < 0):
        ap.error("crop-batch must be positive and limit nonnegative")
    extract(args) if args.command == "extract" else train(args)


if __name__ == "__main__":
    main()
