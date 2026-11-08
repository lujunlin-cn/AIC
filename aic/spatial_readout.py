"""Small shared-encoder readouts with typed losses and resumable real training.

GPU features must be genuine verified P2 grids. Tests use labelled fixtures,
never evidence for real-model integration or research gains.
"""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from aic.scale_policy import atomic_json, canonical_sha, sha
from aic.spatial_research import PARAMETER_CAP, PARENT_ELEMENTS, preference_pairs, source_stat


class UtilityHead(nn.Module):
    def __init__(self, dim, kind="linear", rank=32, subject_conditioning=True):
        super().__init__()
        if kind not in {"linear", "mlp32"}:
            raise ValueError("only preregistered linear or rank-32 MLP")
        self.net = nn.Linear(dim, 1) if kind == "linear" else nn.Sequential(nn.Linear(dim, rank), nn.GELU(), nn.Linear(rank, 1))
        self.register_buffer("mu", torch.zeros(dim))
        self.register_buffer("sigma", torch.ones(dim))
        self.kind, self.dim = kind, dim
        self.subject_conditioning=subject_conditioning

    def forward(self, x):
        norm=(x-self.mu)/self.sigma
        if not self.subject_conditioning:
            norm=torch.cat((norm[...,:-3],torch.zeros_like(norm[...,-3:])),dim=-1)
        return self.net(norm).squeeze(-1)

    def extra_elements(self):
        # Conservative P2 contract counts serialized normalization buffers too.
        return sum(p.numel() for p in self.parameters()) + sum(p.numel() for p in self.buffers())


def typed_loss(pred, target, mask, label_type, objective, tau=.05, margin=.25, explicit_pairs=None):
    if label_type == "crop_iou" and objective not in {"regression", "soft_target"}:
        raise ValueError("IoU is geometric supervision, not new human preference")
    if label_type == "mos" and objective not in {"regression", "pairwise"}:
        raise ValueError("MOS must retain its own supervision semantics")
    if label_type == "subject_ranking":
        if objective!="pairwise" or explicit_pairs is None or len(explicit_pairs)!=len(pred):
            raise ValueError("subject ranking requires explicit pair IDs, never scalar/best_box conversion")
        losses=[]
        for p,valid,pairs in zip(pred,mask,explicit_pairs):
            pairs=np.asarray(pairs,dtype=np.int64)
            if not len(pairs) or pairs.ndim!=2 or pairs.shape[1]!=2 or pairs.min()<0 or pairs.max()>=len(p):
                raise ValueError("invalid subject-conditioned directed pairs")
            if not valid[torch.as_tensor(pairs,device=valid.device)].all():
                raise ValueError("subject pair points to masked candidate")
            if len(pairs)>128:pairs=pairs[np.linspace(0,len(pairs)-1,128).astype(int)]
            idx=torch.as_tensor(pairs,device=p.device)
            losses.append(F.softplus(-(p[idx[:,0]]-p[idx[:,1]])).mean())
        return torch.stack(losses).mean()
    if pred.shape != target.shape or mask.shape != target.shape or not mask.any(1).all() or tau <= 0:
        raise ValueError("invalid completion/candidate mask or temperature")
    if objective == "regression":
        return (((pred-target)**2)*mask).sum(1).div(mask.sum(1)).mean()
    if objective == "soft_target":
        logits = pred.masked_fill(~mask, -torch.inf)
        truth = (target/tau).masked_fill(~mask, -torch.inf).softmax(-1)
        log_prob = logits.log_softmax(-1).masked_fill(~mask, 0)
        return -(truth*log_prob).sum(1).mean()
    losses = []
    for p, y, valid in zip(pred, target, mask):
        pairs = preference_pairs(y.detach().cpu().numpy(), valid.cpu().numpy(), margin)
        if len(pairs):
            # Deterministic bounded pair sample. No validation-derived margin.
            if len(pairs) > 128:
                pairs = pairs[np.linspace(0, len(pairs)-1, 128).astype(int)]
            idx = torch.as_tensor(pairs, device=p.device)
            losses.append(F.softplus(-(p[idx[:, 0]]-p[idx[:, 1]] )).mean())
    if not losses:
        raise ValueError("no strict human MOS preference pairs in batch")
    return torch.stack(losses).mean()


def feature_path(root, row):
    return Path(root)/"features"/(canonical_sha({"dataset": row["dataset_source"], "source": row["source_id"],
        "image": row["image_id"], "ratio": row["target_ratio"], "subject": row.get("subject_id"),
        "image_sha256": row["image_sha256"]})+".npz")


def read_feature(root, row, identity):
    path = feature_path(root, row)
    receipt = json.loads(path.with_suffix(".json").read_text())
    if (receipt["image_sha256"] != row["image_sha256"] or receipt["feature_identity_sha256"] != identity
            or receipt["manifest_row_sha256"] != canonical_sha(row) or receipt["npz_sha256"] != sha(path)):
        raise ValueError("feature/input/encoder provenance drift")
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k].copy() for k in z.files}
    if not np.isfinite(result["x"]).all() or result["x"].shape[0] != len(result["y"]):
        raise ValueError("invalid cached real visual features")
    return result


def batch(root, rows, identity, device):
    data = [read_feature(root, row, identity) for row in rows]
    max_k = max(len(d["y"]) for d in data); dim = data[0]["x"].shape[1]
    x = np.zeros((len(data), max_k, dim), np.float32); y = np.zeros((len(data), max_k), np.float32); mask = np.zeros_like(y, bool)
    for i, d in enumerate(data):
        k = len(d["y"]); x[i, :k] = d["x"]; y[i, :k] = d["y"]; mask[i, :k] = d["mask"]
    return (torch.as_tensor(x, device=device), torch.as_tensor(y, device=device), torch.as_tensor(mask, device=device)), data


def fit_train_normalization(model, root, rows, identity):
    # Streaming equal-row moments; training sampler additionally balances video
    # sources. Never read Dev/Confirm for feature statistics or label thresholds.
    mu = np.zeros(model.dim, np.float64); square = np.zeros_like(mu)
    for row in rows:
        if row["split"] != "train":
            raise ValueError("normalization may use Train only")
        data = read_feature(root, row, identity); x = data["x"][data["mask"]].astype(np.float64)
        mu += x.mean(0); square += (x*x).mean(0)
    mu /= len(rows); square /= len(rows)
    sd = np.sqrt(np.maximum(0, square-mu*mu)); sd[sd < 1e-5] = 1
    with torch.no_grad():
        model.mu.copy_(torch.tensor(mu, dtype=torch.float32, device=model.mu.device))
        model.sigma.copy_(torch.tensor(sd, dtype=torch.float32, device=model.sigma.device))


def rank_correlation(a, b):
    from scipy.stats import spearmanr
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return None
    value = float(spearmanr(a, b).statistic)
    return value if np.isfinite(value) else None


def evaluate_head(model, rows, root, identity, device, batch_size=16, bootstrap=2000):
    model.eval(); groups = defaultdict(list); predictions = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            chosen_rows = rows[start:start+batch_size]; (x, y, mask), data = batch(root, chosen_rows, identity, device)
            score = model(x).masked_fill(~mask, -torch.inf).cpu().numpy()
            for r, d, p in zip(chosen_rows, data, score):
                p = p[:len(d["y"])]; best = int(p.argmax()); quality = d["y"]; valid = d["mask"]
                record = {"row": r, "candidate": best, "utility": p[valid].tolist(), "quality": float(quality[best]),
                          "bbox_xywh": d["boxes"][best].tolist(), "oracle_quality": float(quality[valid].max())}
                ranked=np.argsort(-p,kind="stable");ranked=ranked[valid[ranked]]
                record["oracle_at_k_model_ranked"]={str(k):float(quality[ranked[:k]].max()) for k in (1,2,4,8)}
                if r["label_type"] == "mos":
                    pairs = preference_pairs(quality, valid)
                    record["pairwise_accuracy"] = float((p[pairs[:, 0]] > p[pairs[:, 1]]).mean()) if len(pairs) else None
                    record["SRCC"] = rank_correlation(p[valid], quality[valid])
                elif r["label_type"]=="subject_ranking":
                    pairs=d["pairs"]
                    record["pairwise_accuracy"]=float((p[pairs[:,0]]>p[pairs[:,1]]).mean())
                    record["top1_human_best_subset"]=float(quality[best]==2)
                predictions.append(record)
                groups[(r["dataset_source"], r["route"], r["split"], r["label_type"])].append(record)
    report = {}
    for key, records in groups.items():
        src = [(r["row"]["dataset_source"], r["row"]["source_id"]) for r in records]
        out = {"label_type": key[3], "split": key[2], "media_kind": records[0]["row"]["media_kind"],
               "evaluation_geometry": "pointwise_integer_windows" if key[3] == "crop_iou" else "human_annotated_native_crops",
               "top1_quality": source_stat([r["quality"] for r in records], src, bootstrap),
               "GT_visible_oracle": source_stat([r["oracle_quality"] for r in records], src, bootstrap),
               "oracle_regret": source_stat([r["oracle_quality"]-r["quality"] for r in records], src, bootstrap)}
        out["GT_visible_Oracle_at_K_model_ranked"]={str(k):source_stat([r["oracle_at_k_model_ranked"][str(k)] for r in records],src,bootstrap) for k in (1,2,4,8)}
        out["candidate_top1_counts"]=dict(__import__("collections").Counter(r["candidate"] for r in records))
        for name in ("SRCC", "pairwise_accuracy"):
            kept = [(r, s) for r, s in zip(records, src) if r.get(name) is not None]
            if kept:
                out[name] = source_stat([r[name] for r, _ in kept], [s for _, s in kept], bootstrap)
        if key[3]=="subject_ranking":
            # Ordinal selection levels are NOT MOS/IoU/cardinal utility.
            for name in ("top1_quality","GT_visible_oracle","oracle_regret","GT_visible_Oracle_at_K_model_ranked"):
                out.pop(name,None)
            out["top1_human_best_subset_rate"]=source_stat([r["top1_human_best_subset"] for r in records],src,bootstrap)
            out["subject_macro_preference_accuracy"]=float(np.mean([r["pairwise_accuracy"] for r in records]))
            out["bootstrap_unit"]="image; all subject annotations travel together"
        report["/".join(key)] = out
    return report, predictions


def dev_score(report, label_type):
    values = []
    for r in report.values():
        metric = r.get("pairwise_accuracy") if label_type in {"mos","subject_ranking"} else r.get("top1_quality")
        if metric and metric["mean"] is not None:
            values.append(metric["mean"])
    if not values:
        raise ValueError("no valid independent Dev metric")
    # Dataset/route macro, never weight by number of frames or pairs.
    return float(np.mean(values))


def save_checkpoint(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp"); torch.save(payload, temporary); temporary.replace(path)


def train_readout(rows, root, identity, spec, device="cpu", resume=False):
    """An executable training loop; no Base/Position weights are loaded/changed."""
    train = [r for r in rows if r["split"] == "train" and r["label_type"] == spec["label_type"]]
    dev = [r for r in rows if r["split"] == "dev" and r["label_type"] == spec["label_type"]]
    if not train or not dev:
        raise ValueError("real Train and source-heldout Dev features required")
    out = Path(root)/"runs"/spec["name"]; out.mkdir(parents=True, exist_ok=True)
    ckpt = out/"last.pt"
    contract = canonical_sha({"spec": spec, "features": identity, "rows": rows})
    random.seed(spec["seed"]); np.random.seed(spec["seed"]); torch.manual_seed(spec["seed"])
    rng = np.random.default_rng(spec["seed"])
    dim = read_feature(root, train[0], identity)["x"].shape[-1]
    model = UtilityHead(dim, spec["kind"],subject_conditioning=spec.get("subject_conditioning",True)).to(device)
    total = PARENT_ELEMENTS+model.extra_elements()
    if total > PARAMETER_CAP:
        raise ValueError("P2 + readout + normalization buffers exceeds 9B")
    optimizer = torch.optim.AdamW(model.parameters(), lr=spec["learning_rate"], weight_decay=spec["weight_decay"])
    start, best, history = 0, -float("inf"), []
    if ckpt.exists():
        if not resume:
            raise FileExistsError("readout exists; explicit --resume required")
        state = torch.load(ckpt, map_location=device, weights_only=False)
        if state["contract"] != contract:
            raise ValueError("checkpoint input/config identity differs")
        model.load_state_dict(state["state"]); optimizer.load_state_dict(state["optimizer"])
        start, best, history = state["step"], state["best"], state["history"]
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"].cpu())
        if device.startswith("cuda"):
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    else:
        fit_train_normalization(model, root, train, identity)
    by_dataset = defaultdict(lambda: defaultdict(list))
    for r in train:
        by_dataset[r["dataset_source"]][r["source_id"]].append(r)
    datasets = sorted(by_dataset); sources = {d: sorted(by_dataset[d]) for d in datasets}
    started = time.monotonic()
    for step in range(start, spec["steps"]):
        chosen = []
        for _ in range(spec["batch_size"]):
            d = datasets[int(rng.integers(len(datasets)))]; s = sources[d][int(rng.integers(len(sources[d])))]
            group = by_dataset[d][s]; chosen.append(group[int(rng.integers(len(group)))])
        (x, y, mask), batch_data = batch(root, chosen, identity, device)
        model.train(); prediction = model(x)
        pairs=[d["pairs"] for d in batch_data] if spec["label_type"]=="subject_ranking" else None
        loss = typed_loss(prediction, y, mask, spec["label_type"], spec["objective"], spec["temperature"], spec["MOS_tie_margin"],pairs)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite real readout training loss")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        if not torch.isfinite(norm):
            raise ValueError("nonfinite readout gradients")
        optimizer.step()
        if (step+1) % spec["eval_every"] == 0 or step+1 == spec["steps"]:
            report, _ = evaluate_head(model, dev, root, identity, device, spec["batch_size"], 200)
            score = dev_score(report, spec["label_type"])
            history.append({"step": step+1, "train_loss": float(loss.detach()), "dev_dataset_macro": score, "gradient_norm": float(norm)})
            improved = score > best
            if improved:
                best = score
            state = {"state": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step+1, "best": best,
                     "contract": contract, "spec": spec, "dim": dim, "feature_identity": identity, "history": history,
                     "numpy_rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
                     "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else [],
                     "total_inference_elements": total, "base_optimized": False, "reference": "frozen_P2_not_missing_B3"}
            if improved:
                save_checkpoint(out/"best.pt", state)
            save_checkpoint(ckpt, state)
            print(json.dumps({"run": spec["name"], **history[-1]}), flush=True)
    # Best checkpoint is locked before Confirm. No checkpoint selection using Confirm.
    best_path = out/"best.pt"; best_state = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(best_state["state"]); model.eval()
    atomic_json(out/"training_receipt.json", {"contract": contract, "feature_identity_sha256": identity,
        "best_checkpoint_sha256": sha(best_path), "selected_on": "Dev_source_macro_only",
        "steps": spec["steps"], "extra_elements": model.extra_elements(), "total_inference_elements": total,
        "duration_seconds_this_attempt": time.monotonic()-started, "history": history,
        "real_visual_feature_training": not spec.get("unit_fixture_only",False)})
    return best_path
