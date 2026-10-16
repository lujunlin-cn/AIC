"""TRAIN-only probability calibration and observation diagnostics on bound public axes.

No official inputs, no protected sources, no GPU, no checkpoint selection.
The returned GT-assisted diagnostics are not deployable decisions or input bounds.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def fscore(mask, order, k):
    m = mask.sum(1)
    tp = mask[:, order[:k]].sum(1)
    denom = k + m
    return float(np.divide(2 * tp, denom, out=np.zeros(len(m), float), where=denom > 0).mean())


def prefix_curve(mask, order):
    m = mask.sum(1)
    tp = np.cumsum(mask[:, order], axis=1)
    ks = np.arange(1, mask.shape[1] + 1)
    curve = (2 * tp / (ks[None] + m[:, None])).mean(0)
    return np.r_[0., curve]


def fixed_order(values):
    return np.argsort(-values, kind="stable")


def cell_order(row, cell_values):
    blocks = []
    for j in fixed_order(cell_values):
        cell = np.flatnonzero(row["owner"] == j)
        blocks.append(cell[np.lexsort((cell, abs(row["pts"][cell] - row["pts"][row["ix"][j]])))])
    return np.concatenate(blocks)


def plugin(values, order=None):
    order = fixed_order(values) if order is None else order
    n = len(values)
    pred_m = float(values.sum())
    curve = np.r_[0., 2 * np.cumsum(values[order]) / (np.arange(1, n + 1) + pred_m)]
    return order, int(np.argmax(curve)), pred_m


def basis(row, include_position, include_heads):
    parts = [np.ones((len(row["pts"]), 1))]
    if include_position:
        t = 2 * (row["pts"] - row["pts"][0]) / (row["pts"][-1] - row["pts"][0]) - 1
        parts += [t[:, None] ** j for j in (1, 2, 3)]
    if include_heads:
        parts += [row["heads"][row["owner"]]]
    return np.concatenate(parts, axis=1)


def fit(train, position, heads, ridge):
    mats = [basis(r, position, heads) for r in train]
    x = np.concatenate(mats)
    y = np.concatenate([r["mask"].mean(0) for r in train])
    weights = np.concatenate([np.full(len(r["pts"]), 1 / (len(train) * len(r["pts"]))) for r in train])
    mu = (x * weights[:, None]).sum(0)
    sd = np.sqrt(((x - mu) ** 2 * weights[:, None]).sum(0)).clip(1e-8)
    mu[0], sd[0] = 0, 1
    x = (x - mu) / sd
    regularize = np.r_[0., np.ones(x.shape[1] - 1)]

    def objective(beta):
        logits = x @ beta
        loss = (weights * (np.logaddexp(0, logits) - y * logits)).sum()
        loss += ridge * (regularize * beta ** 2).sum() / 2
        grad = x.T @ (weights * (expit(logits) - y)) + ridge * regularize * beta
        return loss, grad

    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 300, "ftol": 1e-12, "gtol": 1e-8})
    if not result.success:
        raise RuntimeError(result.message)
    return dict(position=position, heads=heads, beta=result.x.tolist(), mu=mu.tolist(),
                sd=sd.tolist(), objective=float(result.fun), iterations=int(result.nit))


def predict(row, model):
    x = basis(row, model["position"], model["heads"])
    return expit(((x - model["mu"]) / model["sd"]) @ model["beta"])


def observation(row, samples):
    pts, mask = row["pts"], row["mask"]
    targets = np.linspace(pts[0], pts[-1], samples)
    ix = np.abs(pts[:, None] - targets).argmin(0)
    hit = np.zeros(len(pts), bool)
    hit[ix] = True
    missed_runs, all_runs, missed_mass = [], [], []
    for ann in mask:
        starts = np.flatnonzero(ann & ~np.r_[False, ann[:-1]])
        ends = np.flatnonzero(ann & ~np.r_[ann[1:], False]) + 1
        missed = [(a, b) for a, b in zip(starts, ends) if not hit[a:b].any()]
        all_runs.append(len(starts))
        missed_runs.append(len(missed))
        missed_mass.append(sum(b - a for a, b in missed) / max(int(ann.sum()), 1))
    return dict(requested=samples, unique_frames=len(np.unique(ix)),
                max_sampling_gap_s=float(np.diff(pts[ix]).max()),
                mean_rater_runs=float(np.mean(all_runs)),
                mean_rater_unobserved_runs=float(np.mean(missed_runs)),
                mean_rater_positive_mass_in_unobserved_runs=float(np.mean(missed_mass)))


def bootstrap(delta, seed, reps):
    rng = np.random.default_rng(seed)
    values = rng.choice(delta, (reps, len(delta)), replace=True).mean(1)
    return dict(mean=float(np.mean(delta)), ci95=np.quantile(values, [.025, .975]).tolist(), sources=len(delta))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    config_path = Path(args.config)
    c = json.loads(config_path.read_text())
    out = Path(c["output"])
    if out.exists():
        raise FileExistsError(f"Preserve prior results: {out}")
    out.mkdir(parents=True)
    dump(out / "config.json", c)
    tick = time.time()
    report = json.loads(Path(c["axis_report"]).read_text())
    identities = json.loads(Path(c["identity_metrics"]).read_text())
    ids = {split: identities[f"{split}_ids"] for split in ("train", "dev")}
    assert not set(ids["train"]) & set(ids["dev"])
    assert all(ids[split] == report["qc_summary"]["pass_ids_by_split"][split] for split in ids)
    axes = {r["video_id"]: r for r in report["axis"]}
    rows, bindings = [], []
    for split in ("train", "dev"):
        for vid in ids[split]:
            axis = axes[vid]
            path = Path(c["data_root"]) / axis["axis_array_relpath"]
            assert sha(path) == axis["axis_array_sha256"], vid
            z = np.load(path)
            pts, mask = z["timestamps"], z["raw_summary_membership"].astype(bool)
            assert mask.shape == (10, len(pts)) and len(pts) > 1 and np.all(np.diff(pts) > 0), vid
            scorepath = Path(c["scores_root"]) / f"{vid}.json"
            s = json.loads(scorepath.read_text())
            ix, vals = np.array(s["frame_indices"]), np.array(s["values"])
            assert s["video_id"] == vid and s["split"] == split
            assert np.array_equal(pts[ix], s["actual_pts"]) and len(set(ix)) == len(ix), vid
            assert vals.shape[0] == len(ix) and np.isfinite(vals).all()
            owner = np.abs(pts[:, None] - pts[ix][None]).argmin(1)
            rows.append(dict(id=vid, split=split, pts=pts, mask=mask, ix=ix, owner=owner, heads=vals[:, :2]))
            bindings.append(dict(video_id=vid, split=split, axis_sha256=sha(path), score_sha256=sha(scorepath)))
    dump(out / "bindings.json", bindings)
    train, dev = [[r for r in rows if r["split"] == split] for split in ("train", "dev")]
    models = {name: fit(train, position, heads, c["ridge"])
              for name, position, heads in [("position", True, False), ("heads", False, True), ("position_heads", True, True)]}
    dump(out / "models.json", models)
    records, oracles, observations, source_summary = [], [], [], []
    for r in rows:
        n = len(r["pts"])
        k = round(c["primary_keep"] * n)
        center_order = np.argsort(abs(np.arange(n) - (n - 1) / 2), kind="stable")
        baseline = fscore(r["mask"], center_order, k)
        def record(arm, order, kk, **extras):
            records.append(dict(video_id=r["id"], split=r["split"], arm=arm,
                                f=fscore(r["mask"], order, kk), kept=kk, frames=n, **extras))
        record("center85", center_order, k)
        record("all", np.arange(n), n)
        order = cell_order(r, r["heads"].sum(1))
        record("aha_raw85", order, k)
        if r["split"] == "dev":
            for name, order0 in [("center", center_order), ("aha_raw", order)]:
                curve = prefix_curve(r["mask"], order0)
                best = int(np.argmax(curve))
                oracles.append(dict(video_id=r["id"], diagnostic=f"GT_K_given_{name}_ordering",
                                    f=float(curve[best]), kept=best, frames=n))
        for name, model in models.items():
            p = predict(r, model)
            order = cell_order(r, p[r["ix"]]) if name == "heads" else fixed_order(p)
            order, opt_k, mhat = plugin(p, order)
            for suffix, kk in [("fixed85", k), ("plugin", opt_k)]:
                record(f"{name}_{suffix}", order, kk, predicted_m=mhat,
                       brier=float(((p-r["mask"].mean(0))**2).mean()),
                       mean_predicted_probability=float(p.mean()),
                       mean_annotation_probability=float(r["mask"].mean()))
            if name != "position":
                pos_order = fixed_order(predict(r, models["position"]))
                record(f"position_at_{name}_plugin_K", pos_order, opt_k)
                record(f"center_at_{name}_plugin_K", center_order, opt_k)
            if r["split"] == "dev":
                curve = prefix_curve(r["mask"], order)
                best = int(np.argmax(curve))
                oracles += [dict(video_id=r["id"], diagnostic=f"GT_K_given_{name}_ordering", f=float(curve[best]), kept=best, frames=n)]
        source_summary.append(dict(video_id=r["id"], split=r["split"], decoded_frames=n,
                                   duration_pts_span_s=float(r["pts"][-1]-r["pts"][0]),
                                   aha_requested=len(r["ix"]), aha_unique_frames=len(set(r["ix"])),
                                   aha_max_sampling_gap_s=float(np.diff(r["pts"][r["ix"]]).max()),
                                   mean_annotation_density=float(r["mask"].mean()),
                                   rater_annotation_densities=r["mask"].mean(1).tolist()))
        if r["split"] != "dev":
            continue
        m = r["mask"].sum(1)
        for keep in c["diagnostic_keeps"]:
            kk = round(keep * n)
            utility = (r["mask"] / (kk + m[:, None])).mean(0)
            full = fscore(r["mask"], fixed_order(utility), kk)
            oracles += [dict(video_id=r["id"], diagnostic="full_GT_fixed_K_upper_bound", keep=keep, f=full, kept=kk)]
            for samples in c["observation_counts"]:
                ix = np.abs(r["pts"][:, None] - np.linspace(r["pts"][0], r["pts"][-1], samples)).argmin(0)
                owner = np.abs(r["pts"][:, None] - r["pts"][ix][None]).argmin(1)
                block = np.bincount(owner, weights=utility) / np.bincount(owner)
                order = fixed_order(block[owner])
                oracles += [dict(video_id=r["id"], diagnostic="GT_block_mean_broadcast_not_upper_bound", keep=keep,
                                 requested=samples, unique_frames=len(set(ix)), f=fscore(r["mask"], order, kk), kept=kk)]
        for samples in c["observation_counts"]:
            observations += [dict(video_id=r["id"], **observation(r, samples))]
    split_arms = {}
    for split in ("train", "dev"):
        split_arms[split] = {}
        for name in sorted({r["arm"] for r in records}):
            rr = [r for r in records if r["arm"] == name and r["split"] == split]
            split_arms[split][name] = dict(f=float(np.mean([r["f"] for r in rr])), macro_keep=float(np.mean([r["kept"] / r["frames"] for r in rr])), videos=rr)
    arms = split_arms["dev"]
    reference_arms = dict(aha_raw85="aha_sum", center85="position_center", all="select_all")
    for current, prior in reference_arms.items():
        expected = identities["results"][str(c["primary_keep"])][prior]
        assert np.isclose(arms[current]["f"], expected["f"], atol=1e-12, rtol=0), (current, arms[current]["f"], expected["f"])
        previous = {row["video_id"]: row for row in expected["videos"]}
        for row in arms[current]["videos"]:
            assert np.isclose(row["f"], previous[row["video_id"]]["f"], atol=1e-12, rtol=0), row["video_id"]
    contrasts = {}
    for candidate, control in c["contrasts"]:
        x = {r["video_id"]: r["f"] for r in arms[candidate]["videos"]}
        delta = np.array([x[r["video_id"]] - r["f"] for r in arms[control]["videos"]])
        contrasts[f"{candidate}_minus_{control}"] = bootstrap(delta, c["bootstrap_seed"], c["bootstrap_replicates"])
    result = dict(status="COMPLETED", scope=c["scope"], config_sha256=sha(config_path), script_sha256=sha(__file__),
                  train_sources=len(train), dev_sources=len(dev), protected_opened=False, training="CPU logistic calibrators only",
                  seconds=time.time()-tick, models=models, arms=arms, split_arms=split_arms, contrasts=contrasts,
                  source_summary=source_summary, empty_gt_raters=sum(int((r["mask"].sum(1)==0).sum()) for r in rows),
                  baseline_reference="All three controls independently reproduce R26 per-source F within 1e-12",
                  empty_gt_convention="Public R26 evaluator uses 0 for both-empty; no official boundary claim",
                  gt_assisted_diagnostics=oracles,
                  observation_diagnostics=observations, limitations=c["limitations"])
    dump(out / "metrics.json", result)
    print(json.dumps(dict(seconds=result["seconds"], arms={k: {"f": v["f"], "keep": v["macro_keep"]} for k,v in arms.items()}, contrasts=contrasts), indent=2))


if __name__ == "__main__":
    main()
