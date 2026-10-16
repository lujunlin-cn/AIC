#!/usr/bin/env python3
"""Independent CPU recomputation of R29 temporal probe; no fitting or GPU."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def features(row, model):
    columns = [np.ones(len(row["pts"]))]
    if model["position"]:
        t = (row["pts"] - row["pts"][0]) * 2 / (row["pts"][-1] - row["pts"][0]) - 1
        columns.extend([t, t * t, t * t * t])
    if model["heads"]:
        expanded = row["head_values"][row["owners"]]
        columns.extend([expanded[:, 0], expanded[:, 1]])
    return np.column_stack(columns)


def probabilities(row, model):
    matrix = features(row, model)
    matrix = (matrix - np.asarray(model["mu"])) / np.asarray(model["sd"])
    logits = matrix.dot(model["beta"])
    return np.exp(-np.logaddexp(0, -logits))


def ordered(values):
    return np.asarray(sorted(range(len(values)), key=lambda f: (-values[f], f)))


def ordered_cells(row, values):
    chunks = []
    for j in sorted(range(len(values)), key=lambda f: (-values[f], f)):
        pool = np.flatnonzero(row["owners"] == j)
        anchor_pts = row["pts"][row["ix"][j]]
        chunks.append(sorted(pool.tolist(), key=lambda f: (abs(row["pts"][f] - anchor_pts), f)))
    return np.concatenate(chunks)


def evaluate(row, frame_order, k):
    selected = set(frame_order[:k].tolist())
    assert len(selected) == k
    # Deliberately compute each annotator independently from frame membership;
    # avoid reusing the probe's vectorized mask-column calculation.
    per_rater = []
    for labels in row["labels"]:
        gt = set(np.flatnonzero(labels).tolist())
        per_rater.append(2 * len(gt & selected) / (len(gt) + k) if len(gt) + k else 0.0)
    return float(np.mean(per_rater)), per_rater


def main():
    tick = time.monotonic()
    config_path = Path("reports/r29/temporal_probe_config.json")
    config = read(config_path)
    out = Path(config["output"])
    metrics, saved_models = read(out / "metrics.json"), read(out / "models.json")
    identities, report = read(config["identity_metrics"]), read(config["axis_report"])
    axes = {a["video_id"]: a for a in report["axis"]}
    assert metrics["config_sha256"] == digest(config_path)
    assert metrics["script_sha256"] == digest("scripts/r29_public_decision_probe.py")
    assert saved_models == metrics["models"]
    assert read(out / "config.json") == config
    ids = {split: identities[f"{split}_ids"] for split in ("train", "dev")}
    assert set(ids["train"]).isdisjoint(ids["dev"])
    rows = []
    axis_details = []
    bindings = {b["video_id"]: b for b in read(out / "bindings.json")}
    for split, sources in ids.items():
        assert sources == report["qc_summary"]["pass_ids_by_split"][split]
        for vid in sources:
            axis = axes[vid]
            path = Path(config["data_root"]) / axis["axis_array_relpath"]
            score_path = Path(config["scores_root"]) / f"{vid}.json"
            assert digest(path) == axis["axis_array_sha256"] == bindings[vid]["axis_sha256"]
            assert digest(score_path) == bindings[vid]["score_sha256"]
            z, scores = np.load(path), read(score_path)
            pts, labels = z["timestamps"], z["raw_summary_membership"]
            ix = np.asarray(scores["frame_indices"], dtype=np.int64)
            heads = np.asarray(scores["values"])[:, :2]
            assert labels.dtype == np.bool_ and labels.shape == (10, axis["decoded_frames"])
            assert int(z["rater_count"]) == 10 and len(pts) == axis["decoded_frames"]
            assert np.isfinite(pts).all() and np.all(np.diff(pts) > 0)
            assert len(set(ix)) == len(ix) and np.all(np.diff(ix) > 0) and ix.min() >= 0 and ix.max() < len(pts)
            assert np.array_equal(pts[ix], scores["actual_pts"])
            assert scores["video_id"] == vid and scores["split"] == split
            assert abs(float(labels.mean()) - axis["raw_positive_fraction"]) < 1e-15
            owners = np.abs(pts[:, None] - pts[ix][None, :]).argmin(axis=1)
            assert np.isfinite(heads).all() and heads.shape == (len(ix), 2)
            rows.append(dict(id=vid, split=split, pts=pts, labels=labels, ix=ix,
                             head_values=heads, owners=owners))
            axis_details.append(dict(video_id=vid, split=split, frames=len(pts), raters=10,
                                     aha_unique_samples=len(ix), axis_sha256=digest(path), score_sha256=digest(score_path)))

    train = [r for r in rows if r["split"] == "train"]
    normalization = {}
    for name, model in saved_models.items():
        mats = [features(row, model) for row in train]
        source_means = [m.mean(axis=0) for m in mats]
        mu = np.mean(source_means, axis=0)
        second = np.mean([(m * m).mean(axis=0) for m in mats], axis=0)
        sd = np.sqrt(np.maximum(0, second - mu * mu)).clip(1e-8)
        mu[0], sd[0] = 0.0, 1.0
        assert np.max(abs(mu - model["mu"])) < 1e-12
        assert np.max(abs(sd - model["sd"])) < 1e-12
        beta = np.asarray(model["beta"])
        objective, grad = 0.0, np.zeros(len(beta))
        for row, mat in zip(train, mats):
            x = (mat - mu) / sd
            target = row["labels"].mean(axis=0)
            logits = x.dot(beta)
            objective += float(np.mean(np.logaddexp(0, logits) - target * logits)) / len(train)
            predicted = np.exp(-np.logaddexp(0, -logits))
            grad += x.T.dot(predicted - target) / len(target) / len(train)
        objective += config["ridge"] * float(np.dot(beta[1:], beta[1:])) / 2
        grad[1:] += config["ridge"] * beta[1:]
        assert abs(objective - model["objective"]) < 1e-12
        assert np.max(abs(grad)) < 1e-6
        normalization[name] = dict(max_mean_error=float(np.max(abs(mu - model["mu"]))),
                                   max_sd_error=float(np.max(abs(sd - model["sd"]))),
                                   objective=objective, gradient_max_abs=float(np.max(abs(grad))),
                                   training_source_weight=1 / len(train), refit=False)

    maximum_f_error, source_checks, same_k_checks, oracle_checks, observation_checks = 0.0, 0, 0, 0, 0
    recomputed = {}
    baseline_rows = []
    source_diagnostics = []
    for row in rows:
        n = len(row["pts"])
        k85 = round(config["primary_keep"] * n)
        center = ordered(-abs(np.arange(n) - (n - 1) / 2))
        model_probs = {name: probabilities(row, model) for name, model in saved_models.items()}
        model_orders = {name: ordered_cells(row, p[row["ix"]]) if name == "heads" else ordered(p)
                        for name, p in model_probs.items()}
        candidates = {"center85": (center, k85), "all": (np.arange(n), n),
                      "aha_raw85": (ordered_cells(row, row["head_values"].sum(axis=1)), k85)}
        for name, p in model_probs.items():
            order = model_orders[name]
            plug = np.r_[0.0, 2 * p[order].cumsum() / (np.arange(1, n + 1) + sum(p))]
            kplug = int(np.argmax(plug))
            candidates[name + "_fixed85"] = (order, k85)
            candidates[name + "_plugin"] = (order, kplug)
            if name != "position":
                candidates[f"position_at_{name}_plugin_K"] = (model_orders["position"], kplug)
                candidates[f"center_at_{name}_plugin_K"] = (center, kplug)
                same_k_checks += 2
        for arm, (order, k) in candidates.items():
            f, raters = evaluate(row, order, k)
            saved = next(x for x in metrics["split_arms"][row["split"]][arm]["videos"] if x["video_id"] == row["id"])
            error = abs(f - saved["f"])
            maximum_f_error = max(maximum_f_error, error)
            assert error < 1e-12 and saved["kept"] == k and saved["frames"] == n
            recomputed[(row["id"], arm)] = f
            source_checks += 1
            if row["split"] == "dev" and arm in ("center85", "all", "aha_raw85"):
                old_name = {"center85": "position_center", "all": "select_all", "aha_raw85": "aha_sum"}[arm]
                prior = next(x for x in identities["results"]["0.85"][old_name]["videos"] if x["video_id"] == row["id"])
                assert abs(f - prior["f"]) < 1e-12
                baseline_rows.append(dict(video_id=row["id"], arm=arm, f=f, r26_f=prior["f"], per_rater_f=raters))
        if row["split"] != "dev":
            continue
        # Recompute observation misses with sets of GT-positive frame IDs.
        missed_by_samples = {}
        for count in config["observation_counts"]:
            points = np.linspace(row["pts"][0], row["pts"][-1], count)
            chosen = np.abs(row["pts"][:, None] - points).argmin(axis=0)
            sampled = set(chosen.tolist())
            masses, runs, missed_runs = [], [], []
            for labels in row["labels"]:
                positive = np.flatnonzero(labels)
                cut = np.flatnonzero(np.diff(positive) != 1) + 1
                segments = np.split(positive, cut) if len(positive) else []
                missed = [segment for segment in segments if set(segment.tolist()).isdisjoint(sampled)]
                runs.append(len(segments)); missed_runs.append(len(missed))
                masses.append(sum(len(segment) for segment in missed) / max(len(positive), 1))
            saved = next(o for o in metrics["observation_diagnostics"] if o["video_id"] == row["id"] and o["requested"] == count)
            assert len(sampled) == saved["unique_frames"]
            assert abs(np.mean(masses) - saved["mean_rater_positive_mass_in_unobserved_runs"]) < 1e-12
            assert abs(np.mean(runs) - saved["mean_rater_runs"]) < 1e-12
            assert abs(np.mean(missed_runs) - saved["mean_rater_unobserved_runs"]) < 1e-12
            observation_checks += 1
            missed_by_samples[str(count)] = float(np.mean(masses))
        source_diagnostics.append(dict(video_id=row["id"], positive_run_missed_mass=missed_by_samples))
        m = row["labels"].sum(axis=1)
        for keep in config["diagnostic_keeps"]:
            count = round(keep * n)
            marginal = np.mean(row["labels"] / (count + m[:, None]), axis=0)
            best_f, _ = evaluate(row, ordered(marginal), count)
            saved = next(o for o in metrics["gt_assisted_diagnostics"] if o["video_id"] == row["id"] and o["diagnostic"] == "full_GT_fixed_K_upper_bound" and o["keep"] == keep)
            assert abs(best_f - saved["f"]) < 1e-12
            oracle_checks += 1

    contrast_checks = {}
    for candidate, control in config["contrasts"]:
        deltas = np.asarray([recomputed[(vid, candidate)] - recomputed[(vid, control)] for vid in ids["dev"]])
        rng = np.random.default_rng(config["bootstrap_seed"])
        indices = rng.integers(0, len(deltas), size=(config["bootstrap_replicates"], len(deltas)))
        ci = np.quantile(deltas[indices].mean(axis=1), [.025, .975])
        key = f"{candidate}_minus_{control}"
        assert abs(deltas.mean() - metrics["contrasts"][key]["mean"]) < 1e-12
        assert np.max(abs(ci - metrics["contrasts"][key]["ci95"])) < 1e-12
        contrast_checks[key] = dict(mean=float(deltas.mean()), ci95=ci.tolist())

    review = dict(
        status="PASS_WITH_SCIENTIFIC_LIMITS", reviewer="independent decision_theory agent",
        method="Read producer code; independently reconstruct all masks and saved-model predictions, no import of producer, no optimizer/refit, no GPU or remote access.",
        checks=dict(source_axes=len(rows), train_sources=len(train), dev_sources=len(ids["dev"]),
                    axis_hashes=True, sorted_unique_in_bounds_sampled_frames=True, all_original_decoded_frames=True,
                    ten_separate_rater_F_then_source_macro=True, saved_model_normalization_and_gradient=normalization,
                    per_source_arm_checks=source_checks, max_f_difference=maximum_f_error,
                    same_K_control_checks=same_k_checks, r26_control_checks=len(baseline_rows),
                    observation_diagnostic_checks=observation_checks, fixed_K_GT_oracle_checks=oracle_checks,
                    paired_source_bootstrap_checks=contrast_checks),
        actual_r26_source_controls=baseline_rows, per_source_observation_checks=source_diagnostics,
        findings=[
            "No implementation/result discrepancy found in recomputed source-arm F, stored TRAIN standardization/objective, or paired controls.",
            "TRAIN normalization gives each source weight1/26; intercept unregularized. Soft mean-rater BCE equals mean per-rater BCE, but does not make plugin F Bayes-optimal.",
            "primary fixed85 content increment is negative with CI crossing zero; the predeclared +.005 and positive CI gate is not met.",
            "Plugin K maximizes 2*prefix_sum(p)/(K+sum(p)); it is explicitly a ratio-of-expectations approximation. Empty action value is set0, harmless for evaluation here because all 400 annotators have positive GT, but not a generally valid deployment treatment of P(M=0).",
            "Same-K position and center controls exist per source for both head-containing plugin arms; fixed85 uses the same exact rounded frame count throughout.",
            "Brier against mean-rater target measures squared error to that consensus; it differs from per-rater Brier by a label-variance term and alone does not establish calibration.",
            "All original decoded frames retained on hash-bound R22 QC arrays. This review does not re-decode media or rederive annotations from publisher raw files.",
            "GT fixed-K oracle is a valid temporal-only upper bound for that exact source, common annotator action and unrestricted original-frame subset; it is not an input-observation learnability bound.",
            "GT-K-given-ordering uses GT to choose K and is nondeployable; block-mean broadcast is correctly called one diagnostic recipe, not an upper bound.",
            "Positive-run misses mean zero sampled frames inside an annotation-positive connected interval, with frame-mass weighting. They do not show missing physical events, causal impossibility of inference, or encoder failure. Different sample counts are not nested phase-matched interventions.",
            "Observation12/24/48 concerns these invented uniform schedules; actual AHA uses approximately1fps and often many more observations, so these misses do not diagnose AHA's input loss directly.",
            "DEV14 is exposed development evidence, not independent confirmation. Pointwise bootstrap intervals are exploratory; no multiplicity correction or official benefit claim is supported."
        ],
        allowed_conclusion="On this exposed source-disjoint public split and frozen feature/query/nearest-PTS broadcast contract, the two saved AHA scalar heads with these fixed logistic/cubic calibrators did not demonstrate incremental temporal F beyond the position control at the declared practical gate. Plugin budget has point gains but uncertainty overlaps zero and matched-K position still explains much of the result.",
        disallowed_conclusions=["AHA visual or hidden representation has no content information", "All AHA prompts or full AHA variants fail", "The competition has no semantic signal", "Any exact posterior decision rule is ineffective", "12/24/48 positive-run misses explain official35-point plateau", "Public temporal changes imply official joint-score improvement"],
        provenance=dict(script_sha256=digest(__file__), producer_sha256=digest("scripts/r29_public_decision_probe.py"),
                        config_sha256=digest(config_path), metrics_sha256=digest(out / "metrics.json"),
                        python=platform.python_version(), numpy=np.__version__, wall_seconds=time.monotonic()-tick,
                        command="/home/hajimi2025/miniforge3/bin/python scripts/r29_temporal_independent_review.py", exit_status=0)
    )
    path = Path("reports/r29/temporal_independent_review.json")
    path.write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": review["status"], "checks": review["checks"], "output": str(path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
