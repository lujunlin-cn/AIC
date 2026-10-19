"""Synthetic-only checks of the R85 paired experiment and cache boundaries.

The small cache is intentionally uninformative: every legal candidate has IoU 1.
Successful execution must therefore leave the research gate closed and must not
claim an official score or create a competition package. No real data is read.
"""
from argparse import Namespace
import copy
import json
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import pytest

from aic.crop_observation import PUBLIC_TABLES, candidate_boxes, mean_iou
from scripts import r85_crop_observation_probe as probe
from scripts.r85_build_public_manifest import build_public_manifest


torch = pytest.importorskip("torch")
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module", autouse=True)
def cpu_runtime():
    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    previous_rng = torch.get_rng_state()
    torch.set_num_threads(2)
    yield
    torch.set_rng_state(previous_rng)
    torch.use_deterministic_algorithms(previous_deterministic)
    torch.set_num_threads(previous_threads)


def test_moments_use_only_selected_train_rows_and_handle_constant_dimensions():
    values = np.array([[[1, 4], [3, 4]], [[900, -900], [800, -800]],
                       [[5, 4], [7, 4]]], dtype=np.float32)
    ids = np.array([0, 2])
    mu, sd = probe.train_moments(values, ids)
    np.testing.assert_array_equal(mu, [4, 4])
    np.testing.assert_allclose(sd, [np.sqrt(5), 1])
    assert mu.dtype == sd.dtype == np.float32
    # Even a corrupt DEV row cannot enter train-only normalization.
    values[1] = np.nan
    changed_mu, changed_sd = probe.train_moments(values, ids)
    np.testing.assert_array_equal(changed_mu, mu)
    np.testing.assert_array_equal(changed_sd, sd)
    with pytest.raises(ValueError, match="nonfinite training features"):
        probe.train_moments(values, np.array([0, 1]))


def test_permutation_moves_only_added_candidate_features():
    old = np.arange(12, dtype=np.float16).reshape(2, 3, 2)
    added = np.arange(24, dtype=np.float16).reshape(2, 3, 4)
    ids = np.array([1, 0])
    mu, sd = np.array([1, 2], dtype=np.float32), np.array([2, 4], dtype=np.float32)
    local_mu = np.array([2, 3, 4, 5], dtype=np.float32)
    local_sd = np.array([1, 2, 3, 4], dtype=np.float32)
    permutations = np.array([[2, 0, 1], [1, 2, 0]])
    normal = probe.batch_features(old, added, ids, mu, sd, local_mu, local_sd)
    shuffled = probe.batch_features(old, added, ids, mu, sd, local_mu, local_sd,
                                    permutation=permutations)
    expected_old = (old[ids].astype(np.float32) - mu) / sd
    np.testing.assert_array_equal(normal[..., :2], expected_old)
    np.testing.assert_array_equal(shuffled[..., :2], expected_old)
    np.testing.assert_array_equal(shuffled[..., 2:],
                                  np.take_along_axis(normal[..., 2:], permutations[..., None], axis=1))
    assert not np.array_equal(shuffled[..., 2:], normal[..., 2:])
    np.testing.assert_array_equal(probe.batch_features(old, None, ids, mu, sd,
                                                      permutation=permutations), expected_old)


def test_statistics_pair_sources_and_do_not_weight_sources_by_frame_count():
    rows = [{"table": "rv_dev", "source": "rv:a"},
            {"table": "rv_dev", "source": "rv:a"},
            {"table": "rv_dev", "source": "rv:b"},
            {"table": "live_dev", "source": "live:a"},
            {"table": "live_dev", "source": "live:b"}]
    candidate = probe.source_report([0.25, 0.75, 1.0, 0.75, 0.25], rows)
    assert candidate["rv_dev"] == {"rv:a": 0.5, "rv:b": 1.0}
    # Deliberately reverse mapping insertion order: pair by source identity.
    other = {"rv_dev": {"rv:b": 0.75, "rv:a": 0.25},
             "live_dev": {"live:b": 0.5, "live:a": 0.5}}
    cfg = {"diagnostic": {"bootstrap_seed": 3, "bootstrap_repeats": 256},
           "engineering_gate": {"mean_iou_delta": 0.01, "positive_source_fraction": 2 / 3}}
    stats = probe.paired_comparison(candidate, other, cfg)
    assert stats["rv_dev"] == {"sources": 2, "mean_delta": 0.25,
                                "source_paired_ci95": [0.25, 0.25],
                                "strictly_positive_source_fraction": 1.0, "gate_pass": True}
    assert stats["live_dev"]["mean_delta"] == 0
    assert stats["live_dev"]["strictly_positive_source_fraction"] == 0.5
    assert stats["live_dev"]["gate_pass"] is False
    assert stats["live_dev"]["source_paired_ci95"][0] < 0
    assert stats["live_dev"]["source_paired_ci95"][1] > 0
    # Replicating all frames of one source must not change its paired weight.
    duplicated = probe.source_report([0.25, 0.75, 1.0, 0.75, 0.25, 0.25, 0.75],
                                      rows + rows[:2])
    assert probe.paired_comparison(duplicated, other, cfg) == stats
    mismatch = copy.deepcopy(other)
    mismatch["rv_dev"]["rv:unknown"] = mismatch["rv_dev"].pop("rv:a")
    with pytest.raises(ValueError, match="paired source coverage missing/mismatched: rv_dev"):
        probe.paired_comparison(candidate, mismatch, cfg)


def test_actual_head_cpu_forward_backward_and_checkpoint_round_trip(tmp_path):
    torch.manual_seed(17)
    model = probe.Head.make(torch, d=5377).float()
    x = torch.randn(1, 129, 5377)
    before = next(model.parameters()).detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
    prediction = model(x)
    assert prediction.shape == (1, 129)
    assert torch.isfinite(prediction).all()
    loss = torch.nn.functional.mse_loss(prediction, torch.ones_like(prediction))
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in gradients)
    assert sum(float(g.abs().sum()) for g in gradients) > 0
    optimizer.step()
    assert not torch.equal(before, next(model.parameters()))
    checkpoint = tmp_path / "synthetic_head.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = probe.Head.make(torch, d=5377).float()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    restored.eval()
    with torch.inference_mode():
        torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)
    assert probe.model_state_hash(restored) == probe.model_state_hash(model)


@pytest.fixture
def synthetic_cache(tmp_path):
    """Build six legal public-schema rows from temporary synthetic table/image files."""
    t5, rv = tmp_path / "T5", tmp_path / "RV"
    (t5 / "tables").mkdir(parents=True)
    (rv / "d1").mkdir(parents=True)
    for table in PUBLIC_TABLES:
        dataset, split = table.split("_")
        units = []
        for i in range(1 if split == "train" else 2):
            video_id = f"synthetic_{dataset}_{split}_{i}"
            units.append({"ds": dataset, "vid": video_id, "W": 4, "H": 4, "ratio": [1, 1],
                          "keys": [0], "J": np.array([0], dtype=np.int64),
                          "gf": np.array([0], dtype=np.int64),
                          "gt": np.array([[[0, 0, 4, 4]]], dtype=np.float64)})
            image_dir = (rv / "d1" if dataset == "rv" else t5 / "keyframes_train") / video_id
            image_dir.mkdir(parents=True)
            Image.new("RGB", (4, 4), (i * 60, 20, 30)).save(image_dir / "0.png")
        (t5 / "tables" / f"{table}.pkl").write_bytes(pickle.dumps(units))
    rows, _ = build_public_manifest(t5, rv)
    assert len(rows) == 6
    root = tmp_path / "features"
    root.mkdir()
    (root / "rows.jsonl").write_text(probe.canonical_rows(rows))
    manifest = tmp_path / "synthetic_manifest.jsonl"
    manifest.write_text(probe.canonical_rows(rows))
    cfg = json.loads((REPO / "configs/R85_REAL_CROP_OBSERVATION.json").read_text())
    cfg["synthetic_cpu_test"] = True
    cfg["paired_training"].update(steps=1, batch=1)
    cfg["diagnostic"]["bootstrap_repeats"] = 64
    cfg["baseline_reproduction"].update(
        reference="synthetic all-candidates-perfect fixture; not a research result", train_rows=2, dev_rows=4,
        dev_sources={"rv_dev": 2, "live_dev": 2},
        source_macro_iou={"rv_dev": 1.0, "live_dev": 1.0},
        center_source_macro_iou={"rv_dev": 1.0, "live_dev": 1.0}, absolute_tolerance=1e-7)
    config = tmp_path / "synthetic_config.json"
    probe.write_json(config, cfg)
    rng = np.random.default_rng(123)
    arrays = {key: rng.standard_normal((6, 129, depth)).astype(np.float16)
              for key, depth in probe.ARRAYS.items()}
    # A large DEV shift catches accidental normalization across the full cache.
    dev_ids = np.array([i for i, row in enumerate(rows) if row["split"] == "dev"])
    for values in arrays.values():
        values[dev_ids] += np.float16(6)
    targets = np.stack([mean_iou(candidate_boxes(r["W"], r["H"], r["ratio"]), r["gt_xyxy"])
                        for r in rows]).astype(np.float32)
    np.testing.assert_array_equal(targets, np.ones((6, 129), dtype=np.float32))
    buffers = dict(arrays, target=targets, done=np.ones(6, dtype=np.bool_),
                   row_hashes=np.asarray([probe.row_feature_hash(arrays, targets, i) for i in range(6)], dtype="S64"))
    for key, values in buffers.items():
        np.save(root / f"{key}.npy", values, allow_pickle=False)
    identity = {"manifest_sha256": probe.sha(manifest), "config_sha256": probe.sha(config),
                "vision_sha256": "1" * 64, "processor_sha256": "2" * 64,
                "model_config_sha256": "3" * 64, "device": "cpu", "gpu_name": None,
                "rows": 6, "smoke_only": False, "synthetic_cpu_test": True}
    meta = {"identity": identity, "status": "COMPLETE", "completed_rows": 6,
            "rows_sha256": probe.sha(root / "rows.jsonl"),
            "cache_sha256": {f"{key}.npy": probe.sha(root / f"{key}.npy") for key in buffers},
            "vision_parameters": cfg["encoder"]["expected_parameters"],
            "performance_claim": False, "official_test_labels_used": False}
    probe.write_json(root / "extraction.json", meta)
    probe.validate_resume_cache(root, rows, meta)
    return Namespace(root=root, rows=rows, meta=meta, arrays=arrays, targets=targets, cfg=cfg,
                     config=config, dev_ids=dev_ids, tmp_path=tmp_path)


def trainer_args(cache):
    torch.manual_seed(41)
    control = probe.Head.make(torch, d=2305).float()
    checkpoint = cache.tmp_path / "synthetic_frozen_A.pt"
    torch.save({"state_dict": control.state_dict(), "mu": np.zeros(2305, dtype=np.float32),
                "sd": np.ones(2305, dtype=np.float32)}, checkpoint)
    return Namespace(config=cache.config, features=cache.root, baseline_head=checkpoint,
                     baseline_head_sha256=probe.sha(checkpoint),
                     output=cache.tmp_path / "paired_results", device="cpu")


def test_full_paired_training_cpu_preserves_controls_and_stops_promotion(synthetic_cache, capsys):
    cache = synthetic_cache
    args = trainer_args(cache)
    assert probe.train(args) is None
    result = json.loads((args.output / "results.json").read_text())
    assert result["status"] == "PUBLIC_PAIRED_EXPERIMENT_COMPLETE"
    assert result["engineering_gate_pass"] is False
    assert result["candidate_content_alignment_supported"] is False
    assert result["new_competition_package"] is None
    assert result["official_score_prediction"] is None
    assert result["official_test_labels_used"] is False
    assert result["TimeLens2_weights_loaded_this_run"] is False
    assert result["device"] == "cpu"
    assert result["config_sha256"] == probe.sha(args.config) == cache.meta["identity"]["config_sha256"]
    assert result["baseline_head_sha256"] == args.baseline_head_sha256
    assert result["feature_identity"] == cache.meta["identity"]
    tags = {f"{arm}_{seed}" for seed in cache.cfg["paired_training"]["seeds"] for arm in ("B", "C")}
    assert len(tags) == 4
    assert {p.stem for p in args.output.glob("*.pt")} == tags
    assert {p.stem.removesuffix("_dev") for p in args.output.glob("*_dev.npz")} == tags
    tr = np.array([i for i, row in enumerate(cache.rows) if row["split"] == "train"])
    old_mu, old_sd = probe.train_moments(cache.arrays["old"], tr)
    dev_row_ids = np.array([cache.rows[i]["row_id"] for i in cache.dev_ids])
    head_count = sum(p.numel() for p in probe.Head.make(torch, d=5377).parameters())
    assert result["head_parameters"] == head_count
    expected_parameters = (cache.cfg["parameter_inventory"]["TimeLens2_reported_parameters"]
                           + cache.meta["vision_parameters"] + head_count)
    assert result["provisional_deployment_parameters"] == expected_parameters
    assert result["below_9b_by_reported_timelens_inventory"] is (expected_parameters <= 9_000_000_000)
    for tag in sorted(tags):
        checkpoint_path = args.output / f"{tag}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        arm, seed = tag.split("_")
        assert checkpoint["arm"] == arm
        assert checkpoint["seed"] == int(seed)
        assert checkpoint["head_parameters"] == head_count
        assert checkpoint["config"] == cache.cfg
        assert checkpoint["feature_identity"] == cache.meta["identity"]
        assert checkpoint["baseline_head_sha256"] == args.baseline_head_sha256
        np.testing.assert_array_equal(checkpoint["mu"], old_mu)
        np.testing.assert_array_equal(checkpoint["sd"], old_sd)
        key = "roi" if arm == "B" else "crop"
        local_mu, local_sd = probe.train_moments(cache.arrays[key], tr)
        np.testing.assert_array_equal(checkpoint["local_mu"], local_mu)
        np.testing.assert_array_equal(checkpoint["local_sd"], local_sd)
        assert result["timings"][tag]["checkpoint_sha256"] == probe.sha(checkpoint_path)
        with np.load(args.output / f"{tag}_dev.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(saved["row_id"], dev_row_ids)
            np.testing.assert_array_equal(saved["mean_iou"], np.ones(4))
            np.testing.assert_array_equal(saved["center_iou"], np.ones(4))
            assert saved["picked"].shape == (4,)
            assert np.all((saved["picked"] >= 0) & (saved["picked"] < 129))
        # A trained artifact reloads into the actual deployment head and reproduces picks.
        loaded = probe.Head.make(torch, d=5377).float()
        loaded.load_state_dict(checkpoint["state_dict"], strict=True)
        picks = probe.predict(torch, loaded, cache.arrays["old"], cache.arrays[key], cache.dev_ids,
                              (checkpoint["mu"], checkpoint["sd"], checkpoint["local_mu"], checkpoint["local_sd"]), "cpu")
        with np.load(args.output / f"{tag}_dev.npz", allow_pickle=False) as saved:
            np.testing.assert_array_equal(picks, saved["picked"])
    for seed in cache.cfg["paired_training"]["seeds"]:
        b, c = result["timings"][f"B_{seed}"], result["timings"][f"C_{seed}"]
        assert b["initial_state_sha256"] == c["initial_state_sha256"]
        assert b["batch_sequence_sha256"] == c["batch_sequence_sha256"]
        for comparison in ("B", "A", "permuted"):
            for domain in ("rv_dev", "live_dev"):
                stats = result["comparisons"][f"C_{seed}-{comparison}"][domain]
                assert stats["sources"] == 2
                assert stats["mean_delta"] == 0
                assert stats["source_paired_ci95"] == [0, 0]
                assert stats["strictly_positive_source_fraction"] == 0
                assert stats["gate_pass"] is False
    assert json.loads(capsys.readouterr().out)["engineering_gate_pass"] is False


@pytest.mark.parametrize("smoke_marker", ["status", "identity"])
def test_formal_training_rejects_smoke_cache(synthetic_cache, smoke_marker):
    cache = synthetic_cache
    args = trainer_args(cache)
    if smoke_marker == "status":
        cache.meta["status"] = "SMOKE_COMPLETE"
    else:
        cache.meta["identity"]["smoke_only"] = True
    probe.write_json(cache.root / "extraction.json", cache.meta)
    with pytest.raises(ValueError, match="formal paired training requires complete, non-smoke extraction"):
        probe.train(args)
    assert not args.output.exists()


def test_formal_training_rejects_wrong_frozen_A_checkpoint_sha(synthetic_cache):
    args = trainer_args(synthetic_cache)
    args.baseline_head_sha256 = "0" * 64
    with pytest.raises(ValueError, match="frozen A checkpoint SHA mismatch"):
        probe.train(args)
    assert not args.output.exists()


def test_formal_training_rejects_config_change_after_cache_creation(synthetic_cache):
    args = trainer_args(synthetic_cache)
    cfg = copy.deepcopy(synthetic_cache.cfg)
    cfg["paired_training"]["steps"] += 1
    probe.write_json(args.config, cfg)
    with pytest.raises(ValueError, match="config changed after feature extraction"):
        probe.train(args)
    assert not args.output.exists()


def test_resume_rejects_reordered_rows_even_with_updated_rows_sha(synthetic_cache):
    cache = synthetic_cache
    (cache.root / "rows.jsonl").write_text(probe.canonical_rows(cache.rows[::-1]))
    cache.meta["rows_sha256"] = probe.sha(cache.root / "rows.jsonl")
    with pytest.raises(ValueError, match="resume ordered rows differ"):
        probe.validate_resume_cache(cache.root, cache.rows, cache.meta)


@pytest.mark.parametrize("key,shape,dtype", [
    ("old", (6, 128, 2305), np.float16),
    ("target", (6, 129), np.float64),
    ("done", (0,), np.bool_),
    ("row_hashes", (6,), "S32"),
])
def test_resume_rejects_wrong_array_shapes_or_dtypes(synthetic_cache, key, shape, dtype):
    cache = synthetic_cache
    np.save(cache.root / f"{key}.npy", np.zeros(shape, dtype=dtype), allow_pickle=False)
    cache.meta["cache_sha256"][f"{key}.npy"] = probe.sha(cache.root / f"{key}.npy")
    with pytest.raises(ValueError, match=f"resume array shape/dtype changed: {key}"):
        probe.validate_resume_cache(cache.root, cache.rows, cache.meta)


def test_resume_rejects_changed_completed_cache_bytes(synthetic_cache):
    cache = synthetic_cache
    crop = np.load(cache.root / "crop.npy", mmap_mode="r+", allow_pickle=False)
    crop[0, 0, 0] += np.float16(1)
    crop.flush()
    with pytest.raises(ValueError, match="completed cache SHA changed: crop.npy"):
        probe.validate_resume_cache(cache.root, cache.rows, cache.meta)


def test_resume_rejects_incomplete_rows_in_completed_cache(synthetic_cache):
    cache = synthetic_cache
    done = np.load(cache.root / "done.npy", mmap_mode="r+", allow_pickle=False)
    done[1] = False
    done.flush()
    cache.meta["cache_sha256"]["done.npy"] = probe.sha(cache.root / "done.npy")
    with pytest.raises(ValueError, match="completed cache has incomplete rows"):
        probe.validate_resume_cache(cache.root, cache.rows, cache.meta)


def test_interrupted_resume_checks_completed_row_hash_and_allows_unfinished_rows(synthetic_cache):
    cache = synthetic_cache
    cache.meta["status"] = "EXTRACTING"
    done = np.load(cache.root / "done.npy", mmap_mode="r+", allow_pickle=False)
    done[1] = False
    done.flush()
    crop = np.load(cache.root / "crop.npy", mmap_mode="r+", allow_pickle=False)
    crop[1, 0, 0] += np.float16(1)
    crop.flush()
    # This row was never published complete and will be recomputed.
    probe.validate_resume_cache(cache.root, cache.rows, cache.meta)
    crop[0, 0, 0] += np.float16(1)
    crop.flush()
    with pytest.raises(ValueError, match="interrupted cache completed row changed: 0"):
        probe.validate_resume_cache(cache.root, cache.rows, cache.meta)
