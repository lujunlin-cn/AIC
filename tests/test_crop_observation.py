"""Synthetic-only R85 coordinates/provenance tests; no real videos or labels."""
import copy
import hashlib
import json
import math
from pathlib import Path
import pickle

import numpy as np
from PIL import Image
import pytest

from aic.crop_observation import (PUBLIC_TABLES, candidate_boxes, crop_extent,
                                  load_manifest_image, make_row_id, mean_iou,
                                  pool_grid_2x2, validate_manifest_rows)
from scripts.r85_build_public_manifest import build_public_manifest, main, write_manifest


@pytest.mark.parametrize("W,H,ratio,first,last,axis", [
    (12, 4, [1, 1], [0, 0, 4, 4], [8, 0, 12, 4], 0),
    (4, 12, [1, 1], [0, 0, 4, 4], [0, 8, 4, 12], 1),
    (12, 4, [3, 1], [0, 0, 12, 4], [0, 0, 12, 4], None),
])
def test_candidate_x_y_and_equal_aspect_axes(W, H, ratio, first, last, axis):
    boxes = candidate_boxes(W, H, ratio)
    assert boxes.shape == (129, 4)
    assert boxes.dtype == np.float64
    np.testing.assert_allclose(boxes[0], first)
    np.testing.assert_allclose(boxes[-1], last)
    np.testing.assert_allclose(boxes[64], (np.array(first) + np.array(last)) / 2)
    if axis is None:
        np.testing.assert_array_equal(boxes, np.tile(first, (129, 1)))
    else:
        assert np.all(np.diff(boxes[:, axis]) > 0)
    np.testing.assert_allclose((boxes[:, 2] - boxes[:, 0]) / (boxes[:, 3] - boxes[:, 1]), ratio[0] / ratio[1])


def test_candidates_keep_fractional_native_coordinates():
    boxes = candidate_boxes(17, 10, [9, 16])
    assert boxes[0, 2] == 5.625
    assert boxes[1, 0] == pytest.approx((17 - 5.625) / 128)
    assert np.all(boxes[:, 2] <= 17)
    assert np.all(boxes[:, 3] <= 10)


@pytest.mark.parametrize("W,H,ratio", [(0, 10, [1, 1]), (True, 10, [1, 1]),
                                      (10, 10, [0, 1]), (10, 10, [True, 1]),
                                      (10, 10, [math.inf, 1])])
def test_invalid_native_geometry_is_rejected(W, H, ratio):
    with pytest.raises(ValueError):
        candidate_boxes(W, H, ratio)


def test_mean_iou_is_continuous_and_annotator_mean():
    boxes = [[0, 0, 10, 10], [10, 0, 20, 10]]
    gt = [[0, 0, 10, 10], [5, 0, 15, 10]]
    np.testing.assert_allclose(mean_iou(boxes, gt), [2 / 3, 1 / 6])


@pytest.mark.parametrize("box", [[0, 0, 0, 4], [0, 0, 4, 0], [0, 0, math.nan, 4],
                                 [0, 0, math.inf, 4], [False, 0, 4, 4], [-1, 0, 4, 4]])
def test_iou_and_crop_reject_invalid_area_coordinates(box):
    with pytest.raises(ValueError):
        mean_iou([box], [[0, 0, 4, 4]])
    with pytest.raises(ValueError):
        crop_extent(Image.new("RGB", (8, 8)), box)


def test_extent_defaults_to_ceil_native_size_without_rounding_source_box():
    pixels = np.zeros((8, 12, 3), dtype=np.uint8)
    pixels[:] = np.arange(12, dtype=np.uint8)[None, :, None] * 15
    image = Image.fromarray(pixels)
    box = [0.25, 1.25, 8.65, 4.55]
    crop = crop_extent(image, box)
    assert crop.size == (9, 4)
    expected = image.transform((9, 4), Image.Transform.EXTENT, tuple(box), resample=Image.Resampling.BICUBIC)
    np.testing.assert_array_equal(np.asarray(crop), np.asarray(expected))
    rounded = image.crop(tuple(round(v) for v in box)).resize((9, 4), Image.Resampling.BICUBIC)
    assert not np.array_equal(np.asarray(crop), np.asarray(rounded))


def test_extent_preserves_non_square_native_sampling_grid_and_allows_explicit_size():
    image = Image.new("RGB", (40, 30), (12, 34, 56))
    assert crop_extent(image, [1, 2, 33, 20]).size == (32, 18)
    assert crop_extent(image, [1.25, 2.5, 33.25, 20.5], output_size=(64, 36)).size == (64, 36)
    with pytest.raises(ValueError, match="outside"):
        crop_extent(image, [0, 0, 41, 30])
    with pytest.raises(ValueError):
        crop_extent(image, [0, 0, 40, 30], output_size=(0, 3))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_pool_constant_grid_and_tiny_fractional_roi(dtype):
    grid = np.broadcast_to(np.array([2, -3], dtype=dtype), (3, 5, 2)).copy()
    boxes = [[0, 0, 13, 7], [0.123, 0.456, 0.124, 0.457], [9, 1.5, 12.1, 6.9]]
    pooled = pool_grid_2x2(grid, 13, 7, boxes)
    assert pooled.dtype == dtype
    np.testing.assert_allclose(pooled, np.tile([2, -3] * 4, (3, 1)), rtol=1e-6)


def test_pool_resolves_tl_tr_bl_br_quadrants():
    grid = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=np.float64)
    np.testing.assert_allclose(pool_grid_2x2(grid, 4, 8, [[0, 0, 4, 8]]), [[1, 2, 3, 4, 5, 6, 7, 8]])


def test_pool_fractional_roi_matches_hand_calculated_overlap_areas():
    grid = np.array([[1, 3], [5, 7]], dtype=np.float64)[..., None]
    np.testing.assert_allclose(pool_grid_2x2(grid, 4, 4, [[1, 1, 4, 4]]), [[3, 13 / 3, 17 / 3, 7]])


def test_pool_mirror_covariance_in_x_and_y():
    grid = np.arange(4 * 5 * 2, dtype=np.float64).reshape(4, 5, 2)
    boxes = np.array([[0.25, 1.1, 8.7, 5.5], [4.2, 2.1, 9.3, 7.8]])
    original = pool_grid_2x2(grid, 10, 8, boxes).reshape(2, 4, 2)
    xboxes = boxes.copy()
    xboxes[:, [0, 2]] = 10 - boxes[:, [2, 0]]
    mirrored_x = pool_grid_2x2(grid[:, ::-1], 10, 8, xboxes).reshape(2, 4, 2)
    np.testing.assert_allclose(mirrored_x, original[:, [1, 0, 3, 2]])
    yboxes = boxes.copy()
    yboxes[:, [1, 3]] = 8 - boxes[:, [3, 1]]
    mirrored_y = pool_grid_2x2(grid[::-1], 10, 8, yboxes).reshape(2, 4, 2)
    np.testing.assert_allclose(mirrored_y, original[:, [2, 3, 0, 1]])


def test_pool_full_crop_grid_and_native_equivalent_are_identical():
    grid = np.arange(3 * 5, dtype=np.float32).reshape(3, 5, 1)
    a = pool_grid_2x2(grid, 5, 3, [[0, 0, 5, 3]])
    b = pool_grid_2x2(grid, 100, 60, [[0, 0, 100, 60]])
    np.testing.assert_allclose(a, b)


def test_pool_rejects_nonfinite_or_invalid_inputs():
    with pytest.raises(ValueError, match="finite"):
        pool_grid_2x2(np.full((2, 2, 1), math.nan), 4, 4, [[0, 0, 4, 4]])
    with pytest.raises(ValueError, match="outside"):
        pool_grid_2x2(np.ones((2, 2, 1)), 4, 4, [[0, 0, 5, 4]])
    with pytest.raises(ValueError, match="float32 or float64"):
        pool_grid_2x2(np.ones((2, 2, 1), dtype=np.int32), 4, 4, [[0, 0, 4, 4]])


@pytest.fixture
def public_tables(tmp_path):
    t5, rv = tmp_path / "T5", tmp_path / "RV"
    (t5 / "tables").mkdir(parents=True)
    (rv / "d1").mkdir(parents=True)
    units = {}
    for number, table in enumerate(PUBLIC_TABLES):
        ds, split = table.split("_")
        vid = ("001" if split == "train" else "081") if ds == "rv" else "live_" + split
        unit = {"ds": ds, "vid": vid, "W": 8, "H": 4, "ratio": [1, 1],
                "keys": [1, 3, 7], "J": np.array([2, 0, 1]), "gf": np.array([7, 1]),
                "gt": np.array([[[2, 0, 6, 4], [0, 0, 4, 4]],
                                [[4, 0, 8, 4], [1, 0, 5, 4]]], dtype=np.float64)}
        units[table] = [unit]
        (t5 / "tables" / f"{table}.pkl").write_bytes(pickle.dumps([unit]))
        frames = (rv / "d1" if ds == "rv" else t5 / "keyframes_train") / vid
        frames.mkdir(parents=True)
        for frame in (1, 7):
            Image.new("RGB", (8, 4), (number * 30, frame, 10)).save(frames / f"{frame}.png")
    return t5, rv, units


def _rewrite_table(t5, name, units):
    (t5 / "tables" / f"{name}.pkl").write_bytes(pickle.dumps(units))


def test_manifest_joins_original_source_frame_not_shard_row_and_records_hashes(public_tables):
    t5, rv, units = public_tables
    rows, meta = build_public_manifest(t5, rv)
    assert len(rows) == 8
    assert len({r["row_id"] for r in rows}) == 8
    assert meta["source_train_dev_overlap"] == []
    assert meta["training_executed"] is False
    assert all(t["selected_keyframes_without_gt"] == 1 for t in meta["tables"])
    for row in rows:
        assert row["frame_id"] in (1, 7)
        assert row["image_sha256"] == hashlib.sha256(Path(row["image_path"]).read_bytes()).hexdigest()
        assert row["table_sha256"] == hashlib.sha256(Path(row["table_path"]).read_bytes()).hexdigest()
        expected_index = 0 if row["frame_id"] == 7 else 1
        assert row["gt_xyxy"] == units[row["table"]][0]["gt"][:, expected_index].tolist()
        assert row["table_gt_index"] == expected_index
        assert load_manifest_image(row).size == (8, 4)
        if row["dataset"] == "live":
            assert "keyframes_train" in row["image_path"]
    validate_manifest_rows(rows)


def test_row_identity_is_independent_of_ratio_scaling_and_table_order():
    assert make_row_id("rv:001", 7, [9, 16]) == make_row_id("rv:001", 7, [18, 32])
    assert make_row_id("rv:001", 7, [9, 16]) != make_row_id("rv:001", 8, [9, 16])


def test_manifest_rejects_duplicate_row_identity(public_tables):
    t5, rv, units = public_tables
    _rewrite_table(t5, "rv_train", units["rv_train"] * 2)
    with pytest.raises(ValueError, match="duplicate row_id"):
        build_public_manifest(t5, rv)


def test_manifest_rejects_source_train_dev_leakage_including_unlabeled_unit(public_tables):
    t5, rv, units = public_tables
    dev = copy.deepcopy(units["rv_dev"])
    dev[0]["vid"] = "001"
    dev[0]["J"] = np.array([], dtype=np.int64)
    _rewrite_table(t5, "rv_dev", dev)
    with pytest.raises(ValueError, match="source train/dev leakage"):
        build_public_manifest(t5, rv)


@pytest.mark.parametrize("change", ["dimension", "gt", "dataset"])
def test_manifest_rejects_native_size_invalid_gt_or_unapproved_dataset(public_tables, change):
    t5, rv, units = public_tables
    altered = copy.deepcopy(units["rv_train"])
    if change == "dimension":
        altered[0]["W"] = 9
    elif change == "gt":
        altered[0]["gt"][0, 0, 2] = 9
    else:
        altered[0]["ds"] = "official"
    _rewrite_table(t5, "rv_train", altered)
    with pytest.raises(ValueError):
        build_public_manifest(t5, rv)


def test_manifest_consumer_detects_image_mutation(public_tables):
    t5, rv, _ = public_tables
    rows, _ = build_public_manifest(t5, rv)
    Image.new("RGB", (8, 4), (99, 99, 99)).save(rows[0]["image_path"])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        load_manifest_image(rows[0])
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        validate_manifest_rows(rows)


def test_manifest_consumer_rejects_edited_schema_source_split_gt_and_duplicates(public_tables):
    t5, rv, _ = public_tables
    original, _ = build_public_manifest(t5, rv)
    for field, value in (("schema_version", "unknown"), ("table", "rv_confirm2"),
                         ("split", "dev"), ("W", 10), ("row_id", "fabricated"),
                         ("gt_xyxy", [[0, 0, 9, 4]]), ("coordinate_scale", "normalized"),
                         ("table_sha256", "0" * 64), ("label_source", "unknown")):
        edited = copy.deepcopy(original)
        edited[0][field] = value
        with pytest.raises(ValueError):
            validate_manifest_rows(edited)
    with pytest.raises(ValueError, match="duplicate row_id"):
        validate_manifest_rows(original + [copy.deepcopy(original[0])])
    leaked = copy.deepcopy(original[0])
    leaked["split"], leaked["table"] = "dev", "rv_dev"
    with pytest.raises(ValueError, match="source train/dev leakage"):
        validate_manifest_rows(original + [leaked])


def test_manifest_jsonl_meta_and_default_no_overwrite(public_tables, tmp_path):
    t5, rv, _ = public_tables
    rows, meta = build_public_manifest(t5, rv)
    output = tmp_path / "public.jsonl"
    written = write_manifest(rows, meta, output)
    assert written["manifest_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert len(output.read_text().splitlines()) == 8
    assert json.loads(output.with_suffix(".meta.json").read_text())["rows"] == 8
    before = output.read_bytes()
    with pytest.raises(FileExistsError):
        write_manifest(rows, meta, output)
    assert output.read_bytes() == before


def test_cli_prepares_synthetic_manifest_without_training(public_tables, tmp_path, capsys):
    t5, rv, _ = public_tables
    output = tmp_path / "cli.jsonl"
    assert main(["--t5-dir", str(t5), "--rv-frames", str(rv), "--output", str(output)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["rows"] == 8
    assert result["training_executed"] is False
