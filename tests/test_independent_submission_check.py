"""Synthetic-only checks for current and historical submission contracts.

Runs with standard-library unittest; no model, video, or label data is read.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from scripts.independent_submission_check import check


class IndependentSubmissionCheckTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="aic_checker_synthetic_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.index_path = self.root / "index.jsonl"
        self.submission_path = self.root / "predictions.jsonl"
        self.meta = {"video_id": "synthetic", "width": 100, "height": 100,
                     "frame_count": 10, "targetRatioWH": [1, 1]}
        self.row = {"video_id": "synthetic", "targetRatioWH": [1, 1],
                    "model_params_b": 0.450283377,
                    "predictions": [{"frame": 0, "bboxes": [0, 0, 100]}]}
        self.write()

    def write(self, rows=None, index=None):
        # NaN/Infinity are intentionally serializable here to exercise rejection.
        for path, values in ((self.index_path, [self.meta] if index is None else index),
                             (self.submission_path, [self.row] if rows is None else rows)):
            path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")

    def current(self, **kwargs):
        return check(self.index_path, self.submission_path, contract="current", **kwargs)

    def test_current_only_parameter_field_matches_supplied_count(self):
        result = self.current(expected_parameters=450_283_377)
        self.assertTrue(result["valid"])
        self.assertEqual(result["model_params_b"], 0.450283377)
        self.assertEqual(result["parameter_count_evidence"], "caller_supplied_count")
        self.assertFalse(result["weights_loaded"])
        self.assertEqual(result["predictions"], 1)

    def test_parameter_cap_and_zero_parameter_endpoints(self):
        for count in (0, 1, 9_000_000_000):
            with self.subTest(count=count):
                self.row["model_params_b"] = count / 1_000_000_000
                self.write()
                self.assertTrue(self.current(expected_parameters=count)["valid"])

    def test_invalid_parameter_declarations(self):
        values = (None, True, False, "0.45", -1, 9.000000001,
                  float("nan"), float("inf"), -float("inf"), 10 ** 400)
        for value in values:
            with self.subTest(value=value):
                self.row["model_params_b"] = value
                self.write()
                with self.assertRaises(ValueError):
                    self.current()

    def test_missing_parameter_field_cannot_use_legacy_bypass(self):
        del self.row["model_params_b"]
        self.row["model_size_mb"] = 50
        self.write()
        for require_size in (True, False):
            with self.subTest(require_size=require_size):
                with self.assertRaisesRegex(ValueError, "model_params_b"):
                    self.current(require_size=require_size)

    def test_exact_parameter_consistency_across_videos(self):
        second = {**self.row, "video_id": "synthetic-2"}
        index = [self.meta, {**self.meta, "video_id": "synthetic-2"}]
        self.write(rows=[self.row, second], index=index)
        self.assertEqual(self.current()["videos"], 2)
        second["model_params_b"] += 1e-12
        self.write(rows=[self.row, second], index=index)
        with self.assertRaisesRegex(ValueError, "differs across rows"):
            self.current()

    def test_supplied_count_mismatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "supplied parameter count"):
            self.current(expected_parameters=450_283_378)

    def test_supplied_count_is_integer_and_within_cap(self):
        for value in (True, False, -1, 1.0, "1", 9_000_000_001):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "expected_parameters"):
                    self.current(expected_parameters=value)

    def test_bytes_do_not_determine_current_eligibility(self):
        self.row["model_size_mb"] = 20_000
        self.write()
        self.assertTrue(self.current()["valid"])
        self.assertEqual(self.current()["parameter_count_evidence"], "declaration_only")
        with self.assertRaisesRegex(ValueError, "not weight bytes"):
            self.current(expected_bytes=20_000_000_000)

    def test_optional_current_size_is_nonnegative_and_finite(self):
        self.row["model_size_mb"] = 0
        self.write()
        self.assertEqual(self.current()["model_size_mb"], 0)
        for value in (None, True, False, "1", -1, float("nan"), float("inf")):
            with self.subTest(value=value):
                self.row["model_size_mb"] = value
                self.write()
                with self.assertRaises(ValueError):
                    self.current()

    def test_optional_current_size_presence_and_value_are_consistent(self):
        second = {**self.row, "video_id": "synthetic-2"}
        index = [self.meta, {**self.meta, "video_id": "synthetic-2"}]
        self.row["model_size_mb"] = 20_000
        self.write(rows=[self.row, second], index=index)
        with self.assertRaisesRegex(ValueError, "present on some rows only"):
            self.current()
        second["model_size_mb"] = 20_001
        self.write(rows=[self.row, second], index=index)
        with self.assertRaisesRegex(ValueError, "differs across rows"):
            self.current()
        second["model_size_mb"] = 20_000
        self.write(rows=[self.row, second], index=index)
        self.assertEqual(self.current()["model_size_mb"], 20_000)

    def test_predictions_required_but_empty_list_is_valid(self):
        self.row.pop("predictions")
        self.write()
        with self.assertRaisesRegex(ValueError, "predictions must be a list"):
            self.current()
        for value in (None, {}, "", []):
            with self.subTest(value=value):
                self.row["predictions"] = value
                self.write()
                if value == []:
                    self.assertEqual(self.current()["predictions"], 0)
                else:
                    with self.assertRaisesRegex(ValueError, "predictions must be a list"):
                        self.current()

    def test_duplicate_json_keys_are_rejected(self):
        raw = self.submission_path.read_text(encoding="utf-8")
        raw = raw.replace('"model_params_b": 0.450283377',
                          '"model_params_b": 10, "model_params_b": 0.450283377')
        self.submission_path.write_text(raw, encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON object key"):
            self.current()

    def test_invalid_frames_and_order_are_rejected(self):
        for frame in (True, False, -1, 10, 1.5, "1"):
            with self.subTest(frame=frame):
                self.row["predictions"] = [{"frame": frame, "bboxes": [0, 0, 100]}]
                self.write()
                with self.assertRaisesRegex(ValueError, "frame out of range"):
                    self.current()
        for frames in ([1, 1], [2, 1]):
            with self.subTest(frames=frames):
                self.row["predictions"] = [{"frame": f, "bboxes": [0, 0, 100]} for f in frames]
                self.write()
                with self.assertRaisesRegex(ValueError, "duplicate/unsorted frame"):
                    self.current()

    def test_invalid_crop_geometry_and_numeric_types(self):
        crops = ([False, 0, 100], [0, 0, True], ["0", 0, 100], [0, 0, 0],
                 [-1, 0, 100], [1, 0, 100], [0, 0, 101], [0, 0],
                 [0, 0, float("nan")], [0, float("inf"), 10])
        for crop in crops:
            with self.subTest(crop=crop):
                self.row["predictions"][0]["bboxes"] = crop
                self.write()
                with self.assertRaises(ValueError):
                    self.current()

    def test_current_height_computation_avoids_intermediate_overflow(self):
        ratio = [1e308, 1e308]
        self.meta["targetRatioWH"] = ratio
        self.row["targetRatioWH"] = ratio
        self.write()
        self.assertTrue(self.current()["valid"])

    def test_current_rejects_height_underflow(self):
        ratio = [1, 1e-308]
        self.meta["targetRatioWH"] = ratio
        self.row["targetRatioWH"] = ratio
        self.row["predictions"][0]["bboxes"] = [0, 0, 1e-308]
        self.write()
        with self.assertRaisesRegex(ValueError, "crop height must be positive"):
            self.current()

    def test_current_has_strict_bounds_while_legacy_retains_tolerance(self):
        for crop in ([5e-8, 0, 100], [0, 5e-8, 100]):
            with self.subTest(crop=crop):
                self.row["predictions"][0]["bboxes"] = crop
                self.write()
                with self.assertRaisesRegex(ValueError, "crop out of bounds"):
                    self.current()
                self.assertTrue(check(self.index_path, self.submission_path,
                                      require_size=False)["valid"])

    def test_ratio_must_match_without_boolean_coercion(self):
        for ratio in ([True, True], [1, 2], [1, 0], [1], "1:1"):
            with self.subTest(ratio=ratio):
                self.row["targetRatioWH"] = ratio
                self.write()
                with self.assertRaisesRegex(ValueError, "targetRatioWH"):
                    self.current()

    def test_current_validates_index_metadata(self):
        for field, value in (("width", True), ("height", 0), ("frame_count", 1.5),
                             ("targetRatioWH", [1, 0]), ("video_id", 1)):
            with self.subTest(field=field, value=value):
                self.write(index=[{**self.meta, field: value}])
                with self.assertRaises(ValueError):
                    self.current()

    def test_coverage_and_duplicates_remain_checked(self):
        second_meta = {**self.meta, "video_id": "synthetic-2"}
        self.write(index=[self.meta, second_meta])
        with self.assertRaisesRegex(ValueError, "missing video IDs"):
            self.current()
        self.write(rows=[self.row, copy.deepcopy(self.row)])
        with self.assertRaisesRegex(ValueError, "unknown/duplicate video_id"):
            self.current()
        self.write(index=[self.meta, self.meta])
        with self.assertRaisesRegex(ValueError, "duplicate index video_id"):
            self.current()

    def test_blank_lines_and_empty_files_remain_invalid(self):
        for raw in ("", "\n", json.dumps(self.row) + "\n\n"):
            with self.subTest(raw=raw):
                self.submission_path.write_text(raw, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.current()

    def test_legacy_default_keeps_existing_byte_contract(self):
        self.row.pop("model_params_b")
        self.row["model_size_mb"] = 50
        self.write()
        result = check(self.index_path, self.submission_path, 50_000_000)
        self.assertEqual(result, {"valid": True, "videos": 1, "predictions": 1,
                                 "model_size_mb": 50.0, "checker": "independent_submission_check_v1"})
        with self.assertRaisesRegex(ValueError, "weight bytes"):
            check(self.index_path, self.submission_path, 50_000_001)

    def test_legacy_omission_behavior_is_preserved(self):
        self.row.pop("model_params_b")
        self.write()
        with self.assertRaisesRegex(ValueError, "missing model_size_mb"):
            check(self.index_path, self.submission_path)
        self.assertIsNone(check(self.index_path, self.submission_path, require_size=False)["model_size_mb"])

    def test_modes_do_not_silently_accept_invalid_options(self):
        with self.assertRaisesRegex(ValueError, "contract must be"):
            check(self.index_path, self.submission_path, contract="typo")
        with self.assertRaisesRegex(ValueError, "requires contract=current"):
            check(self.index_path, self.submission_path, expected_parameters=450_283_377)

    def test_cli_current_accepts_valid_count_and_rejects_over_cap(self):
        script = Path(__file__).resolve().parents[1] / "scripts" / "independent_submission_check.py"
        command = [sys.executable, str(script), "--index", str(self.index_path),
                   "--submission", str(self.submission_path), "--contract", "current",
                   "--parameter-count", "450283377", "--allow-missing-size"]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["contract"], "current")
        self.row["model_params_b"] = 9.000000001
        self.write()
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("model_params_b", result.stderr)


if __name__ == "__main__":
    unittest.main()
