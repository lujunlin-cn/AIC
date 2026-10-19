"""Only synthetic records: no official labels or video files are used."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import warnings
import zipfile

from scripts.check_current_package import check_package, main


class CurrentPackageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.index = self.root / "index.jsonl"
        self.package = self.root / "synthetic.zip"
        self.index.write_text(json.dumps({"video_id": "synthetic", "width": 20,
                                         "height": 20, "frame_count": 3,
                                         "targetRatioWH": [1, 1]}) + "\n")
        self.row = {"video_id": "synthetic", "targetRatioWH": [1, 1],
                    "model_params_b": 9,
                    "predictions": [{"frame": 1, "bboxes": [2, 3, 10]}]}

    def write_package(self, name="predictions.jsonl", extra=None):
        with zipfile.ZipFile(self.package, "w") as archive:
            archive.writestr(name, json.dumps(self.row) + "\n")
            if extra:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", UserWarning)
                    archive.writestr(extra, "{}\n")

    def validate(self, **kwargs):
        values = dict(parameter_count=9_000_000_000, expected_videos=1)
        values.update(kwargs)
        return check_package(self.index, self.package, **values)

    def test_two_checkers_accept_and_record_exact_hashes(self):
        self.write_package()
        result = self.validate()
        self.assertTrue(result["project_checker"]["valid"])
        self.assertTrue(result["independent_checker"]["valid"])
        self.assertEqual(result["zip_sha256"], hashlib.sha256(self.package.read_bytes()).hexdigest())
        self.assertFalse(result["weights_loaded"])
        self.assertFalse(result["raw_video_reproduction_verified"])
        self.assertIsNone(result["score"])

    def test_wrong_parameter_count_cannot_pass(self):
        self.write_package()
        with self.assertRaises(ValueError):
            self.validate(parameter_count=8_999_999_999)

    def test_incomplete_index_cannot_pass_expected_coverage(self):
        self.write_package()
        with self.assertRaisesRegex(ValueError, "expected 2"):
            self.validate(expected_videos=2)

    def test_current_field_is_required(self):
        self.row.pop("model_params_b")
        self.row["model_size_mb"] = 50
        self.write_package()
        with self.assertRaises(ValueError):
            self.validate()

    def test_frozen_zip_digest_is_enforced(self):
        self.write_package()
        with self.assertRaisesRegex(ValueError, "SHA256 differs"):
            self.validate(expected_zip_sha256="0" * 64)
        digest = hashlib.sha256(self.package.read_bytes()).hexdigest()
        self.assertTrue(self.validate(expected_zip_sha256=digest)["valid"])

    def test_archive_layout_rejected(self):
        for name, extra in [("nested/predictions.jsonl", None),
                            ("../predictions.jsonl", None),
                            ("predictions.jsonl", "notes.txt"),
                            ("predictions.jsonl", "predictions.jsonl")]:
            with self.subTest(name=name, extra=extra):
                self.write_package(name=name, extra=extra)
                with self.assertRaisesRegex(ValueError, "exactly one root"):
                    self.validate()

    def test_bad_count_rejected_before_reading_payload(self):
        for count in [-1, 9_000_000_001, True, 1.5]:
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.validate(parameter_count=count)

    def test_failure_cli_does_not_produce_success(self):
        self.write_package()
        output = self.root / "failure.json"
        code = main(["--index", str(self.index), "--zip", str(self.package),
                     "--parameter-count", "9000000001", "--expected-videos", "1",
                     "--output", str(output)])
        self.assertEqual(code, 2)
        self.assertFalse(json.loads(output.read_text())["valid"])


if __name__ == "__main__":
    unittest.main()
