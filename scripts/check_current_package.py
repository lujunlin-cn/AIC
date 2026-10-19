#!/usr/bin/env python3
"""Read-only, two-checker validation of the current AIC prediction ZIP.

This checks the project's ZIP convention and the declared/supplied parameter
count. It does not read labels, run inference, load weights, or predict a score.
The supplied count must come from a separate audit of every inference model.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aic.contract import ContractError, load_index, validate_submission_file
from scripts.independent_submission_check import check


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_package(index_path: str | Path, zip_path: str | Path, *,
                  parameter_count: int, expected_videos: int,
                  expected_zip_sha256: str | None = None) -> dict:
    if type(parameter_count) is not int or not 0 <= parameter_count <= 9_000_000_000:
        raise ValueError("parameter_count must be an integer in [0, 9000000000]")
    if type(expected_videos) is not int or expected_videos <= 0:
        raise ValueError("expected_videos must be a positive integer")
    index_path, zip_path = Path(index_path).resolve(), Path(zip_path).resolve()
    index = load_index(index_path)
    if len(index) != expected_videos:
        raise ValueError(f"index contains {len(index)} videos, expected {expected_videos}")
    zip_sha = sha256_file(zip_path)
    if expected_zip_sha256 is not None:
        if (len(expected_zip_sha256) != 64
                or any(c not in "0123456789abcdef" for c in expected_zip_sha256.lower())):
            raise ValueError("expected_zip_sha256 must contain 64 hexadecimal characters")
        if zip_sha != expected_zip_sha256.lower():
            raise ValueError("ZIP SHA256 differs from the frozen expected SHA256")
    with tempfile.TemporaryDirectory(prefix="aic-current-check-") as temporary:
        extracted = Path(temporary) / "predictions.jsonl"
        with zipfile.ZipFile(zip_path) as archive:
            members = archive.infolist()
            if len(members) != 1 or members[0].filename != "predictions.jsonl" or members[0].is_dir():
                raise ValueError("project ZIP convention requires exactly one root predictions.jsonl")
            # Read the one allowed member; never extract arbitrary archive paths.
            with archive.open(members[0]) as source, extracted.open("wb") as target:
                shutil.copyfileobj(source, target)
        project = validate_submission_file(
            extracted, index_path, stage="current",
            actual_model_params_b=parameter_count / 1_000_000_000,
        )
        project.raise_for_errors()
        independent = check(index_path, extracted, contract="current",
                            expected_parameters=parameter_count)
        predictions_sha = sha256_file(extracted)
    return {
        "valid": True,
        "scope": "current_format_and_caller_supplied_parameter_count_only",
        "official_checker_executed": False,
        "weights_loaded": False,
        "raw_video_reproduction_verified": False,
        "score": None,
        "index_sha256": sha256_file(index_path),
        "zip_sha256": zip_sha,
        "predictions_sha256": predictions_sha,
        "expected_videos": expected_videos,
        "parameter_count": parameter_count,
        "project_checker": project.to_dict(),
        "independent_checker": independent,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", required=True)
    parser.add_argument("--zip", required=True)
    parser.add_argument("--parameter-count", required=True, type=int)
    parser.add_argument("--expected-videos", required=True, type=int)
    parser.add_argument("--expected-zip-sha256")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    if args.output and Path(args.output).resolve() in {Path(args.index).resolve(), Path(args.zip).resolve()}:
        parser.error("--output must not overwrite the index or ZIP")
    try:
        report = check_package(args.index, args.zip, parameter_count=args.parameter_count,
                               expected_videos=args.expected_videos,
                               expected_zip_sha256=args.expected_zip_sha256)
        code = 0
    except (ContractError, ValueError, OSError, zipfile.BadZipFile, RuntimeError) as error:
        report = {"valid": False, "error": str(error),
                  "official_checker_executed": False, "score": None}
        code = 2
    rendered = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
