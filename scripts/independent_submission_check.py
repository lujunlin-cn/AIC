#!/usr/bin/env python3
"""Dependency-free second checker for an AIC JSONL submission.

This intentionally does not import ``aic.contract`` so it can catch a shared
validator bug. It checks the frozen metadata index, complete video coverage,
finite JSON values, frame order/uniqueness, target ratio, legal [x,y,w]
geometry, and a caller-supplied model count when supplied.

Use ``--contract current`` for new submissions: all rows must declare
``model_params_b`` in [0, 9], including every inference component. The default
``legacy`` contract preserves old callers and their historical MB field. This
checker never loads weights; matching a supplied count is not a weight audit.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def finite_tree(value):
    if isinstance(value, dict):
        return all(finite_tree(v) for v in value.values())
    if isinstance(value, list):
        return all(finite_tree(v) for v in value)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _finite_number(value)
    return True


def _finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON object key: {key}")
        result[key] = value
    return result


def _valid_ratio(value):
    return (isinstance(value, list) and len(value) == 2
            and all(_finite_number(v) and v > 0 for v in value)
            and _finite_number(value[1] / value[0])
            and value[1] / value[0] > 0)


def read_jsonl(path, *, strict=False):
    rows = []
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            raise ValueError(f"blank line {number}")
        value = json.loads(line, object_pairs_hook=_unique_object if strict else None)
        if not isinstance(value, dict) or not finite_tree(value):
            raise ValueError(f"invalid/nonfinite JSON object at line {number}")
        rows.append(value)
    if not rows:
        raise ValueError("empty JSONL")
    return rows


def check(index_path, submission_path, expected_bytes=None, require_size=True,
          *, contract="legacy", expected_parameters=None):
    """Check one submission; the current contract ignores legacy require_size.

    ``expected_parameters`` is an integer supplied by the caller after a
    separate audit. File bytes cannot establish a model's parameter count.
    """
    if contract not in {"legacy", "current"}:
        raise ValueError("contract must be legacy or current")
    if contract == "current" and expected_bytes is not None:
        raise ValueError("current contract uses expected_parameters, not weight bytes")
    if expected_parameters is not None:
        if contract != "current":
            raise ValueError("expected_parameters requires contract=current")
        if type(expected_parameters) is not int or not 0 <= expected_parameters <= 9_000_000_000:
            raise ValueError("expected_parameters must be an integer in [0, 9000000000]")
    index_rows = read_jsonl(index_path, strict=contract == "current")
    if contract == "current":
        for meta in index_rows:
            if not isinstance(meta.get("video_id"), str) or not meta["video_id"]:
                raise ValueError("index video_id must be a nonempty string")
            if any(type(meta.get(k)) is not int or meta[k] <= 0
                   for k in ("width", "height", "frame_count")):
                raise ValueError("index dimensions and frame_count must be positive integers")
            if not _valid_ratio(meta.get("targetRatioWH")):
                raise ValueError("index targetRatioWH must contain two finite positive numbers")
    index = {r["video_id"]: r for r in index_rows}
    if len(index) != len(index_rows):
        raise ValueError("duplicate index video_id")
    rows = read_jsonl(submission_path, strict=contract == "current")
    seen_ids = set()
    sizes = []
    params = []
    predictions = 0
    for row in rows:
        vid = row.get("video_id")
        if contract == "current" and (not isinstance(vid, str) or not vid):
            raise ValueError("submission video_id must be a nonempty string")
        if vid not in index or vid in seen_ids:
            raise ValueError(f"unknown/duplicate video_id: {vid!r}")
        seen_ids.add(vid)
        meta = index[vid]
        if ((contract == "current" and not _valid_ratio(row.get("targetRatioWH")))
                or row.get("targetRatioWH") != meta.get("targetRatioWH")):
            raise ValueError(f"targetRatioWH mismatch for {vid}")
        if contract == "current":
            count_b = row.get("model_params_b")
            if not _finite_number(count_b) or not 0 <= count_b <= 9:
                raise ValueError(f"model_params_b must be a finite number in [0, 9] for {vid}")
            params.append(float(count_b))
            if (expected_parameters is not None
                    and params[-1] != expected_parameters / 1_000_000_000):
                raise ValueError(f"model_params_b differs from supplied parameter count for {vid}")
            if "model_size_mb" in row:
                size = row["model_size_mb"]
                if not _finite_number(size) or size < 0:
                    raise ValueError(f"optional model_size_mb must be finite and nonnegative for {vid}")
                sizes.append(float(size))
        # Historical packages may omit model_size_mb; this never bypasses the
        # parameter declaration required by the current contract.
        elif "model_size_mb" in row or require_size:
            if "model_size_mb" not in row or not isinstance(row["model_size_mb"], (int, float)):
                raise ValueError(f"missing model_size_mb for {vid}")
            sizes.append(float(row["model_size_mb"]))
            if not 0.0 < sizes[-1] <= 9216.0 or not math.isfinite(sizes[-1]):
                raise ValueError(f"invalid model_size_mb for {vid}")
        rw, rh = map(float, meta["targetRatioWH"])
        last = None
        frames = set()
        if contract == "current" and not isinstance(row.get("predictions"), list):
            raise ValueError(f"predictions must be a list for {vid}; [] is legal")
        for pred in row.get("predictions", []):
            if not isinstance(pred, dict):
                raise ValueError(f"prediction object expected for {vid}")
            frame = pred.get("frame")
            if type(frame) is not int or not 0 <= frame < int(meta["frame_count"]):
                raise ValueError(f"frame out of range for {vid}: {frame!r}")
            if frame in frames or (last is not None and frame < last):
                raise ValueError(f"duplicate/unsorted frame for {vid}")
            frames.add(frame); last = frame
            crop = pred.get("bboxes")
            if not isinstance(crop, list) or len(crop) != 3:
                raise ValueError(f"bboxes must be [x,y,w] for {vid}")
            if contract == "current" and not all(_finite_number(v) for v in crop):
                raise ValueError(f"crop coordinates must be finite numbers, excluding bool, for {vid}")
            x, y, w = map(float, crop)
            h = w * (rh / rw) if contract == "current" else w * rh / rw
            if not all(math.isfinite(v) for v in (x, y, w, h)) or x < 0 or y < 0 or w <= 0:
                raise ValueError(f"invalid crop for {vid}")
            if contract == "current" and h <= 0:
                raise ValueError(f"crop height must be positive for {vid}")
            tolerance = 0.0 if contract == "current" else 1e-7
            if (x + w > int(meta["width"]) + tolerance
                    or y + h > int(meta["height"]) + tolerance):
                raise ValueError(f"crop out of bounds for {vid}")
            predictions += 1
    missing = sorted(set(index) - seen_ids)
    if missing:
        raise ValueError(f"missing video IDs: {missing[:5]} (total {len(missing)})")
    if contract == "current":
        if any(p != params[0] for p in params):
            raise ValueError("model_params_b differs across rows")
        if sizes and len(sizes) != len(rows):
            raise ValueError("optional model_size_mb present on some rows only")
        if sizes and any(size != sizes[0] for size in sizes):
            raise ValueError("optional model_size_mb differs across rows")
        return {"valid": True, "videos": len(rows), "predictions": predictions,
                "model_params_b": params[0], "expected_parameters": expected_parameters,
                "model_size_mb": sizes[0] if sizes else None,
                "contract": "current", "weights_loaded": False,
                "parameter_count_evidence": ("caller_supplied_count" if expected_parameters is not None
                                             else "declaration_only"),
                "checker": "independent_submission_check_v2"}
    if not require_size and not sizes:
        return {"valid": True, "videos": len(rows), "predictions": predictions,
                "model_size_mb": None, "checker": "independent_submission_check_v1"}
    if len(sizes) != len(rows):
        raise ValueError("model_size_mb present on some rows only")
    if max(sizes) - min(sizes) > 1e-9:
        raise ValueError("model_size_mb differs across rows")
    if expected_bytes is not None and not math.isclose(sizes[0], expected_bytes / 1_000_000, abs_tol=1e-9):
        raise ValueError("model_size_mb does not equal supplied weight bytes")
    return {"valid": True, "videos": len(rows), "predictions": predictions,
            "model_size_mb": sizes[0], "checker": "independent_submission_check_v1"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--submission", required=True)
    ap.add_argument("--contract", choices=("legacy", "current"), default="legacy",
                    help="新提交使用 current（model_params_b，最多 9B）；默认 legacy 保留历史 MB 校验")
    ap.add_argument("--weight-bytes", type=int)
    ap.add_argument("--parameter-count", type=int,
                    help="current：独立权重审计得到的全部推理参数数；checker 本身不加载权重")
    ap.add_argument("--allow-missing-size", action="store_true",
                    help="仅影响 legacy：允许整包省略 model_size_mb（锚点包 LFM_V8_B3_SEMIFINAL 的写法："
                         "该包在复赛已获官方分 34.75 且无此字段）")
    args = ap.parse_args(argv)
    print(json.dumps(check(args.index, args.submission, args.weight_bytes,
                           require_size=not args.allow_missing_size,
                           contract=args.contract, expected_parameters=args.parameter_count), indent=2))


if __name__ == "__main__":
    main()
