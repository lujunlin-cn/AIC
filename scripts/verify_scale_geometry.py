#!/usr/bin/env python3
"""Run deterministic CPU geometry contract cases and save their evidence."""

from __future__ import annotations

import argparse
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

from aic.scale_geometry import deployed_box_xyxy, maximum_window, scale_crop_from_center, transpose_xywh


def run(cases: int = 1200, seed: int = 20261108) -> dict:
    rng = random.Random(seed)
    checked = {"A_9_16": 0, "B_16_9": 0}
    max_center_drift = 0.0
    for index in range(cases):
        width, height = rng.randint(240, 2400), rng.randint(240, 1800)
        name, ratio = ("A_9_16", (9, 16)) if index % 2 else ("B_16_9", (16, 9))
        center = (width / 2, height / 2)
        maximum_width, _ = maximum_window(width, height, ratio)
        full = scale_crop_from_center(width, height, ratio, center, 1.0)
        if full[2] != math.floor(maximum_width + 1e-9):
            raise AssertionError("scale=1 differs from maximum legal AIC window")
        crop = scale_crop_from_center(width, height, ratio, center, 0.63)
        dx = abs(crop[0] + crop[2] / 2 - center[0])
        dy = abs(crop[1] + crop[2] * ratio[1] / ratio[0] / 2 - center[1])
        max_center_drift = max(max_center_drift, dx, dy)
        if max(dx, dy) > 1.0:
            raise AssertionError("fixed center drift exceeded one pixel")
        if crop[0] < 0 or crop[1] < 0 or crop[0] + crop[2] > width:
            raise AssertionError("crop is outside image bounds")
        if crop[1] + crop[2] * ratio[1] / ratio[0] > height + 1e-9:
            raise AssertionError("crop height is outside image bounds")
        left, top, right, bottom = deployed_box_xyxy(crop[:3], ratio)
        if right > width + 1e-9 or bottom > height + 1e-9:
            raise AssertionError("deployed AIC projection is outside image bounds")
        transposed, dimensions = transpose_xywh(
            [full[0], full[1], full[2], full[2] * ratio[1] / ratio[0]], width, height
        )
        restored, restored_dimensions = transpose_xywh(transposed, *dimensions)
        original = [full[0], full[1], full[2], full[2] * ratio[1] / ratio[0]]
        if restored_dimensions != (width, height) or any(
            abs(float(left_value) - float(right_value)) > 1e-9
            for left_value, right_value in zip(original, restored)
        ):
            raise AssertionError("transpose geometry did not round-trip")
        checked[name] += 1
    return {
        "schema_version": "aic.scale_geometry_contract.v1",
        "status": "passed",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(),
        "device": "cpu",
        "cases": cases,
        "cases_by_geometry": checked,
        "seed": seed,
        "scale_one_matches_maximum_window": True,
        "center_drift_tolerance_px": 1.0,
        "observed_max_center_drift_px": max_center_drift,
        "aic_deployed_projection_bounds_checked": True,
        "transpose_roundtrip_checked": True,
        "official_aic_test_data_accessed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20261108)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run(args.cases, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
