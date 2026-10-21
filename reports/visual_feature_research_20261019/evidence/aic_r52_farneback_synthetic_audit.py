#!/usr/bin/env python3
"""R52 numerical audit on one synthetic translation; no real videos/labels."""
import hashlib
import json
import platform
from pathlib import Path

import cv2
import numpy as np


def main():
    seed = 20261019
    rng = np.random.default_rng(seed)
    height, width = 90, 160
    original = rng.integers(0, 256, size=(height, width), dtype=np.uint8)
    original = cv2.GaussianBlur(original, (5, 5), 0.9)
    dx, dy = 3.0, 2.0
    affine = np.float32([[1, 0, dx], [0, 1, dy]])
    shifted = cv2.warpAffine(original, affine, (width, height),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    # Match the Farneback arguments in AIC scripts/r52_optical_flow_probe.py.
    kwargs = dict(pyr_scale=0.5, levels=2, winsize=15, iterations=2,
                  poly_n=5, poly_sigma=1.2, flags=0)
    margin = 16
    results = []
    for name, scale, dtype in (("uint8_0_255", 1.0, np.uint8),
                               ("float32_0_255", 1.0, np.float32),
                               ("float32_0_1_R52", 1 / 255.0, np.float32)):
        first = (original.astype(dtype) * scale).astype(dtype)
        second = (shifted.astype(dtype) * scale).astype(dtype)
        flow = cv2.calcOpticalFlowFarneback(first, second, None, **kwargs)
        inner = flow[margin:-margin, margin:-margin]
        error = np.linalg.norm(inner - np.array([dx, dy], np.float32), axis=-1)
        magnitude = np.linalg.norm(inner, axis=-1)
        results.append({"input": name, "dtype": str(first.dtype),
                        "input_min_max": [float(first.min()), float(first.max())],
                        "median_flow_xy": np.median(inner, axis=(0, 1)).astype(float).tolist(),
                        "mean_flow_xy": inner.mean(axis=(0, 1)).astype(float).tolist(),
                        "median_magnitude": float(np.median(magnitude)),
                        "mean_epe": float(error.mean()),
                        "median_epe": float(np.median(error)),
                        "p95_epe": float(np.quantile(error, .95)),
                        "fraction_epe_le_0_1": float(np.mean(error <= .1)),
                        "fraction_magnitude_gt_1": float(np.mean(magnitude > 1)),
                        "finite": bool(np.isfinite(flow).all())})
    out = {"experiment": "R52_FARNEBACK_SYNTHETIC_INPUT_SCALE_AUDIT",
           "scope": "Numerical input validity only; no AIC gain or real-data reproduction claim",
           "opencv_version": cv2.__version__, "numpy_version": np.__version__,
           "python": platform.python_version(), "platform": platform.platform(),
           "seed": seed, "image_hw": [height, width], "expected_flow_xy": [dx, dy],
           "expected_magnitude": float(np.hypot(dx, dy)), "interior_margin_pixels": margin,
           "interior_pixels": (height - 2 * margin) * (width - 2 * margin),
           "farneback_kwargs": kwargs,
           "texture_sha256": hashlib.sha256(original.tobytes()).hexdigest(),
           "shifted_sha256": hashlib.sha256(shifted.tobytes()).hexdigest(),
           "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
           "results": results}
    path = Path('/tmp/aic_r52_farneback_synthetic_audit.json')
    path.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps(out, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
