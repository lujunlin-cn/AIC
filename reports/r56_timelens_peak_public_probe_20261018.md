# TimeLens2 peak prompt smoke (2026-10-18)

A second, structurally different prompt was run on all 12 public annotation clips using TimeLens2-8B, 2 fps and 224-to-224 pixel budget reduced to 50,176 pixels/frame to fit the RTX 4080. It completed with no inference error and produced `reports/r56_timelens_peak_smoke_20261018.json`.

The prompt asked for narrow intervals around decisive outcomes. On the same 18 public views, binary interval decoding gives macro `0.720168` (`+0.015752` over center), below the cached generic prompt (`0.721327`, `+0.016911`) and well below R54 residual (`0.729414`, `+0.024997`). Only 4/18 views beat center. The prompt variant is rejected for deployment; the result supports retaining the generic interval prompt and changing its decoder instead.

Remote smoke command used `/root/autodl-tmp/aha_env/bin/python`, TimeLens2-8B, `CUDA_VISIBLE_DEVICES=0`, `max_pixels=50176`; no official data or labels were used.
