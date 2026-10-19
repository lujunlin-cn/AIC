# R105 AViMoS saliency as AIC crop utility (2026-10-20)

## Protocol

The R104 visual decoder produced four per-candidate features on the R70 native crop rows: saliency mass inside the legal window, centroid distance, within-window entropy and window area. The R70 current-frame candidate representation and native crop labels were retained. A 168-source held-out evaluation compared identical utility heads with zero features, AViMoS features, and source-permuted AViMoS features.

## Result

| variant | held-out source macro IoU |
|---|---:|
| control | 0.5747144 |
| AViMoS features | **0.5782359** |
| source-permuted features | 0.5734110 |

The actual saliency feature delta is +0.0035215 over control; permutation removes the gain. This is the strongest of the three requested route probes, but a source-split blend calibration exposed overfitting: a simple saliency rank blend reached 0.5885 on its tuning half but only 0.5536 on the held-out half, below the baseline holdout 0.5630.

## Decision

Keep AViMoS as a research candidate, but do not make a blind official package from the overfit blend. The positive full-head result supports a fresh source-level confirmation or a target-compatible crop head; the blend result warns that the current 168-source public split can select unstable saliency coefficients. The three direct submission packages remain the prevalidated R70, R71 and R89 candidates until a deployment-path inference is completed.

Evidence: `metrics.json`, `blend_metrics.json`; code `scripts/r105_avimos_aic_probe.py`, `scripts/r105_train_cached.py`, `scripts/r105_calibrate_blend.py`.
