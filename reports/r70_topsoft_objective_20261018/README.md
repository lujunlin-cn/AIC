# R70 topsoft objective and utility driven temporal selection (2026-10-18)

## Spatial objective

The original R70 readout was trained with per-candidate MSE. Four fixed objective variants were trained on the same public RetargetVid/LIVE source split, same 2305-dimensional V9 feature contract, seed, batch and 1200 steps: MSE, pairwise, listwise expected-IoU, and topsoft candidate weighting. No official test labels were used.

The source-macro deltas over the center candidate were:

| objective | RV dev | LIVE dev | mean |
|---|---:|---:|---:|
| MSE | +0.06729 | +0.07073 | +0.06901 |
| pairwise | +0.06884 | +0.06390 | +0.06637 |
| listwise | +0.06987 | +0.06731 | +0.06859 |
| topsoft | +0.07121 | +0.06892 | +0.07006 |

Topsoft is a small, consistent source-level improvement over the MSE variant on the fixed split, so it was used for one official inference package. This is public source evidence, not official score evidence.

The topsoft package is `submissions/R70_TOPSOFT_SPATIAL_R54TIME/`: 426/426 videos, 199037 predictions, 101797 changed retained boxes, 8.854531425B parameters, zero point fallback. ZIP SHA is `da5fcc5dccbfcd960b112852295dbbba35be89aaad61c20fcbb81d1ba8fed4d2`. Independent checker, final contract and ZIP roundtrip pass.

## Utility driven temporal decision

The topsoft head's per-keyframe top candidate score was interpolated on the original frame axis. A fixed parameter-free joint rule was applied:

```text
T = rank(center + 0.20 * (rank(interval_inside) - 0.5))
U = rank(interpolated R70 top candidate score)
J = T * (0.5 + 0.5 * U)
```

The exact `round(.85*N)` budget and full V9 spatial boxes were retained. The resulting package `submissions/R70_TOPSOFT_UTILITY_TEMPORAL/` has 426/426 videos, 199037 predictions, mean 5.122 swaps per video relative to R54, 377 changed videos, and 8.854531425B parameters. ZIP SHA is `aadb42cadc97a9880b63cc7367ea5285d53a365ea4802040a934131e1440883f2`. All local checks pass; official score is unknown.

This is a mechanism probe rather than a score claim. It tests whether predicted cropability should enter the temporal decision as an expected utility factor. If it falls, the likely failure is public-to-official spatial utility transfer or correlation between temporal value and cropability; if it rises, the next step is to keep the joint objective and calibrate only the utility observation, not scan arbitrary weights.

Remote source artifacts:

- `/root/autodl-tmp/experiments/R70_OBJECTIVE_VARIANTS/metrics.json`
- `/root/autodl-tmp/experiments/R70_OFFICIAL_UTILITY_DUMP_TOPSOFT_20261018/`
