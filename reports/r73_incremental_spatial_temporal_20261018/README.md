# R73 incremental spatial utility temporal residual (2026-10-18)

## Question

R72 used the absolute predicted best-crop score (`max_score`) as a temporal residual. That is not the marginal spatial value of a frame: a frame can have a high absolute crop score while its center crop is already equally good. The public native-crop objective therefore predicts two quantities per keyframe:

```text
absolute utility: max_score
marginal utility: max_score - center_score
```

The second quantity is aligned with the intervention made by the V9/B3 spatial branch: improvement over the legal center window.

## Public calibration check

On the held-out public native crop rows used by R70 (`1546` frame/ratio rows; no official test data), the topsoft readout was compared to native candidate IoU:

| predicted signal | target | Spearman | Pearson |
|---|---|---:|---:|
| `max_score` | `max(native IoU)` | 0.0761 | 0.0960 |
| `center_score` | `center(native IoU)` | 0.4485 | 0.4486 |
| `max_score-center_score` | `max(native IoU)-center(native IoU)` | 0.4092 | 0.4773 |

Among the top 10% rows by predicted signal, the actual marginal IoU was `0.5449` for the incremental signal versus `0.2210` for absolute `max_score`. This is a row-level calibration diagnostic, not a joint temporal score estimate.

## Decoder

The exact R54 temporal score and `round(.85*N)` budget are reconstructed and checked. Only the temporal ranking changes:

```text
S_R54 = rank(center) + .20 * (rank(interval_inside) - .5)
S_R73 = S_R54 + .20 * (rank(max_score - center_score) - .5)
```

The full V9 parent boxes are copied byte-for-byte. Official labels and scores are not read.

## Package

`submissions/R73_INCREMENTAL_SPATIAL_UTILITY_TIME_R54_V9_B3/`

- 426/426 videos, 199037 predictions
- exact 85% budget
- 5004 frame swaps versus R54, 399 videos changed
- total parameters `8,854,531,425` (`8.854B`)
- independent checker and local contract valid
- ZIP SHA256: `ae70c89199b7a43a7e4c9f6798952ea6b44400a0e68fbe946283018047edbb26`

## Limits and decision

This is a high-risk candidate. The calibration check supports using marginal crop gain instead of absolute cropability, but there is no same-video temporal+spatial public GT and no official score. A submission result would test transfer of this decoder mechanism; it must not be described as a measured gain before official feedback. The tie-aware signed/ clipped versions are identical because the 129-candidate set includes the center candidate, so `max_score-center_score >= 0` by construction.
