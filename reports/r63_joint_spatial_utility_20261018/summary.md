# R63 corrected joint spatial utility candidate (2026-10-18)

R63 is the final high-disagreement arm.  It uses the verified per-video FPS
and the exact R54 centre expression, then ranks frames by

```
rank(center) + 0.20 * rank(TimeLens2 interval) + 1.0 * rank(V9/R26 geometry disagreement)
```

The exact centre expression was kept because algebraically equivalent floating
forms can change a boundary tie.  A gamma-zero reproduction matches R54 on all
426 frame sets.

Package: `submissions/R63_JOINT_V9_DISAGREEMENT_G1/`

* 426 videos, 199,037 predictions; exact 85% budget, sorted unique frames.
* Relative to R54: 12,572 swaps (425 videos), 36 non-contiguous videos.
* Mean normalized selected position `.56669` (R54 `.52344`).
* Prediction SHA256: `5432d76ba12225ad95529f9a7a6dcc14726ea9dc49e6cbf9ee62ca2c1f5aa476`.
* ZIP SHA256: `463c9c2c7d7c70137aa4da44ec14768816a89353d5fb3a44dcd65d5bb699f556`.
* Total declared parameters `8,854,531,425` (8.855B).

R59/R61 drafts are superseded because they used an index without FPS; R63 is
the uploadable artifact.

**Status correction (2026-10-18): INVALID.** The geometry rank in this
artifact is a stable ordinal rank. Most geometry deltas are tied, so it adds a
frame-index position prior and the package is not a valid test of spatial
disagreement. Do not upload it; use the tie-aware R67 rebuild instead.
