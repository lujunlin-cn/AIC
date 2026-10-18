# R64 corrected joint spatial stability control (2026-10-18)

R64 is the opposite control to R63.  It uses verified FPS and the utility

```
rank(center) + 0.20 * rank(TimeLens2 interval) - 1.0 * rank(V9/R26 geometry disagreement)
```

It favours spatially stable frames.  A gamma-zero reproduction matches R54 on
all 426 frame sets, so the change is attributable to the spatial term.

Package: `submissions/R64_JOINT_V9_STABILITY_GM1/`

* 426 videos, 199,037 predictions; exact 85% budget, sorted unique frames.
* Relative to R54: 22,741 swaps (all 426 videos), 107 non-contiguous videos.
* Mean normalized selected position `.44301` (R54 `.52344`).
* Prediction SHA256: `60ba42b8dc24725f843c69a65c03c602b66f4fb08c186af7a3550521898eaca4`.
* ZIP SHA256: `84320541b0250384613124133c39c478686837b83f35f7780ebe4734f8658216`.
* Total declared parameters `8,854,531,425` (8.855B).

R60/R62 drafts are superseded by this exact-centre corrected package.
