# R62 corrected joint spatial stability control (2026-10-18)

R62 is the opposite, corrected mechanism control to R61.  It uses verified
FPS and the utility

```
rank(center) + 0.20 * rank(TimeLens2 interval) - 1.0 * rank(V9/R26 geometry disagreement)
```

It favours spatially stable frames.  The first R60 draft used an index without
FPS and is invalid; R62 is the only package to consider.

Package: `submissions/R62_JOINT_V9_STABILITY_GM1/`

* 426 videos, 199,037 predictions, exact 85% budget and sorted unique original
  frames.
* Relative to R54: 22,742 swaps (all 426 videos), 107 non-contiguous videos.
* Mean normalized selected position `.44303` (R54 `.52344`).
* Prediction SHA256:
  `b461d5bacefb8a0f1aa51b31e4c73120d5af01678311390c89ca988340fb125c`.
* ZIP SHA256:
  `cebcf0da9ad92a5bb2af85523eb46e5f4f74d79040e7475df63d50e35328c480`.
* Total declared parameters `8,854,531,425` (8.855B).

R61 and R62 distinguish whether spatial disagreement is a content transition
signal or a crop instability signal.  They are high-risk controls and should
replace, rather than supplement, the invalid R59/R60 drafts.
