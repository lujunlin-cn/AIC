# R61 corrected joint spatial utility candidate (2026-10-18)

R61 is the corrected, high-risk version of the joint decoder.  It uses the
verified per-video FPS from `r47_timelens2_index_enriched_426.jsonl`, then
ranks each frame by

```
rank(center) + 0.20 * rank(TimeLens2 interval) + 1.0 * rank(V9/R26 geometry disagreement)
```

The first R59 draft used an index without FPS and is invalid; R61 is the only
package to consider.  All other inputs and the mechanism are unchanged.

Package: `submissions/R61_JOINT_V9_DISAGREEMENT_G1/`

* 426 videos, 199,037 predictions, exact 85% budget and sorted unique original
  frames.
* Relative to R54: 12,576 swaps (425 videos), 33 non-contiguous videos.
* Mean normalized selected position `.56671` (R54 `.52344`).
* Prediction SHA256:
  `c203b5a968d7282b2b66ea543d6974ee52c7585b73202834bf0ea07deabb9b0c`.
* ZIP SHA256:
  `ba5e28ef56df49c1ed1a94b2c25493b3eea40de43a359dc3c8c18c75fe2e8b90`.
* Total declared parameters `8,854,531,425` (8.855B).

The hypothesis is that frames where the crop readout changes strongly mark a
useful content transition.  The competing stability hypothesis is R62.
