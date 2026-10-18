# R59 joint spatial utility candidate (2026-10-18)

R59 is a deliberately high-risk deployment candidate.  It starts from the
R54 TimeLens2 centre residual and adds a frame utility derived from the
disagreement between the V9 B3 crop and the audited R26 B3 crop on the same
original frame.  A frame receives

```
rank(center) + 0.20 * rank(TimeLens2 interval) + 1.0 * rank(V9/R26 geometry disagreement)
```

where ranks are source-video percentile ranks.  The coefficient `1.0` was
fixed before packaging; there was no coefficient sweep.  The selected frame's
V9 box is emitted unchanged.  The package keeps exactly `round(0.85*N)`
original frame IDs per video.

The mechanism hypothesis is that a large spatial readout change marks a
content/state transition where the crop model has useful information, and
that those frames are more valuable than the static frames discarded by the
centre window.  The competing explanation is that V9/R26 disagreement is
mostly camera motion, shot boundary, or model instability; the official score
could therefore fall sharply.  No official labels or scores were used.

## Artifact and checks

Package: `submissions/R59_JOINT_V9_DISAGREEMENT_G1/`

* 426 videos, 199,037 predictions, exact per-video 85% budget.
* No duplicate or out-of-range frame IDs; predictions sorted by original frame.
* ZIP contains only `predictions.jsonl`; extraction hash matches source.
* Total declared inference parameters: `8,854,531,425` (`8.855B`), below the
  9B limit: TimeLens2 `8,767,123,696` + V9 vision-only B3 `87,407,729`.
* Prediction SHA256:
  `1d0f4c84d2dfad3e9f27e4bb70dc6c8d9b231bc010a1362b6838eaf10b635770`
* ZIP SHA256:
  `d8c99c1ea4e3d51a808ef341253e080ed813c44995f7543c9c9a815458907566`

Relative to R54, R59 changes 12,880 frame swaps (all 426 videos), with 40
videos becoming non-contiguous.  Mean selected normalized position moves from
R54 `.52344` to R59 `.56746`; landscape and portrait swaps are similar
(29.88 and 31.10 per video respectively).  Relative to R26 it changes 17,937
swaps.  This is a materially different policy rather than a renamed R54.

## Interpretation and upload guidance

This is a joint time-space utility test, not an isolated spatial score test:
R59 changes both selected frames and the inherited V9 boxes.  A positive score
would support using spatial disagreement as a temporal acquisition signal; a
large negative score would reject this aggressive coupling while leaving R54,
V9-center, and fixed-late controls interpretable.  It is suitable for one
high-risk submission slot after the safer R26/R54/R47 controls.
