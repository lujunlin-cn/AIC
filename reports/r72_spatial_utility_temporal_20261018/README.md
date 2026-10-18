# R72 spatial-utility temporal residual (2026-10-18)

## Hypothesis

R70's native-crop readout predicts a frame-level maximum legal-window utility. A
frame with a reliably high maximum utility is more likely to contribute a useful
scored crop when retained. The test is whether this signal adds information to
the frozen R54 temporal ranking.

## Frozen intervention

For every official video, reconstruct the exact R54 score:

```text
R54 = rank(centre-of-video) + 0.20 * (rank(TimeLens interval membership) - 0.5)
```

The R70 head is run on the frozen 1-second keyframes. Its `max_score` over 129
legal crop windows is linearly interpolated to the original frame axis and
ranked within each video. The candidate uses one predeclared residual:

```text
R72 = R54 + 0.20 * (rank(R70 max crop utility) - 0.5)
```

The 0.20 coefficient equals the existing R54 interval residual and was fixed
before reading the official prediction differences. The exact 85% budget and
all V9 B3 boxes are retained; only the selected frame set changes.

## Evidence and limits

- R70 utility dump: 426/426 videos, 8,848/8,848 keyframes, 0 fallback.
- Full 129 candidate logits are saved per keyframe in the remote experiment
  dump and the local diagnostic copy; frame summaries include max score, center
  score, top-two gap, entropy, and standard deviation.
- R72 changes 4,120 frame selections in total (mean 9.67 per video, median 6),
  across 393 videos; mean overlap with R54 is 97.94%.
- The selected R54 frames have only a small higher mean utility than frames
  rejected by R54 (+0.0086 in the uncalibrated head output), so the intervention
  is a high-risk probe rather than a confirmed gain.

No official labels, score feedback, or test pseudo-labels were used. The public
native-crop utility validation does not identify the joint temporal/crop score;
therefore this package is an official measurement candidate, not a local score
claim.

## Artifact

`submissions/R72_SPATIAL_UTILITY_TIME_R54_V9_B3/` contains the exact-budget ZIP,
manifest, prediction SHA and local contract checks. Total declared inference
parameters are `8,854,531,425` (`8.854531425B`).
