# R47 V9 vision-only spatial candidate

## Decision question

Does the already audited V9 vision-only B3 crop runtime produce a different
official result from the R26 cached B3 boxes when the exact R26 centre-85 frame
mask is held fixed? This is a spatial candidate, not a temporal parameter scan.

## Construction

- Source: remote `LFM_V9_B3_VISIONONLY_SEMIFINAL/predictions.jsonl`, SHA recorded
  in the candidate manifest.
- Temporal mask: `K=round(.85*N)`, `start=(N-K)//2`, copied from R26.
- Spatial boxes: V9 full-frame predictions, selected framewise without changes.
- Components: SigLIP2 NaFlex vision tower (85,843,200), V8 B3 utility head
  (1,511,425), YuNet (53,104), total **87,407,729** parameters.
- No official labels, official score, test pseudo-labels, or score-derived
  data were used.

## Audit

The package has 426/426 videos and 199,037 predictions. The independent
submission checker passes (`model_size_mb=87.407729`), and ZIP extraction is
byte-identical. The selected frame IDs are exactly the R26 centre mask. V9
boxes differ from the R26 artifact on 37,499 of 234,159 full-frame boxes,
which makes this a real spatial intervention rather than a renamed copy.

Candidate: `submissions/R47_V9_B3_CENTER85/`. It is packaged but not uploaded;
the official score is unknown.
