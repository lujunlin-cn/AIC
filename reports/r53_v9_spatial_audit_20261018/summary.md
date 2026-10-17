# R53 V9 spatial intervention audit (2026-10-18)

## Decision question

Is the packaged `R47_V9_B3_CENTER85` candidate a materially different spatial
policy from the known R26 B3 parent, or only a renamed copy?  The audit compares
the full-frame R26 B3 artifact with the full-frame V9 vision-only artifact and
then restricts the statistic to the exact R26 centre-85 frame mask.

## Result

The audit is geometry-only and uses no AIC labels.  On all 234,159 frames the
mean box IoU between R26 and V9 is approximately `0.99804`, with about 9.68%
of frames changed.  On the 199,037 frames selected by the R26 exact centre-85
mask the mean is approximately `0.99801`, with about 9.95% changed.  The
intervention is larger for portrait targets (about 18% changed, mean IoU
approximately `0.99649`) than landscape targets (about 7% changed, mean IoU
approximately `0.99856`).

These statistics establish that V9 is a real, small spatial intervention. They
do not establish that V9 boxes have higher native IoU, because no AIC native
crop GT is available. The public RetargetVid group experiment already supplies
the relevant negative control: aggregating multiple detected faces reduced
portrait IoU and never beat the single-face path on 20 exposed sources.

## Action

Do not create another unlabeled box-jitter, EMA, or group heuristic package.
`R47_V9_B3_CENTER85` remains the only justified spatial isolation upload. Its
paired TimeLens2/V9 package tests a different temporal mechanism jointly. If
V9 scores below R26, stop this spatial replacement family; if it improves,
inspect portrait/landscape and disagreement strata before changing geometry.

Reproduce with:

```bash
python scripts/r53_v9_spatial_audit.py \
  --r26 artifacts/r26/b3_allframes.jsonl \
  --v9 reports/r47_v9_parent_full_20261017.jsonl \
  --out reports/r53_v9_spatial_audit_20261018/metrics.json
```
