# R78 direct deployed-box utility readout (2026-10-18)

## Research question

Can the temporal utility be trained against the IoU of the **actual V9 B3 output crop**, instead of using the best of the 129 R70 candidate windows? This isolates the utility/output mismatch found in the R70 reassessment without changing the V9 spatial box.

## Label contract

For every public native-crop row, `x` is the existing V9 vision-grid candidate feature tensor `[129,2305]` and `y[j]` is native GT IoU for legal candidate `j`. Let `j_B3 = argmax(B3_head(x))`, using the audited `student_head.pt` at `/root/autodl-tmp/evac/data/aic/experiments_910a/LFM_V9/vision_only_model/student_head.pt`. The target is exactly:

```text
y_deploy = y[j_B3]
```

This is a public native crop target for the deployed V9 box; no official test labels or official scores are used. Data are the R70 full shards (12,822 rows, 1,663 sources; `rv_train/live_train` versus source-disjoint `rv_dev/live_dev`).

## Results

The best practical input is the existing R70 topsoft 129-score vector plus the B3 output position (`j_B3/128`). A 50,561-parameter scalar MLP gives:

- dev MSE `0.04678`, MAE `0.16730`;
- row Spearman `0.47468`;
- source-macro Spearman `0.15947`;
- source-disjoint evaluation, 156 sources.

Using selected V9 features plus all 129 B3 logits gives lower source-robustness (`0.15610` source-macro Spearman) and 1.31M parameters. Source-balanced sampling and same-source pairwise loss did not improve the source-macro result (`0.12723` and `0.14866`).

## Deployment recipe

At each official R70 one-second keyframe:

1. read `ratios.t.utility.scores` (129 topsoft scores) from the existing utility dump;
2. derive the normalized B3 output offset from the V9 B3 box at that keyframe, as in R76;
3. feed `[scores[0:129], offset/(NC-1)]` to the 50k head;
4. interpolate the scalar utility on the original frame axis;
5. use it as the spatial factor in the R77 multiplicative temporal ranking while copying V9 B3 boxes unchanged.

The head adds about 50k parameters and keeps total inference parameters below 9B.

## Limits and stop conditions

This is a mechanism and deployment-feasibility result, not an official score result. Public native crop domains may not match the AIC test domain; source-macro correlation is modest; no same-video temporal GT exists to measure the final joint objective. Do not tune the R77 multiplier or package this head as an official improvement before a fixed independent gate. If the R70 official spatial package falls, close this transfer route; if it rises, use this output-aligned head as the next controlled joint candidate.

Remote artifacts remain under `/root/autodl-tmp/experiments/R78_DIRECT_OUTPUT_UTILITY/`; the `.pt` files are intentionally not copied into Git.

## Nested source check

To remove early-stopping leakage, the R70 training sources were split by source into 1,196 inner-train and 299 inner-tune sources; the 168-source R70 dev remained untouched. The same 50,561-parameter topsoft+position head reached external MSE `0.04825`, MAE `0.16949`, row Spearman `0.47334`, and source-macro Spearman `0.15850`. This preserves the mechanism signal with a genuinely held-out outer source set, although it remains a public spatial transfer result rather than a joint AIC score.
