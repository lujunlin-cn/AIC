# R82 direct deployed-box utility ensemble (2026-10-19)

## Decision question

Can source-level uncertainty reduce the domain risk of the direct deployed-box IoU head? The deployment target remains the native IoU of the audited V9/B3 crop selected at a public row. This experiment changes only the utility readout: five independently seeded 52,097-parameter heads are averaged, and the temporal factor uses `mean - 0.5 * ensemble_std`.

The beta value was fixed for this candidate after the ensemble hypothesis was defined; the external R70 development sources remain an evaluation set. Because the beta comparison was inspected on that external set after training, R82 is an exploratory high-risk candidate, not a confirmation result.

## Contract and provenance

- Public target: `y[argmax(B3_head(x))]`; no official labels, scores, or test-derived data were used.
- Feature: 129 R70 topsoft logits, exact public B3 argmax position, and max/top2-gap/std/mean/entropy/argmax summaries (136 dimensions).
- Inner/tune/external source split: 1,196 / 299 / 168 sources; the external set was untouched during head checkpoint selection.
- B3 checkpoint SHA256: `0def920daada215b8779b1808e1f60a6de437f10aa1ea2cf1521df39fbd3eda1`.
- R70 topsoft checkpoint SHA256: `80aa02bcf12db970dfb120e35021f594cacf91d685a479f524c9d729b7330253`.
- R70 topsoft logits SHA256: `6e1968e35a989554cac6fe9761ca4254adce069423c088c4ad2d4ffd885bdae2`.
- Public labels archive SHA256: `d4eb3337af56c1b65d288cd0fed81cc7ccec377f4013c7000dfa1a12f3ef54d9`.
- Official V9/B3 full-frame artifact SHA256: `95b9219676a5eece74fbce7bc108d08ba2b84759d0f754bf5660cad9a44c8e8a`.

## External result

The single nested R78 head had source-macro Spearman `0.16869`. The five-head ensemble gave:

| readout | MSE | row Spearman | source-macro Spearman |
|---|---:|---:|---:|
| ensemble mean | 0.04695 | 0.47742 | 0.16816 |
| mean − 0.25·std | 0.04742 | 0.47723 | 0.18073 |
| **mean − 0.5·std** | 0.04805 | 0.47678 | **0.19120** |

A paired 10,000-resample bootstrap over the 156 external sources gives the source-macro difference `(mean − .5·std) - mean` of `+0.02304`, 95% interval `[+0.00856,+0.03812]`. This supports a risk-sensitive domain effect on this public transfer set; it does not identify the official AIC gain.

## Candidate package

[R82_DIRECT_OUTPUT_UTILITY_ENSEMBLE_RISK05_MULT_TIME_R54_V9_B3](../../submissions/R82_DIRECT_OUTPUT_UTILITY_ENSEMBLE_RISK05_MULT_TIME_R54_V9_B3) contains 426/426 videos and 199,037 predictions. It copies the R54/V9 B3 boxes and changes only the temporal frame ranking:

```text
rank(R54 temporal value)
* (0.5 + 0.5 * midrank(mean_direct_utility - 0.5 * ensemble_std))
```

- Total parameters: `8,856,303,335` (`8.856303335B`), within the 9B limit.
- Five direct heads add `260,485` parameters; all are counted.
- Relative to R54: 2,294 frame swaps; 191 videos differ from R81, with mean R81/R82 frame overlap `0.99624`.
- ZIP SHA256: `de99b3c5fcaee15966d893425e1b39a896645546092a4d00dc00133bc31529fd`.
- Prediction SHA256: `b186208540716b2465dd2c6766f12365e8c0aa2d8ccc39c61883d12f0becb6f4`.
- Independent checker: 426 videos, 199,037 predictions, valid; ZIP roundtrip passed; not uploaded.

## Interpretation and next action

The ensemble reduces source-transfer error only after applying a conservative uncertainty penalty. That is consistent with a shift in calibration between public crop sources and AIC videos, but it could also be a finite-split artifact. R82 is therefore the next single high-risk verification candidate if a submission slot is available. Keep R54 (36.22 official) as the safety anchor and do not treat the public result as an official score prediction. If R82 falls, close the direct-utility temporal branch; if it rises, retain the ensemble and investigate whether the beta can be fixed by a second source-held-out split before making more candidates.
