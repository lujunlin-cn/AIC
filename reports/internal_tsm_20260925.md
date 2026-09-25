# Canonical internal TSM probe (2026-09-25)

This experiment is a temporal proxy study only. It does not report AIC
`F_video`: the TVSum data has no composition crop ground truth.

The historical A1 implementation shifts the final `[T,D]` embedding cache.
This probe uses a separate encoder/cache: ResNet18 stem + `layer1` is run on a
sampled frame sequence, `temporal_shift_feature_map` (fold divisor 8) shifts
the intermediate `[T,64,H,W]` map, and `layer2`--`layer4` plus global average
pooling produce the 512-D features. One frame of context is included on each
chunk boundary. Cache metadata is
`canonical_internal_tsm_layer1_fold8_v1`.

The chunked encoder agrees with a full-sequence reference on `XzYM3PfTM4w`
(222 sampled frames): maximum absolute feature error `1.81e-5`, mean absolute
error `1.59e-6`. Removing the shift changes features substantially (mean
absolute difference `0.08127`). The operation adds zero parameters. The
measured GPU probe was 0.2246 s/batch without the shift versus 0.5240 s/batch
with the chunked implementation; this includes chunk boundary overhead and is
not a final production latency benchmark.

The same 27 train / 16 development videos, proxy-v2 labels, Temporal U-Net,
seed, optimizer and 20-epoch budget were used for the full control:

| comparison | dev macro F1 | threshold |
| --- | ---: | ---: |
| A0 control | 0.14185 | 0.45 |
| internal TSM | 0.14843 | 0.40 |

This single split is only a probe. Five source-group folds (43 non-lockbox
videos) provide the stability check. With one global threshold of 0.30,
A0 is `0.14631 ± 0.03251` and internal TSM is `0.13800 ± 0.06299` (mean ±
sample standard deviation). Fold paired differences (internal minus A0) are
`[-0.03609, +0.04901, +0.01742, -0.01249, -0.05937]`.

Using each fold's exploratory development threshold gives internal TSM
`0.14474 ± 0.07039` at threshold 0.35 and A0 `0.14631 ± 0.03251` at threshold
0.30. The thresholds are not lockbox choices; they are shown only to expose
calibration sensitivity.

Finally, models trained on the 27-video training partition were evaluated
once on the seven-video frozen local lockbox. At thresholds chosen before the
lockbox, internal TSM scores `0.17455` (0.30) / `0.17514` (0.35), while the A0
control scores `0.21113` (0.30) / `0.21068` (0.35). The lockbox result does not
support replacing the ResNet18 fallback.

For an additional comparison against the strongest previously retained A0
checkpoint (`A0_006`, trained with the project trainer rather than the small
probe trainer), lockbox macro F1 is `0.18591` at threshold 0.30 and `0.18203`
at 0.35. Internal TSM remains lower (`0.17455`/`0.17514`). This comparison is
reported as a separate checkpoint audit because its optimizer/training loop is
not identical to the controlled fold trainer.

Artifacts:

- `reports/internal_tsm_correctness_20260925.json`
- `reports/internal_tsm_folds_summary_20260925.json`
- `reports/a0_control_folds_summary_20260925.json`
- `reports/internal_tsm_lockbox_eval_20260925.json`
- `scripts/extract_internal_tsm_features.py`

Decision: keep the implementation and cache for future variants, but lower
canonical internal TSM priority. It has no stable temporal-proxy gain, larger
fold variance, and no evidence of improving the frozen lockbox.
