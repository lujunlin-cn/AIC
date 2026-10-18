# B3 original asset recovery audit (2026-10-18)

## Requested asset

The historical V8 B3 checkpoint is identified by the frozen config as:

`/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt`

The associated runtime also needs the LFM2.5-VL-450M vision tower. The archived B3 head identity prefix recorded in `reports/r8/source_and_label_semantics.json` is `a6e7fc4bac697614`.

## Inventory evidence

The path is absent on the current AutoDL clone at all relevant locations:

- `/root/autodl-tmp/evac/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt`
- `/root/autodl-tmp/experiments/LFM_V8/arms/B3_s0/head_s.pt`
- `/root/AIC/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt`
- `/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt`

The bounded inventory found 67 candidate `.pt` files in the 4–12 MB range. Shape and state-dict inspection found only V7 and V9 A0/A1/KD variants with the `2305 -> 129` head contract; no file matched the B3 identity. The available V7 heads are explicitly different training arms. The archived V8 B3 submission ZIP contains only `predictions.jsonl`, so it cannot recover weights, logits, or features. Git history/branches contain config and code only. Historical server endpoints were unreachable during this audit.

## Consequence

The original B3 checkpoint cannot be truthfully restored from the current workspace. A retrained head from public data would be a new checkpoint and must be named as such. V7 must not be relabeled B3. Exact B3 spatial replay remains available only from the cached R26/R28 predictions.

A reconstruction attempt would require rebuilding the V8 multi-geometry training pool: native RV features, rotated RV features, LIVE-YT-VC features, manifests, and the exact training seed/selection state. Even if successful, byte or prediction equivalence would need a separate comparison. The original B3 plus TimeLens2 combination is `9,217,407,073` parameters and violates the current `<=9B` rule; the recovered B3 is therefore useful for a small temporal model or audit, not for a TimeLens2+B3 submission.

## Status

- exact original B3 head: **not recovered**
- exact LFM base: **not recovered**
- cached B3 predictions: available and replay-audited
- V9 vision-only replacement: available and used by R54/R70
- reconstruction from public data: possible as a new experiment, not an original-asset recovery
