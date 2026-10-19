# R102 AIKU temporal transfer pilot (2026-10-20)

## Question

Does public AIKU summary supervision teach a reusable temporal readout that improves a target-domain temporal model, after the target input projection is reinitialized? This is a transfer probe, not a deployment claim: AIKU supplies 1024D pre-extracted features without source video or a feature extractor.

## Protocol

- AIKU full asset: zip SHA256 `0e179afc43461b92ff4719f2e719c62464320164da772eaf81f0f1923d769272`.
- Pilot used the first 600 of 6,000 AIKU groups (`train600.h5`, SHA256 `444d3f66e0c003166c76fbfe30a56377859794c7428e398a4377d7a995bd415d`) because the remote SSH link throttled multi-gigabyte transfer. The full H5 remains local and was not modified.
- AIKU model: 1024→64 projection, dilated temporal convolutions, continuous `gtscore` regression plus `gtsummary` auxiliary loss; 3 epochs, 480/120 video split.
- Target: R92 PHD2/GIF RGB features, first of four views, 5,886 fragments and 448 held-out source videos. Input projection was reinitialized; temporal trunk and readout were transferred.
- Target-only and transfer used the same seed, optimizer and 800 target steps. Audio and official test labels were excluded.

## Result

| target model | held-out source macro fixed-85% F |
|---|---:|
| target-only | 0.415304 |
| AIKU temporal transfer | 0.418717 |
| delta | **+0.003413** |

The transfer model's training source macro was 0.410951 versus 0.411860 for target-only. A 20,000-replicate source bootstrap over the 448 paired held-out sources gave a 95% interval `[+0.00080,+0.00610]`; the gain is therefore not explained by fitting the target sources more closely in this pilot.

## Decision

AIKU provides a weak positive transfer signal and remains a candidate for a proper target-domain temporal student. It is not yet a competition package: the source feature contract is unavailable at official inference, the pilot uses 10% of AIKU groups, and the target feature model is not the complete R54 deployment path. A next experiment must use a target-compatible student and source-level confirmation before packaging.

Evidence: `metrics.json`; code `scripts/r102_aiku_transfer_probe.py`.
