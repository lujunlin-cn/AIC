# R70 official inference and package (2026-10-18)

## Question

The first R70 official run showed high CPU and near-zero GPU utilization. The original path decoded/preprocessed images on CPU, copied each vision grid back to CPU, built 129 candidate masks in a Python loop per frame, and invoked the utility head one frame at a time. This was a throughput implementation failure, not evidence that the vision tower was unnecessary.

## Fix and verification

Code commits:

- `50b116f`: cache candidate masks, batch candidate pooling and utility head on GPU.
- `a7c9e6d`: bounded one-batch CPU prefetch overlapping decode/processor with CUDA.
- `b75982f`: cast processor pixel tensors to FP16 before host-to-device transfer and use `inference_mode`.
- `bb70411`, `3a309ad`: package builder accepts the flat official point output directory.

The batched pooling helper was compared with the original `feature_from_grid` on landscape, portrait and square random grids: maximum absolute difference was `0.0`. Batched and per-frame head outputs also matched in the numerical check.

The 10-video profile with batch 32 completed in `2.7 s`, with no fallback. The full run used one process, `--batch 32 --prefetch-workers 1`, and completed in `174.3 s` after resuming 31 already completed videos. GPU samples reached `97–100%` during active batches; four concurrent CPU-bound processes were not used.

## Official inference result

- input: frozen official keyframe point metadata and R54 parent time predictions
- videos: `426/426`
- keyframes: `8,848/8,848`
- fallback points: `0`
- status: all `ok_r70`
- parent predictions: `199,037`
- boxes changed on retained frames: `102,981`
- official labels or score feedback read: `false`

Remote output: `/root/autodl-tmp/experiments/R70_OFFICIAL_POINTS_BATCH32/`.

## Package

Local package: [`submissions/R70_SPATIAL_UTILITY_R54TIME_BATCH32/`](../../submissions/R70_SPATIAL_UTILITY_R54TIME_BATCH32/)

- parent time set: `R54_TIMELENS_CENTER_RESIDUAL_V9_B3`
- only spatial boxes changed; exact R54 frame set and budget retained
- parameters: `8,854,531,425` (`8.854531425B`), within the `<=9B` rule
- predictions SHA256: `5d34388eb39a2d8f49ba9cd29420cbb5c2e450f2c88ec2aee16be11cbeeff3da`
- ZIP SHA256: `a5d3031cb6ab896d73e83e032be0e385bdb17efa3c0024af06036dcc63179b7a`
- local contract: valid, `426/426`, `199,037`
- independent checker: valid, `426/426`, `199,037`
- ZIP contents: exactly `predictions.jsonl`, round-trip SHA matches
- official upload: not performed

## Interpretation

This run answers the resource bottleneck: the prior GPU idling was caused by CPU preprocessing and per-frame CPU readout work. It does not establish an official score improvement. R70 still tests the public native-crop readout hypothesis; the official score is the next required measurement.
