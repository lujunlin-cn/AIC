# VideoMAEv2 / InternVideo2 follow-up (2026-09-26)

## Scope

This is a bounded representation probe on the local QVHighlights-derived weak-label 40 train / 10 validation subset. It excludes the official evaluation set, uses no test labels, and has `official_aic_gt=false`. The 40/10 manifests have zero shared original YouTube source IDs. The full 800/89 local manifest requires `QVH_SOURCE_PURGE_V1` (775/89 after removing 25 train clips from 23 shared original sources).

## VideoMAEv2 corrected probe

`VIDEOMAE_QVH_PROBE_V2_20260926` uses the model-declared ImageNet normalization, 2 FPS sampling, 16-frame clips, stride 4, mean weak timeline targets, timestamps in every cache, and identical clip-mean controls for the existing representations. VideoMAEv2-Base loaded with 86,227,200 parameters and source weight 344,924,592 B. Extraction took 325.48 s and peak allocated VRAM was 2,552,064,000 B.

| frozen representation + same temporal head | best weak val F1@0.5 | Spearman | empty rate |
|---|---:|---:|---:|
| VideoMAEv2-Base | 0.54444 | 0.41459 | 0.30 |
| DeiT-S clip mean | 0.40000 | 0.38757 | 0.90 |
| ResNet18 clip mean | 0.33571 | 0.42129 | 0.00 |
| ViT-B clip mean | 0.46667 | 0.42070 | 0.70 |

This is a weak-label development comparison with ten videos and checkpoint selection on the same validation set. Paired VideoMAE minus DeiT F1 is +0.1444 with a 95% bootstrap interval [-0.1557, +0.3889]; Spearman is +0.0270 with interval [-0.0030, +0.0765]. The apparent F1 advantage is therefore uncertain and may be calibration/label interaction. The train-mean timeline baseline reaches Spearman 0.55405, higher than all fitted representations, showing that this protocol is not a sufficient semantic benchmark. VideoMAEv2 is not a submission candidate from this result.

The first VideoMAE run used 0.5 normalization and an incomplete cache schema; it is retained as `VIDEOMAE_QVH_PROBE_20260926_failed_shape`/the prior directory and excluded from all conclusions.

## InternVideo2 compatibility

`InternVideo2 Stage1-1B-224p-K700` loaded on an authorized V100 with standard PyTorch FP16 after an explicit LayerScale name adapter. The source file is 2,042,600,861 B, SHA256 `a615568ca9f386509373e4943a5924adfb36f640c2e7c460930c277072e48caf`, and the loaded headless encoder has 1,020,710,144 parameters. Eight-frame forward mean latency was 0.2095 s at 2,267 MiB; 16-frame interpolated positional mode was 0.5385 s at 3,084 MiB. Both outputs were finite. This is compatibility only: no temporal head, human labels, local score, or AIC submission bundle exists.

## Decision

Keep SUB_C (DeiT-S + YuNet) as the measured official candidate and A0 as the engineering fallback. VideoMAEv2 is a promising bounded representation for a larger, source-disjoint or human-labeled experiment, but this ten-video weak-label result does not justify replacing the official candidate. InternVideo2 remains an L-tier reference; its raw score would need to exceed 7.37778 to beat 6.64 after a 0.90 coefficient.
