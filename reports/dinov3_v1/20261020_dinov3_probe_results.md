# DINOv3 v1 matched probes — 2026-10-20

## Scope

This run uses only public PHD2/GIF fragment selections and the existing
RetargetVid/LIVE native crop tables. It does not read official test labels,
official scores, or the test prediction directory. The remote host was
`autodl-container-2c1e409ad1-4d8375b1` (RTX 4080 32 GB), with the archived
DINOv3 ViT-B/16 weight SHA256
`73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`.

## PHD2 cache and label contract

The resumed extractor completed all 5,886 clean fragments in 2,087.9 s. Each
row stores 8 CLS vectors, timestamps, union labels and user vote fractions;
absence outside a selected interval was not treated as a confirmed negative
by the source metadata. The cache is remote at
`/data/aic/features/DINOV3_PHD2_V1/`.

The same DINO CLS + three-dilation TCN, source split and fixed 85% decoder was
run for union-binary and soft-vote targets. Paired results over three source
seeds were:

| seed | union F | soft F | union − soft |
|---:|---:|---:|---:|
| 20261020 | 0.431730 | 0.431403 | +0.000327 |
| 20261021 | 0.407743 | 0.409287 | −0.001544 |
| 20261022 | 0.354469 | 0.350107 | +0.004362 |

Mean union was `0.397981` versus soft `0.396932`; the paired difference is
only `+0.001049` and changes sign once. Union AP was higher in all three
runs, but this does not identify a reliable competition contract. A later
3,000-step matched ablation on seed 20261020 gave position-only `0.457027`,
DINO CLS `0.438451`, DINO+position `0.433453`, and DINO+first-difference
`0.435161`. The position-only result is close to the center policy while all
DINO variants are lower. This is evidence against promoting the current
PHD2→DINO temporal route; it is not evidence that DINOv3 lacks visual
information on the official test domain.

## Native crop spatial probe

The spatial extraction covered 12,822 table rows, 11,554 unique keyframes and
168 source-disjoint development sources. All modes used the same 129 legal
windows, letterbox-aware DINO patch masks, native crop IoU labels and a
590,593-parameter candidate head. The source-macro center baseline is
`0.497588`; candidate oracle is `0.803418`.

| mode | source-macro top IoU | delta vs center | patch permutation control |
|---|---:|---:|---:|
| DINO patch | 0.484223 | −0.013365 | — |
| CLS broadcast + position | 0.101462 | −0.396125 | — |
| patch spatial permutation | 0.261900 | −0.235687 | — |

The patch arm is materially better than the two controls but still below the
fixed center decoder. Thus DINO patch tokens contain some candidate-dependent
signal, yet the current readout/training contract does not transfer enough to
replace R70. The negative CLS and shuffle controls also validate that the
patch arm is not merely a position-only artifact. Because the patch arm failed
the preregistered `+0.005` source-level gate, no DINO spatial submission was
made.

## Decision and next experiment

The pre-registered DINOv3 v1 gate is not met. Stop the current DINO temporal
recipe and the current spatial regression head. Preserve the cache and code
for a structurally different, target-aligned readout test: a pairwise/listwise
candidate ranking objective with per-frame candidate normalization and a
source-disjoint RV/LIVE report. If that objective does not exceed center on
both domains, close DINOv3 as a deployable route and redirect effort to the
AViMoS/low-level composition representation or a new target-domain temporal
supervision source. No official package is justified by these probes.

## Reproduction

```bash
# temporal contract
/root/miniconda3/bin/python /root/AIC/scripts/dinov3_temporal_label_contract_probe.py \
  --cache /data/aic/features/DINOV3_PHD2_V1 \
  --out /data/aic/experiments/DINOV3_PHD2_V1/label_contract_probe.json \
  --steps 1200 --device cuda

# temporal position/difference ablation
/root/miniconda3/bin/python /root/AIC/scripts/dinov3_temporal_ablation.py \
  --cache /data/aic/features/DINOV3_PHD2_V1 \
  --out /data/aic/experiments/DINOV3_PHD2_V1/temporal_ablation.json \
  --steps 3000 --seed 20261020 --device cuda

# spatial patch/CLS/shuffle probe
/root/miniconda3/bin/python /root/AIC/scripts/dinov3_spatial_patch_probe.py \
  --t5-root /data/aic/experiments/T5_CROPHEAD_V3 \
  --rv-frames /root/autodl-tmp/evac/data/aic/experiments/QWEN_RV200_KEYFRAMES \
  --out /data/aic/experiments/DINOV3_SPATIAL_V1 \
  --dinov3-repo /data/aic/pretrained/dinov3/dinov3-main \
  --weights /data/aic/pretrained/dinov3/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth \
  --batch-size 128 --steps 800 --device cuda
```
