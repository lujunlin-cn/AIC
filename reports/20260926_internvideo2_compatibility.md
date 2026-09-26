# InternVideo2 1B bounded compatibility probe — 2026-09-26

The existing 2.04 GB checkpoint runs on an authorized V100 using standard PyTorch FP16. This is a compatibility reference, not a trained highlight candidate; no quality/official-score conclusion follows. Only a QVHighlights **training** clip was read.

| Configuration | Input | Output | Mean forward latency | Peak allocated VRAM | Loaded parameters |
|---|---|---|---:|---:|---:|
| Original 8-frame convention | 1×3×8×224×224 | 1×768 | 0.209517 s | 2267.245 MiB | 1,020,710,144 |
| 16-frame positional interpolation | 1×3×16×224×224 | 1×768 | 0.538493 s | 3084.064 MiB | 1,023,593,728 |

Both outputs are finite. Latency is the mean of three synchronized forwards after warmup, excludes decoding, and is not full-video inference latency. Inputs use deterministic 2 FPS frame selection and bicubic resize/ImageNet normalization solely for this engineering probe. Full fair comparison needs the same clip aggregation/evaluator as other encoders.

## Assets and provenance

- Existing checkpoint: `/data/aic/pretrained/internvideo2_stage1_1b/1B_ft_k710_ft_k700_f8.pth`
- File bytes: **2,042,600,861**, SHA256 `a615568ca9f386509373e4943a5924adfb36f640c2e7c460930c277072e48caf`.
- File tensor dtype BF16; explicitly converted to FP16 for V100 execution.
- Neural size tier for this file: **L**, expected coefficient **0.90**. It needs raw official score >7.37778 to exceed current SUB_C=6.64. No score exists for this model.
- Local compatible implementation was absent. Fetched `https://github.com/OpenGVLab/InternVideo2` through existing local Clash proxy, checkout `3d521087215c9024199b0512370a29dee1c0fee6`, into `/data/aic/tmp/InternVideo2`.
- Upstream source license is MIT. This does not replace a separate weight license/provenance audit before release.
- Encoder source SHA256 `315936c8eac45890b7f78df8c27e5b234a915454dbf329d7ed79f7c06c217c80`; position helper `d0dbe1f563306c22e5a25042034356a2b0a733a6cba2d613d7baaf44848c4226`.
- Environment: `/opt/miniconda3/envs/cv/bin/python` (Python 3.12.0; user site packages participate). Added `einops==0.8.0`. No FlashAttention installation.
- Physical GPU **7**, verified idle before probes.

## Explicit compatibility adapter

The wrapper imports only the upstream vision encoder, bypassing unrelated LLaVA imports. FlashAttention/fused MLP/fused RMSNorm flags are false. An import stub cannot be instantiated and therefore cannot silently execute. Upstream `LayerScale.gamma` checkpoint names map to the clean implementation's `LayerScale.weight` without changing values. All encoder + attention projector weights match; trained `fc_norm` is loaded separately. Only the K700 classifier weight/bias is omitted. There are **zero missing encoder weights**.

The 16-frame variant applies the upstream 8→16 temporal positional interpolation; its parameter count is slightly larger due to positional-table expansion. It is not a new pretrained 16-frame checkpoint. Original checkpoint remains unmodified. No FP16 competition bundle has been exported yet.

## Reproduction

```
cd /home/supie/AIC
nvidia-smi
CUDA_VISIBLE_DEVICES=7 timeout 180s /opt/miniconda3/envs/cv/bin/python scripts/probe_internvideo2_smoke.py --frames 16 --output /data/aic/experiments/INTERNVIDEO2_SMOKE_20260926/16f.json
```

`--frames 8` repeats the original temporal shape. Reports: `reports/internvideo2_smoke_20260926/{8f,16f}.json`; remote outputs `/data/aic/experiments/INTERNVIDEO2_SMOKE_20260926/`.

Next justified step is a bounded frozen-feature probe using identical target/validation semantics as the image controls. Do not launch full 130 GB extraction or fine-tuning based only on this compatibility result. Official AIC score and local highlight validation score remain null.
