# VideoMAEv2 matched QVHighlights probe (2026-09-26)

This is a bounded representation diagnostic on the QVHighlights-derived
training package. It did not read the official evaluation set, train on test
videos, or produce an AIC submission. The labels are
`qvh_seed_timeline_linear_v1`, so `official_f_video` and `competition_score`
remain null.

The first run is retained under
`/data/aic/experiments/VIDEOMAE_QVH_PROBE_20260926_failed_shape` and marked
invalid because its channel normalization shape was wrong and its cache lacked
timestamps. It is excluded from all results. V2 fixed those issues, used the
ImageNet mean/std from the VideoMAE package, added timestamps and tie-aware
average ranks, and preserved the original frame index/timestamp/label arrays.

## V2 protocol

- 40 train and 10 validation clips; train/validation YouTube source IDs are disjoint.
- 2 FPS, 16 sampled frames per clip, stride 4, 72 clips per 300-frame video.
- Clip targets are the mean of the frame weak labels over each clip.
- VideoMAEv2-Base directly encodes the clip. A0, DeiT-S, and ViT-B controls
  average their existing per-frame cache features over the same spans.
- All variants use the same Temporal U-Net, BCE, seed `20260926`, 20 epochs,
  and fixed prediction/target thresholds of 0.5.
- Checkpoint selection uses the same 10 validation videos; results are
  development diagnostics rather than held-out estimates.

## Results

| representation | best clip F1 | mean Spearman | empty rate | best epoch |
|---|---:|---:|---:|---:|
| VideoMAEv2-Base | 0.54444 | 0.41459 | 0.30 | 18 |
| ViT-B/16 control | 0.46667 | 0.42070 | 0.70 | 2 |
| DeiT-S control | 0.40000 | 0.38757 | 0.90 | 13 |
| ResNet18 control | 0.33571 | 0.42129 | 0.00 | 4 |

A train-only normalized temporal-position baseline reached F1 `0.29571` and
Spearman `0.55405`. This shows that the weak labels contain strong position
structure, so representation differences must not be read as semantic gains.
Paired bootstrap deltas against DeiT-S were broad: VideoMAEv2 F1 `+0.14444`
with 95% CI `[-0.15569, 0.38889]`, and Spearman `+0.02701` with CI
`[-0.00300, 0.07645]`.

VideoMAEv2 has 86,227,200 parameters and a 344,924,592-byte source FP32
checkpoint (about 172.5 MB if exported to FP16, before the temporal head), so
it is S under the user parameter-count tiers and an estimated M under FP16 file-size tiers; no complete FP16 bundle has yet been exported. Encoder extraction took 325.48 s; total run time
was 370.14 s. The recorded 2,552,064,000-byte peak is the encoder extraction
peak only, not a full end-to-end peak after temporal-head fitting.

The matched clip probe is useful for deciding whether a video representation
deserves a larger controlled study. It is not evidence that VideoMAEv2 beats
DeiT-S on the AIC evaluator, nor does it justify an M-tier submission by
itself.

## Reproduction

Remote artifacts:

- `/data/aic/experiments/VIDEOMAE_QVH_PROBE_V2_20260926/metrics.json`
- `/data/aic/experiments/VIDEOMAE_QVH_PROBE_V2_20260926/analysis.json`
- `/data/aic/experiments/VIDEOMAE_QVH_PROBE_V2_20260926/config.json`
- `/data/aic/experiments/VIDEOMAE_QVH_PROBE_V2_20260926/provenance_control_audit.json`

The source model SHA256 is
`ebffa1874066ea227330016e58a848e9e2bb1ff5605746459bded1122a42176d`; the metrics, analysis and config SHA256 values
are respectively `cd19dfca8ff130c7034d1f06cc93ed61a82e226e492e5174359e1251557845a0`,
`0d75d5daa6c417ecf64d9881466cc1e3cda4e21be024bbb8234aa76add551d37`, and
`8800fc6ee35600cd34071acc6385bf2280750a66b8fe83bf97cecd2e2342cf4c`.
