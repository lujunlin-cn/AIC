# 2026-09-26 bounded model-base inventory

Historical inventory followed by a 2026-09-26 live update: VideoMAEv2-Base and InternVideo2 Stage1-1B are now downloaded; see update below. Earlier absence statements describe the initial scan only. The remote scan was performed on `supie` in
`/data/aic/pretrained`, `/data/aic/weights`, and the remote HuggingFace cache.

## What is already present

| asset | bytes | measured role |
|---|---:|---|
| `vit_b_16-c867db91.pth` | 346,328,529 | torchvision ViT-B/16 source weight; existing M-tier reference |
| `deit_small_patch16_224_headless.pt` | 86,722,803 | source DeiT-S ImageNet weight; frozen release is 46,618,447 B |
| `resnet18_headless.pt` | 44,782,331 | source ResNet18 weight; frozen A0 release is 25,685,169 B |
| `yunet/face_detection_yunet_2023mar.onnx` | 232,589 | existing spatial observer |

No UniVTG, VideoMAE(V2), InternVideo2, ActionFormer, or TriDet checkpoint is
present in the remote cache or project weight directories. The local
`/home/hajimi2025/.cache` contains unrelated models (Nemotron shards,
ResNet50-IBN, MobileNetV2, OSNet), none of which is a verified video temporal
backbone for this project.

## Candidate families

| family | actual/estimated parameters and FP16 size | V100 / size assessment | integration cost and decision |
|---|---:|---|---|
| VideoMAEv2-Base | ViT-B video encoder, about 86M parameters; HF public `model.safetensors` is 344,924,592 B (FP32); FP16 export is about 172.5 MB before temporal head | Expected to run with standard PyTorch on V100 at small clip batch (throughput not yet benchmarked); M-tier after FP16 export | Public HF model is `cc-by-nc-4.0`; expects 16 frames (`tubelet_size=2`, 224p) and custom remote code. Requires a new clip cache and temporal pooling adapter. **Best bounded M-tier visual probe, but no weight currently present and license must be accepted for the contest.** |
| InternVideo2 Stage2 1B | approximately 1B parameters; FP16 is roughly 2 GB plus runtime state | L-tier; V100 inference may require batch 1/offload and is outside the 100–500 MB target | HF checkpoint is gated (`OpenGVLab/InternVideo2-Stage2_1B-224p-f4`), so no unauthenticated download was attempted. Apache-2.0 code/checkpoint card, but large video-text stack and new preprocessing. **Reference only; do not prioritize before M-tier.** |
| InternVideo2 6B | approximately 6B; FP16 >12 GB | L-tier and unsuitable for a single V100 inference bundle without aggressive sharding/offload | No local weight. **Reject for current competition loop.** |
| ActionFormer | Source instantiation from the public configs gives 6,943,497 params (ANet TSP, 13.89 MB FP16) or 29,251,612 params (THUMOS I3D, 58.50 MB FP16) | Head itself is S-tier; with DeiT-S total remains roughly 60–105 MB depending on config | Requires pre-extracted I3D/TSP/SlowFast features and action-segment labels. It is a temporal localization head, not a visual backbone. Adaptation to frame highlight scores is possible but needs a new head/evaluator. **Low-size temporal-head experiment; not an M-tier visual candidate.** |
| TriDet | Source instantiation gives 12,811,555 params (ANet TSP, 25.62 MB FP16) or 15,989,604 params (THUMOS I3D, 31.98 MB FP16) | Head itself is S-tier; with ViT-B it is roughly 204 MB FP16 (M-tier), with DeiT-S roughly 78 MB | Requires external pre-extracted action features and temporal action GT. Boundary-aware outputs are not directly AIC frame/crop predictions. **Potential bounded temporal-head comparison; no local checkpoint and no direct raw-video path.** |
| UniVTG | No verified local checkpoint; architecture is a multimodal grounding/saliency head over pre-extracted SlowFast/CLIP features (hidden dimension defaults around 256; exact bundle size depends on feature and text branches) | Head likely S-tier, but the released CLIP-B/16/R50 feature encoders are separate and would push the full bundle into M/L | Public release provides Google Drive checkpoints, not a local/HF weight in this workspace. It needs text/query features and moment-retrieval labels; it is not a generic raw-video highlight backbone. **Do not treat as direct AIC candidate; at most a temporal ranking-head research probe after labels are mapped.** |
| FineGym | Dataset/benchmark, not a standalone backbone | No model-size entry | FineGym supplies action-quality annotations; it does not provide a drop-in AIC inference model. **Data source only, not a model candidate.** |

### Size implications relative to the official submissions

The current DeiT-S + YuNet candidate is 46,851,036 B. A VideoMAEv2-Base
FP16 backbone would be about 172.5 MB, before adding the temporal and spatial
weights, so it is an M-tier reference. A ViT-B + TriDet/ActionFormer head would
also be M-tier (roughly 200 MB) and would test temporal-head capacity separately
from the visual encoder. InternVideo2-1B is already L-tier; it cannot be a
100–500 MB candidate even if the temporal head is small.

## Recommended bounded order

1. Keep the already measured ViT-B bundle as the immediate M-tier reference.
2. If one new M-tier visual probe is justified, use VideoMAEv2-Base only after
   license review and a 16-frame cache/throughput smoke test; do not download
   InternVideo2 first.
3. For a low-cost temporal-head diagnosis, adapt TriDet or ActionFormer on
   existing DeiT/ViT-B features. This tests temporal architecture, not a new
   visual representation, and requires AIC-compatible frame-score labels.
4. Keep UniVTG and FineGym out of the direct submission path until a query-free
   label mapping and raw-video inference path are implemented.

Sources inspected: remote filesystem inventory; ActionFormer public config/model
instantiation (`happyharrycn/actionformer_release`, MIT); TriDet public config/
model instantiation (`dingfengshi/TriDet`, MIT); UniVTG README/model source
(`showlab/UniVTG`, MIT); HF API/model card for
`OpenGVLab/VideoMAEv2-Base` (344,924,592-byte public file, CC-BY-NC-4.0) and
`OpenGVLab/InternVideo2-Stage2_1B-224p-f4` (Apache-2.0, gated).

## Live continuation update

- VideoMAEv2-Base actual loaded parameters: 86,227,200; source file 344,924,592B. It is S under parameter-count tiers and M under FP16-file-size tiers (FP16 full bundle not yet exported). Corrected V2 clip probe pending; old normalization run invalid.
- Authorized InternVideo2 **Stage1-1B-224p-K700**, not the Stage2 variant above: source file 2,042,600,861B, SHA256 `a615568ca9f386509373e4943a5924adfb36f640c2e7c460930c277072e48caf`. Loaded headless+fc_norm 8-frame encoder has 1,020,710,144 parameters; V100 FP16 smoke passes. See `20260926_internvideo2_compatibility.md`. L under both size interpretations; no highlight score yet.
- The user prioritizes **100–500M parameters**. File MB and parameter M are distinct; neither ViT-B nor VideoMAEv2-Base tests that parameter interval. A later bounded ViT-L/video-large model should be audited under both interpretations.
