# Video foundation state before submission-candidate work

Official user-reported raw scores: SUB_A 1.01, SUB_B 6.08, SUB_C 6.64. SUB_C is the measured control; A0 remains an engineering fallback. B/C temporal selections are identical; YuNet is the isolated spatial change.

VideoMAEv2-Base corrected 40/10 weak-label clip probe: F1 .54444, Spearman .41459; matched DeiT .40000/.38757. Paired intervals cross zero; no promotion. The train-only position prior has Spearman .55405, exposing label-position bias. See `20260926_videomae_qvh_probe.md`.

InternVideo2 Stage1-1B K700: V100 FP16 8f .2095s / 2267MiB and 16f .5385s / 3084MiB; compatibility only, not a submission candidate. Source bytes 2,042,600,861; loaded headless 8f parameters 1,020,710,144. See `20260926_internvideo2_compatibility.md`.

Native QVH protocol `splits/qvh_native_bounded_v1.json`: 96 train / 24 dev / 40 reserved holdout. Native saliency is query-conditioned; missing ratings must not silently become negative highlights. No crop GT. Existing frozen-head human sanity on ten exposed videos is not independent OOD.

Current regression: 95 passed. New candidate D/E work must preserve SUB_C YuNet dense_v1 and source-frame JSONL construction. Training on native QVH changes supervision relative to historical SUB_C; a matched retrained DeiT control is required. Video clip sampling is an additional variable and must be explicit. No official upload is authorized in this phase.
