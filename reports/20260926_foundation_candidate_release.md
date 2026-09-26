# Foundation candidate release (2026-09-26)

## Gate

The native-QVH train/dev protocol used 96 train and 24 DEV source videos. The 40-video holdout and all official test labels were excluded. A matched retrained DeiT-S control scored F1 `0.76543`, Spearman `0.11364`, NDCG `0.96287`. VideoMAEv2-Base scored Spearman `0.06824` and NDCG `0.96109`, so SUB_D was rejected before official inference.

InternVideo2 Stage1-1B scored F1 `0.76609`, Spearman `0.23474`, NDCG `0.96763`. Paired deltas versus DeiT were Spearman `+0.12111` (95% CI `[-0.03795, 0.29911]`) and NDCG `+0.00476` (CI `[-0.01045, 0.01849]`). This is a positive exploratory signal with uncertainty; it is not a proven replacement for SUB_C.

## Frozen candidate

`SUB_E_INTERNVIDEO2_NATIVE_V1` uses InternVideo2 Stage1-1B K700, the native-QVH-trained Temporal U-Net head, fixed threshold `0.35`, 2 FPS, original-PTS interpolation, and the unchanged YuNet `true_face_smooth` spatial path. All code, checkpoint, detector, and index hashes are frozen in the remote manifest `/data/aic/experiments/NATIVE_CANDIDATE_V1/SUB_E_frozen_manifest.json`.

The complete 174-video raw inference was run as two fixed parity shards on physical GPUs 2 and 4 and merged in original index order. This is execution parallelism only; no test-specific decision was made. The merged JSONL contains 174/174 videos and 87,637 predictions, with zero empty videos. Project validation, independent validation, and unzip regression all passed with zero errors.

- Weights: `2,049,501,387` bytes; `1,022,373,889` parameters; L tier (500M–9B), expected size coefficient `0.90`.
- ZIP: `/data/aic/official_test_20260926/submissions/SUB_E_INTERNVIDEO2_NATIVE_V1_FINAL/upload.zip`
- ZIP SHA256: `342d36e229f2502d863e8944a4906030050e16282cc94f5cb81138bc9a0d419a`
- Predictions SHA256: `a999f51a78a1832c0171b46a691c3baf8a7ba90c0b0d88c5ab6a68b0a4362b4e`
- Official score: `null` (not submitted)

The ZIP contains exactly one root file, `predictions.jsonl`. It is READY_TO_UPLOAD as an engineering artifact. Because the paired DEV interval crosses zero and the L tier needs raw score above `7.3778` to beat SUB_C's weighted `6.64`, the candidate is a high-information submission option rather than a guaranteed improvement.
