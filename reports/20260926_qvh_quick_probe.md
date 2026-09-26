# QVH native candidate quick probe

Protocol `NATIVE_FOUNDATION_CANDIDATES_V1` uses 96 source-disjoint train videos and 24 DEV videos from native QVHighlights annotations. The 40-video holdout was not materialized or read. Targets are masked query-conditioned human saliency; no unannotated clip is silently converted to a negative.

Matched retrained DeiT-S control: F1 `0.76543`, Spearman `0.11364`, NDCG `0.96287`. VideoMAEv2-Base: F1 `0.76543`, Spearman `0.06824`, NDCG `0.96109`; paired deltas are F1 `0`, Spearman `-0.04540` (95% CI `[-0.16453, 0.07504]`), NDCG `-0.00178` (CI `[-0.01210, 0.00788]`). VideoMAE is therefore `NOT READY`.

InternVideo2 Stage1-1B: F1 `0.76609`, Spearman `0.23474`, NDCG `0.96763`; paired versus DeiT deltas are F1 `+0.00066` (CI `[-0.02093, 0.02242]`), Spearman `+0.12111` (CI `[-0.03795, 0.29911]`), NDCG `+0.00476` (CI `[-0.01045, 0.01849]`). This is a positive exploratory signal, but every paired CI crosses zero, so it is `PROMISING`, not proven superiority.

The protocol has additional confounds relative to historical SUB_C: native QVH supervision, two-second anchors, and clip context. These results are not official AIC scores and do not inspect the native holdout or official test labels.
