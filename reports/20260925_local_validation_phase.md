# 2026-09-25 local validation phase

本报告只记录本地可复现证据。由于尚未取得 AIC 联合 temporal/crop GT、官方 evaluator 或排行榜反馈，本文所有 TVSum 数值都是 `TVSum temporal proxy`；`official_f_video` 与 `competition_score` 保持 `null`。

## Frozen protocol

`splits/local_protocol_v1.json` 固定 TVSum v3 的 27 train、16 dev、7 local lockbox。manifest SHA-256 为 `b8db2bb3f1e36e2ecc32a1c307531e43d69bae3f74605c9112b8eda8b679e565`，assignment hash 为 `e032429c87604f327f3298d41ca117beace9ea127d340fab684d22949f88dc79`。每个视频是独立 source group；没有可靠 category metadata，因此 v1 不声称 category stratification。

TRAIN 用于训练，DEV 只用于 checkpoint、threshold 和 postprocess 选择，LOCKBOX 只在候选冻结后一次性比较。lockbox 结果没有反向改变配置。GT 定义固定为 `tvsum_summary_mean_norm_ge_0.5_v1`，连续 score、二值 proxy 和作者 15% summary evaluator 分开保存。

## Frozen candidate results

下表的 lockbox 配置均在 lockbox 读取前由 train/dev 或预先注册的 fold 协议确定。7 个视频的 bootstrap 区间很宽，只用于显示不确定性。

| candidate | frozen policy | dev macro F1 | lockbox macro F1 | lockbox bootstrap 95% CI | weight bytes | tier |
|---|---|---:|---:|---:|---:|---|
| A0_006 | raw, threshold 0.40 | 0.151875 | 0.114304 | [0.0482, 0.1886] | 25,685,169 | S |
| A0_006 (pre-registered fold threshold) | raw, threshold 0.30 | — | 0.185913 | [0.1112, 0.2526] | 25,685,169 | S |
| A1_005 | feature-level shift, threshold 0.35 | 0.144745 | 0.155284 | [0.0624, 0.2583] | 25,685,169 | S |
| B0 ViT-B/16 | raw, threshold 0.30 | 0.159280 | 0.194505 | [0.1114, 0.3172] | 174,986,447 | M |
| Bs0 DeiT-S/16 | median window 9, threshold 0.35 | 0.140241 | 0.291867 | [0.1351, 0.4824] | 46,618,447 | S |
| canonical internal TSM | threshold 0.35 | 0.148430 | 0.175143 | [0.0785, 0.2792] | 0 neural parameters | S cache |

The internal TSM CI is not repeated in this table because its report was produced by a separate probe evaluator; its seven per-video values are `[0.2198, 0.4146, 0.2712, 0, 0.0426, 0.2133, 0.0645]`, giving mean 0.175143 and sample std 0.147493. The A0 threshold 0.30 row is a pre-registered calibration sensitivity check; the direct dev-selected A0 configuration remains threshold 0.40.

The DeiT-S bundle has 23,280,257 parameters and was exported as one complete FP16 file. A raw-video run over all seven lockbox videos produced 9,325 predictions and passed the local JSONL validator. A one-video center-crop run took 15.16 s with 1,324,096 KB maximum resident memory on the remote V100 environment. The raw inference path now accepts the same frozen postprocess config used by cache evaluation.

The fixed median-9/threshold-0.35 policy was repeated with two additional temporal-head seeds. Seed 20260925/20260926/20260927 gave DEV macro F1 `0.14024/0.10558/0.13890` and lockbox macro F1 `0.29187/0.22378/0.28128`. The lockbox mean across these three seeds is `0.26565` with sample standard deviation `0.03664`; this is positive evidence for the route, but also shows enough initialization variance that the single seed must not be called a stable winner yet.

## Zero-parameter postprocess

For A0_006, Gaussian smoothing (window 5, sigma 1, threshold 0.40) improved DEV from 0.151875 to 0.154742 but fell to 0.112405 on lockbox. Gap filling and minimum run length also fell on lockbox. Rank normalization reached 0.115942 on lockbox but was lower on DEV and is not promoted. Raw thresholding remains the conservative fallback. A1 Gaussian was similarly not selected because it was not the DEV winner; its lockbox value is recorded only as an exploratory diagnostic.

## Split and failure evidence

The five source-group folds remain high variance. The canonical internal TSM fixed-threshold mean is 0.13800 ± 0.06299 versus the A0 control 0.14631 ± 0.03251; paired internal-minus-A0 differences are `[-0.03609, 0.04901, 0.01742, -0.01249, -0.05937]`. This is mixed and does not justify replacing A0. DeiT-S five-fold frozen policy (median 9, threshold 0.35) is 0.14389 ± 0.05173; it is a promising S-tier challenger, but this is still TVSum-only and lacks an external dataset.

The zero fold and low-scoring lockbox videos are over-selection/calibration failures rather than empty-output failures: the tested candidates had empty prediction rate 0 on lockbox, while prediction rates often exceeded the 3–8% proxy target rate. Per-video reports preserve score quantiles, rates, precision/recall and Spearman for future domain/error grouping.

## OOD, spatial and teacher status

SumMe was attempted through the public ModelScope repository. The repository contained five sample MAT files and an LFS pointer for `raw/SumMe.zip`; `git lfs pull` made no progress for more than 90 seconds and was stopped. No raw SumMe video features were used, so no cross-dataset score is claimed. YouTube Highlights was not downloaded because the available recovery is multi-GB and the current sprint already has a frozen S-tier comparison.

Spatial output is end-to-end legal for center, saliency and subject modes, including DeiT-S, and the validator catches frame/coordinate/ratio errors. TVSum has no legal crop GT, so spatial IoU, temporal/spatial oracle and AIC `F_video` remain `null`. The subject mode is still a saliency proxy. No local VLM teacher weights were available; no pseudo-labels or external API calls were used.

## Decisions

* Keep A0_006 as the <=100 MB engineering fallback; use center crop and a threshold explicitly sourced from train/dev.
* Keep B0 as an M-tier semantic reference until a measured joint score can pay the size coefficient.
* Keep DeiT-S as the strongest S-tier temporal challenger by current frozen lockbox evidence, but require another source-group protocol and an OOD dataset before replacing the fallback.
* Lower canonical internal TSM priority. The implementation is correct and parameter-free, but its paired folds do not show stable gain and its lockbox value is below the strongest A0 control.
* Do not promote any postprocess policy from the current seven-video lockbox.

## Next queue

1. Acquire an actual AIC sample/index/evaluator and any legal crop GT; run center fallback first.
2. Reproduce A0 and DeiT-S with a second independent split/seed, keeping v1 lockbox untouched.
3. Obtain a usable SumMe or YouTube Highlights raw subset and report dataset-native ranking/summary metrics separately.
4. Add crop GT validation for center/saliency/subject if a small legal source can be obtained.
5. Only after those checks run a bounded local VLM teacher pilot, then consider distillation.
