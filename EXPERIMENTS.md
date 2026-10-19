# 实验历史

## R98 — AIM 2024 / UNISAL human-saliency native-crop probe（2026-10-19）

- **Hypothesis:** temporal human visual attention can provide a spatial observation independent of foreground statistics.
- **Change:** frozen UNISAL `weights_ft_mit1003.pth`; saliency centroid and 50%-mass centroid converted to legal crop; fixed LIVE T5 keyframes and native crop GT.
- **Data:** 148 LIVE source dev; no audio; no official labels.
- **Result:** Qwen point `0.52594`, UNISAL centroid `0.48871` (`-0.03723`, 95% CI `[-0.06317,-0.01117]`), mass50 `0.49230` (`-0.03364`, CI `[-0.05943,-0.00794]`).
- **Decision:** close direct saliency-center replacement; keep only attention-as-secondary-feature or target-supervised local decoder as future distinct mechanisms.
- **Report:** `reports/r98_unisal_human_saliency_native_crop_20261019/summary.md`.

## R99 — Low-rank local-global context TCN（2026-10-19）

- **Hypothesis:** same-source global visual context changes local frame value through a real interaction, not a constant linear offset.
- **Change:** rank-32 Hadamard interaction between per-frame RGB embedding and same-fragment mean; matched zero-interaction, linear-global and cross-source permutation controls.
- **Data:** R92 RGB view, 5,886 PHD2/GIF fragments, 448 sources, observed-selection labels; fixed 85% source-macro F.
- **Result:** control `0.415121`, linear global `0.414338`, interaction `0.415452`; interaction delta `+0.000331`, 95% CI `[-0.000916,+0.001633]`; permutation delta `+0.000006`.
- **Decision:** not ready for package; do not scan rank/loss. Reopen only with denser/full event observation and a new source-level confirmation protocol.
- **Report:** `reports/r99_lowrank_context_tcn_20261019/summary.md`.

## R100 — R70 pooled temporal context spatial proxy（2026-10-19）

- **Change:** fixed R70 current candidate tensor; neighbouring candidate-pool mean as context; rank-32 interaction vs zero-context control and source-permuted context.
- **Data:** 12,822 public native-crop rows, 188 RV/LIVE dev units.
- **Result:** control `0.587141`, interaction `0.587888`, delta `+0.000747`, CI `[-0.003092,+0.004446]`; permutation delta `+0.000374`.
- **Decision:** close this proxy; it does not test a genuinely new low-resolution encoder.

## R101 — R70 current candidate × ResNet18 neighbouring context（2026-10-19）

- **Change:** fixed R70 `[129,2305]` current candidate features plus frozen ImageNet ResNet18 (11,176,512 parameters) embeddings from up to two neighbouring keyframes; rank-32 interaction, zero-interaction and source-permutation controls.
- **Data:** public RetargetVid/LIVE native crop rows, 188 source units in dev; no audio, no official labels.
- **Result:** control `0.588335`, interaction `0.588532`, delta `+0.000197`, CI `[-0.001942,+0.002463]`; permutation delta `+0.001546`.
- **Decision:** close current observation/supervision/readout recipe; no package and no rank/fusion scan.
- **Reports:** `reports/r100_spatial_context_utility_20261019/summary.md`, `reports/r101_spatial_resnet_context_20261019/summary.md`.

## R86/R87/R88（2026-10-19，官方回分与阻塞修复）

| 实验 | 关键变化 | 结果 | 结论 |
|---|---|---:|---|
| R86 | R54 时间集合固定；T5 H1 应用于 `points_V9_A1raw/points` | 官方 **34.88**，相对 R54=36.22 为 **-1.34** | 混杂实验：同时换了点源和映射，不能归因于 H1；归档 |
| R88 | R54 时间集合固定；同一 H1 应用于 `points_B3_vo/points` | 426/426、199,037、8.854531425B；local contract valid；ZIP `db6408cd...` | R86 的最小纠错候选；尚未上传/无官方分 |
| R87 BF16 | AHA causal cache + PHD2 `gif_soft` 源级留出 | trace/full pilot finite_fraction=1.0；32 源留出 Ridge row AP=0.3486，source-macro AP=0.0952 | 数值阻塞已修复；监督/目标域迁移证据不足，不出包 |

R86 的详细错配审计、R88 manifest 和 R87 BF16 trace 见
`reports/r86_r87_observation_and_transfer_20261019/summary.md`。

## V6_NEXT10H_E4_E3_E5_E1AUTO（2026-09-30，代理指标，非官方分）

所有数值为 MrHiSum 弱标签 / YouTube-Highlights match_label / PM400 clip_context 公共标注上的代理指标。confirm=MrHiSum fresh reserve 4,008 一次性使用。

| 实验 | 配置 | 关键数值 | 判定 |
|---|---|---|---|
| E4 | T1 TemporalTCN(1152,128) 740,609 参数 ×3 种子 | dev 0.3188±0.0058；confirm 0.3220/0.3250/0.3289；配对 delta +0.034/+0.037/+0.041，聚类 CI 下界全>0 | 弱域过门，候选 |
| E4 | T2 U-Net+rank(λ=.5) ×3 种子 | confirm 0.3025/0.3043；s1 CI 下界 -4e-05 | 否决 |
| E4/E3 | T1_s3 → YTH 迁移 n=32 六域 | macro AP 0.6162 vs 基率 0.4722；Spearman 0.1296 | TRANSFER_INDICATIVE |
| E3 | 官方链 selftest | rgb_q∈[0,255]，audio absmean 134.3 | DEPLOYABLE |
| E5 | LFM2.5-VL-450M 零样本 P1/P2/P3 | 0/8+0/4 可解析；语义正确、坐标退化 | SMOKE_FAIL，负结果 |
| E1-AUTO | 教师 pilot 32 视频/640 帧 | 56.0 min，0.19qps，parse 100%，保留 90.6% | 达扩展标准 |
| E1-AUTO | A/B 对照 3 种子 | A 0.8360 / B 0.8429 / Δ+0.0069（±0.04 摆动） | 噪声内，不宣称收益 |

产物：`reports/20260930_v6_next10h_research.md`、`reports/v6_next10h/*`；实验目录
V100 `/data/aic/experiments/{V6N_E4,V6N_E3_YTH,V6N_E1A,V6N_E5}`、910A `/data/aic/experiments/V6N_E1A`。

本项目所有TVSum数值均按binary proxy、ranking或author-style summary单独命名，不是AIC官方F_video。当前虽已取得RetargetVid crop GT，仍没有AIC联合GT；`official_f_video`和`competition_score`保持null。早期章节保留为历史，最新证据见末节。

## OFFICIAL_TEST_20260926（冻结推理闭环）

正式评测集只用于 intake、完整解码核验和冻结候选推理；没有训练、微调、人工标注、逐视频调参、teacher 伪标签或第三方 API。三候选均在同一 174-video index 上运行，输出通过项目 validator、独立 checker 和 ZIP 解压回归。官方联合 GT 不可用，因此不登记官方分数。

| run_id | candidate | weight bytes | threshold | predictions | empty rate | runtime | result |
|---|---|---:|---:|---:|---:|---:|---|
| `OFFICIAL_TEST_20260926_SUB_A_001` | A0 ResNet18 + repaired Temporal U-Net + center | 25,685,169 | 0.40 | 1,809 | 0.862069 | 742.213 s | completed, READY_TO_UPLOAD |
| `OFFICIAL_TEST_20260926_SUB_B_001` | DeiT-S/16 + repaired Temporal U-Net + center | 46,618,447 | 0.35 | 8,133 | 0.557471 | 742.764 s | completed, READY_TO_UPLOAD |
| `OFFICIAL_TEST_20260926_SUB_C_001` | DeiT-S/16 + repaired Temporal U-Net + YuNet `true_face_smooth` | 46,851,036 | 0.35 | 8,133 | 0.557471 | 1,959.029 s | completed, READY_TO_UPLOAD |

Canonical archives and hashes are recorded in `reports/20260926_official_test_inference.md`; per-candidate manifests and weight inventories are under `submissions/20260926/SUB_A|SUB_B|SUB_C/` and remain ignored from Git.

## SPATIAL_GROUP_003（2026-09-25）

- Hypothesis: area/confidence weighted top-three face centers reduce single-face fixation in multi-person crops.
- Control: center, true_face, true_face_smooth; same 20 DHF1K/RetargetVid videos, two ratios, six raters, fixed alpha=.25 and detector.
- Result: group `.48004/.75115`, group+EMA `.48106/.75216` (1:3 / 3:1); paired group+EMA minus single-face+EMA `-0.00264`, 95% CI `[-0.00726, 0]`, positive source fraction `0/20`.
- Decision: reject this concrete group-center rule; retain explicit modes and E2E tests. This source set is exposed development evidence, not an independent test.

## 本轮评估修复

- GT 与 prediction threshold 解耦：固定 `tvsum_summary_mean_norm_ge_0.5_v1`，prediction threshold 单独调节。
- Local validation protocol v1 已冻结在 `splits/local_protocol_v1.json`。其 manifest hash 与 source-group 分区绑定；不可将 lockbox 结果用于调参，校验命令为 `python scripts/validate_local_protocol.py`。
- `run_epoch` 改为 per-video 统计和 video-macro 选模；同时保存 micro、precision、recall、selection rate、empty rate、quantiles、MAE、Spearman。
- `TemporalUNet` / `A0Model` 接收真实 `lengths`，消除 right-padding 对 GroupNorm、pooling、interpolation 的影响。旧路径实测 5→12 最大差 0.3392、8→12 最大差 0.5733，修复后差为 0。
- MAT/TSV audit：50 个 `user_anno` 均为 `(20,nframes)`，2 秒是收集协议，发布数据已经重复到帧级；旧均匀展开对现有版本是 identity，最大差 <5e-8。

## 参考基线

`PROXY_BASELINES_001`（16-video val，预算只由 train target rate 推导）：all-negative `0.00000`、all-positive `0.08715`、constant train mean `0.00000`、uniform budget `0.05225`、random budget `0.06708`、linear ridge `0.03383`。linear ridge 的 mean Spearman 为 `0.2866`，但固定 0.5 selection 仍不如 all-positive，说明 calibration 是独立问题。

## Repaired A0/A1

| run | change | fixed-0.5 macro F1 | dev threshold | tuned macro F1 | fp16 bytes | time |
|---|---|---:|---:|---:|---:|---:|
| A0_004 | A0, batch=2, lengths-aware | 0.01316 | — | — | 25,685,169 | 10.76s |
| A0_005 | A0, batch=1, lengths-aware | 0.04828 | 0.40 | 0.15010 | 25,685,169 | 29.81s |
| A0_006 | A0, proxy-v2 labels, batch=1 | 0.05661 | 0.40 | **0.15188** | 25,685,169 | 29.95s |
| A1_004 | A0 + feature-level embedding shift | 0.09607 | 0.30 | 0.12685 | 25,685,169 | 27.31s |
| A1_005 | A1, proxy-v2 labels, batch=1 | 0.12108 | 0.35 | 0.14474 | 25,685,169 | 30.19s |

At fixed 0.5, A1_005 looks stronger; after dev-only threshold calibration A0_006 is stronger on the same split. This is a calibration/post-processing result, not proof that feature-level shift is harmful in general.

## Source-group split stability

Five deterministic source-group folds were trained with the same seed, batch=1 and 15-epoch budget. Fixed threshold 0.30 gives A0 `[0.08529,0.20472,0.11486,0,0.16796]`, mean `0.11457`, std `0.07900`; A1 `[0.09380,0.21172,0.09111,0,0.14274]`, mean `0.10787`, std `0.07764`. Per-fold threshold tuning gives optimistic A0 `0.14983±0.03692` and A1 `0.15422±0.03665`; those thresholds were selected on each validation fold and are not lockbox evidence. Fold variance is large and prevents a stable A1 win claim.

## Feature Bank grouped ablation

Before training, dimension audit found恒零 dims 11, 23–28, 31 and duplicate/near-duplicate pairs (0,8), (0,9), (5,22), (7,16). With the same 32D head and repaired batch=1 protocol: motion-only A2_002 `0.02724`, quality-only A2_003 `0.01892`, composition-only A2_004 `0.01613`, audio-zero A2_005 `0.00000`. The current bank is rejected; this does not test real waveform audio.

## Canonical internal TSM probe

`temporal_shift_feature_map` shifts channels on a ResNet intermediate `[B,T,C,H,W]` map. GPU 7 probe: mean output difference `0.12085`, parameter-free, average batch time 2.524ms→2.973ms (`+17.8%`). A full recache/end-to-end training has not yet been run; historical A1 is only final-embedding shift.

## ViT B0

`B0_probe_vit_b16` uses frozen torchvision ViT-B/16 ImageNet-1K features (768D) and the same Temporal U-Net. Fixed-0.5 macro F1 `0.06760`; dev threshold 0.30 `0.15928`. A real raw-video → B0 bundle → saliency crop → JSONL validator run passed. FP16 bundle is `174,986,447` bytes (M tier under current decimal assumption), so it remains a probe until multi-fold and joint AIC evaluation justify the size penalty.

## Spatial and decoder diagnostics

`center|saliency|subject` are now explicit raw inference modes. On one real TVSum video, all modes produced valid JSONL; no crop IoU was claimed. Remote PyAV 15.1 cache/raw features differed by at most ~2.5e-5 and probabilities by ~2.1e-7; local PyAV 18.1 produced larger drift, so the remote environment is the reproducibility target.

## Loss ablation

`A0_012` changed only BCE-with-logits to SmoothL1 on sigmoid scores, keeping proxy-v2 data, batch=1, seed and Temporal U-Net fixed. Fixed-0.5 macro F1 was `0.11679`; dev threshold 0.40 reached `0.14143`, below A0_006 BCE `0.15188`. BCE remains the current loss control; ranking loss and simpler head were not run in this sprint.

## Frozen local lockbox and post-processing

`splits/local_protocol_v1.json` freezes 27 train / 16 dev / 7 lockbox videos and its manifest/assignment hashes. Lockbox runs use one config selected before reading lockbox. A0_006 raw threshold 0.40 is `0.114304` on lockbox (bootstrap 95% CI `[0.0482,0.1886]`); the pre-registered fold threshold 0.30 sensitivity check is `0.185913`. Gaussian smoothing improved dev by `0.00287` but fell to `0.112405` on lockbox, so raw A0 remains fallback. Full details are in `reports/postprocess_A0_006_20260925.md` and `reports/20260925_local_validation_phase.md`.

## Canonical internal TSM

The new cache applies `temporal_shift_feature_map` after ResNet18 layer1, before layer2--4. Chunk/full max error is `1.81e-5`; removing the shift changes features by mean absolute `0.08127`; parameter increment is zero. On five source-group folds, A0 is `0.14631±0.03251` and internal TSM `0.13800±0.06299` at fixed threshold 0.30, with paired mean difference `-0.00830`. The frozen lockbox is `0.175143` at threshold 0.35 versus the strongest A0_006 control `0.182033` at the same threshold. The implementation is retained, but the route is currently deprioritized. See `reports/internal_tsm_20260925.md`.

## DeiT-S S-tier probe

`Bs0_deit_s_probe` uses timm DeiT-S/16 ImageNet-1K frozen embeddings (384D) and the same Temporal U-Net. The complete FP16 bundle is `46,618,447` bytes (`23,280,257` parameters). DEV median-9 smoothing at threshold 0.35 is `0.140241`; the same frozen policy is `0.291867` on the seven-video lockbox (bootstrap 95% CI `[0.1351,0.4824]`). Five source-group folds give `0.14389±0.05173`. A raw-video run over all lockbox videos produced 9,325 predictions and passed the validator; this is the strongest current S-tier temporal challenger, but has no OOD or AIC joint-score evidence.

## B0 M-tier reference

The existing frozen ViT-B/16 reference is `0.159280` on DEV and `0.194505` on lockbox at threshold 0.30, with a `174,986,447` byte bundle. It remains a semantic reference until its raw joint gain is shown to offset the M-tier size coefficient.

## OOD and teacher blockers

The SumMe ModelScope repository cloned metadata and five sample MAT files, but its raw videos remain an unavailable LFS object; `git lfs pull` made no progress for more than 90 seconds and was stopped. No SumMe score was produced. No local VLM teacher weights were present, so no external API or pseudo-label run was started.

## Representation generalization continuation（2026-09-25）

本段为当前证据；以上章节是逐阶段历史，原7条现只称comparison_holdout_v1。MAT category已实际取得；SumMe raw和RetargetVid真实crop GT已取得，旧blocker不再代表全部当前状态。

- `TVSUM_MATLAB_CHECK_001`：10例与未修改作者knapsack/summary函数经Octave逐帧mask一致。
- `RG_DEV_001`：不重训复核5个历史representation，新增ranking/summary；结果见完整representation报告。
- `RG_NCV_*`：90次真正训练，3model×2repeat×5outer×3seed，总1032.39s，最长14.73s。Binary F1 A0/DeiT/ViT-B=.16084/.16869/.17301；summary=.21521/.23073/.23064；Spearman=.43216/.41439/.44127。DeiT−A0 summaryCI跨0，不能升级。
- `SUMME_OOD_001`：因原文件非单调PTS失败，保留日志；`SUMME_OOD_002`显式标注ordinal协议后6条raw+JSONL验证完成。Native human meanF1=.28779/.21340/.17264。
- `SUMME_MATLAB_CHECK_001`：18个已保存预测与原作者evaluator比较，最大误差1.11e-16。
- `SUMME_CV_OOD_001`：相同6条，全部30个CV heads/模型、不调参/不挑head；summary=.25752/.15345/.12961；证据受小样本和mirror时轴约定限制。
- `SPATIAL_GT_001` / `SPATIAL_GT_NATIVE_002`：20条完整源视频、6固定方法，保存首版并补全作者negative clamp后重计分。自有结果不变；真实IoU已可报告，不能再记null。参数未在本批GT调节。
- `DENSE_E2E_001`：8种结构/空间组合，原视频到3600条预测全部有效，最终crop与benchmark完全一致。YuNet额外232589bytes正确计量。
- `SUMME_EXPANSION_ACQUIRE_001`：次批预固定压缩大小ranks9–16、15分钟上限；独立OOD追加复核，不改变任何模型参数。
- `SUMME_CV_OOD_002`：第二批8条全部strict alignment，30冻结heads均评估；summary=.17883/.11955/.15411。合并14条=.21255/.13408/.14361；DeiT−A0 CI[−.13175,−.02734]，不是普遍架构结论。
- `SUMME_OOD_003`：同8条历史完整bundle，summary=.14689/.12124/.14905，JSONL974/2852/7019条全部valid。合并14条summary=.20727/.16074/.15916。
- `SUMME_BASELINES_001`：无GT预算泄漏，constant-first/uniform/random32 meanF1=.12578/.09987/.13755；A0-CV−random+.07501 CI[.02407,.13841]，DeiT/ViT相对随机CI跨0。
- `CROSS_DATASET_DUPLICATE_001`：700对视频无SHA重复/稀疏pHash flag，限于每视频9帧的筛查覆盖。
- `ENGINEERING_RELEASE_001`失败保留：CUDA未初始化即调用显存统计。002/003修复后3候选在DHF1K3样例和TVSum历史DEV2视频均完成；DEV实际阈值输出1263/2297/1263帧，全valid、无新调参。

逐run完整config/hash/seed/split/checkpoint/command/environment和指标追加到registry；机器可读统计、per-video/failure CSV/JSON和图像在 `reports/representation_generalization/`。仍未运行VLM pilot，不重开TSM/Feature Bank v1/后处理sweep。

## Linear head controlled continuation

`RG_LINEAR_001`新增60次真实nested训练，只改head；600条配对输入完全一致。A0/DeiT linear的TVSum summary=.21324/.21781，Spearman=.32930/.33129，均没有胜过U-Net。`SUMME_LINEAR_OOD_001`沿用已有raw提取缓存，全30heads各评估14条：.14771/.12230；A0下降CI不跨0。拒绝当前固定预算线性替换，不否定其他小head；该外部数据已经暴露，不能称新lockbox。

## RG_RANK_001（2026-09-25）

- Hypothesis: within-video pairwise supervision improves ranking/15%-budget selection without adding inference weights.
- Control: repaired Temporal U-Net, same repeated nested source-group folds, 3 seeds, 20 epochs; only loss changed to `BCE + 0.1*pairwise_logistic`.
- Result: 30 A0 and 30 DeiT-S outer evaluations completed remotely. A0: proxy F1 `.155797±.042412`, Spearman `.426467±.108241`, NDCG@15 `.651274±.064576`, summary F1 `.217504±.028750`. DeiT-S: `.172148±.031270`, `.424183±.065153`, `.669796±.045305`, `.233615±.013983`; empty prediction `.0300/.0033`.
- Paired per-video deltas DeiT-S−A0: F1 `+.01635`, Spearman `-.00228`, NDCG@15 `+.01852`, summary `+.01611`; positive video fractions `52/44/52/50%`. No replacement decision; no AIC official score.
- Reproducibility: remote code hash synchronized at commit `e9788a4`; Python `/data/miniconda3/bin/python`; GPU physical 2/4; results `/data/aic/experiments/RG_RANK_001`.

## EVAL_INTAKE_001（2026-09-26）

- Hypothesis: a strict intake manifest and frozen release runner prevent frame/PTS/path/model-size mistakes when the private evaluation set arrives.
- Change: added `AIC_EVAL_INTAKE_V1`, `prepare_eval_set.py`, `run_eval_candidates.py`, and direct raw-video `release_batch.py`; no training, labels, threshold search, or official score.
- Evidence: local and remote A0 release smoke exit 0; loaded bytes `25,685,169`; final JSONL validator valid; local tests `83 passed`.
- Decision: accepted as the only entry path for a new evaluation set. Preserve original index/archive and never overwrite intake or prediction runs.
## 2026-09-26 official-score diagnosis and scaling

### OFFICIAL_SCORE_DIAG_20260926

- **Hypothesis:** the official gap is caused by a combination of visual representation, calibration, and spatial crop quality; the B/C pair can isolate the spatial contribution.
- **Change:** compare frozen release manifests and platform feedback without touching the official test set.
- **Result:** A0/center `1.01`, DeiT-S/center `6.08`, DeiT-S/YuNet `6.64`; B/C selected-frame lists are identical and 8,130/8,133 bboxes differ.
- **Conclusion:** YuNet contributes `+0.56` raw in this controlled official differential. A→B is system-level evidence, not a pure parameter intervention.
- **Record:** `reports/20260926_official_score_diagnosis.md`.

### DATA_SCALE_20260926

- **Hypothesis:** the 27-video TVSum training set is too small for the current DeiT temporal head.
- **Control:** DeiT-S frozen features, same 16-video validation set, Temporal U-Net, BCE, seed `20260926`, 20 epochs; deterministic train prefixes of 7/14/27 videos.
- **Result:** D25 `.0527846` (epoch 16, 10.95s), D50 `.0421875` (epoch 17, 13.78s), D100 `.1214247` (epoch 11, 19.43s) video-macro F1@0.5.
- **Conclusion:** one TVSum split is too noisy for a monotonic curve; D100 is higher but this is not a promotion or an AIC score.

### QVH_DEITS_QUICK_5PCT

- **Hypothesis:** QVHighlights-derived weak temporal supervision can provide a different-domain temporal signal.
- **Change:** deterministic 40/10 matched clip pilot, 2 FPS DeiT-S frozen features, same Temporal U-Net and 20-epoch budget.
- **Result:** 37/9 clips after missing-source filtering; best weak-label validation macro F1 `.67846`, Spearman about `.393`, 29.25s.
- **Conclusion:** ingestion and head training work on the new domain. The labels are seed weak labels, not human GT or AIC F_video; no backbone finetune has been claimed.

### QVH_FINETUNE_PROBE_20260926

- **Hypothesis:** unfreezing the final DeiT-S block can improve task adaptation at no inference parameter-count increase.
- **Status:** bounded 2-epoch raw-video last-block and frozen-head controls are running on the same 37/9 weak-label subset. Results will be added when both runs finish.

## 2026-09-26 continuation: integrity before scaling

`QVH_SOURCE_AUDIT_20260926`: 23 original source IDs cross the full 800/89 split. Purge 25 train clips, keep 89 val unchanged, producing versioned 775/89 protocol. Current 40/10 timeline pilot has no shared source IDs.

`INTERNVIDEO2_SMOKE_8F/16F_20260926`: loaded checkpoint and FP16 standard kernels pass V100; 8f .209517s / 2267.245MiB, 16f .538493s / 3084.064MiB. Engineering-only; no score.

`VIDEOMAE_QVH_PROBE_20260926`: invalid normalization and incomplete cache schema; preserved, excluded from comparisons. Corrected V2 uses ImageNet normalization, timestamps, tie-aware ranking, and identical clip-mean targets for image controls.

## Native human QVH continuation (2026-09-26)

Native public annotations now match 6,384 train / 1,354 val raw videos; first 5/5 per split pass full decode, duration, monotonic PTS and saliency schema audits. Official native train/val original source intersection is zero. A new bounded protocol `splits/qvh_native_bounded_v1.json` freezes 96 train / 24 dev / 40 holdout, one clip per original source, excluding prior weak-data/audited sources from holdout. This is QVH query-conditioned human supervision, not AIC joint GT.

Frozen weak-trained clip heads evaluated on human-rated clips in the previously exposed ten-video weak validation set: Spearman VideoMAE .23530 / DeiT .17771 / A0 .12251 / ViT-B .02553. VideoMAE−DeiT delta +.05759, paired 95%CI [-.21716,+.33909]; no upgrade. Nine queries belong to native TRAIN and one native VAL: this is a label sanity check, not native validation or OOD. No human labels were used to change checkpoints or thresholds. Reports: `reports/qvh_human_sanity_20260926/`.

Cache provenance bug fixed for future extraction: explicit encoder identity no longer overwritten by ResNet18. Existing caches preserved with external provenance audit.

## NATIVE_CANDIDATE_V1 (running)

Hypothesis: native-human QVH supervised heads on frozen video encoders improve ranking over matched DeiT without changing spatial pipeline. Frozen protocol: `configs/NATIVE_FOUNDATION_CANDIDATES_V1.json`; 96 train/24 dev, no holdout, 20 epochs, seed 20260926, BCE on rated clips only, fixed .35 prediction threshold. Physical GPUs6/5 for DeiT/VideoMAE; 2/4 independent InternVideo shards. Outputs `/data/aic/experiments/NATIVE_CANDIDATE_V1`. Per-job timeout7200s. New data/anchor context confounds vs old SUB_C explicitly retained; no model upgrade while running.

### NATIVE_FOUNDATION_CANDIDATES_V1 gate (2026-09-26)

VideoMAEv2-Base did not beat the matched retrained DeiT-S control on native-QVH DEV ranking (Spearman `.06824` vs `.11364`; NDCG `.96109` vs `.96287`) and is rejected for release. InternVideo2-1B reached Spearman `.23474` and NDCG `.96763`, with paired deltas `+.12111` and `+.00476`; both CIs cross zero. It is retained as a promising L-tier engineering candidate only. Full release uses an immutable manifest and never reads the 40-video holdout.

### SUB_E_INTERNVIDEO2_NATIVE_V1 release (2026-09-26)

InternVideo2-1B + native-QVH Temporal U-Net + unchanged YuNet `true_face_smooth` completed frozen 174-video inference. Two fixed parity shards were merged in index order; project/independent/unzip checks all passed. 2,049,501,387 loaded bytes, 1,022,373,889 parameters, 87,637 predictions, zero empty videos. ZIP is READY_TO_UPLOAD; official score remains null.

### Official SUB_E score (2026-09-27)

The frozen InternVideo2 Stage1-1B K700 candidate received official platform score `34.43`. This is a platform score only; official raw/size decomposition was not supplied. It validates the video-native foundation direction and changes the next search toward higher-ceiling bases and controlled post-training.

## 2026-09-27 — SUB_F Stage2 release and K710 gate

- `INTERNVIDEO2_STAGE2_NATIVE_V1`: fixed threshold `.35`, 2 FPS, native QVH 96/24 head, YuNet `true_face_smooth`; official test was used only for frozen inference. 174/174 videos, 87,781 predictions, zero empty videos. Loaded bytes `2,827,511,457`, parameters `1,020,930,561`, L tier. Project validator, independent checker, and unzip regression passed. ZIP `/data/aic/official_test_20260926/submissions/SUB_F_INTERNVIDEO2_STAGE2_NATIVE_V1_FINAL/upload.zip`, SHA256 `a7622bae7b78712549a344f5b5d61fd1762c395a734d82a2c85173bc344a6537`; official score remains null.
- `INTERNVIDEO2_K710_NATIVE_V1`: full 96/24 fit-only after 120/120 feature extraction, no holdout access. Best epoch 12 F1 `.76543`, Spearman `.19818`, NDCG `.96576`; lower than Stage2 (`.21433/.96962`), so no official inference and no candidate promotion.

## 2026-09-27 — K710 official package

- `SUB_G_INTERNVIDEO2_K710_NATIVE_V1`: fixed K710 encoder, native-QVH head checkpoint, threshold `.35`, 2 FPS, YuNet `true_face_smooth`; no test-specific tuning. Full output has 174/174 videos, 87,585 predictions, zero empty videos. Weight bytes `2,049,516,187`; L tier; ZIP SHA256 `a5be193b72e0f18d4a91aabc94f3fd237aa229803db74abc6086100527f61366`. All validators and unzip regression passed. Official platform score was subsequently reported as `34.42`; raw/size decomposition remains unavailable.

## 2026-09-27 — Official score root-cause diagnosis

- New feedback: G `17.46`, E01 `17.46`, Stage2 `34.42`. The remote artifact audit shows shard0 packages are 87-line even-ID subsets and FINAL packages are 174-line merged outputs. This explains the exact near-half score pattern far better than a representation failure.
- K700/K710/Stage2 full outputs use the same spatial policy and have nearly identical selected-frame sets; K700↔K710 changes occur in only 3 videos. No model conclusion is made from G=17.46 until the uploaded file hash is confirmed.
- Report: `reports/20260927_official_score_root_cause.md`.

## 2026-09-27 — K710 full-score correction

- User feedback confirms the K710 174-video FINAL package scored `34.38`. The earlier G=`17.46` was a single-shard coverage failure; E01=`17.46` shows the same near-half pattern.
- Full official scores are now K700 `34.43`, Stage2 `34.42`, K710 `34.38`. The three frozen systems have nearly identical selected-frame sets and identical YuNet boxes on common frames, so further same-protocol foundation swaps are low-information.
- Report: `reports/20260927_official_score_root_cause_v2.md`.

## 2026-09-27/28 — 全选诊断、时间对照与空间裁剪候选

- `B0_ALL_SELECT_YUNET_V1`：全选 + YuNet，不加载编码器。与 V0 174/174 个视频、87,781/87,781 个框完全一致。53,104 参数，232,589 B。ZIP `ab6e7c0ff2841742fa53a8a0ec609cc09e1e036adc53f068a869a3138d6bc31b`。
- `ALL_SELECT_DIAG_V1`（native QVH dev，裁剪固定，2 s 片段集合 F1 代理）：
  - 全选 .2997（hi）/ .4171（rel）。
  - V0 用 train 阈值 .65 得 .3166，CI [−.023, +.054]。
  - E1 未标注=0 重训 .1916，CI [−.196, −.015]；E2 为 .1520。
  - 不冻结时间选择。见 `reports/20260927_temporal_ablation_results.csv`。
- `TIME_MAPPING_CHECK_V1`：训练用 floor(t/2) 对应标签，推理把分数放在锚点时刻，存在 +0.99 s 系统偏移，与 2 s 分辨率是两回事。+1 s 修正后 Spearman .1757 → .1601，登记为缺陷。
- `SPATIAL_SUBJECT_002`（RetargetVid，真值都是最大窗口）：face_scaled − true_face_smooth 平均 −.0488，CI [−.081, −.023]，最差 −.406。这是平均估计，不是上界。
- `SUBJECT_SCALE_DEVCHECK_V1`（QVH dev）：9:16 与 3:4 下 100% 合法且位于 B0 窗口内，平均尺度 .970 / .962，中心抖动中位数约 .001/帧。
- `SUBJECT_SCALE_V1`：12 分片跑在 GPU 1/2/4/5/6/7，最长 2,265 s，基础窗口与 B0 差值为 0。产出：
  - S2：`4d39831ad9c1b3ec652c97f2bb52429d084fdeb7c7196d62dc2ce16e8e1a5420`，19,463,858 参数 / 78,077,396 B，平均尺度 .916。
  - S1：`7c4ef54a49e76823c11ce827f3a541a8a9188671a44f370d167276980099f976`，0 参数。
  - Q2 内部探针：`9c9460fb760c88ff626e07c71d6f3b375149f41bdd639fde44ec53b96a12d235`。
- `QWEN_QVH_DEV_AUDIT_V1`：阻塞，GPU 4–7 被非本项目服务占用。

## 2026-09-28 晚 — 教师 T0/T1/T2（指南 v2）

- `OFFICIAL_SCORE_20260928_QWEN_POINT_BATCH`：登记 NOFACE 42.83、POINT 40.09（绑定 zip f73c399f/684e566f），N0/P0/B0 冻结。
- `OBS_CACHE_RETARGET_ALL200`：170 个新视频（dev2=031-100 + confirm2=601-700）的观测缓存；与旧 30 个合成全部 200 个 RetargetVid 可用集。
- `T0_N0_GATE_AUDIT_V1`：门控/关键帧/坐标审计，重建与已评分包逐帧一致（max_abs 0）。
- `QWEN_RV200_POINT_D1`（1 s 点）、`QWEN_RV200_POINT_D2`（0.5 s 加密，共享帧复用）、`QWEN_RV200_REGION_D1`（区域提示）：分别为 7,266/8,162/8,514 次推理。
- `T0_T2_TEACHER_DIAG_V1` / `T1_T2_TEACHER_DIAG_V2`：confirm2 首次使用。DENSENF−N0 +0.0135 [+.0096,+.0176]；REGIONNF_fit−N0 −0.0239 [−.034,−.015]。
- `QWEN32B_OFFICIAL_V2`：region_d1 因 T1 在公共集被否定在 23/174 中止；point_d2 官方加密推理进行中。
- `T3_YTH_VAL_PROXY_V1` / `QWEN32B_OFFICIAL_V2/temporal`：只删不增时间判断的弱标签代理与官方推理，进行中。

## 2026-09-28 — 平台反馈与最大窗口位置实验

- 平台分数：B0 34.42、S2 30.37、S1 30.13、Q2 27.40（12:46）。
- `OBS_CACHE_RETARGET_V1` / `OBS_CACHE_OFFICIAL_V1`：逐帧缓存 YuNet 观测与 COCO 检测，B0 可从缓存逐帧复现（max_abs 0）。
- 新增确认集 DHF1K 021–030：从 RAR 用 bsdtar 解开，这是首次使用。
- `MAX_WINDOW_PATH_P1A_V1`：L1 DP / 每镜头固定窗口 / 不平滑，都与 B0 持平。REJECTED。
- `MAX_WINDOW_CAND_P1B_P2_V1`（PyAV 15 复核版为 `_V2_PYAV15`）：
  - COV m1.1：confirm +.031，CI [−.014, +.081]，不交付。
  - QWEN：dev +.029，confirm +.067 [+.015, +.125]，合并 30 视频 +.042 [+.007, +.081]。
  - QWENNF：合并 +.028 [−.001, +.061]。
- `QWEN_SUBJECT_POINT_OFFICIAL_V1`：官方 3,106 个关键帧解析全部成功，模型时间 729 s。第 1 次运行在视频 97 上因 PyAV 17 的 trc 问题崩溃；之后全部 174 个视频改用 PyAV 15 的 PNG 重跑。
- 候选包：
  - `MAX_WINDOW_QWEN_POINT_V1`：`684e566f801d2ed4e7b07f4f362d03dad20213631b75a79fc9d213a9a0151e7c`，改动 174 个视频 / 83,483 帧。
  - `MAX_WINDOW_QWEN_NOFACE_V1`：`f73c399f28879de12c3b3a2e796970221c8abd9265ceda48b087190e94b3a96c`，改动 168 个视频 / 75,688 帧。
- `TEMPORAL_CONTRACT_V1`：对齐契约与标签状态盘点。

## 2026-09-28 夜 — 46.84 之后（指南 v3）

- `OFFICIAL_SCORE_20260928_TEACHER_V2_BATCH`：TEMP 46.84（`631d0104…`）、dense 43.76（`972b3ed1…`），SHA 从文件重算。
- `PACKAGE_DENSE_TEMP_V3_20260928`：DENSE bbox × TEMP 掩码，`c702de92…`，174v/78,992f，契约 0 差；DENSE 改动 91.6% 在 TEMP 保留帧。`scripts/mask_combo_release.py`、`scripts/verify_combo_zip.py`。
- `T4_TEMP_BOUNDARY_V3`：转换边界 0.25 s 复核（YTH val 193 次、官方 122 次 32B 查询，模型时间 239 s / 158 s）。YTH F 代理 −0.0014 [−.0034,+.0003]，恢复帧正例率 43.9% < 基线 51.3%。REJECTED，不出包。
- `T5_CROPHEAD_V3`：LIVE-YT-VC 1,800 视频观测缓存（GPU2）与 1 s Qwen 点（val 178 / train 1,622）；RetargetVid 关键帧候选表。结果见 probe 报告。
- `OFFICIAL_SCORE_20260928_DENSE_TEMP_V3`：DT_V3 平台 47.88（用户 22:33），相对 TEMP +1.04，交互 +0.11。新最高已知成绩。
- `T5_CROPHEAD_V3`：H1（x a=.85 b=+.025；y a=.65 b=+.025）dev 选中；LIVE-YT-VC val +0.0062 [+.0034,+.0093]，RV confirm2 +0.0148 [+.0097,+.0200]；H2 MLP 更弱且随 seed 波动。PROMOTED，包 `519a718a…`（174v/78,992f，identity 复现 N0 0 差，keys==TEMP）。探针墙钟 72 s（CPU），LIVE 点推理 178+1,622 视频。

## 2026-09-29 V4 十小时周期（母本 DT_V3 47.88）

公共集数字都不是 AIC 官方分。RV = RetargetVid 6 名标注者；dev2 与 confirm2 均为复用验证集。

- `V4_R0_H1_XONLY`：诊断包。55 个 x 轴视频取 T5 行，119 个 y 轴视频取 TEMP 行。`6ecb766e…`，官方分 null。
- `V4_H1_AUDIT`：按轴、点位置、人数、有脸分层。见 `reports/20260929_h1_transfer_audit.csv`。
- `V4_E1_OBS025`：d4 关键帧（官方 11,753 个，新增 5,765 个）。
  - dev2 +.0067 [.0044,.0091]；confirm2 +.0063 [.0041,.0087]。ADAPT 为 +.0039，未选。
  - 晋级，包 `f575c9ba…`。
- `V4_S1_KEYFRAME_INTERP`：同镜头线性插值，无查询。
  - dev2 +.0110 [.0079,.0145]；confirm2 +.0119 [.0084,.0160]；LIVE val（仅报告）+.0150。
  - 晋级，包 `501cebce…`。
- `V4_S2_KEYFRAME_MEDIAN3`：三点中值。
  - dev2 +.0053 [.0036,.0072]；confirm2 +.0044 [.0028,.0062]。在插值之上只多 +.0005。
  - 晋级，包 `d62ed910…`（可选）。
- `V4_E2_CONTEXT_POINT`：dev2 −.0277 [−.0379,−.0178]，12 好 / 55 差。
  - 主要错误是向中心拉，以及与邻帧混淆。
  - 否决；未跑 confirm2 与官方推理。
- `V4_E3_EVENT_TEMP`：片段角色判断。
  - YTH val 人工 −.0009 [−.0100,+.0076]；YTH train（新）+.0014 [−.011,+.012]。否决。
- `V4_E3B_EVENT_DROP_ONLY`：在 dev 结果之后预注册，只在 YTH train 上检验：人工 −.0037 [−.0152,+.0051]。否决。
- `V4_E4_VISUAL_RERANK`：pilot，32B 看实际裁剪从 3–5 个同尺寸候选中选一个，正序、倒序两问一致才采纳（10,448 次查询，3,414 s）。
  - dev2 +.0093 [.0019,.0173]；x +.0174，y +.0012。
  - 关键帧层面：DENSE .704，选中 .712，候选 oracle .778。选择器只拿到 oracle 增益的约 11%，y 轴为负。
  - confirm2 +.0072 [−.0011,+.0161]，x +.0179，y −.0034，否决（7,078 次查询，4,528 s）；官方推理已中止，不出包。

## 2026-09-29 V5 周期（母本 INTERP 48.67，正式最佳）

- `V5_P0_XRERANK_OFFICIAL`：E4 式视觉复核点接回 INTERP 管线（只 x 行 x 分量，y 与非空间行逐字节不变）。
  - dev2 +.0093 [.0036,.0161]（x +.0187）；confirm2 +.0094 [.0040,.0163]（x +.0188）；y 严格 0。
  - E4 sanity 复现：dev2 .009302 vs .0093，confirm2 .007228 vs .0072。
  - 晋级并打包 `QWEN32B_INTERP_XRERANK_V5_FINAL`（zip `027fd12a…`，predictions `4f7cfe52…`，174v/78,992f，四重身份守卫全过，vs INTERP 差异 21 视频/10,273 帧）。官方分待评。
- `V5_H3_VISUAL`：冻结 DINOv2 ViT-B/14 + 33 常量网格候选评分头，G（26 几何）/V（7 基础+视觉）/VQ（26+视觉）匹配消融；rv 160/40/200 + live 1,416/148/172 单元全 axis0，DINOv2 特征 2,136 单元全缓存；Huber .25、AdamW 3e-4、≤2000 步、dev 关键帧 IoU 选优；预注册双门（gate1 数据集均衡 dev、gate2 rv_dev+confirm2 管线配对）。训练中（cuda:2 seed 1）。
- `V5_P2_GROUND_PILOT`：Qwen 点名 1-4 主体短语 → GroundingDINO-tiny 逐关键帧落位 → qwen 顺序 + 覆盖率 ≥.6 选身份 → 自由轴中心替换。
  - confirm2 24 视频分层：v0 无条件替换 −.0657 [−.1063,−.0317]（7 好/16 差）；v1b 限幅 ≤.75 窗宽 −.0385 [−.0621,−.0200]（5/18）；四分层全负。
  - 否决：单帧 top-1 框中心噪声 > DENSE 蒸馏点；"再看一眼"在候选空间有效（P0 +.0093）、开放词表框空间无效。
- `V5_EXT_PM400_AVEPM`：PM-400 直链 12×3 全 502（协议留档）；社区 GDrive 缓存 68.07 GB 下载中（14:51 时 22.5%，ETA ~17:50）；`portrait_reframe_pilot_v1` 补标任务（320 源/86 类，215/51/54，AVE-PM 标签仅上下文不冒充 GT）。
- `V5_H3_VISUAL`（结果）：三头 2000 步训练完成（G 561s / V 991s / VQ 5,981s，seed 1，best dev IoU .6194/.6272/.6304）。
  - gate1 视觉价值 +.0110 ✅；gate2：rv_dev V +.0117 [−.0118,+.0370]、VQ +.0110 [−.0112,+.0361] CI 未过 ✗；
    confirm2 VQ +.0190 [+.0071,+.0313]、y 轴 +.0172 ✅；live_val（全新）VQ +.0113 [+.0020,+.0207]（仅报告）。
  - 预注册判定 NOT_PROMOTED（gate2 需双集全过）；不出包。G 纯几何四集全正（rv_dev CI_low +.0056）。
  - 消融表 `reports/20260929_visual_scorer_ablation.csv`；registry V5_H3_VISUAL。
- `V5_P0_XRERANK_OFFICIAL`（官方回分 18:25）：**48.49，比母本 INTERP 48.67 低 0.18**。RV x 代理 +1.9 IoU 未迁移；空间类改动连续第三次官方迁移失败（缩放 −4 / 居中 −4.29 / 本包 −0.18）。正式最佳保持 INTERP 48.67。

## 2026-09-29 深夜：V6 E0 审计+接口修复+E2E；E1 工具链；E2 MrHiSum pilot

- `V6_E0_AUDIT`：XRERANK 官方分更正 48.49（配置内 correction_log）；RV 同源双比例合并后 source-group 级 CI 重算（单元级 12/12 逐位自校验过）；gate1 按完整预注册重判 = **FAIL**（rv_dev 单独 best(V,VQ)−G = −0.0031，点估计失败非 CI 宽度）；gate2 rv_dev 源组级 CI_low −0.0114 < −0.002 仍 FAIL。H3 V5 不出包维持，依据更早更强。exposure ledger v1 三点裁定：无 holdout_confirm_reserved/"227" 记录；live_val=T5 confirm3 已暴露（V5 report-only 合规）；跨库同组拷贝继承暴露身份。`reports/20260929_v6_e0_evidence_audit.md`。
- `V6_E0_INTERFACE`：H3 部署接口 5 项预登记缺陷 + E2E 新发现 3 项全部修复（X 生成并存入 unit、--kind G/V/VQ、轴可选拼接器 G1/G2/G3、--exact-mother 34 列去重、ds=(0,0) 域审计；追加：unit 缺 X、34 列 broadcast、官方关键帧 1fps 缓存 vs dense 0.5s 网格——从源 mp4 补渲染 2,882 帧，provenance 在档）。
- `V6_E0_E2E`（官方 174 真实数据端到端）：tables 174 units / 5,586 kf / 34 列，`exact_mother_added_frac` 均值 **0.879**（量化母本吸附缺陷的影响面）；score VQ+V 全量 rerank + 域审计；splice 双轴诊断包 G1/G2/G3 + 独立校验 + ZIP 往返全过（vid12 x 轴 VQ 148/148 帧变；vid0 y 轴 V 578/630 帧变）。诊断包 NOT_FOR_UPLOAD。证据 `reports/v6_e0_e2e/`。
- `V6_E1_TOOLCHAIN`：portrait_reframe_pilot_v2 三件套（schema v2：119 竖屏源同方向更正、连续窗口高度、双标注 0.1h 第三人规则）+ 冻结 temporal manifest v2（30/320，sha256 排序规则）+ 自包含标注工具（托管浏览器实测：视频加载/帧行/时间段行/导出→校验器 0 错误；实测发现并修复时间段 falsy-zero 无法闭合 bug 与 0 长段垃圾行，已加 <0.05s 取消守卫）+ validate_annotations.py（永不合并行）。人工标签未到，训练不启动，不用伪标签冒充 GT。
- `V6_E2_MRHISUM_PILOT`（冻结 release `mrhisum_feat_subset_v1`）：全量导出 27,877 视频（去量化 v1 映射 (q+0.5)*2/255−1）；**特征提取复现**=从 asia 镜像重下 train0026（md5 对 download plan 一致）重跑 extract_shard，15 视频 rgb/audio/labels **逐位 IDENTICAL**；**部署复现**=27,877 视频全量前向两进程 sha256 逐位一致（`87d6d15b…`，5,624,693 steps）；**训练不可逐位**：4 runs dev Spearman 0.2774–0.2909（极差 ~0.013），根因 `upsample_linear1d_backward_out_cuda` 无确定性实现（warn_only 下告警在档）；TemporalUNet(1152) 头 1,712,897 参数 = 0.0017B（≤0.1B 档 k=1.00）。
- `V6_SEMIFINAL_K700`：按 V6 指令建立复赛候选 `SUB_SF_K700_SEMIFINAL_V1`——基座 SUB_E_INTERNVIDEO2_NATIVE_V1（InternVideo2 Stage1-1B 原生 QVH 头 + true_face_smooth YuNet，官方 34.43 = K700，最佳已回分 ≤9B 系统）。预测逐字节承接（platform 对该内容已回 34.43），唯一变更是元数据：全行 `model_params_b=1.022373889`（encoder+时间头+YuNet 总参数，≤9B 档 k=0.90，`size_coefficient_params` 实测）。身份链断言全过：源 predictions sha256=回分记录值、内容等价（除新增字段）、index sha256 与冻结清单一致、独立校验+ZIP 往返 valid。zip `87cdb7ff…`（1,053,965 B），READY_TO_UPLOAD_SEMIFINAL，未上传，不预报分数。教师系（Qwen3-VL-32B+YuNet≈33.36B>9B）不合复赛上限，仅作研究基线。构建脚本 `scripts/v6_semifinal_k700_package.py`。

## 2026-10-07 R11 四层路线 Day-1（空间主攻；S2a 检测器源收口）

计划 `reports/20261007_round11_plan.md`（v1 `0ad156b…` → v2 `3dc6713…`，用户裁定四层转 R11、空间优先；v2 补 P-O 观察线、S2 扩三源、外部方法对照）。执行地 kmsp05/910A（通道 `ssh -p 44000`）。day-1 判定全文 `reports/r11/20261007_day1_verdicts.md`。

- `R11_S1_SCOREFIELD_VITERBI`：B3_s0 分数场（四池 8,972 行）L1 Viterbi，dev(live_dev) 选 λ=0.25，locked-out live_confirmation **+0.00243 [+.00043,+.00450]**；rv_dev +0.00391 跨 0。晋级门（≥+0.005 双池）未过，关闭线（<+0.002）未触发=灰区。⚠️ 源内帧序置换对照 +0.00355 ≥ 真实 +0.00228——增益不依赖帧序，更像平滑正则。裁定：解码层微弱真实正收益（零参数部署端免费后处理）；「跨帧裁剪轨迹」机制无撬动级证据，P1A+S1 两次持平后轨迹支路永久关闭，空间主杠杆压到 S2。资产 `aic/r11_trajectory.py`（λ=0 恒等断言）。
- `R11_S2C_SPEECH_AUDIT`：R10 AST-527 全池 1,963 源重读（概率域口径修正后）：speech 主导 41.7%（819 源）≥30% 门 → **S2c Light-ASD 臂按预注册恢复**；BGM 主导 36.4%。⚠️ 首版 logits 域误读（11.1%）已修正。
- `R11_V0_DENSE_PREMISE`：三个 dense 观察缓存路径均不存在 → NOT_RUN 合法记录；dense-view KD 臂保持关门。
- `R11_V1_EVENT_ROLES`：Qwen2.5-VL-7B 生成式角色链（PHD2 媒体+R10 槽时间轴，修 3 管线 bug 后）输出单调角色（全 SETUP/全 RESULT）——孤立 16s 窗设计下 7B 无判别力，**V1-7B 挂起**。32B 线同日判死：vllm-ascend 官方从未支持 910A（v0.7.3/v0.11.0 硬件表+FAQ 明文），0.18.0 引擎初始化 hang，transformers 32B generate 在 10-04 重建栈 hang（71-78s 加载正常、首 forward 卡 `Tensor.to()` eventfd，TEFUSION 死循环轮询 `GetFinishedCompilationTask`，py-spy 双栈定位）；910-ops 9.0.0 补装+env 顺序修正后仍 hang，根因未定，排障挂起（详见 day-1 判定六/七）。7B 下午可跑/晚间不可跑与 910-ops 装卸循环时间线吻合，栈状态与重建基线存在未表征偏差，未再盲试。
- `R11_S2A_DETECTOR_EXTRACT`：RT-DETR r50vd NPU 前向崩 AsStrided（910A 实为 lspci 19e5:d801，npu-smi 显示不可信）→ CPU 32 分片。PHD2 池 3,917/3,917 frag 全量落盘（`r11_s2a/p0..p31`），coverage 0.766 [0.717–0.830] 无分片低于 0.60 门，0 错误。教训入脚本：HF_ENDPOINT 先于 import；当前栈严禁 jit_compile=False；RT-DETR logits 无 background 列。
- `R11_S2A_READOUTS`（判定 `reports/r11/20261007_s2a_readout_verdict.md`，数据 `reports/r11/s2a_readouts.json`）：⚠️ day-1 提取跑在 PHD2（keep-mask 协议池、无空间候选），readout 数据合同要求 B3 分数场同池 → 补 live 池（LIVE-YT-VC 351 源×30 帧）RT-DETR 提取 10,530/10,530 帧（32 分片 ~35min，口径与 PHD2 提取一致）。三读数（dev=live_dev 选参、eval=live_confirmation 单次、S1 λ=0.25 同口径）：
  - a. 新池 oracle 抬升：**+0.0000045** vs 门 +0.02 —— **FAIL，低 3 个数量级**（top3 口径 +0.0000058 同量级）。机制：9-16 窗自由轴 span≈437px、129 格点距≈3.4px，检测窗与最近格点窗几乎重合——格点已致密覆盖检测位置（与 F3 33 格点 regret 0.0055 同构）。dense 候选支路按预注册关闭。
  - b. B3 argmax 与检测框一致率（IoU≥0.5）：0.506（dev 0.502）——半数重合。
  - c. 检测先验条件重排：dev β*=0.5 → eval **+0.00454 [+.0021,+.0071]**，CI>0 但 <+0.005 门——压线未过，弱正灰区，不满足打包条件，不出包不占名额。
  - **裁定：S2a 检测器源收口关闭**。S2 三源余 S2c（Light-ASD，前提门已过）与 S2b 跟踪连续性（CPU 零依赖；PHD2+live 两池检测框均为其现成资产）。
- 过程教训（本轮新增）：① 执行前先核对读数协议的数据合同再跑提取（day-1 池错位）；② LIVE-YT-VC 帧号=解码顺序计数（流首帧 pts 非零 +4 偏移），PyAV seek 取不到首关键帧前帧；③ `pgrep -f` 链式任务自匹配死锁，用日志标记判完成。提取/读出脚本 `scripts/r11_s2a_live_extract.py`、`scripts/r11_s2a_readouts.py`（commit `9e5ec4f…`）。

## 2026-10-07 深夜：R12 启动（NPU 重启后探针矩阵 + P0-a 对接审计进行中）

计划链 `reports/20261007_round12_r45_plan.md`（v1，fecffac）→ `20261007_round12_frontier_addendum.md`（v2，3ab6db6）→ `20261007_round12_frame_ranking_frontier_v3.md`（v3，9410741）。用户指令：批准 R12，等 910A NPU 恢复再做后面操作。进度报告 `reports/20261007_round12_progress.md`。

- `R12_NPU_PROBE_REBOOT`：整机重启完成后 910A 探针矩阵（Qwen2.5-VL-7B 单卡 generate 金标准）。probe1 卡 0 OOM（被两非我方进程各占 16.3GB，未触碰）；probe2 卡 6 默认 SDPA **崩溃**（`can not cast format when output is input`，ERR01007，sdpa_attention.py:158）——**新坑：910A 必须 `attn_implementation="eager"`**；probe3 卡 6 eager：加载 56.8s 正常、基本算子真实执行，但 generate 24 token >14min 零产物、AICore 三采样 0%、裸点输出=**判定八同签名，整机重启未恢复**。根因指向更新：判定八的「host 残留态」假设削弱，更可能是 **CANN 9.0.0 算子编译层固有缺陷**（TBE 编译任务永返 size=0）。升级选项（华为工单 / CANN 8.3.RC2 双栈降级实验）属用户裁定，未执行。CPU 批处理教训：32 分片曾被外部信号整批扫掉（零异常无 OOM），**setsid 重启后恢复**——910A 长跑 CPU 批处理一律 setsid 入册。
- `R12_P0A_PHASH_BANK`（进行中）：426×PHD² 对接审计素材链打通——官方 426 intake（duration/fps/sha256 全）+ keyframes 指纹**已算完**（426 行）；PHD² 在盘 13,993 源（399GB，文件名=YouTube ID）+ selections train/test（110,611+10,222 vid，键=YouTube ID，per-user GIF 区间）。指纹库 32 分片（每 2s 一探针，上限 400/视频）断点续跑中。匹配脚本 `scripts/p0a_match.py`（Hamming 矩阵→候选→offset 复验→GIF 区间 join）已冒烟。预注册双结局：A 对上子集→官方域监督解锁+窗口切割函数→复赛镜像池；B 整批不过→与 V10 一致化，对接线永久关闭。V10 当年 174 档案重读：时长匹配（71 scored）+首帧 mae≈25+中后段发散 70-80+cover_share=0 →「非连续裁剪」的精确形态在案。
- `R12_FRONTIER_V3`：TRINITY（ECCV 2026，2608.29577）Table 2 = Mr.HiSum 最全 fine-tuned SOTA 表；AHA 在 Mr.HiSum 低于专用监督基线（64.19 vs 67.20@ρ50）→ 降级为 LoRA 初始化备选；R12-GATT 改指 TRINITY 配方（冻结 Siglip2+Transformer d5/h512/1D-RoPE+BCE、标签保峰聚合）；新增 R12-EMO（情感峰值辅助视角，批次按视角隔离）。

## 2026-10-08 凌晨：P0-a 对接审计收口（结局 B）+ P0-b Mr.HiSum 排序双臂启动

用户约束：NPU 已恢复但让位 32B V1 线（r11_v1_event_roles_32b_hf 满占 8 卡），本轮只准 10 分钟探针复验 + CPU 动作。

- `R12_P0A_MATCH_VERDICT`（判定 `reports/r12/20261007_p0a_match_verdict.md`，产物 summary/matched 同目录）：指纹库完成（32/32 分片，13,937 vid、1,394,739 探针，0 解码错误）→ 全库匹配+常数 offset 复验：**426 集仅 1/426 过验证**（vid31↔9cnBxSB1jUE，offset 103.0s，med_ham 14，inlier 0.667）且窗口内无 GIF 区间 → **结局 B，对接线按预注册永久关闭**，与 V10 初赛 174 集审计（非连续裁剪）一致化。边界如实记录：在盘 13,993=全集 11.6%，本判定不外推到未在盘 88.4%；复验协议为常数 offset 模型（变速会被拒真，V10 全轴对齐已独立对冲）。监督解锁转 QV/user-vote/EMO 三臂。
- `R12_NPU_PROBE4_ABORTED`：探针按判定九配方更新（use_fast=False + 推理 jit_compile=False + eager）后无卡可跑——8 卡各被 32B V1 进程占 20-22GB（剩 ~10GB < 7B FP16 15.2GB）。裁定：不抢卡，复验以判定九验证记录（7B 24s/源 ok=3、32B hang 解除）为准，探针脚本待有空卡窗口执行。
- `R12_P0B_RANK_PILOT`（启动）：Mr.HiSum 帧级排序，损失单变量 pairwise（Mr.HiSum 官方配方，帧对采样 |Δgtscore|≥0.1、softplus）vs MSE（B3 现状口径）。模型=TRINITY 式轻全局排序头（1152→512 投影 + 3 层 Transformer 8 头 + 1D RoPE + 逐帧标量头）。数据全在盘：yt8m 特征（V6 逐位复现）+ gtscore（Most-Replayed 曲线）+ 官方 split 27,892/2,000/2,000。**对照口径=官方基线 50.80/67.20（同 yt8m 特征）；TRINITY 40.98/69.06 自有 CLIP 特征不可比**。协议：mAP@ρ=15/50，test 2000 单次终评。冒烟过（300 视频 1 epoch 5.5min 含预载；RoPE 维度 bug 首跑修复）。全量双臂过夜后台（3 epoch，CPU 32 线程×2）。

## 2026-10-08 凌晨二批：NPU 窗口验证双 PASS（探针 + M01 smoke）

用户批准执行排程清单；32B V1 线完活退出后卡 1-7 空闲（卡 0 有 2 个非我方残留进程，避开）。

- `R12_NPU_FINAL_PROBE`（**PASS**）：判定九配方（`use_fast=False` + 推理 `jit_compile=False` + eager，卡 1，timeout 600）——`PROBE_RESULT gen_ok=True gen_s=3.4 total_s=35.2`（加载 30.9s、24 token 3.4s、输出内容正常）。对照：故障态同探针 >14min 零产物、判定九基线 24s/源。**NPU 推理栈完全恢复，判定九配方实证**。排程 cron 已删（用户手动批准提前执行）。
- `R12_M01_SMOKE_R12`（**PASS**）：`v11_m01_npu_smoke.py --cards 2`（VideoMAEv2 ViT-B fwd+bwd，safetensors 直载路径）——grad_norm 0.0149 非零有限、update ok、save/reload 一致 7.2e-05（fp16 量级）、peak 5.8GB、`"pass": true`。**NPU 训练栈可用**（R12-VMAE 臂的栈前提复验通过）。
- 特征资产盘点（R12-GATT 前置）：B3 同款 SigLIP pooled 特征在盘——`LFM_V8/{live_feats,rv_feats_rot,diag_feats_dense}`（53G/5.1G/53G，per-frag `pooled (768,) f16` + grid）。**GATT 头与 B3 同表示换头=干净单变量**；B3 时间监督合同在 `v8_s_train_multidata.py`，下轮精读复刻后再训（与 P0-b 损失判定联动定配方）。

## 2026-10-08 凌晨三批：全批准执行（VMAE 免提取裁定 + GATT 方案 + AHA 资产落地）

用户「全部批准执行」（VMAE 提取 + GATT 构建 + AHA 下载）。

- `R12_VMAE_FEATS_ALREADY_ON_DISK`（**降本裁定**）：`LFM_V11/phd2_vmae_feats/` 6,001 npz 全量在盘（V11 已提取；每 frag `ord [8,768]` 16 帧 linspace 窗 + `rep [8,768]` 静态控制）——原「8 卡 1-2h 提取」取消。臂改两档：VMAE-F（冻结特征轻头，CPU，rep 条件=运动贡献对照）先行；VMAE-E2E（M01 桥 NPU 微调）视 F 档读数再定。⚠️ `v8_s_train_multidata.py` 经核实为**空间头**（129 格点+IoU），B3 时间头合同需从 `r8_temporal_dualbase.py` 一线考古——GATT 数据合同下轮完成。
- `R12_GATT_EXEC_PLAN`（`reports/r12/20261008_gatt_vmae_exec_plan.md`）：架构固定=TRINITY 式（768→512+3 层 Transformer+1D RoPE，与 p0b 同构）；**配方冻结条件预注册**=P0-b 终评损失胜者（pairwise→GATT 直接 ranking）；本地只做预检，判定走平台差分。
- `R12_AHA_ASSETS`：官方权重=**LoRA adapter**（951MB zip：adapter_model.safetensors 1.2GB + tokenizer；base=`lmms-lab/llava-onevision-qwen2-7b-ov`；modules_to_save 含 connector/mm_projector/lm_head + **informative/relevance/uncertainty 三解耦头**）——本地 Google Drive 下载完成（80s，12MB/s），scp 910A 进行中；基座 16GB 级 hf-mirror 下载已启动（`/data/aic/pretrained/llava-onevision-qwen2-7b-ov`）。R12-AHA 成本结构更新：需基座+LoRA 合成，推理 24GB 级单卡可容。

## 2026-10-08 上午：AHA smoke PASS（R12-AHA 臂栈验证闭环）+ NPU 自由窗口批注

用户「NPU 空闲，随便你用」授权。VMAE 提取裁定不做（PHD2 预检特征已在盘，live 池特征属打包链用料，避免提前烧 1.5h NPU 在未过预检的臂上）。

- `R12_AHA_SMOKE_V1`（**PASS**，12 次迭代调试链全部固化）：aha49 专用 venv（transformers 4.49.0+peft 0.12.0+torch 2.9.0 降回 pin）+ 基座 llava-onevision-qwen2-7b-ov（15GB）+ LoRA adapter 合成加载 52.6s → **8.63B 参数、relevance/informative/uncertainty 三解耦头全部在位、NaN=0、NPU forward 0.3s、logits finite**。配方坑固化：①bitsandbytes aarch64 装不上→stub（需补 __spec__）；②**PYTHONPATH 覆盖 CANN TBE 路径=OpCompileProcessor init failed 根因（追加非覆盖）**；③AHA 代码需 RANK/WORLD_SIZE 等单卡分布式 env；④AHA forward 仅在 labels 非 None 时把 logits 放入 tuple；⑤generate override 与 peft 双传 return_dict（smoke 用 forward 绕过）。产物 `scripts/aha_smoke_v1.py`。
- AHA Mr.HiSum 官方复现（追 +8.3% 数字）需**原始视频**（--input_dir yt8m/videos，我方只有 frame features h5）——验证集子集下载（~数百 GB）为用户数据管线决策项，不在本轮。
- 二分记录：transformers 5.x 加载 OneVision 权重直接 shape mismatch（4.49 venv 是必选，非可选项）。
- P0-b 中期信号：pairwise 臂 ep0 step 20k val mAP15/50=**26.23/61.04** vs mse 臂同期 21.6/55.9——**pairwise 初步领先 +4.6/+5.1**，待终评（明日）。

## 2026-10-08 下午：四步流程准备全部就绪（watcher 已挂，待 P0-b 终评自动执行）

用户批准「终评→冻结 GATT 配方→r8 考古→GATT/VMAE-F 训练」全流程。

- `R12_B3_FEAT_CONTRACT_FIX`（考古重大修正）：**B3 的 pooled 768d = LFM2VL-450M vision tower**（`v8_live_extract.py` vision()：帧长边 640、min/max_tiles=1、grid mean、fp16 eager），**不是 SigLIP**——此前表述全部更正。另两条考古：`r8_temporal_dualbase.py`=QVH 域（9,066 窗 saliency）、`v8_t_train_value.py`=DHF1K 域（rank-percentile 标签）——B3 的 PHD2 域时间头原脚本未定位，GATT 采用自足合同（平台差分判定，不依赖本地复刻）。
- `R12_P0C_LFM_POOL`（完成）：PHD2 域 B3 同表示特征提取——`scripts/p0c_phd2_lfm_pooled.py`（vision() 逐字移植 + patch_siglip2_npu 修复版；与 live_feats 的差异=patch，已声明），冒烟 6/6 零错误 → 8 卡全量 **6,001/6,001 npz**（~30min），每 frag `{ts[8], pooled[8,768]f16}`。
- `R12_GATT_TRAIN_SCRIPT`（就绪+冒烟过）：`scripts/r12_gatt_vmaef_train.py` 双特征同框（--feat vmae|lfm，单变量=特征）× 三损失开关（--loss pairwise|mse|bce，P0-b 胜者冻结，平局→bce/TRINITY 口径）。数据合同：index_clean 5,886 frag + selections GIF-union slot 标签（slot keep-rate 0.371 非退化）+ 源级 md5 80/20（4,721/1,165）——冒烟 5,886 全命中特征、训练循环正常。
- `R12_WATCHER_CHAIN`（已部署）：`scripts/run_after_p0b.sh` setsid 常驻——轮询双 P0B_FINAL → 冻结 loss → LFM 池就绪检查（已满足）→ 自动启动 4 臂全量（vmae/lfm × seed 0/1，50 epoch，GATT_FINAL 幂等跳过）。日志 `r12_p0b/chain.log`。

## 2026-10-09 中午：P0-b 迁 NPU 八卡灌满（用户裁定单一权威结果）

用户否决「CPU 慢版+NPU 快版交叉验证」（两套数值可能不一致=脏），裁定：CPU 双臂杀掉停用，8 卡 NPU 灌满。配置矩阵升级：**pairwise×4 seeds（卡 0-3）+ mse×4 seeds（卡 4-7）**，终评=4-seed 均值±std（seed 方差直接可见，对齐平台噪声管理口径）。

- `R12_P0B_NPU8`：`p0b_mrhisum_rank.py` 加 `--device` 开关（权重 CPU 初始化后 .to(npu)，避开 NPU InplaceNormal 初始化 bug，jit_compile=False 训练按 M01 先例）。冒烟：500 样本 18s（**~40 倍加速**）、P0B_FINAL 管道通。8 卡全量：3 epochs ≈ 45min/臂，实测 ~2,000 步/min（CPU 41 步/min 的 **~50 倍**）。
- `R12_WATCHER_V2`：`run_after_p0b.sh` v2——等 8/8 P0B_FINAL → 4-seed 均值判胜（diff≥0.5→pairwise，<0.5→bce，else mse）→ `verdict.json` 落盘 → 自动启 GATT 8 臂 NPU（vmae/lfm × seed 0-3，GATT_FINAL 幂等）。`r12_gatt_vmaef_train.py` 同步加 --device。
- ETA：P0-b 终评 ~12:45；GATT 8 臂 ~15:00 前后全部出 dev-AP。CPU 双臂 ep0 中期曲线留档（pair 28.14/61.48 vs mse 25.73/58.62 @25k，方向旁证不作为判定依据）。

## 2026-10-09 下午：P0-b 终评判定 pairwise 胜出 + GATT 8 臂上线

- `R12_P0B_VERDICT`（**frozen_loss=pairwise**）：8/8 NPU 终评（test mAP@ρ=50）——pair 4 seeds [58.95,58.69,59.72,59.30] 均值 **59.17±0.38**，mse [56.56,53.74,52.18,53.51] 均值 **54.00±1.60**，diff **+5.17**≥0.5 预注册规则→pairwise；pairwise 又高又稳（seed std 小 4 倍）。mAP15 25.66 vs 18.14 同向。对照官方基线 50.80/67.20：轻头在 yt8m 特征上 mAP15 仍大幅落后（预检只判损失单变量，不作 SOTA 宣称），主战场=迁我方域。
- watcher v2 的冻结段踩坑：python `glob.glob` 不支持 bash 花括号 `{pair,mse}`→匹配 0 文件→均值段异常→LOSS 空→首发射 8 臂全挂 `--loss` 空参。**手动读数判胜+手写 verdict.json+手动发射**（watcher 修正留档）。
- `R12_GATT8_ONLINE`：GATT 8 臂（vmae/lfm × seed 0-3，pairwise，50 epochs，NPU 8 卡）调试链 3 波修复（pairwise_loss randint 需 device= 参数；slot_ap 需显式 device 形参）后健康运行——**首 epoch vmae s0 dev_ap=62.26**（2.9min/epoch，随机基线≈keep-rate 口径），预计 ~15:30 全部完成。

## 2026-10-09 下午：GATT 8 臂终态 + R12_GATT_LFM_SEMIFINAL 包交付

- GATT 8 臂终态（50 epochs NPU，best_dev_ap）：lfm 64.89/67.70/66.00/64.52（均值 65.78）vs vmae 63.93/63.27/65.02/63.68（63.98）——**lfm 胜 +1.8**，B3 同表示特征优于 VMAE 时空特征（表示墙不在视觉塔）。vmae 臂预检判负（平台复核留给差分包，不单独出包）。过拟合出现（train>dev 5-8 点）但 best-dev ckpt 兜底。
- 判定十一下一颗：frozen 组合=GATT 头（6,703,617 参数）+ LFM2VL pooled 1s keyframe + pairwise 损失（P0-b 4-seed 判定）。
- `R12_GATT_LFM_SEMIFINAL` 包交付：`/data/aic/semifinal_20261001/submissions/R12_GATT_LFM_SEMIFINAL/`（zip 1,383,849B，sha256 5381b13266df71f2...，本地副本 downloads/）。核验：426 行、keep 192,928 帧、**keep 集与 B3_TEMPORAL 在 426/426 视频全不同（真换了打分器）**、bbox 逐字节承自 B3_SEMIFINAL、457.0M 参数/930.6MB=0.90 档同父包。单变量=时间打分器 TCN→GATT；对照=34.88；回分差分归因：正→头架构杠杆；负+预测集有差异→平台不认；无差异→工程问题。
- GATT 推理特征=官方 temporal_feats 缓存（B3_TEMPORAL 当时推理的 1s keyframe pooled，零提取成本复用）。

## 2026-10-09：R12_GATT_LFM 官方回分 35.17（+0.29 vs 34.88）——时间轴首次平台转正

- 官方回分 35.17。预测集差异 426/426（keep 集全换）仍净 +0.29 = 打分器打包效应（架构+监督+损失三子变量捆绑）。规则更新：**比赛已取消 k_size（用户明示），后续不再使用 k_size 口径**。
- 距 45 差 9.83；单换打分器幅度不够 → 下一级杠杆=监督源。计划与分析全文 `reports/r12/20261009_r12_gatt_verdict_next.md`：P1 R12-QV/UV 监督源升级（同链换标签，2 seeds 各）、P2 keep 0.60/0.70 零训练快包、P3 容量/特征增强（P1 正后叠加）、P4 S2c 读出判定（今晚）/AHA 挂起。
- 冻结资产：GATT best ckpt `lfm_pairwise_npu_s1_best.pt`（67.70）；推理链（temporal_feats→GATT→掩码→`r12_gatt_package.py`）全通，后续包边际成本≈换标签重训+一条组包命令。

## 2026-10-08 16:30：P1/P2/P3 全量发射（用户批准，8 卡灌满）

- **P2 快包 ×2 已出**（frozen ckpt lfm_s1_best，13s/包）：`R12_GATT_LFM_KEEP070SEMIFINAL`（sha b2554d20…，169,567 帧）、`R12_GATT_LFM_KEEP060SEMIFINAL`（sha c6082a76…，145,764 帧）。单变量 vs 35.17=keep 深度（manifest 已声明）。本地副本 downloads/。
- **P1-UV 2 臂**（cards 0,1）：trainer 新增 `--labels vote`（逐用户 keep 指示均值→软标签；单用户 src 与 union 等价，差异仅多用户 src）。tag=lfm_vote_npu_s0/s1。
- **P3 3 臂**（cards 2,3,4）：d6（`--n-layers 6`）、e150（epochs 150）、concat（lfm pooled ‖ vmae ord 1536d，`--feat concat`）。
- **P1-QV 主攻**（cards 5-7 + 待臂完 0-4）：QV saliency 实际形态=仅 relevant_windows 内有标注（3 人 0-4 分、2s clip）→ 路线 B：窗口内 clip=3 人均值/4 软标签（多 query 取 max）、窗口外=0。8,619 段（train 7,100+val 1,519，全部 150s 切段）**文件对齐率 100%**，keep_rate=0.296。提取=r12_qv_extract.py（vision() 逐字 p0c 合同+grab/retrieve 顺序解码+生产者线程流水），8 分片断点续跑；pilot 20 段测吞吐后全自动（r12_qv_chain.sh：臂完→全量 8 卡→2 seeds 训练）。脚本 commit 5e8e916。

## 2026-10-08：P2 keep 深度差分回分——0.80 确认为最优，P2 关闭

- KEEP060 = 33.82（−1.35 vs 35.17）、KEEP070 = 35.10（−0.07）、KEEP080 = 35.17。三点曲线：keep 0.80 局部最优。
- 归因：①0.70→0.80 仅 −0.07（噪声级）= 排序器在头部 70-80% 区间近乎无判别力；②0.60 大降 1.35 = 被剪的 60-80% 区间含真实正例，当前排序质量撑不起更深剪枝。**45 分需要排序质量跃升（监督源），不是剪枝策略微调**——keep 维度关闭，0.80 保持默认；仅在 P1 监督源出正、排序质量跃升后重测。

## 2026-10-08 18:05：UV/P3 五臂判负关闭，卡全部支援 QV

- 全程 best_dev_ap（vs union 基线 64.52/64.89/66.00/67.70，均值 65.78）：vote s0=65.33@ep6、vote s1=64.76@ep8、d6=64.35@ep0、e150=64.47@ep2、concat=64.30@ep7。**五臂全部在前 8 epoch 见顶后回落 3-7 点，无翻盘迹象**（被杀时 ep17-37）。
- 判定：①UV vote 软标签无增益（83% 单用户池下 vote≈union，差异部分是噪声）→ R12-UV 关闭；②容量臂 ep0-2 即见顶=PHD2 frag 池（~4,700 样本）撑不起 d6/e150/concat → P3 在 PHD2 域内关闭（QV 联合池出正后可重测）。
- 行动：五臂 kill，cards 0-4 → shard 3/4 提取上 card 0/1（~42min 全齐）→ chain2 phase5 自动开 QV 训练 2 seeds。涨分收敛到 QV 单杠杆。

## 2026-10-08 19:50：QV 全链完成——两 seed 包就绪，待平台差分

- 提取 8,619/8,619 全齐（8 分片零错误）。训练 2 seeds：**QV val AP 峰值 40.74（s0@ep4）/ 41.52（s1@ep3）**，此后过拟合回落（train 66-68）；随机基线 ~30（正例率 0.296）→ 域内超随机 +37%（PHD2 域 +82%，域内信号偏弱）。
- 部署适配：训练 token=2s clip，部署 1s keyframe → 组包脚本 `--feat qv` 分支：stride-2 打分+最近前置 token 广播（与训练分布对齐）。
- **两包出好**：`R12_QV_LFM_S0`（sha e6a73354…，192,933 帧）、`R12_QV_LFM_S1`（sha 527c59ee…，192,766 帧）。**keep 集帧级 sym_diff vs 35.17 版 = 24.5%/23.5%**——排序真差异显著，非噪声重排。单变量=监督源（manifest 已如实声明 QV 监督+stride-2 适配）。
- 判定待平台：≥+0.3 → QV 监督成立，叠加 QV+PHD2 联合与 TVSum 系扩源；≤0 → 外部 saliency→部署域迁移证伪，转入联合/ensemble 形态前先停。
- 途中修：trainer 短段零 pad（np.stack 形状崩）、deploy-check 预算口径、组包 $S 命名覆盖、manifest 文案分支。commit 链见 git log。

## 2026-10-08 20:30：QV 平台判定——失败关闭（S0=35.08 / S1=34.94）

- 双 seed 一致小负（−0.09/−0.23，sym_diff 24%）→ 非 seed 噪声，是「QV saliency→部署域」迁移无效。R12-QV 关闭。
- **关键证据链闭合：PHD2 keep / Mr.HiSum / QV saliency 三种外部监督，平台分全部锁在 34.9-35.2**——平台对「换什么监督」不敏感（V10 finding-3「AP-tied 两头平台差 0.88」再次印证）。瓶颈假设转向：特征表征/底座（450M 单帧 pooled 特征承载不了帧级重要性判别）。
- 用户情报：**对手已冲到 45**（+9.83）。45=掩码质量需比随机好 ~38%，我们当前 ~7%。启动全文档复盘（模型选型/底座/输出形态三轴）。

## 2026-10-09 深夜：R13_F3_MRHISUM_PAIR 平台判定——32.74（−2.43），监督源五连败收口

- Mr.HiSum 27.9k pairwise 头（域内 ap50 65.78 ≈ 官方基线 67.20）平台 32.74，
  打破「34.9-35.2 锁定区间」下限。fps 广播 bug 嫌疑已排除（25/25 帧数元数据精确）。
- **监督源维度穷尽关闭**：PHD2 35.17 / QV 35.08 / VLM-zs 34.59 / KD 34.61 /
  Mr.HiSum 32.74——五种监督全部 ≤35.17，唯任务同源的 PHD2 出正。
  域内质量与平台分脱钩再 +1（65.78 域内 = 最差平台）。
- F1 配方审计闭环（pairwise 唯一正确损失，BCE/MSE 证伪）但该能力不迁移平台。
- 剩余未测轴：①PHD2 域 × pairwise 损失臂（R12 五臂未测）②27B 域内蒸馏
  （JSON 率 85.4% 过门槛，推理合规）③输出形态/空间已关闭。

## R15d（2026-10-15）官方指标天花板前沿 + (d,R̄) 不可识别性（解析，无训练）

- Hypothesis：`keep_scan_closure.md` §3 反解出的"空间轴不是瓶颈（R̄=1.00）"可被证伪。
- Change：无模型改动；纯解析反解官方公式 `F_i = 2·S_IoU/(N_pred+N_gt)`。
- Result：**证伪成立**。测得的 6 个官方分数无法识别 (d, R̄)，d 只在 `[0.21029, +∞)` 上有下界；
  文档采用值 0.2103 是该下界，即 R̄ 的最大值，从最有利情形外推。
  - 现役 p=0.8534 的**排序天花板 = 39.54 分**（d=0.2103；d=1.0 时为 37.50），**已在 89.8%**。
  - 现役 keep 率下**不可能达到 40 分**（需 keep ≤0.841）。
  - lift = ρ/p 实测仅 **1.053**（p=0.8534）～**1.161**（p=0.5451, d=0.4）；随机 ≡ 1.000。
  - 修好空间到 R̄=1 值 **+0.00**（R̄=1 世界）～**+9.42**（R̄=0.784 世界）。
- Size：无变化。
- Training Time：0。
- Conclusion：撤回 C3。P0 改做**随机排序探针**测 d（可识别，成本 2 次提交）。
  P1 用 AHA relevance 头把 lift 从 1.05 推向 ≥1.17。证据 `scripts/r15d_ceiling_frontier.py`、
  `docs/r15d_ceiling_and_identifiability.md`。

## R21（2026-10-15）Claude 截图方向实物审计 + R17/R18/R19 补充（CPU 只读，0 GPU，0 训练）

- Hypothesis：近期用于 steering 的本地数字（内容上限、CLIP 优于 VPB、均匀窗失败、shot 优势）
  是否为可复现的实物事实。
- Change：无模型改动。读原脚本/日志/缓存/既存 checkpoint + CPU 冻结回放 + 数学反例。
  远端 `R21_CLAUDE_AUDIT_20261015`，commit `d45cc09`，干净源码 `a7e4a51`，3 子任务全 exit 0（4.2 s）。
- Result：**多数不成立。**
  - **REJECTED**：`0.3340` 是内容上限（实为源全在 train 且用于选模的泄漏读数）；
    残差 B 是"纯内容"（常数预测得 B=1.0 未用图像；位置 ramp 得 0.882349）；
    平均曲线相关高 ⇒ 无内容（完美预测时同样 =1）；CLIP `0.7404` 优于 VPB
    （仅 19 个非恒定段有效、14 段同源；**全视频一致训练均值特征仍得 0.699287**，
    位置曲线 0.731791，与原始值只差 0.008639）；LFM 已被排除（NOT IDENTIFIED）。
  - **不可用**：`−0.0994` 均匀窗失败（812 行标签行号读 2508 行 RGB，124/125 源错配）；
    `0.7085` shot 优势（未做镜头边界检测，跨片段 Pearson ≠ 帧内 Spearman）。
  - **新增阻断**：1979/2508 MrHiSum `gif_hard` 全零（全池 79.07%，非 87%），构造器未恢复；
    2508 行 = 1329 规范源；VPB 头被改成 ReLU 且漏 norm；确认池耗尽。
  - **R17**：整轴训练 block − 短窗 block = +0.002011，**未过 0.005 开发门**；
    头对位置控制仅 +0.0033。
  - **R18**：32 源中 29 过轴一致性门，**3 源永久隔离**；映射规则 7610/8000 匹配；
    exposure_v3 记 35,419 事件 / 511 角色冲突。无晋级。
  - **R19**：完整 10 人 GT 下块排序 > 点排序（keep0.85 **+0.006746** CI[0.002214,0.011612]；
    keep0.15 block48−block12 **+0.100858** CI[0.081642,0.121294]；rater 不一致下限 0.337965），
    **但是 GT 辅助 oracle，非可学增益**；3 源恢复尝试 `PROBE_FAILED` HTTP 403，隔离维持。
- Size：无变化。
- Training Time：R17 12 臂共 173.13 s；R21 3 子任务 4.2 s。
- Conclusion：**本地尺子全部失效**，方向改为先重建测量仪器。
  **对 E1.1 的修正（我独立复核）**：R21 泄漏结论**不适用** R16 的 T1\*/T1S\* 臂 ——
  `public_uniform_group_v1.json` 实测 121/40/42 源、三向交集全 0（裸 11 字符 ID 比对），
  `T1S_uniform_s13_v1` 的 42 eval 源 ∩ train = 0。故 E1.1 判决成立**但理由更换**：
  计划表 §1 的「模型也塌 → 内容不存在，35.52 终局」比
  `r15b_content_signal_verdict.py` 自身判决文本更激进，且与 R21 冲突；
  采用脚本版本 —— **"本协议下可学信号弱" ≠ "任务无内容信号"**。
  8 条臂全部落在控制带 [0.2404, 0.2470] 的 ±0.008 内。残留待核：`manifest_sha256` 声明值
  `a5de5742…` 与实测 `016a3f03…` 不一致。计划见 `docs/r21_direction_and_plan_20261015.md`。

---

## 阶段 0（2026-10-15）R21 阻断清单收口：0.1 / 0.2 / 0.3（CPU 只读，0 GPU，0 训练，0 提交）

- **Hypothesis**：R21 列出的五项阻断中，生成器/哈希/额度三项是文书与溯源问题，
  闭掉它们不需要算力，但决定阶段 1 之后的实验是否可解释。
- **Change**：不改任何模型与数据；只做溯源核对与文档收口。
- **Result**：

  **0.1 提交额度** —— 官方额度是**每天 5 次**，不是总量稀缺。用户 2026-10-06 更正。
  `reports/r6/claim_ledger.md` 记「memory 已存」但该 memory 文件从未写盘（本次补写
  `memory/aic-submission-quota-daily.md`）。此前按"总量稀缺"设计的 hour-36 一次性决策、
  D/S-T 诊断包与真候选互斥的 slot_strategy **作废**。
  ⇒ **随机排序探针可负担**，从"需先确认额度"变为可执行。

  **0.2 `manifest_sha256` 不一致 —— 虚惊，撤销**。`T1S_uniform_s13_v1/config.json` 声明的
  `a5de5742f51a…` 指向 `R16_EXEC/temporal_rgb_v1/manifest.jsonl`（**特征池清单**），
  不是划分文件；该路径下 `public_uniform_group_v1.json` 实测 `016a3f034086…`，本就不同名。
  另：两个划分副本（`016a3f03` 与 `48cb0c2c`）字节内容不同但**分区逐字段完全相同**
  （121/40/42，`indices` 968/320/336，`positive_rows` 344/107/142）——差异来自格式化。
  ⇒ R16 源级划分**认证通过**，§2 的"待核实"项关闭，T1/T1S 8 条臂读数继续有效。

  **0.3 MrHiSum `gif_hard` —— 裁定 `PLACEHOLDER_OR_FAILURE`**。决定性证据在
  `/root/AIC/scripts/r14_assemble_sft_v2.py:107`：

  ```
  else:                                    # source == "mrhs"
      ...
      gif_hard = [0] * 12          # 字面常量，不是计算出来的
      gif_soft = soft
  ```

  同文件 PHD2 分支（84–100 行）**确实计算** `gif_hard`。上游 `r13_sft_data_build.py:129`
  对 `mrhs` 写 `"gif_hard": None`，而 `r13_grpo_pool.py:27-30` **依赖该 `None`**
  把 MrHiSum 排除出 GRPO 池 ⇒ v2 把 `None` 换成 `[0]*12`，语义从显式"无标签"
  静默变成貌似合法的"这 12 帧都不是高光"。
  MrHiSum 原生标注只有 `gtscore` / `change_points` / `gt_summary`，**无帧级硬标签**。
  三个流传的全零比例在不同分母上均复现：**0.9%**=仅 PHD2 子组 / **87%**=holdout-125 /
  **79.07%**=完整池（1983/2508）。`gif_soft` 完好（0% 全零）。
- **Size**：无变化。
- **Training Time**：0（纯 CPU 只读溯源 + 文书）。
- **Conclusion**：R21 五项阻断中 **1 项裁定（占位符）、1 项撤销（虚惊）、1 项闭掉（额度）**，
  余下 2 项（源身份未绑定 / 缓存合同破裂 / 确认池耗尽）由阶段 1 仪器重建处理，0.4 后台核实中。
  文档同步收口：`STATUS.md` 阻断 1、`DECISIONS.md` 新增「2026-10-15（补）」节。
  **副作用**：任何依赖 `gif_hard` 的历史数字（含 GRPO 池构成）都建立在占位符上，需重算；
  弱监督线因 `gif_soft` 完好不受影响。

---

## R22（2026-10-15）阶段 1：仪器重建 —— 64 公共源有界恢复 + PTS 轴审计（CPU，0 GPU，0 训练，0 提交）

- **Hypothesis**：R21 判定本地尺子全失效，根因之一是缓存合同破裂（行号错配）。
  重建一把**源级绑定**的尺子，是后续一切实验可解释的前提。
- **Change**：新增 `scripts/r22_build_instrument.py`。对 R20 已冻结的 64 源清单
  （`candidate_sources.jsonl`，32 train / 16 dev / 16 protected）做有界恢复（2 GB / 2400 s）、
  完整解码、PTS 轴审计、10 人原始秒标注栅格化。

### 结果

| 项 | 值 |
|---|---|
| 媒体到手 | **64 / 64**（1.395 GB，wall 280.17 s） |
| 轴门判决 | **通过 54 / 失败 10** |
| 通过分布 | train 26 / dev 14 / protected 14 |
| 失败分布 | train 6 / dev 2 / protected 2 |

**10 个 `AXIS_GATE_FAIL` 的性质**：全部由 `raw_interval_outside_media` 触发
（标注区间越过媒体末尾），Δ(decoded − annotation) **全部在容差内**（最大 0.178 s < 0.5 s）：

| 源 | Δ(s) | 越界条数 | 最大溢出(s) |
|---|---|---|---|
| FaKQvJBhMPg | −0.178 | 2 | +0.168 |
| B-KSdR2ct00 | −0.060 | 9 | +0.060 |
| uVcyJg_3Fj8 | −0.016 | 1 | +0.016 |
| HqxTRzf11tc | −0.067 | 1 | +0.067 |
| x4iCAIPRDVQ | −0.060 | 10 | +0.060 |
| UXc2y7lR5t0 | −0.100 | 10 | +0.100 |
| U6Xnw2RGjsI | −0.120 | 2 | +0.120 |
| q0o0k2uD0co | −0.048 | 6 | +0.048 |
| 4zFiORtQrlA | −0.030 | 1 | +0.030 |
| 8J_erRJya-k | −0.128 | 3 | +0.128 |

⇒ 是**真实的标注/媒体边界不一致**（标注最末段略微越过媒体末尾），不是解码错误。
54 个通过者的 `outside=0` 且 `invalid=0`。**这 10 源按 R20 规则永久隔离，不自动替换。**

**标签完整性**：54 个通过源中 `onehot` 阳性率 mean 0.1710 / median 0.1616，
`raw` mean 0.1696 / median 0.1618，**全零 0 / 54** —— 与 MrHiSum 的 79.07% 全零形成对照。
每个源产出 `axis_arrays/<vid>.npz`（含 `timestamps` / `raw_summary_membership` / `rater_count`）
并记录 `media_sha256` + `axis_array_sha256`，绑定 `(source_id, media_hash, actual_pts, preprocess)`。

### 过程中修掉的三个执行层缺陷（都会污染读数，需留档）

1. **探测丢弃失败记录** —— `probe()` 只保留最后一个 host 的尝试，镜像的偶发失败被
   静默吞掉，整体误判为 `NEED_MEDIA`。改为每 host 重试 3 次、保留全部记录。
2. **镜像 429 限流被记成"源不可用"** —— 首轮 64 源里 **24 个 HTTP 429**
   （另 2 个 timeout）。429 是**传输层**错误，直接落 `failed` 会把限流污染成
   数据可得性结论。改为对 429/5xx/timeout 指数退避重试（4/12/28 s + 抖动），
   重试仍败才落 `failed` 并保留全部尝试记录。**改后退避版 64/64 成功、0 失败。**
3. **`--resume` 字节双计** —— 恢复时预载旧成功记录，本轮又 append 同一批源，
   64 源被计成 99 条而误触 2 GB 上限中断。改为不预载（`download()` 本就靠
   文件存在性幂等返回 `already_present`），只带过轴判决与既往失败原因。

> 另：`av` 只在 `/root/autodl-tmp/aha_env/bin/python`（av 13.0.0 / numpy 2.5.4），
> `/root/miniconda3/bin/python` 没有 → 全程用 aha_env 解释器。

- **Size**：无变化（纯 CPU，未训练任何模型）。
- **Training Time**：0（无训练）。构建 wall 280.17 s。
- **Conclusion**：阶段 1 达成验收 —— 64 源、54 个源级绑定的可用源、
  `label_status` 明确（无全零）、两次运行可复现。
  **阶 2 判决实验（冻结 VPB point-vs-block × seed 13/17）的输入条件已满足。**
  报告：`reports/r22/r22_instrument_report.json`。

---

## R22b（2026-10-15）附带发现：现役 A025 的 p 在磁盘上复现不出来（CPU 只读，0 提交）

- **Hypothesis**：文档记现役 `p = 0.8534`，但这个数一直只是被引用、从未在盘上复算过。
- **Change**：对 8 个包逐一用 `intake_426.jsonl` 的 `frame_count` 复算 pooled `p = N_pred/N_tot`。
- **Result**：

| 包 | 文档 p | 实测 pooled p | 差 |
|---|---|---|---|
| KF045 | 0.4691 | 0.469092 | **+0.0000** |
| KF050 | 0.5451 | 0.545117 | **+0.0000** |
| KF055 | 0.6252 | 0.625225 | **+0.0000** |
| KF060 | 0.6433 | 0.643311 | **+0.0000** |
| KF090 | 0.9416 | 0.941582 | **−0.0000** |
| KF092 | 0.9550 | 0.955013 | **−0.0000** |
| KF095 | 0.9624 | 0.962414 | **−0.0000** |
| **A025（现役）** | **0.8534** | **0.839126** | **−0.0143** |

**KF 族 7/7 精确复现 → 口径确定是 pooled `N_pred/N_tot`，我的复算方法正确。
A025 是唯一的例外。**

- 盘上 A025 实测 `N_pred = 196,489`，`frame_count` 合计 `234,159`。
  要得 p=0.8534 需要 **N_pred = 199,831（差 3,342 帧）或 N_tot = 230,243（差 3,916 帧）**。
- `jobs.log` 显示 A025 的 zip 被重建**至少三次**，`pred_frames` 分别
  195080 / 195551 / **196489** —— **0.8534 对应的 199,831 哪一版都不是**。
- 且 `keep_frac=0.85 × 234159 = 199,035` 也 ≠ 196,489，隐含 keep ≈ 0.840。
- 旁证：`pkg_vpb_gifens_a0.15` / `a0.35` / `a025` 三个包 pooled p 依次
  0.839071 / 0.838665 / 0.839126 —— 若后缀是 keep，0.15 与 0.35 应差 20 个点，
  实际只差 0.0004 ⇒ **后缀不是 keep**（疑为融合权重 α）。但 `audit2.py:2` 读的是
  `scores_fuse_gifens_a025`，即该路代码里 `a025` 又是 gifens 的 α。
  **α 与 keep 的记账在两处互相矛盾，无法从磁盘判定。**

- **Size**：无变化。
- **Training Time**：0。
- **Conclusion**：A025 的 `p = 0.8534` **在本机不可复现**（缺 3,342 帧的中间 build）。
  它对项目的影响是**可算出方向**的：把 p 从 0.8534 修正到 0.839126，
  `F_ceil(p) = 34.75(1+d)/(p+d)` 全部上移 —— d=0.2103 时 **39.54 → 40.08**。
  而"40 分是否可达"的精确阈值：天花板=40 需 `d* = (F−40p)/(40−F)`，
  - p=0.8534 → d\*=0.1170 **< d 下界 0.21029 ⇒ 40 恒不可达**
  - p=0.8391 → d\*=0.2257 **> 下界 ⇒ d∈[0.2103,0.2259) 时 40 可达（窄窗）**

  ⇒ 现役的 p 究竟是 0.8534 还是 0.8391，**决定"现役 keep 率下 40 分不可达"这条结论
  是真结论还是窄窗内的近似失效**。
  **处置：该结论降级为"依赖一个未经磁盘复算的 p"**，在阶段 2 之前不据此做任何设计决策。
  最省的解法是**随机排序探针**（`ρ≡p` 精确成立，只含 d 一个未知量），
  它同时钉死 d 并给出真实的 p。

---

## R22c（2026-10-15）阶段 1 第 3/4 项：缓存重绑 + 头身份恢复（CPU 只读，0 GPU，0 训练，0 提交）

计划 §4 阶段 1 有 4 项。R22/R22b 只完成前 2 项（64 源、源级绑定）。
本节补齐第 3 项（缓存重绑）与第 4 项（头身份恢复），并交出两个可复算的执行器：
`scripts/r22_rebind_cache.py`、`scripts/r22_head_identity.py`。

**Hypothesis:**
VPB 评分头的「被审计副本用 ReLU、漏 `norm=(mu,sd)`」若可逆，则恢复身份后
**应当精确重现 checkpoint 自带的旧读数** `best_holdout_spearman=0.5696324319689352`。
一次精确命中会同时验证头身份、输入归一化、缓存可重绑、行对齐、holdout 划分五件事。

**Change:**
不训练、不拟合、不出预测。只用 `vp_score_head.pt` 的 state 在**训练时干净缓存视图**
上复算那条已记录的读数，并横向对照四种身份组合。

**关键前提：`627 × 4 = 2508`。**

盘上缓存与训练时**不是同一份**：

| shard | mtime | 盘上行数 | 训练时应为 | 说明 |
|---|---|---|---|---|
| `shard_0of4.npz` | 10-13 02:14 | 627 | 627 | 干净 |
| `shard_1of4.npz` | 10-13 02:14 | 627 | 627 | 干净 |
| `shard_2of4.npz` | 10-13 02:14 | 627 | 627 | 干净 |
| `shard_3of4.npz` | **10-13 13:35** | **819** | 627 | **被追加 192 行** |

而 `vp_score_head.pt` 存于 **10-13 06:12**，**早于** 13:35 的追加 ⇒
**训练时缓存一定是 627/627/627/627 = 2508**，与 `sft_data_v2.jsonl` 行数相等。
追加的 192 条全部是**已存在 vid 的重复**（192/192 都在前 627 段出现过），
且**逐值与首次出现不同（0/192 相同）** ⇒ 不是幂等重跑，是重算得出了不同的数。
分片方式是 **strided**（`rows[i::4]`），已逐位核实四个 shard 前 627 条分别等于
`rows[0::4] / rows[1::4] / rows[2::4] / rows[3::4]`。

**Result —— 第 4 项验收 PASS，精确到 0 差异：**

| 身份组合 | holdout spearman | Δ vs ckpt |
|---|---|---|
| **A `/255.0` + `(x−mu)/sd` + GELU（恢复真身）** | **0.5696324320** | **+0.000000** |
| A2 有归一化 但 ReLU | 0.5736808712 | +0.004048 |
| B 无归一化 + ReLU（被审计副本） | 0.5616186094 | −0.008014 |
| C 无归一化 + GELU | 0.5562483757 | −0.013384 |
| A′ 恢复真身 但用 spatial 特征 | 0.0259223256 | −0.543710 |
| B′ 审计副本 用 spatial 特征 | −0.0173808474 | −0.587013 |

`ckpt.report.best_holdout_spearman` 与复算值**字面相等**（`abs(Δ) < 1e-6`，
实测 `+0.000e+00`）。一处**已定位的真身**与三处**已定位的副本**：

```
train_vp_head.py:86     nn.Sequential(Linear(d,512), nn.GELU(), Linear(512,1))   ← 真身
train_vp_head.py:106    norm = lambda a: (a - mu) / sd                          ← 真身
extract_features.py:65  np.stack(imgs).astype(float32) / 255.0                  ← 真身
e01_vpb_uniform_eval.py:42  nn.ReLU()，:110 完全不做标准化                      ← 副本
e01_vpb_uniform_eval.py:93  /127.5 - 1.0                                        ← 副本
e01_eval.py:32          nn.ReLU()，不做标准化                                   ← 副本
resid_e03.py:46         nn.ReLU()，:96 未施加 ckpt 的 norm                      ← 副本
```

**副产物：`spatial` 特征对 `gif_hard` 几乎无信号**（0.026）—— 与真身 0.570 相差
0.544。这是一个此前没有的、**在已验证的干净视图上**取得的读数。

**第 3 项：缓存合同破裂的机制复原。**

`train_vp_head.py:56` 有 `assert len(vids) == len(rows)`，训练路径被保护。
但评估路径**没有**这条断言：

```
e04_compare.py:118-124   concat(shards) → argsort(order) → train_eval(fts, ...)
resid_e03.py:85-93       同上，且用 ScoreHead()=ReLU 且未施加 ckpt 的 norm
```

两者都是**无长度断言、无截断**，把 2700 行缓存直接喂给索引域 <2508 的 `hold_idx`。
⇒ **静默行号错配、不报错**。这就是 `0.3340` 与 `−0.0994` 那类翻车的**具体机制**：
不是概念性的"可能错配"，是"断言只写在训练路径、评估路径被丢掉"。
受影响读数 `E0.3 VPB head` 与 `E0.4 VideoPrism-B 列` **均已在 R21 判废**；
本节新增的是机制证据，不是新判决。

**附带发现：抽特征本身不可复现。**

拿**当前 jpg**用 `videoprism_base_f16r288.fp16.pt` 重算，与盘上缓存比：

| 来源 | 重算 vs 缓存 maxabs |
|---|---|
| `shard_0of4` 前 627 行（**从未被改写过**） | 0.0111 / 0.0136 / 0.0151 / 0.0334 |
| `shard_3of4` 前 627 行 | 0.0107 / 0.0141 / 0.0148 / 0.1206 |
| `shard_3of4` 追加段 | 0.0107 / 0.0142 / 0.0148 / 0.1298 |

**未被改写的 shard0 缺口一样大**（fp16/fp32 权重实测完全相同，最大差 0.000000；
帧目录 mtime 全为 10-12，早于两次抽特征）⇒ 缺口**不是**追加或精度造成的，
而是**抽取路径本身已不可复现**。单行最大缺口 **0.1298 ≈ 1.45 个特征标准差**。

**但这个缺口不足以解释已作废的读数**：在已验证的干净视图上做同尺度扰动，
0.0236σ（≈0.014）对应的 spearman 变化**恰好是 0.000**，要到 0.033 才 −0.0032、
0.12 才 −0.0088。⇒ 那些作废仍立于 R21 的泄漏/行号/指标混比理由上，不受此削弱。

- **Size**：无变化。
- **Training Time**：0（纯 CPU 只读）。
- **Conclusion**：阶段 1 四项全部收口。
  (1) 头身份**可恢复且已恢复**，旧读数已**精确重现**；
  (2) 缓存**可重绑**，训练时干净视图已冻结并可复算；
  (3) 评估路径的行号错配从"怀疑"升级为**已定位的机制**（断言缺失）；
  (4) 新增一条独立测量缺陷：抽特征不可复现（单行最高 1.45σ），
      但经扰动实验证明它**不足以**解释已作废读数。
  ⇒ 阶段 2 的输入条件（干净视图 + 正确头身份）**已具备**。

---

## R23 — 随机排序探针：读数形式更正、最优点更正、包已建成

- **Hypothesis**：把空间输出逐字节复制现役 B3（`Rbar` 恒定）而把时间选择换成
  与内容无关的固定伪随机抽样，则 `rho ≡ p` 精确成立，官方分中**只剩 `d` 一个未知量**，
  于是可凭 2 次提交把 GT 密度钉死 —— 这**不依赖任何本地标签有效性**。
- **Change**：(1) 读数形式由计划的聚合闭式改为**逐视频精确期望**
  `E[S(p)] = mean_i{F_anchor,i·(1+d)·p_i/(p_i+d)}`；(2) 主读数点由 `p=0.5/0.8`
  改为 **`p=0.25/0.50`** 的嵌套双点。
- **Metric**：本臂**不追求分数**，追求 `d` 的可识别性。读出量 = 两个估计量
  （单点反解 `d` 与比值 `r=S(.25)/S(.50)` 反解 `d`）的**自洽性**。
- **Size**：0（`model_size_mb` 整字段省略，与锚点包同构）。
- **Training Time**：0（纯 CPU 只读打包，<60 s）。
- **Result**：包已建成并三层验证通过。

**为什么原式的读数形式是错的（本轮实质发现）**

独立性一步给出 `E[Rbar|命中] = Rbar_i`，于是 `F_anchor,i` 与 `p` 无关，
`Rbar_i` 的异质性在宏平均里**精确抵消**。计划的
`S_rand(p)=F_ANCHOR·(1+d)·p/(p+d)` 只在 `Rbar_i ≡ const` 时等于逐视频形式；
而官方 `F_video` 是**视频级宏平均**（规则 §3.4），所以逐视频形式才对。
⇒ 此前挂在探针上的最后一条残余风险（`Rbar` 抽样噪声）**退役**。

**为什么 p 要取 0.25 而不是 0.5**

`d∈[0.2103, 1.0]` 上 S 的跨度：`p=.25 → 8.94`、`.30 → 8.69`、`.50 → 6.44`、`.80 → 2.41`。
`p=.25` 对 `.80` 是 **3.7 倍**灵敏度。嵌套比值 `S(.25)/S(.50)` 的动态范围
`0.7716 → 0.6000`，是 `S(.5)/S(.8)` 的 2 倍，对 `d_i` 异质性也约 2 倍稳健。

**闭环（最有说服力的一步）**

用官方口径精确模拟打分（含 **cv 很大的异质 `Rbar_i`**，再校准 `F_anchor=34.75`），
真值 `d` 的真值→反解：`0.21029→0.21579`、`0.25→0.26895`、`0.30→0.29719`、
`0.50→0.50194`、`1.00→0.99966`；比值式反解 `0.21029→0.23206` …
`1.00→0.99940`。两个估计量互相自洽，且**强异质 `Rbar` 下仍回真值** ⇒
"`Rbar` 抵消"不是纸上推导，是被数值验证过的。

**副产物**：比值 `r` **不含 `F_ANCHOR`**，因而同时是锚点 `34.75` 的一次独立测量。
两个估计量在同一份官方分上必须自洽 —— 这是锚点与读数形式的**交叉检验**。

**真包验证**：426/426 全帧覆盖、`missing_bbox_source=0`、嵌套逐视频零违例、
`bboxes` 58536/58536 与 B3 逐字节一致、独立 validator 双 arm `valid:true`、
zip 内 `namelist==['predictions.jsonl']` 与 B3 同构。

**资产**：远端 `/root/autodl-tmp/experiments/R23_RANDPROBE/`
`R23_RANDPROBE_P250`（p=0.24996384，58536 帧，zip `51910acc96f14f70`）、
`R23_RANDPROBE_P500`（p=0.4999734696，117077 帧，zip `98862a8f78526209`）。
**未上传**，待用户提交。

- **Conclusion**：探针从"计划里的一步"变成"可直接提交的两个已验证包"。
  读数形式、最优点、声明尺寸三处均已更正并留证。
  唯一的开放项是**官方额度时序**（是否先做阶段 2 判决实验），由用户决定。


## 2026-10-15：R23 官方复赛回分与后续执行

用户报告已提交 R23，P250=18，P500=26（原报告精度；未独立登录平台核验）。现役最佳仍为35.52。
复赛比初赛难，禁止跨赛段比较，初赛分数不参与当前选模。
撤回“两个随机探针唯一识别 d”的强结论：在逐视频 d_i 异质时，总分只提供加权约束；
rho=p 是随机抽样的期望关系，有限样本不是恒等式，Rbar_i 也不会从宏平均中无条件消失。
18/26 只作为同赛段官方观测保留，不宣称精确密度或已证实的排序天花板。
按用户授权执行 R24：现算 VPB point/block × seeds13/17、公平 CLIP 对照、AHA 真实视觉 smoke。

## R24 — 2026-10-15 干净对照与AHA有界尝试

重新抽取26train/14dev公共VideoXum的VPB与CLIP特征，12全轴采样；两监督×两seed×两表示共8臂。
固定300epoch最终头，无dev选epoch；768→512 GELU→1、train-only标准化、soft BCE。
F@.85：VPB point .296544/.298774；block .294371/.296831；CLIP point .292495/.286712；block .279953/.284288。
位置/同分控制.297467，随机.285002，全选.292570。block−point均值−.002058，两seed负，门失败。
14预留源未打开；不晋级，不替换比赛包。不支持CLIP替换，但CI跨零，不作总体优劣推断。
AHA2.455秒失败于缺wandb导入；未加载、无视觉输出，不算smoke通过。
原帧指标独立重算8×3精确相等；checkpoint重放误差<1e−5。每头约.55–.70秒，外层45min timeout。
完整协议/命令/日志/SHA/逐源结果：[reports/r24/summary.md](reports/r24/summary.md)。

## 2026-10-15：R24 AHA依赖修复与真实视觉通过

按用户指示安装缺失包，补齐并SHA256校验SigLIP视觉权重。wandb仅作为导入依赖，运行禁用在线日志，无需登录。
FP16已成功加载并精确验证三个评分头，但首次前向出现非有限输出；保留失败证据，不声称定位到具体层。
同脚本改为基座原生BF16后：真实公共TRAIN两不同帧与首帧重复测试通过，视觉嵌入/三个头输出均有限。
不同帧最大响应差2.05078125，重复差0；总20.81秒，峰值18,632,761,344B（17.35GiB）。
Decision：运行可用门通过，质量门未测；不替换比赛包，不扩量蒸馏。先验证离线teacher与人工标签的相关性。
限制：实际加载参数17,346,192,448B，不认定满足比赛权重上限；单帧smoke不代表视频任务质量。
完整环境、命令和失败/通过证据：reports/r24_aha_repair/summary.md。


## 2026-10-15：R25 AHA原生BF16复赛提交包

Hypothesis：AHA原生因果评分头在相同85%预算、相同B3空间框下可能改善高光排序。
Change：仅替换时间评分为relevance probability＋informativeness probability；1fps、完整视频因果上下文、每块8帧，未量化BF16；无训练，无测试标签。
Control：426同赛段视频、固定B3空间框。R23为随机抽帧，不是AHA。
完整AHA8,030,363,168参数，空间450,283,377参数，总8,480,646,545参数≤9B。
旧9GB文件上限解释撤回，以用户确认官方参数口径为准；初赛分数不参与比较。
Result：2026-10-15用户报告复赛官方35.18，较35.52低0.34、较全帧34.75高0.43；未独立核验平台上传SHA，不晋级。
BF16不同token/head矩阵形状造成数值差，放宽容差尝试未采用；最终恢复原生完整token评分头形状，全426视频重新计算，原生前向交叉检查<1e-5。
记录：reports/r25_aha_submit/summary.md；不是以模型容量或单帧smoke宣称提升。


## 2026-10-15：R25回分与原因审计

Hypothesis：35分平台期可能涉及预算/位置偏差、AHA任务适配和共享空间误差，不能由模型容量解释。
Analysis：本地ZIP及426视频双校验再次通过；实际87.5076%保留、宏平均88.7453%，比逐视频精确85%多5870帧。
首采样帧仅28/426保留，前/中/后三分之一宏平均保留77.12%/98.77%/90.30%，297条仅一个保留区间。
上游取点、头变换一致；generic query、等权融合、cache和文本顺序与作者路径有差异，尚未测定因果影响。
Conclusion：维持35.52，不宣称35是天花板或空间已证实为主因；预注册公共dev位置控制、精确预算和固定掩码空间对照。
Blocker：AutoDL40206域名及公共DNS直连均关闭，raw头分数分析未执行。未训练、未打开预留14、未生成或上传新候选。
证据：reports/r25_score_analysis/summary.md、package_audit.json、official_score_event.json；复现脚本scripts/r25_analyze_package.py。


## 2026-10-16：R26选帧/预算/位置/空间诊断完成

Hypothesis：R25平台期涉及预算实现、因果位置与任务适配，空间待隔离。
Change/Control：复用426原评分+B3；精确85%仅截断末cell；固定原mask仅换中心空间；26TRAIN/14DEV以冻结R25协议比较head及位置控制，未打开预留14。
Result：sum公共F@.85=.290883，prefix/train_position=.297467，center=.288381；对上述控制CI均跨零。info/时间rho中位+.635、rel/时间−.827。静态同图16帧sum范围.107329，chunk8/1总分最大差.017685。
Budget：204907→199037帧，消除5870超额；同一mask空间控制204907帧不变，116620框改变。三ZIP426/426双validator无issues、参数合规，官方分null，未上传。
Training：0；公共推理488.76s，GPU0 RTX4080，峰值allocated17860467712B。现役35.52保留，不因本地公开摘要指标晋级或宣称容量问题。
Evidence：reports/r26/summary.md、diagnostics.json、public_metrics.json；配置configs/R26_DIAG_20261016.json。

## R26 官方回分与R27中心窗口实验（2026-10-16）

用户报告中心85+B3=36.04、精确AHA+B3=35.07、R25mask中心空间=30.23；本地ZIP SHA与冻结manifest重核一致，平台上传SHA未独立核验。分别说明中心排序高0.97、预算修复低0.11、B3空间高4.95。仅比较复赛，不能将联合IoU增益当纯时间语义精度或空间误差占比。中心85晋级安全基线，旧35.52保留。

R27假设：最佳中心时间窗收益依赖预算/端点位置；仅改变keep(.75/.95)或shift(−.05/+.05，keep=.85)，保持B3所有框。配置在运行前保存；CPU缓存重组7.93s，4×426视频双validator通过，复建最佳199037帧及框完全一致；新包预测数175623/222454/199037/199037，450283377参数。0训练、0新增GPU推理，预留14源未打开。当前四项官方分null，不能宣称性能提升。顺序95、75、前移、后移，全部不超36.04则停止扫描。

Evidence: reports/r26/official_analysis.md、official_score_events.json；reports/r27/summary.md、run.json与四份manifest；configs/R27_CENTER_WINDOW_20261016.json；scripts/r27_center_windows.py。

## 2026-10-16：R27扫描收口，R28转空间主体评分

用户报告R27_CENTER075_B3=35.87、R27_CENTER095_B3=35.39，较中心85+B3=36.04分别−0.17/−0.65；本地ZIP SHA一致，平台上传SHA未独立核验。停止预算/位置扫描，EARLY/LATE暂停，保留安全基线。
R28实际完成426视频234159帧空间缓存回放，最大坐标差0；x/y/全画面126/141/159视频，4913使用关键帧全ok_head，无回退。记录路径中的B3_s0头/LFM基座缺失，不能声称原视频部署通过。历史插值/加密/检测先验不重复；B3已是candidate-utility，优先主体评分、视觉表示、真实原生几何标注。
Qwen3.5-4B两分片738tensor映射完整；公共4源×3帧×2ratio工程pilot配置先冻结，独立环境，20min超时，无训练、无官方视频推理、预留14源未打开。读数及运行状态见reports/r28；公共pilot不直接晋级或预测官方分。

## R29 — 2026-10-16：效用决策、有效监督与空间选择的判别收口

父代码`d3e6a40e417321be67229f2add54aca34f80d858`。主协议`configs/R29_BREAKTHROUGH_20261016.json`；完整推导、逐源结果、失败例与证据边界见`reports/r29/summary.md`。本轮没有正式模型训练、官方测试训练/标注、protected14访问、新提交包或官方新回分。以下数字均非官方联合成绩。

### R29_DECISION_THEORY：未知GT数量会改变最优排序与预算

Hypothesis：最终目标是条件期望`E[2ΣY_f IoU/(K+M)|X]`，未知M与联合crop收益使边际高光概率或期望之比不总是充分决策量。固定K按`max_b E[Y_f IoU(b,B_f)/(K+M)|X]`选top K；可变K须重新计算。已知M时有精确简化，不需假设帧独立。
Result：P(1110)=.6、P(0001)=.4且K=1，前三帧边际概率更高，但选其中一帧E[F]=.30、选第四帧=.40。独立p=(.9,.4)且完美crop时插件全选，真实单选.78高于全选.746667。240次全动作穷举、51200个已知M等式、40640个增删等式、81个独立排序、24个多项式解码对照全部通过，最大精确误差0。
Cost/Conclusion：CPU3.7933s、exit0；48个合成概率解码约.00779s。证明近似可能改变动作，不证明项目后验可识别、可校准或官方可涨分。协议`reports/r29/decision_theory_protocol.json`，代码`scripts/r29_decision_theory.py`，结果`reports/r29/decision_theory.json`。

### R29_PUBLIC_DECISION_PROBE_20261016：已有AHA两标量未过内容增量门

Hypothesis：若两标量包含可迁移效用，在相同85%原帧预算下，TRAIN-only位置＋头应超过位置；若主要损失来自概率/数量校准，插件应改善自身固定预算并超过同K位置控制。预先冻结实用门：主对照delta≥.005且源级配对95%区间下界>0。
Change/Control：26TRAIN源宏平均BCE拟合三个固定logistic读出（cubic位置、info＋rel、二者），ridge=.001；真实PTS最近采样广播、精确原帧K。14已暴露DEV源，十人先逐人F再源内平均，源级配对bootstrap10000次、seed20261016；无DEV选模。
Result：位置固定85 F=.297467；位置＋头=.294206，差−.003262、95%CI[−.008499,+.001084]，未过门。位置＋头插件=.302208，较自身固定85差+.008003[−.008254,+.025267]，较同K位置差+.003536[−.008390,+.016137]；全选=.292570，不能宣称校准挽救成功。插件是期望分子/分母近似，不是精确后验解码。
Observation：12/24/48均匀取点完全未命中run中的正例质量分别35.412%/14.075%/4.479%，但这些run不是物理事件、三组采样非嵌套干预；实际AHA 1fps命中40源×10人全部正run。命中一点不证明观察到关键动作/音频，稀疏12帧漏采不足以解释当前AHA。
Verification/Cost：CPU全命令1.0249s、exit0（计算内部.5878s）；只训练小logistic校准器，无encoder/正式训练。独立实现复算520源×arm最大F差0，160同K、42R26控制、42观测与42GT辅助固定K界及四个bootstrap对照通过；不是独立新源确认。
Conclusion：停止当前两标量校准配方；未否定完整AHA特征、其他合法query/观测/监督。证据`reports/r29/public_decision_probe/metrics.json`、`temporal_independent_review.json`、`temporal_probe_environment.json`。

### R29_R28_SPATIAL_CONTINUATION：保留中断历史后完成原pilot

R28的`INTERRUPTED_SSH_RESULTS_NOT_RETRIEVED`原记录保留；新40188/westd端点恢复13条旧预测，核验原图SHA与已冻结prompt，仅补11条，共四个已暴露RV源×3帧×2ratio=24项。不是新源确认，也没有同帧B3对照。
Result：连续IoU源宏平均Qwen=.599645、center=.579117；x轴=.456114/.415057，y轴=.743177/.743177。20/24输出中心、y轴12/12中心。原生RV像素口径另报.601083/.580877，不能混用指标口径。24/24解析有效；独立复算最大误差2.22e−16。
Cost：GPU0 RTX4080、先查空闲；总27.2867s、新11条生成14.0396s、peak allocated9358655488B、exit0。运行时唯一模型参数4539265536，仅公共pilot组件，不是新部署组合申报。没有训练或新比赛候选。
Conclusion：生成坐标路径存在选择失败，提示唯一数值例`center=.5`的锚定、坐标接口与视觉理解均为竞争解释，尚不能归因LFM。记录`reports/r29/spatial_completion/spatial_continuation/metrics.json`与`reports/r29/spatial_review.md`；原R28日志及13条原件另存`reports/r29/recovered_r28/`。

### R29_SPATIAL_GT_ASSISTED_DIAGNOSTIC：候选充分性与选择误差分开

Hypothesis：在固定最大窗几何下，当前center输出错过已有高IoU偏移候选；竞争解释为该几何本身不足。先冻结探索协议，再逐项穷举129合法自由轴偏移，六人工框平均连续IoU；仅四个已暴露源，原Qwen及其最近格投影均固定，无新模型推理或GT选prompt。
Result：all中心/Qwen/投影/oracle=.579117/.599645/.600223/.868315；x=.415057/.456114/.457268/.854424；y=.743177/.743177/.743177/.882205。034 f240/f360的x轴Qwen IoU=0，合法候选GT辅助最优=.947777/.966675。
Cost/Conclusion：CPU.00597s。这些帧上的固定几何容纳合适候选，当前自由坐标选择未找到；oracle不证明输入可学、LFM/B3失败、独立泛化或官方收益。下一步优先原生x/y监督和同输入B3控制，不能用teacher框代人工GT。协议`configs/R29_SPATIAL_GT_ASSISTED_DIAGNOSTIC.json`，结果`reports/r29/spatial_oracle.json`，失败例`reports/r29/spatial_failure_cases.png`。

### R29_AHA_TRANSFER_PRIMARY_SOURCE_CHECK：教师目标与部署合同并不等价

一手论文2509.16421v2及作者checkout`41473169504ae3793811066ffd398bc69094e499`核查：relevance监督为归一化YouTube replay、query由原标题模板生成；informativeness正例为每caption片段50%至随机50–75%理解点，属于局部阶段启发式，不等于人工精彩度或全片中心捷径。原始序列化预处理未重建，不冒充训练数据重现。
作者inference对raw logvariance取exp后融合，论文描述clamped logvariance；query专用sink重初始化在当前源码被注释。R25/R26的泛化query、info＋rel等权和增长past不能称完整论文推荐系统；没有证据证明补融合/cache就提分。三head均未学习native crop效用。
Conclusion：结合公共同K对照，停止当前直接替换配方；保留能被合法TRAIN选定、同观测同K位置控制的源外增量推翻的边界。不以论文mAP替代联合F、不用官方回分搜query/fusion。CPU只读，无训练/GPU/重复下载/参数重计。证据`reports/r29/aha_transfer_check.md`及`aha_transfer_evidence.json`。

**阶段决策**：主攻可追溯有效原生监督、任务分离读出与源外内容增量；空间候选读出作备选。中心85＋B3用户报告36.04继续安全基线；本轮新官方Score=null，无独立确认、无新可上传候选。新监督阶段协议已写但未执行，不能把标注/训练计划登记成已完成实验。

## R30 — 2026-10-16：冻结 VPB 静态/动态内容探针

假设：在固定12点原生PTS与精确85%预算下，VPB静态表示或相邻变化提供位置之外的可迁移时间信息。使用已绑定公共VideoXum 26 TRAIN/14 DEV、十人原始标签；仅TRAIN拟合ridge-logistic，DEV一次评估。静态DEV F=.292491，位置读出=.297399，中心=.288381；动态=.260556，静态+动态=.278525。相对中心的源级bootstrap：静态 +.004109 CI[-.012593,.021734]，动态 -.027826 CI[-.050982,-.005046]，静态+动态 -.009857 CI[-.027008,.007853]。动态TRAIN=.304126而DEV=.260556，显示768维差分读出过拟合。

结论：未达到预设内容增量门（delta≥.005且CI下界>0），停止高维动态配方，不启动大时序训练。该结果不能证明视觉动态没有信息；下一步仅做低维强正则事件状态探针，若仍失败则转音频/事件结构或空间部署合同。结果与限制见`reports/r30/summary.md`。

## R31 — 2026-10-16：低维事件状态 probe

假设：R30 的高维相邻差分可能只是读出过拟合，低维状态摘要可保留可迁移事件变化。固定 VideoXum 26 TRAIN/14 已暴露 DEV、相同12点观测、精确85%预算和位置控制，TRAIN-only 强正则读出。DEV F：center=.288381，position=.297399，low-dimensional=.302285，PCA4=.298328，low+PCA=.301449；low 相对 center +.013904，源级 bootstrap 95% CI=[-.000339,+.031144]。DEV 已在R24/R26暴露，VideoXum标签不是AIC标签，且没有AIC对应特征缓存，故不部署、不作为独立确认。结论：低维摘要值得保留为机制线索，但内容增量未被识别；转向已有AIC冻结时序链路生成可部署候选。证据`reports/r31/summary.md`。

## R32 — 2026-10-16：AIC冻结 temporal_k0.90 + B3空间候选

假设：中心85的官方优势可能来自预算/位置决策；已有AIC 426视频冻结 temporal_k0.90 链路提供不同时间选择，配合已验证B3空间框可形成真实可上传对照。使用远端 `/root/autodl-tmp/evac/data/aic/semifinal_20261001/temporal_k0.90/predictions.jsonl`，不访问GT、不训练、不使用官方回分；输出426/426视频、207292预测帧，实际总保留率207292/234159=.885262。参数申报：B3 450283377 + temporal TCN 246401 = 450529778（0.450529778B），不把文件bytes当资格限制。独立checker与本地contract均通过；ZIP SHA=`9d6f148b4e42c48ff8553edd052f226503d4d304a1c046a3a496dc16aab3a434`。这是已生成、未上传、无官方分数的新候选，保留R26中心85+B3作为回滚基线。证据`submissions/R32_VT_K090_B3/manifest.json`。

## R33–R36 — 2026-10-16：远端两条冻结时间头的四个提交候选

远端 AIC 426 缓存已核验为完整426视频；`temporal_vt_replay`与`temporal_vt_exactdp`使用不同冻结时间头，重叠帧上的 B3 框逐项一致。没有GT、官方回分或测试集训练。四包均通过独立checker与semifinal contract：

| 包 | 机制 | 预测帧 | 实际保留率 | 参数 |
|---|---|---:|---:|---:|
| R33_VT_REPLAY_B3 | replay时间头+B3 | 183505 | .783677 | .450529778B |
| R34_VT_EXACTDP_B3 | exact-DP时间头+B3 | 183584 | .784014 | .450529778B |
| R35_VT_UNION90_B3 | 两头并集+B3 | 210827 | .900358 | .450776179B |
| R36_VT_CONSENSUS85_B3 | 两头共识优先，中心85补齐+B3 | 199037 | .850008 | .450776179B |

R35并集为210827帧，R36先保留两头共同帧，再用中心85包的合法B3帧补足逐视频精确预算；这是预算与错误相关性的对照，不是开发集调参。所有包尚未上传、官方分数为null。每包manifest、预测文件和ZIP SHA见对应`submissions/`目录。

## R37–R41 内容驱动时间决策（2026-10-16）

用户拒绝 keep/位置扫参后，基于冻结768D视觉状态构造五个固定机制候选，统一继承B3空间框和每视频floor(0.85N)预算。R37事件转折、R38转折窗口、R39冗余压缩、R40状态新颖性覆盖、R41内容门控弱监督头。五包426/426、198842帧、450529778参数，独立checker全部通过。详细机制与限制见`reports/r37_content_candidates_20261016.md`。打包时暂无本地联合GT或官方分数；当时优先建议验证R41，其次R40。最新回分见2026-10-17追加事件。

### R39–R41 回分追加（2026-10-17）

用户报告复赛官方分：R41门控35.19、R40新颖性34.15、R39冗余34.48；相对R26中心85+B3=36.04依次−0.85/−1.89/−1.56，均不晋级。本地ZIP SHA匹配原manifest，包内426个唯一视频及predictions SHA通过；未独立核验平台或上传字节。三包198842帧、R26为199037帧，不能称严格同预算归因；单一宏平均回分不能分离误删高光、crop收益分布和域迁移。

结论：削弱当前冻结视觉状态启发式及其弱监督门控在比赛域上的增益假设，停止其扫参；不证明动态/内容无信息。R37/R38尚无新回分。下一资源投入转为12源小批盲标，诊断目标语义、前后过程和声音依赖。详细事件与SHA见`reports/r37_content_candidates/official_score_events.json`；没有新训练或新候选。

## R42 — 2026-10-16：小规模人工先验诊断批次（Smoke已完成，空间待校准）

**Hypothesis**：当前中心时间策略长期停在约36分，可能是目标语义、过程/音频可观测性或时间—空间联合效用错位；12个源视频的盲标可以区分这些机制，成本低于继续换模型。

**Data / change**：从 AutoDL 已有 `QVH_NATIVE_CANDIDATES` 按预先冻结的 metadata-only 规则抽取12个公开源视频，每个30秒；导出360张1fps JPEG和12个低码率MP4，约25MB。6个视频双人重叠，6个单人。Label Studio任务默认不包含类别、抽样原因、assignment或QVH query；这些字段仅在独立manifest中用于事后分析。

**Labels**：通用高光等级、事件阶段、是否需要前后过程/声音语言、事件类型、置信度；高光核心另标目标比例构图窗口及原因。首轮不显示QVH query，先播放片段再看逐帧图。

**验收**：任务数360，素材12个视频/360张图；本地目录 `artifacts/annotation_batch_v1/`；配置与操作说明见 `configs/ANNOTATION_BATCH_V1.json`、`configs/label_studio_annotation_v1.xml`、`reports/annotation_batch_v1_runbook.md`、`reports/annotation_protocol_v1.md`。人工结果尚未产生，不报告模型收益或官方分。

**Decision gate**：若高光核心多数需要前后过程，优先增加时序观测/事件结构；若多数依赖声音或语言，转音频/OCR小模块；若人工通用高光与QVH query系统性错位，停止直接混合QVH软标签；若高光与稳定构图窗口经常冲突，改联合时间—空间效用。若四项均不明显，才回到冻结视觉读出和部署合同。

**Protocol v1.1（已冻结）**：先完整播放并建立事件时间线，再按 1 fps 逐帧标注；`0/1/2/3/不确定` 有明确裁决树，不能把证据不足写成 0；`needs_context` 单独记录前后过程与声音/语言依赖。该批次无 `targetRatioWH`，构图框只作自由比例可行性诊断，不作为 native crop GT；只对高光 2/3 画框，双人结果保留为分布，不平均成唯一框。没有人工导出结果前，不启动训练或提交包。

**2026-10-17 播放入口验收**：60个Smoke任务原位补充原生HTML播放器和MP4独立链接；项目新增视频目录读取权限，不同步创建任务。12段30秒MP4均保留，CPU约9秒生成12份VP9/Opus兼容源。公网浏览器实测两个Smoke视频：播放时间前进、声音解码字节数为正、取消静音、跳转15秒、Range请求206；不代表人工听感验收。任务ID及已有1份草稿哈希保持一致，无正式标注被新增。详见`reports/annotation_video_player_20261017.json`。

**2026-10-17 协议v1.2**：按用户实际操作反馈，隐藏并取消`query_relevance`必填，`needs_context`允许0/1留空，补充定义/例子及无高光窗口规则。已有正式标注和草稿均保留，旧字段不回填；等级2/3的信息依赖缺失单独报告。属于标注流程修复，未新增人工GT、未训练；浏览器验证见`reports/annotation_schema_v12_20261017.json`。

### R42_SMOKE_QC_V1 — 2026-10-17：首次正式导出质量检查

固定快照SHA256=`99f97f081afd8d934f6d18873a3d819e8fcacbe1923e6604363116c95e195dfe`。SQLite只读事务导出60任务/60正式标注，另存1份草稿，不合并草稿或重写人工结果。配置`configs/R42_SMOKE_QC_V1.json`，审计脚本`scripts/audit_annotation_smoke.py`；当前为单人两原视频源，独立单位是源视频，无人际一致性或标注准确率。

四核心字段60/60，2/3信息依赖43/43，图片尺寸/源时间/有限坐标合同零错误。两个源2/3为17/30与26/30，不设正例密度目标。52框、39帧有框；11帧多框、1个近零误操作框、1帧等级1留框，以及构图说明/可行性缺失，共18题自动旗标。顺序图视觉复核建议回看51/52事件阶段；未独立听辨音轨，不将语言/情绪相关歧义判为错误。信息依赖37当前帧足够、6前后过程是自我报告，不能当静态表示充分性试验。

**结论与行动**：结构导出通过，时间等级有可解释的内容差异；空间存在把对象框当保留窗口的语义偏差，先做有限修订再扩其余300任务。协议自身的去冗余措辞也可能偏离逐帧F，不能直接将序数等级二值化为官方Y或据此调keep。本次CPU审计exit0，无GPU/训练/候选/官方涨分；全部复现文件及逐题队列见`reports/r42_smoke_quality_20261017/review.md`。原结果保留，未修改线上配置或标签。

### R42_SMOKE_REVISED_V12 / R42_ANNOTATION_V13_EXPANSION — 2026-10-17

用户修订Smoke共14题，60正式/2草稿/2源保留。新版导出SHA256=`9d501a6b1067ff064071572744f3878e8fd01552c1d68c67ada451eb05fbdcf6`，四核心与媒体映射零错误，44/44个等级2/3有依赖；原18题结构旗标缩至6题多框（34/35/38/57/59/60）。CPU审计0.145秒、exit0，见`reports/r42_smoke_revised_20261017/`。不以标签比例判错，不重写答案，6题补修与扩标并行。

V1.3明确标逐时刻事件价值，无高光数量/比例/去冗余配额；补齐裁决树、8个虚构例子、声音/过程依赖、单窗口及未知情形填法。原12源素材不扩量；项目1保留60已完成、新增240题，项目2有240空白题，合计540份任务、360独立图片、6源180图片重叠。分工只依原列表顺序，任务不含query/类别/模型预测。单人预计1.5–3小时，属于排班估算。

部署4.785秒，幂等重跑2.906秒且新增0题。360张JPEG本地读取通过，12源公网图片200/MP4 Range206；Chromium实测两项目说明、视频播放/跳转、0/1跳过依赖与2/3无合理框备注提交路径。拦截annotation/draft写请求，无测试标签保存，60正式/2草稿和旧任务data哈希未变。自有验证session及cookie已撤销，备份私有保留。命令、版本、hash、浏览器耗时/截图和退出状态见`reports/r42_annotation_v13_20261017/deployment.json`；幂等见`idempotence.json`。

**结论**：支持“现有空间操作问题可用有限协议修正降低”的工程判断；尚不能识别时间标签准确性、内容模型能力或官方增益。旧60不是V1.3重标，主双人一致性只用4个双方新标源，另2旧Smoke源单列。待新增人工结果再检验过程/音频/监督错位，不开展新训练或宣称已具备native crop GT。没有新候选包、独立确认或官方涨分。

## R43 — 2026-10-17：InternVideo2 Stage2 6B 事件完成度全量候选

**Hypothesis**：视频原生6B表征和“准备—关键动作—结果/反应”阶段提示，能在冻结中心位置先验之外识别事件完成度；若有效，收益来自时间语义而非视觉状态变化或人工标签校准。

**Change**：每秒锚点取四帧上下文，事件正向提示减routine/等待提示，并加入关键动作、结果、准备阶段；视频内百分位排序后固定`0.75事件+0.25中心秩`。空间逐帧继承远端`LFM_V8_B3_SEMIFINAL`，预算为`round(0.85*N)`原始帧。未使用测试标签、伪标签或V1.3人工标签。

**Execution**：AutoDL RTX 4080 32GB，FP16。batch=88完成379个视频后在video 379因显存峰值/碎片OOM；保留原子结果，以全新进程batch=4续跑，最终426/426完成，0回退，runtime=1495.343s。实际InternVideo2参数6,366,530,804；连同B3空间父模块6,816,814,181，满足<=9B。

**Contract**：426视频、199,037条预测；逐视频`round(0.85*N)`、帧唯一升序且范围合法、targetRatioWH正确；选中帧与远端B3父预测逐框比较0 mismatch。远端和本地`independent_submission_check_v1`通过，ZIP仅含根目录`predictions.jsonl`，解压payload逐字节一致。

**Candidate**：`submissions/R43_IV2_EVENT_COMPLETION_20261017/R43_IV2_EVENT_COMPLETION_20261017.zip`；ZIP SHA256=`f20b91eadd89c2e6a73fd249a0dcba64078203008f80edf5b9a8902947074cb1`；predictions SHA256=`925eb721aed41e320ea8c3217b402df41a8bdbb03adfce8c66888103e37c5525`。用户已上传并回报官方分数34.95，不能宣称提升。

**Conclusion**：工程与部署合同已通过，但R43官方34.95低于R26的36.04。batch=88不是稳态配置，长时间运行仍会触发显存峰值；后续若重跑使用batch=4或实现按OOM重启的分块策略。官方结果削弱直接事件完成度排序路线，停止同类提示和权重扫参。

### R43 官方回分追加（2026-10-17）

用户回报 R43=`34.95`，R26中心85+B3=`36.04`，`Δ=-1.09`。两包视频数、精确预算、预测总数和B3空间框合同一致，故这次比较只改变时间集合。R43与R26平均选帧交集比例`0.8872`，平均每视频替换`53.0`帧（中位数34），没有官方逐视频GT，不能进一步分解漏选与误选。

R43官方结果削弱“视觉视频文本事件完成度可直接替代中心控制”的假设；停止提示、融合权重和同类大模型的扫参。人工诊断显示高光2/3中279/340处于“进行中”，完成/结果与结果后反应合计34/340；另有88/340明确需要前后过程或声音/语言。它们支持检查目标语义和可观测模态，但不构成官方增益估计。R43事件记录与ZIP SHA见`reports/r37_content_candidates/official_score_events.json`。

### R44 设计：可观测性与联合效用诊断（2026-10-17）

R43和R42的联合证据改变路线：直接事件完成度排序替换中心导致官方`-1.09`，而人工等级在同账号重复样本上的二次加权κ仅`.185`。因此先做源级留一的低维嵌套对照（中心→静态RGB→局部前后状态→音频可靠性门控），分别预测通用价值、进行中/结果和依赖类型；未知字段保留mask，空间框不作为native GT。配置为`configs/R44_OBSERVABILITY_UTILITY_20261017.json`，状态`DIAGNOSTIC_001_COMPLETED_NEXT_READOUT_NOT_STARTED`。

首个实用门是源外配对效应稳定为正，并且内容只作为受限中心残差；不通过就停止新增编码器和R43同类提示扫参。只有通过后才在固定B3与精确85%预算上生成候选。当前无训练、无官方标签、无R44上传包。

### R44_DIAG_001 实际结果

12源CPU留一、固定Ridge读出、静态/局部帧变化/音频低级描述子。通用等级MAE：中心1.0668，局部状态0.9695，加入音频1.0100；局部状态在9/12源改善。高光中的进行中/结果MAE：中心0.1978，局部0.2320，音频0.2525；前后/声音依赖MAE：中心0.3932，局部0.3902，音频0.4073。结论：存在可继续验证的局部状态信号，但没有阶段或低级音频门控增量；暂停音频-only分支，不启动大模型或官方候选。

## R42 V1.3 完整标注质量（2026-10-17）

Label Studio项目1/2正式标注分别300/300与240/240，结构审计零错误；两项目均由同一账号完成，不能报告双人准确率。四个真正同协议重叠源120帧，高光等级完全一致率28.3%，相差不超过1级62.5%，阶段一致46.7%，事件类型42.5%。这些标签适合做机制诊断和错误分层，不能直接作为官方Y硬监督。

540帧中高光2/3为340帧；其中279帧标为“进行中”，88帧明确需要前后过程或声音/语言。有限流程旗标包括14个低等级带框、1个多框、1个近退化框、6个高光依赖缺失和若干构图说明缺失；正式标签未自动改写。完整审计见`reports/r42_annotation_v13_full_20261017/summary.md`。
## R44 — 2026-10-17：人工标签校准残差，严格继承 R26

**Hypothesis**：R26 的中心 85% 排序已经是当前安全基线；540 份人工标签中可复现的局部前后状态信息可以作为小幅残差，修正少量时间帧，同时保留 R26 已验证的 B3 空间收益。竞争解释是人工标签语义、同账号重复性和公开源域迁移不足，残差会破坏中心排序。

**Change / control**：源级留一 Ridge 读出，9 个局部静态描述子 + 前后灰度状态 + 已审计 InternVideo2 事件分数；`alpha=10`、`eta=.20` 在查看官方视频前冻结。全片候选来自 B3 全帧父预测；每视频 `round(.85*N)`；空间框逐项从 B3 父预测复制。对照是 R26 `R26_POS_CENTER_EXACT085_B3`，不是 R43 的 6B 时间集合。

**Public diagnostic**：12 源、18 个标注视图，等级 MAE 中心 1.0702、局部+IV2 残差 0.9992；`eta=.20` 固定集合代理 F 平均 `+0.02086`、二值代理 `+0.02384`，9/18 视图为正。该结果是公共人工标签诊断，不是官方联合 F，也没有独立新源确认。InternVideo2 embedding 读出没有转化为集合收益，因此不部署高维 embedding 版本。

**Package result**：426/426 视频、199037 帧、精确预算；419/426 视频相对 R26 改变时间集合，平均替换 14.91 帧；保留帧 B3 框 0 项改变。参数申报 450283377（0.450283377B）；独立 checker 与 ZIP 解压回归通过。候选包 `submissions/R44_HUMAN_CALIBRATED_RESIDUAL_ETA020_B3/`，ZIP SHA `b814699f1bdc71f2fec80e8e0b9465fe600327d44eab731f41f4d75ff6deac9e`，尚未上传，官方分数 null。

**Conclusion**：R26 是父包与回退；R44 是目前唯一完成包级验收的新机制候选。若官方回分不高于 36.04，停止继续扫 `eta`/`alpha`，转向新的可观测信息或空间原生监督；若超过，则以官方回分和上传 SHA 绑定确认，不把公共代理增益当成分数承诺。证据：`reports/r44_human_calibrated_residual_20261017/summary.md`、`reports/r44_residual_002_probe_20261017.json`、`configs/R44_HUMAN_CALIBRATED_RESIDUAL_ETA020_B3.json`。


## R44 官方回分与随机微扰分析（2026-10-17）

**Hypothesis / control**：R44 只在 R26 中心 85% 的边界进行人工标签校准残差重排，B3 框和每视频 `round(.85*N)` 预算保持不变。父包 R26 官方分 36.04。

**Official result**：用户报告 R44=`36.09`，Δ=`+0.05`。R44 426/426 视频、199037 帧，ZIP SHA256=`b814699f1bdc71f2fec80e8e0b9465fe600327d44eab731f41f4d75ff6deac9e`；平台上传 SHA 未独立核验。相对 R26 改变 6352/199037 帧（3.19%），B3 框零改变。

**Interpretation**：固定官方 evaluator 下这是该包的真实差异，但增益小到无法与近邻换帧波动区分。20,000 次、18 个公开人工视图的诊断 null（与 R44 相同交换比例）`P(ΔScore≥.05)=.2369`，不能作为官方 p 值，因为人工标签不是比赛 GT。结论为弱信号，不晋级为稳定收益，也不继续 eta/alpha 扫描。完整分析见 `reports/r44_score_analysis_20261017/analysis.md`。

**Next**：冻结 R26 中间区域，仅允许 R44 变化落在中心窗口两端各 5% 的局部边界带，构建 R45；若边界保护仍无实用收益，停止该残差路线。


## R45_BOUNDARY_GUARDED_RESIDUAL_B3（2026-10-17）

**假设**：R44 的微小正差若有机制，应该局限在 R26 中心窗口的边缘换帧；冻结中心区域可去除长视频的大跳变与随机近邻波动。竞争解释是 R44 的 +0.05 只是偶然交换，边界保护不会复现。

**预注册改变**：从已生成的 R44 集合出发，只保留新增帧和删除帧均位于 R26 中心窗口两端各 5% 全视频边界带内的交换；其余变更回退到 R26。固定每视频 `round(.85*N)`，不扫 eta/alpha；B3 框沿用 R26，公共帧逐项比对。

**结果**：426/426、199037 帧；R44 的 6352 个交换中保留 5942 个，410 个回退；R45 相对 R26 改变 411 个视频。参数 `450283377`（0.450283377B）。结构审计和 ZIP 往返通过。预测 SHA `f71d233a199b81ff48adcea154e1af792fef921022272afc3d27070bca6d0940`，ZIP SHA `f50506802fa3809d41913e2f3751bddc1ca71d2e9ea60c47e067b220a6d07cb2`。

**状态**：已形成可上传候选，尚无官方回分。该包只检验边界保护机制，不能预先承诺涨分；R26 仍为回退，R44 仍是当前最高已报告分。配置、构建脚本和报告见 `configs/R45_BOUNDARY_GUARDED_RESIDUAL_B3.json`、`scripts/package_r45_boundary_guarded.py`、`reports/r45_boundary_guarded_residual_20261017/summary.md`。

## 2026-10-18：R45/R46 高频状态与空间部署审查

| Experiment | Hypothesis | Change | Public evidence | Decision |
|---|---|---|---|---|
| R45_HIGHFREQ_BOUNDARY_PROBE | 4fps变化主要帮助中心窗边缘 | 固定eta=.20、keep=.85、两侧5%边界；源级留一 | center .7038887；full/boundary .7066364；3/12源正 | 仅保留边界作为候选派生规则，停止边界比例扫描 |
| R45_HIGHFREQ_STATE_PROBE | 子秒状态补充局部高光读出 | D1静态、D2 1fps local、D3 4fps、D4+D冻结IV2 | D1/D2/D3/D4 MAE 1.07136/0.98537/0.99133/0.96739；D3集合代理+0.02283，D4+0.01306 | 生成D3与D4完整包，不把代理增量当官方收益 |
| R46_SEMANTIC_TRANSITION_PROBE | 语义锚×状态变化交互提高边界排序 | 固定交互块，alpha=10、eta=.20 | 代理与D4相同，3/12源正；回归略差 | 停止该支线 |
| R46_SPATIAL_ASSET_AUDIT | 可恢复B3 head后再做utility residual | 搜索远端/root与/data的67个.pt及V8 ZIP | 精确V8 B3 head/base缺失；V9 A0/A1和V7近似head不可替代 | 暂缓空间提交，避免伪造归因 |

## 2026-10-18：R47 TimeLens2 事件区间诊断

| Experiment | Hypothesis | Change | Evidence | Decision |
|---|---|---|---|---|
| R47_TIMELENS2_PUBLIC | 直接事件区间建模能补足中心位置排序 | TimeLens2-8B固定无query prompt，2fps/320px，区间转逐秒分数，精确85%代理 | 12源、18视图：center .7044166，TimeLens .7213273，+ .0169107，6/18正；真实视觉smoke合法返回区间 | 机制成立到诊断级，保留为学生蒸馏方向 |
| R47_PARAMETER_AUDIT | TimeLens2能否与现役B3一起提交 | safetensors shape metadata计数 | TimeLens2=8,767,123,696；加B3=9,217,407,073，超过9B | 停止全量推理和直接包构建；不使用违规组合 |

## 2026-10-18：R47 TimeLens2 完整推理与合规空间候选

**Hypothesis**：TimeLens2的事件区间输出提供了中心先验之外的时间证据；若空间组件使用已核验的vision-only V9 B3（87,407,729参数），总系统仍低于9B，可直接检验事件区间对官方F的贡献。

**Execution**：固定prompt、2 fps、最大320×320像素、`temperature=.01/top_p=.001/top_k=1`，分9片完成426个视频。所有响应可解析或有明确空区间，原始响应SHA记录在候选manifest；没有读取官方标签或回分。V9 B3父集为完整426行、234,159个原始帧预测，按同一视频FPS把区间映射到原始frame轴，再选择精确`round(.85*N)`帧。

**Candidates**：

| Candidate | 空间 | 参数 | 预测 | 状态 |
|---|---|---:|---:|---|
| `R47_TIMELENS2_V9_B3` | V9 vision-only B3完整空间预测 | 8,854,531,425 | 426/426、199,037 | 可上传候选；资格依赖把已审计V9 artifact视为实际空间组件 |
| `R47_TIMELENS2_CENTER_EXACT085` | 确定性中心框 | 8,767,123,696 | 426/426、199,037 | 参数严格合规的空间降级对照 |

V9候选ZIP SHA=`436e9156c8fd55a04cae88840da95f177d633ae6b3e08665db8918f29c88c56b`，预测SHA=`13c4719b4c72f3eefe718715514da49ae0d4124fc1b6e63f3dc8db948f2199a8`。中心候选ZIP SHA=`a6952e2f6802a1b28dba738c4bf3f992158a7618da65ba57bc3197ea4e485ac9`，预测SHA=`dc1246b7d3dac69635397188f979e68cfe30d83b351a8ae995f0bbe18d89d665`。原始9分片SHA固定于`reports/r47_raw_shards_20261018/`；不采用其他重复进程产生的不同SHA包。

**Interpretation**：TimeLens2+完整LFM B3为9.217B，明确超限；V9 vision-only组件的实际参数清单已在远端`vision_only_model/MANIFEST.json`核验，故8.8545B候选满足数值上限。候选仍保留“artifact acceptance”资格条件，因为本次构包复用了此前已生成的V9全帧预测，没有重新运行空间forward。中心候选用于区分时间机制与空间降级，不能作为高分预期。

**Reproducibility**：脚本为`scripts/r47_timelens2_official.py`、`scripts/r47_merge_timelens_shards.py`、`scripts/package_r47_timelens2_b3.py`和`scripts/r47_package_timelens2.py`；完整区间合并记录在`reports/r47_timelens2_all426.json`。官方分数仍为空，不把公开18视图代理增量当成官方收益。

同一V9 vision-only空间父集另生成了`R47_V9_B3_CENTER85`：时间集合严格复用R26中心85%，只替换V9 vision-only B3空间预测。它覆盖426/426、199,037帧，参数87,407,729；相对R26的234,159个全帧空间预测有37,499项差异。该包用于识别空间runtime替换的独立贡献，不与TimeLens2增益混淆，官方分数仍为空。
## R48：低频—高频共识门控残差（2026-10-18）

**Hypothesis**：D3高频局部状态的集合代理增益可能来自少数正确的边界纠正，但其回归MAE略差于D2；让高频残差只有在独立的1fps局部读出同向时生效，可以减少镜头切换和噪声导致的错误交换。

**Change**：源级留一 Ridge（`alpha=10`），D2为9维静态+1fps前后状态，D3为同一静态描述子+4fps子秒状态。每视频保留R26精确85%预算，固定`eta=.20`；R48得分为`rank(center)+eta*(rank(D3)-.5)*I[rank(D2),rank(D3)同侧]`。空间框逐项复制R26 B3。

**Public result**：18个标注视图上，D2 `+0.01901`（9正），D3 `+0.02283`（9正），R48 `+0.02360`（9正）；起始变化门控 `+0.01376`（7正）。R48只比D3小幅改善，且使用同一12源和同一账号标注，不能视为独立确认或官方收益。

**Status**：公共判别实验通过“继续构包”门；426视频包正在AutoDL生成。若结构审计通过，R48是D3的机制控制候选；不进行任何进一步阈值/系数扫描。详见`reports/r48_consensus_gated_residual_20261018/summary.md`和`reports/r48_consensus_probe_20261018.json`。

**Package audit**：全量R48已完成，426/426、199037条、精确`round(.85*N)`；相对R26有415个视频变化，平均交换12.65帧，B3框逐项一致。`R48_CONSENSUS_GATED_RESIDUAL_B3` ZIP SHA=`d3efa721eb7c2c5ad4d4fbb2000e7e79b78d667aecfe3355b16dd236ad0aa1d9`；固定5%边界派生包 SHA=`90d6769e771489fc632fef3dbdc41ed306869f4d76d40a361b80924f00f15bb8`。独立checker、根目录ZIP和解压预测SHA通过，均为`PACKAGED_NOT_UPLOADED`。

## 2026-10-18：R49–R52 时间残差扩展判别

这些实验都使用同一12个公开30秒源、整源留一、固定中心85%代理、`eta=.20`和Ridge `alpha=10`；没有读取官方标签，也没有根据结果生成权重/系数扫描。

| 实验 | 改变 | soft代理增量 | 结论 |
|---|---|---:|---|
| R49 | 先预测人工`event_phase`，再按训练折阶段价值形成残差 | +0.00291，2/12源正 | 阶段因观测不足和标注分歧不可稳定恢复，停止 |
| R50 | D3局部状态加全片起止/中位状态、累计变化和运动密度 | +0.02001；去掉原始位置+0.02211 | 低于统一聚合D3 +0.02302，不生成包；全局上下文不是已证增量 |
| R51 | 仅在前后运动都高且两侧连续性过中位数时启用D3 | +0.00654 | “高光必须持续”被削弱；停止该门控 |
| R52 | D3加固定Farneback光流幅度/尾部/方向一致性/散度 | +0.01093 | 简单灰度状态已优于该光流合同，停止 |

报告与可复算脚本：`reports/r49_event_phase_residual_probe_20261018.json`、`reports/r50_global_context_ablation_20261018.json`、`reports/r51_persistent_transition_probe_20261018.json`、`reports/r52_optical_flow_probe_20261018.json` 及对应 `scripts/r49_*.py`–`scripts/r52_*.py`。这些是公共机制诊断，不是官方联合F。

**Action**：D3/R46仍是时间侧首选；R48只作为固定共识门控控制包等待结构审计。研究资源转向TimeLens2合规候选和V9空间独立对照，不继续堆叠时间特征。

## 2026-10-18：R53 输入/条件残差判别与 R54 TimeLens2 中心残差

| 实验 | 机制 | 公共源级结果 | 决定 |
|---|---|---:|---|
| R53_AUDIO | RMS、频带、谱质心、ZCR等低维音频描述子；与RGB局部状态比较 | audio-only `+0.00749`，RGB+audio `+0.01162`；二值分别 `+0.00854/+0.01540` | 低维音频没有超过D3，停止直接音频残差；不生成包 |
| R53_QUALITY | 固定画质/运动交互条件化D2 | D2 `+0.02235`；质量交互 `+0.01498` | 画质门控过拟合，停止 |
| R53_PHASE_PRED | 从RGB预测事件阶段再交互 | `+0.02473`，阶段准确率`.6333`，相对D2仅`+0.00238`，CI跨零 | 不生成包；需新语义输入才能继续 |
| R54_TL_RESIDUAL | TimeLens2区间掩码只作中心排序的固定`eta=.20` residual | 中心`.704417`→硬区间`.721327`（`+.01691`）→R54`.729414`（`+.024997`，10/18正） | 形成提交候选，替代硬区间作为TimeLens2首选 |

R54完整包已通过独立 checker 和 semifinal contract：426/426、199037帧、`8.854531425B`参数；V9空间artifact资格条件与R47相同。包路径为`submissions/R54_TIMELENS_CENTER_RESIDUAL_V9_B3/`，ZIP SHA=`1e0f92f2292da0c057aa9b45a27134783af4def37a60c95ffde6d8fa9a8ae210`。公共区间来自缓存的TimeLens2代理报告，官方测试仅用于合法推理，未参与拟合或调参。

## 2026-10-18：R54/R46 官方回分与时间相位判断

| Experiment | Official score | Controlled change | Evidence | Conclusion |
|---|---:|---|---|---|
| R46_HIGHFREQ_LOCAL_RESIDUAL_B3 | 35.88 | R26 B3固定，仅4fps局部状态时间残差 | 5,636换帧、424/426视频；框0项改变；相对R26 -0.16 | 公共人工标签代理未迁移，停止低级RGB残差族 |
| R54_TIMELENS_CENTER_RESIDUAL_V9_B3 | 36.22 | TimeLens2区间中心残差 + V9 vision-only空间 | 6,152换帧、417/426视频；V9框28,730项改变；相对R26 +0.18 | 联合路线首次超过R26；时间、空间、交互尚未分解 |

R54的变化几乎是视频级连续窗口后移：425/426视频近似连续块，选中位置均值`.49957→.52344`，区间覆盖`.75695→.76531`。这支持“TimeLens估计了后段相位”的解释，同时保留两个竞争解释：V9少量竖屏框修正贡献，或R54主要是位置修正而非语义帧排序。详见`reports/r54_r46_official_postmortem_20261018/summary.md`和`reports/r54_timelens_center_residual_20261018/package_diff_20261018.md`。

## 2026-10-18：激进观测密度实验 R55_DENSE

**Hypothesis**：2fps稀疏观测把动作起点和结果边界混在一起；TimeLens2同权重提高到4fps会产生不同且更准确的事件区间，从而改善视频级窗口偏移。

**Change**：TimeLens2-8B、generic prompt、V9 vision-only B3保持不变；视频输入由2fps提高到4fps，最大每帧像素和确定性解码固定不变。10视频真实smoke已完成，视频0区间`[[3,10]]→[[0,5],[7,9]]`，视频1`[[7,16]]→[[6,16]]`；全量426正在AutoDL `/root/autodl-tmp/r55_dense_full/`执行。

**Status**：这是观测机制实验，不是参数/eta扫描。全量完成后只生成一个4fps候选并绑定其响应SHA；若结果不稳定，保留R54，不继续无结构提高fps。

## 2026-10-18：R56事件端点性候选

R56固定中心锚和`.20` residual，把二值区间替换为区间内归一化端点性`q=2*abs(normalized_phase-.5)`，多区间取max。公共18视图soft proxy比R54高`.001696`，binary高`.002584`，11/18视图为正；该增益主要由一个视图贡献，仍属诊断证据。包已通过426/426、199037、参数8.854531425B、独立checker与contract，ZIP SHA=`43845ef4085e65985c768a7c9391e895d28ed0c89f0d04ef69bc3801da96f455`。其价值是测试“事件端点比事件中段更接近高光”，不应被描述为已确认涨分。

## 2026-10-18：R57 固定后移窗口控制

R54的官方差异近似一个连续窗口后移；同一公共18视图上固定后移2秒的中心窗口 proxy `.736923`，高于中心`.704417`和R54`.729414`。因此构造单一、预先固定的 `R57_FIXED_LATE2S_V9_B3`，每视频精确85%预算，窗口按实际fps后移2秒并裁剪到合法边界，空间使用V9。包426/426、199037、参数0.087407729B，checker通过；预测 SHA=`c1fc69324a7dafa3c2faca2de80af24438c20df466b3dfb5cf9815eaacae84c9`，ZIP SHA=`eac6c8610a43f6bb20b536018984febc683ac97ebf0ea3b986acb50f681f7fce`。该包专门检验“R54主要是后段位置先验”的竞争解释，不是偏移扫描。

## 2026-10-18：TimeLens2 窄结果提示词拒绝

第二个要求“窄结果区间”的prompt在12个公开标注片段上完成真实视觉推理。相同18视图 binary proxy=`.720168`，低于generic prompt`.721327`，仅4/18超过中心；不扩展到426，不形成提交包。当前证据支持保留generic prompt，优先改变观测密度和解码规则。

## 2026-10-18：R58 TimeLens + dense audio novelty（激进正交路线）

**Hypothesis**：TimeLens主要提供视觉事件相位；音频瞬态、掌声/欢呼或语音能量峰可能在视觉窗口外定位高光。对R54每视频精确85%集合，固定交换最低音频新颖度的5%选中帧与最高新颖度的未选帧。音频只用无参数DSP（20 Hz RMS/peak/ZCR/FFT质心、正向log能量变化、局部对比、±0.25 s邻域最大值），不使用测试标签。

**Public mechanism check**：严格在同一12源、18视图人工标注上运行同一5%交换协议：soft代理`−0.00592`、binary`−0.00585`，仅2/18视图为正。该反证比R53低维音频`+0.00749`更接近实际解码规则，说明“用瞬态替换TimeLens窗”大概率损失连续事件帧。

**Test artifact**：426视频中386有音频；相对R54交换9,515帧（386视频），保留R54已有V9框，新帧采用已审计R26 B3全帧框。候选`submissions/R58_TIMELENS_AUDIO_NOVELTY_V9_B3/`通过独立checker（426/426、199,037、8.854531425B），ZIP与预测SHA均写入manifest。由于公共反证和空间混杂，不建议占正式名额；若需要极端波动试水，只能作为最后备选。
# 2026-10-18：超级激进联合路线与密集观测

## R59_TIMELENS4FPS_MASK_EDGE_ENSEMBLE_V9_B3

**假设**：2fps观测会把动作起点、过程和结果合并；同一8B TimeLens2权重改为4fps后，区间边界和端点性应产生不同的可迁移窗口信号。固定解码为 `rank(center)+.20*rank(interval)+.50*rank(endpoint)`，不扫系数；空间沿用已审计V9 vision-only B3。

**结果**：426/426、199037帧、独立checker通过，总参数8,854,531,425（8.855B）。相对V9中心85改变424/426视频、14028次交换（32.93/视频），明显不同于2fps R58的12933次交换。4fps响应SHA=`bf712c77d94cdfd41d07e1d9aa0ec2719bd33328787df43662d96d6a64dc1906`，ZIP SHA=`ef46457ab64b9a6d2a160cf45ba60ec75f2139fe2f74209470cc4da162cdad76`（完整SHA见报告）。

**结论**：这是当前最强的观测机制候选；它尚无官方回分，不能把密集观测的结构差异写成收益。若官方下降，应停止简单加密采样；若上升，下一步优先保留4fps事件边界，而不是继续扫eta。

## R63/R64_JOINT_SPATIAL_DISAGREEMENT

**假设**：V9与R26空间读出发生较大变化的帧，可能是主体/构图状态真正变化的内容转移；反向稳定性臂检验该差异是否只是空间不稳定。两臂均使用真实FPS和与R54完全相同的中心浮点表达式；gamma=0逐视频逐帧复现R54。

**R63** 使用 `rank(center)+.20*rank(TimeLens interval)+1.0*rank(V9-R26 geometry disagreement)`，相对R54交换12572帧、425视频，均值位置`.56669`；ZIP SHA=`463c9c2c7d7c70137aa4da44ec14768816a89353d5fb3a44dcd65d5bb699f556`。

**R64** 使用负空间分歧，交换22741帧、426视频，均值位置`.44301`；ZIP SHA=`84320541b0250384613124133c39c478686837b83f35f7780ebe4734f8658216`。两包总参数均8.855B、精确85%预算。R59–R62旧稿因FPS索引或浮点边界问题标记作废，不能上传。

**结论（后续审计更正）**：R63/R64 均作废、不可上传。stable ordinal rank 对约 90% 的平值几何差异按帧号分配不同秩，混入隐藏位置先验，故它们不能作为空间分歧与稳定性的有效机制对照。

### R63/R64 实现审计更正与 R67 修正版

后续代码审计发现上述两包的`rank(V9-R26 geometry disagreement)`采用stable ordinal rank。几何差异约90%为平值时，平值仍按帧号获得不同秩，导致R63均值位置后移、R64前移；这不是空间分歧/稳定性机制的有效对照，两个包均标记为不可上传。

`R67_JOINT_V9_DISAGREEMENT_TIEAVG_G1`修复了两处归因问题：center和TimeLens interval完全沿用R54 stable rank，只有几何分歧使用平值平均秩；绝对几何差异`<=1e-6`归零以去除序列化噪声。gamma=0输出与R54逐视频逐帧和序列化SHA一致。gamma=1包相对R54交换2427帧（81个视频），均值位置`.519774`，426/426、199037条、总参数`8,854,531,425`；独立checker与contract通过。预测SHA=`4655d73b5e8a93398b75c1ff2c6ae8d6110e061750d0350db23283d6d4490beb`，ZIP SHA=`550bf1c87d2fa03f046b6dfd581882bce6d47850585789ed7e09a175ec2d34bd`。这是有效的空间分歧修正版，但官方回分为 **36.16**（低于 R54=36.22 的 0.06）；无监督空间分歧路线停止，不再作为上传候选。

## 音频正交路线停止

R58音频新颖度候选在同一12源/18视图人工标签上严格复现为soft `-0.00592`、binary `-0.00585`，仅2/18视图为正；且新增帧混用R26 B3空间artifact。保留包作最后风险备份，不作为主提交。


## 2026-10-18：R57/R59官方回分与R63/R64最后名额审计

R57=`34.76`、R59=`35.04`，均低于R54=`36.22`。R57的统一`+2s`窗口后移交换14,015帧并把宏观位置推到`.57061`，反驳全局后段平移；R59的4fps interval/edge解码交换14,028帧并得`35.04`，反驳未校准的密集端点替换。两项公共18视图正代理均未迁移到426视频。

原R63/R64的geometry disagreement使用stable ordinal rank，在约90%平值/数值噪声上按帧号注入隐藏位置先验，故不具备机制效度。修正R67采用tie-aware average rank和`1e-6`噪声阈值，官方=`36.16`（R54 -0.06），已成为有效空间分歧对照；R63/R64及无epsilon R68变体禁止上传。

最后一个高风险候选转向结构不同的R69：同一8.767B TimeLens2完整视频直接输出目标比例全局crop，保留R54时间集合/精确85%预算。全量响应在AutoDL运行，完成后仅在解析率、坐标合同和完整426覆盖通过时形成上传包；该实验直接检验VLM空间grounding能否替代逐帧B3 utility，预期风险高且无官方涨分承诺。


R69全量响应已完成：TimeLens2直接空间 query 在 426/426 视频解析成功。最终发行包为 `R69_TIMELENS_DIRECT_CROP_V1`，保留 R54 帧集合和精确预算，直接使用完整视频 query 的目标比例 crop（每视频一个全局框），不使用旧的 `CENTER_MAXWIN` 中心投影变体；199,037 条预测、8,767,123,696 参数、0 fallback、0 帧集合差异、129,443 个选中帧框替换。independent checker、semifinal contract 和 ZIP 往返均通过。预测 SHA=`aad8d3455490c2d3d43d42bf98c93117b405993452f985fb840126109452dfcf`，ZIP SHA=`f5d6dcadbd511f286c98360a72cfb568146056eb77d8cee6d62194b9e7839871`。该包是结构性高风险空间 grounding 试验，官方分数待上传后回传。

## 2026-10-18：R70 candidate utility readout（进行中）

R70与R69直接构图路线不同。R70固定V9 vision-only塔、processor、129合法最大窗口和native crop GT，只重新训练候选效用head。由于当前克隆缺原始V8 B3 `head_s.pt`，新head明确不称B3复现。

公共源级合同：RetargetVid/LIVE train用于训练，RV/LIVE dev按源留出；官方test不读。实现为`scripts/r70_spatial_utility_pilot.py`，冻结协议为`configs/R70_SPATIAL_UTILITY_PILOT.json`。smoke已通过（视觉tensor零缺失/零多余、候选形状`[129,2305]`、训练链可运行）；完整抽取在AutoDL `/root/autodl-tmp/experiments/R70_SPATIAL_UTILITY/full/`。

 promotion gate：新head必须在RV和LIVE两个留出池同时超过center；未过则关闭当前读出路线。通过后才固定R26 center85时间集合重提取426，形成单一空间候选包。

## R70 official inference and CPU/GPU throughput repair (2026-10-18)

**Hypothesis:** the observed CPU-heavy/GPU-idle behavior is an implementation bottleneck rather than a vision-tower limitation. **Change:** cache the exact V9 candidate mask by geometry/grid shape; pool all 129 candidates and run the utility head as one GPU batch; overlap one CPU decode/processor lookahead batch; convert processor pixels to FP16 before transfer. **Controls:** original feature helper and per-frame head on random grids; 10-video profile. **Result:** max feature difference `0.0`, 10-video batch-32 profile `2.7s`, full 426 inference `174.3s`; active GPU samples reached `97–100%`. **Official output:** 426 videos, 8,848 keyframes, zero fallback. **Package:** R54 time set fixed, 102,981 retained boxes changed, 8.854531425B, checker/contract/ZIP valid; prediction SHA `5d34388eb39a2d8f49ba9cd29420cbb5c2e450f2c88ec2aee16be11cbeeff3da`, ZIP SHA `a5d3031cb6ab896d73e83e032be0e385bdb17efa3c0024af06036dcc63179b7a`. **Conclusion:** CPU preprocessing and per-frame CPU readout caused the prior GPU starvation. The package remains an unuploaded official-score experiment; it cannot be called a gain until platform feedback.

## 2026-10-18：R69 分段构图

**Hypothesis**：完整视频一次 crop 把不同高光阶段的主体/构图平均掉；在同一 R54 选中时间集合内分段询问可减少空间时变误差。

**Control/Change**：固定 R54 时间集合与精确 85% 预算；按原始 frame 连续 run 切 3 段、每段 5% 上下文；同一 TimeLens2-8B、16:9 query；不改时间排序。代码 `scripts/r69_segmented_crop_batch.py`、`scripts/r69_segmented_crop_package.py`，缓存 `reports/r69_segmented_crop_full_20261018.json`，报告 `reports/r69_segmented_crop_20261018/README.md`。

**Result**：1,281/1,281 响应有效，426/426 覆盖、0 fallback；111,306/199,037 框替换。188/426 视频有多个 segment crop；相对全视频 R69 有 102,617 框不同、277 视频变化。smoke 中视频 0 垂直中心变化、视频 1 水平范围变化、视频 2 由窄到全幅，说明输入不是固定全片答案。包 `submissions/R69_SEGMENTED_CROP_R54TIME/`，预测 SHA `df2c8c42fce86357f13c8db5fba4fa5a6b88332285ebb1120c1a448fc5f45eef`，ZIP SHA `0b1de985c709b2b4d31d2c1a57b3d0420a3f66afb9dbad82605ff534e10a8010`，合同/checker/往返通过，官方待上传。

**Decision/Limit**：这是结构性空间决策实验，不是官方收益证据；逐段变化也可能是 VLM 构图不稳定或分段上下文变化。若回分不升，停止继续增加分段数/上下文扫描；若回分升，下一步只按错误视频和切换边界做机制复核。

## 2026-10-18：R71 逐视频自适应预算

**Hypothesis**：TimeLens 区间占用率可反映事件密度；密度低的视频应少取帧，把预算留给密度高的视频，可能降低官方 F 的分母浪费。

**Change**：R54 center+0.20 interval stable rank 和 V9 空间框完全固定，仅把每视频 keep 从精确 85% 改为占用率规则；无官方标签/回分拟合。结果 426/426、195,214 条，删除 3,823 帧，参数 8.854531425B。

**Diagnostic**：10 个公共视窗 soft 增量 +0.0011959（3/10 正），binary -0.00287195（3/10 正）；soft/binary 方向冲突，且不是 AIC GT。结论是机制未识别，保留为高风险备选，不升级 R54、不做规则扫描。

## 2026-10-18：R72 原始 B3 资产恢复审计

**Question**：能否找回 R26/R54 系历史 V8 B3 的真实训练 head/base，以恢复原始空间路线。

**Audit**：期望路径及当前克隆路径均不存在；67 个候选 `.pt` 只匹配 V7/V9 A0/A1/KD 形状/前缀；V8 B3 提交 ZIP 无模型/特征/logits；Git 历史/分支无二进制；旧端点不可达。报告 `reports/r72_b3_asset_recovery_20261018/README.md`。

**Conclusion**：精确原始 B3 不可恢复，缓存 B3 预测可作回放父集；不能把 V7 或新训练 head 称 B3。且 TimeLens2+B3 理论参数约 9.217B，当前 9B 规则下不合规。B3 恢复路线关闭；只有按公开数据重训的新 head 实验可重新立项。

## 2026-10-18：R69 分段构图稳定性门控

**Hypothesis**：R69 的空间变化有价值，但不一致的段间 crop 会制造 native-frame IoU 损失；run 级稳定性门控应保留一致变化并抑制大跳变。

**Fixed rule**：合法自由轴中心相对整视频 crop 的最大偏差 `<=0.12` 且同一 run 内 spread `<=0.12` 才采用全部分段框，否则整个 run 回退整视频 crop。阈值对应允许一个常见的 0.1 量化步，不看官方分数选择。

**Result**：427 runs 中 375 stable、52 unstable；unstable 回退 33378 选中帧。相邻中心总变差从 22.4604 降到 11.9998（-46.5%），p95 邻跳从 0.2000 降到 0.1367；相对无门控 R69 改变 28302 帧。包 `submissions/R69_SEGMENTED_GATE_R54TIME/`，426/426、199037、8.767B、0 fallback，ZIP SHA `f7d17806697bd677580fcddda2fe02c441c1dacb3c7f979cd000823a57c5cf58`，checks 通过。

**Conclusion**：门控确实改变了稳定性结构，但没有 native GT，不能把跳变减少换算成 IoU 增益。该包是无门控 R69 的更安全版本；若官方仍下降，下一步停止段数/阈值扫描。

## 2026-10-18：R70 topsoft spatial objective

**Hypothesis**：逐候选 MSE 没有直接强调“接近最优候选”的排序；soft top-candidate target 可能改善候选 utility 的跨源排序。

**Control**：同一 V9 2305 维特征、同一 source-disjoint RetargetVid/LIVE split、同一 seed/batch/1200 steps；比较 MSE、pairwise、listwise 和 topsoft。

**Result**：topsoft source-macro delta 为 RV `+0.07121`、LIVE `+0.06892`，平均 `+0.07006`；MSE 为 `+0.06729/+0.07073`，平均 `+0.06901`。pairwise/listwise 没有同时超过 topsoft。topsoft 官方空间包 `R70_TOPSOFT_SPATIAL_R54TIME` 已完成，426/426、199037、101797 框变化、8.855B、0 fallback，ZIP SHA `da5fcc5dccbfcd960b112852295dbbba35be89aaad61c20fcbb81d1ba8fed4d2`。

**Limit**：增量很小，仍来自同一公开 split；官方 test 没有 native GT。该目标值得保留为下一空间候选，但不能称已确认提升。

## 2026-10-18：R70 spatial utility driven temporal reorder

**Hypothesis**：官方 F 的时间选择应同时考虑高光价值和选中后可获得的 crop IoU；R70 top candidate score 可作为后者的 observable proxy。

**Fixed decision**：对完整 frame 轴计算 `T=rank(center+.20*rank(interval))`，将 R70 topsoft top-candidate score 插值后得到 `U=rank(utility)`，用 `J=T*(.5+.5U)` 选择精确 `round(.85*N)` 帧。保留完整 V9 空间框，不扫 joint weight。

**Result**：426/426、199037、平均每视频相对 R54 交换 5.122 帧、377 视频变化、8.855B；包 `R70_TOPSOFT_UTILITY_TEMPORAL`，ZIP SHA `aadb42cadc97a9880b63cc7367ea5285d53a365ea4802040a934131e1440883f`，checks 通过，官方分数未知。

**Interpretation**：这是首次把空间 utility 直接放入时间决策的机制包。官方下降将优先指向 utility 跨域校准或时间价值与构图价值的相关性假设；官方上升才继续研究 utility calibration，不扫描任意融合权重。

## 2026-10-18：R72 保守 utility residual 对照

在同一 R70 utility dump 上构造更接近 R54 的联合控制：`R72 = R54_score + .20*(rank(R70 max_score)-.5)`，`.20` 直接继承 R54 interval residual，不采用乘法重标定。精确重建 R54 集合后，共交换 4120 帧、393/426 视频变化，平均重叠 97.94%；V9 B3 框逐字节来自 full parent。包 `R72_SPATIAL_UTILITY_TIME_R54_V9_B3`，8.855B，ZIP SHA `666a83f8dddd691022c66193eff30b54af6845fafadba3bb20ed3a6c99f64d88`，checks 通过。它是联合时间路线的保守对照，官方分数未知。

## 2026-10-18：空间效用驱动时间重排复核、tie-aware 与 output-aligned utility

**R70_UTILITY_DIAGNOSTIC_001**：在公开 RetargetVid/LIVE source-disjoint 留出（168源、1546行）上，用 topsoft 候选分数比较绝对 crop utility 与边际 utility。`max_score` 与实际选中框 IoU Spearman=.293、与实际 `IoU(argmax)-IoU(center)`=.128；`max_score-center_score` 分别为 -.113 和 **.377**；`soft_expected_score` 与实际选中框 IoU=.378。它们是不同空间目标；数据没有同视频时间GT，不能换算官方F。完整结果见 `reports/r70_gain_diagnostic_20261018.json`。

**R74_TIE_AWARE_INCREMENTAL_UTILITY_TIME_R54_V9_B3**：R73的增量utility存在约44.7%关键帧零增益平值，stable argsort会注入帧号顺序。只将utility秩改为exact-tie midrank，保持R54表达式、.85预算、V9 B3框，换帧由5004降至2418。426/426、199037、8.854531425B，ZIP SHA=`ee0e6d8f228941b7da47833ca5ee4f0f72c6e1c961610577bfa66385fd96098a`，checker通过。无官方分数，不占明日2/3/5。

**R76_OUTPUT_ALIGNED_UTILITY_TIME_R54_V9_B3**：R70 129候选utility在每个关键帧沿窗口偏移插值到V9 B3实际输出框位置，再以tie-aware rank加入R54固定residual。官方8848关键帧上 output utility 相对 max utility 均值差=-.03886、std=.08233，仅56.5%近似相同，证明R70 max与最终输出框不等价。R76 426/426、199037、8.854531425B，4251 swaps，ZIP SHA=`18d4986d84b705cd4076c4f640c77155eca5d1ac024583318fac0e7e3e7b2265`，checker通过；未上传。

**R77_OUTPUT_ALIGNED_MULTIPLICATIVE_TIME_R54_V9_B3**：使用 `rank(R54 temporal value) * (0.5 + 0.5*midrank(output-aligned utility))`，以条件独立时的 `E[Y*IoU]` 为决策动机，不扫融合系数。426/426、199037、8.854531425B、2308 swaps，ZIP SHA=`aae5cfa6ffc56e8276aa5aef741781143a23e71ca8f428a608228700aa7d4106`，checker通过；未上传。完整复核见 `reports/r70_utility_temporal_reassessment_20261018/summary.md`。

## 2026-10-19：直接部署框 utility head 与 R81

**R78**：冻结B3 candidate head，在R70公开source-disjoint native crop数据上把目标改为部署候选实际IoU `y[argmax(B3_head)]`。selected feature、confidence、全129 logits和topsoft logits+输出位置四种输入均完成诊断；50,561参数的`topsoft logits + B3 output position`版本row Spearman=`.47468`，source-macro Spearman=`.15947`，公开域结果仍未形成官方联合证据；严格nested external dev row Spearman=.47334、source-macro=.15850。回归头与V9原始分数的LIVE source-macro bootstrap差值为`+.00453`，95% CI `[-.00501,+.01418]`；pairwise分支停止。

**R81_DIRECT_OUTPUT_UTILITY_MULT_TIME_R54_V9_B3**：使用50,561参数直接utility head在R70官方关键帧topsoft logits和审计V9 B3输出位置上推理，按`rank(R54 time)*(0.5+0.5*midrank(output utility))`重排。426/426、199037、8.856094947B（含生成R70 logits的1.511M head）、2332 swaps，使用nested head，ZIP SHA=`80087b6e75fe70887db1f408e6b69baeae20812e5bb3e57ab3723de66dd1dfdb`，checker/roundtrip通过，未上传。报告见`reports/r78_direct_output_utility_20261019/summary.md`。

**Head identity gate**：A0 surrogate smoke与历史V9/B3 point文件明显不一致，停止用A0重跑冒称当前B3。R81直接复用审计V9全帧框和官方R70 logits，避免替换空间artifact；公共B3 logits训练资产仍需SHA/来源复核后才能把R81作为正式上传候选。


## R82_DIRECT_UTILITY_ENSEMBLE_RISK05_MULT_TIME_R54_V9_B3（2026-10-19）

**Hypothesis**：不同源域上的直接部署框 IoU 预测存在校准偏差；五个独立初始化 head 的均值减去不确定性可降低错误高估，把更可靠的空间效用用于时间重排。

**Change**：保留 R54 时间价值、V9/B3 全帧框、R70 topsoft 输入和精确 `round(.85*N)` 预算；仅将 R81 的单头 utility 换为五头 `mean - 0.5*std`。五个 head 全部计入参数。

**Public nested result**：168 外部源（逐行 0.04805 MSE、Spearman 0.47678、source-macro Spearman 0.19120）；mean 基线 source-macro 0.16816，原 R78 单头 0.16869。156 源 paired bootstrap delta `+0.02304`，95% CI `[+0.00856,+0.03812]`。beta 在查看外部结果后比较，故该包仍是 exploratory/high-risk，不能写成确认收益。

**Package**：426/426、199037 predictions、total parameters `8,856,303,335`、2294 swaps vs R54；prediction SHA `b186208540716b2465dd2c6766f12365e8c0aa2d8ccc39c61883d12f0becb6f4`，ZIP SHA `de99b3c5fcaee15966d893425e1b39a896645546092a4d00dc00133bc31529fd`。独立 checker、ZIP roundtrip 通过，未上传。

**Conclusion**：保留 R82 作为 direct-utility 路线的最高风险候选；若官方回分下降，关闭该时间重排分支；若上升，先用第二个独立 source split 固定 beta，再决定是否继续。

## 2026-10-19：R83 官方回分与本地验证有效性审计

**Question**：公共 native crop、人工时间视窗、稳定性和外部 utility 排序，能否作为 AIC 426 视频官方分数的代理？

**Evidence**：R70 官方 `36.42`（相对 R54 +0.20），本地 RV/LIVE source-macro crop utility 平均 +0.07006；R71 官方 `36.43`（+0.21），本地十个视窗 soft +0.00120、binary -0.00287；R82 官方 `36.20`（-0.02），本地外部 source-macro bootstrap +0.02304、95% CI `[+0.00856,+0.03812]`；R69 gate 官方 `24.95`（-11.27），本地邻域稳定性改善 46.5%。

**Conclusion**：本地代理不能承担官方候选排序。R82 是最直接的负识别例，R69 说明官方没有平滑奖励。R70 的同向结果支持空间读出存在迁移信号，但不能把一次官方正回分拆成确定的因果增益。R71 的正回分不能由十个公共窗口的标签方向解释，预算效应仍未识别。详细记录：`reports/r83_official_local_validity_20261019/summary.md`。

**Next mechanism**：转向公开 PHD2/GIF 标签上的低容量 `interval policy`。模型输入全片粗粒度视觉/运动统计，输出归一化中心、宽度和边界置信度，解码成单一连续 85% 集合。它与 frame-rank、utility fusion 结构不同，目标是拟合视频级评测政策。源级留出同时超过中心策略和既有 PHD2 frame-rank 才允许制作一个官方单变量包；只提高 PHD2 AP 而不提高集合效用则关闭。

## 2026-10-19：R83 interval policy probe

**Hypothesis**：低容量模型学习视频级连续区间政策，比逐帧排名更接近官方偏好。

**Protocol**：公开 PHD2/GIF 标签，5 折源级留出（5886 片段、2262 源、留出426源）；已有 LFM 8 槽特征；26,786 参数；输出只能选择两个连续 7/8 窗口；不读取官方 test。

**Result**：center `0.025424`，小型逐帧 BCE `0.025281`，interval policy `0.025424`，interval oracle `0.028871`。policy 完全退化为 center。

**Conclusion**：关闭这个过度受限的 interval 形式以及对窗口公式的继续扫描。不能据此关闭所有小数据路线，因为 8 槽只有两个合法窗口，且该指标是 PHD2 片段 F。下一主攻转到有直接 native crop 监督的低容量空间连续坐标/候选读出，固定 R54 时间集合。

## 2026-10-19：R84 coordinate-consistency spatial readout

**Hypothesis**：R70 候选 utility 的空间排序可以由更小的读出保留；加入坐标一致性损失能减少 flat utility 的跨域错误。

**Change**：固定 V9 vision tower、129 legal windows、RetargetVid/LIVE native crop source split；623,361 参数逐候选 MLP，MSE + top-soft KL + soft-argmax coordinate loss；固定 R54 时间集合和空间输出解码。

**Public result**：合并 source-macro `0.52306` vs center `0.49759`（`+0.02547`）；LIVE row `+0.03135`，RV row `+0.00365`；低于 R70 topsoft 两域平均 `+0.07006`。这是结构诊断，不是官方涨分预测。

**Official candidate**：426/426、199037、0 fallback、112141 changed boxes；8.853590257B；ZIP SHA `526d3e0b41c329806c7746e0f7c1404b0ec8d3d2470495e8470c0f25983934f4`；checks pass，未上传。若只保留一个新空间包，R70 仍优先；R84 用于检验小读出域互补。

## R85 官方榜单分布与目标域迁移复核（2026-10-19）

**Hypothesis**：40分以上的稀疏高分段可能来自评测政策/目标域迁移差异，而不是模型容量；空GT视频是待检验解释，不是已知事实。

**Change**：通过公开排行榜接口获取182条复赛记录；检索Highlight-CLIP、AHA、Cross-category VHD、Audiovisual Highlight Detection、Less is More、audio/visual recurrence、MH-DETR和Saliency-Guided DETR的一手论文摘要与任务条件；对照本项目已有R25/R43/QV/MrHiSum证据。

**Result**：最高47.13，中位数36.44，≥40分42条，≥45分4条，34–38分52条；R71=36.43约第92名。榜单证明存在高价值策略差异，但不能识别官方空GT比例。现有公开人工片段18条均含至少一个2/3高光标签，不能代表官方测试。直接运行AHA、InternVideo2事件完成度和QV/MrHiSum错配监督均未突破；目标域适配尚未被验证。

**Decision**：下一路线是PHD/GIF源级监督适配AHA，并以HL-CLIP式saliency pooling作结构对照；第一阶段固定R54/R70空间输出，使用源级留出联合集合效用作为门。没有真实负视频池前不提交empty-video gate。榜单快照与检索记录见`reports/r85_leaderboard_transfer_review_20261019/`。
# R86/R87 — 2026-10-19 observation path and target transfer

## R86_T5_CROPHEAD_V3_R54TIME

Hypothesis: the source-disjoint native-crop point-to-window mapping already
validated by T5 can improve the current V9 observation path while keeping the
R54 temporal policy fixed. Change: apply frozen H1 coefficients from the four
public RetargetVid/LIVE tables to current V9 keyframe points, then use the
existing hold+EMA decoder. Result: 426/426 videos, 199037 predictions,
106781 kept-frame boxes changed, width unchanged, independent checker passed,
8.854531425B declared parameters. Package ZIP SHA is
`308275cd2a3ac3caef0aaf3ae237e3d2552bc048500cb23f12b79c3cdd8f714a`.
Official score is pending. This is the first new upload candidate after R84;
it changes the spatial observation mapping rather than a budget or threshold.

## R87_AHA_PHD_PILOT

Hypothesis: frozen AHA scores can be calibrated with source-disjoint PHD2 GIF
labels. A 96-fragment train / 48-fragment validation pilot loaded the local
base and adapter after repairing the AHA environment. Consecutive-frame causal
outputs became NaN for many fragments under both static-cache and SinkCache
variants, before any adaptation head was fit. The result is blocked by a
runtime/cache contract and has no valid metric; it is not evidence against
target-domain transfer. No official test data or labels were used.

## R89/R90（2026-10-19）

- **R89_LOWLEVEL_COMPOSITION_R54TIME_B3**：固定 R54 时间集合、预算、TimeLens 排序和 B3 合同；使用 RetargetVid/LIVE 原生 crop GT 训练参数为零的梯度/局部对比/相邻帧差分/清晰度/构图位置 ridge。开发源宏 +0.002497，Confirm 272 源宏 +0.004104；去掉差分 Confirm +0.004108，运动差分未贡献。官方包已生成、未上传，426/426、199037 帧、8.854531425B，ZIP SHA=`e2830a5fb539969e9dbde5e5133728f37d9207bb23449e746990a134b0973d2e`。报告：`reports/r89_lowlevel_composition_20261019/summary.md`。
- **R90_TEMPORAL_STRUCTURE_MOTION_RANK**：TVSum 27/16 源级留出，固定 top85 连续重要性代理。静态 ResNet 512D=0.256248；32D bank=0.246228；bank+差分/3帧均值=0.246542；静态+bank 时序=0.256212；静态+语义 novelty/速度/加速度=0.256299，配对 CI 跨零。低层 bank 及简单运动结构不足以替代轻量语义；关闭直接拼接路线。报告：`reports/r89_lowlevel_composition_20261019/summary.md`。
- **R91_LOWPARAM_RESNET18_B3_V3**：冻结 ResNet18+线性 TVSum 连续读出选时间，V9/B3 全帧空间候选复制；98,584,754 参数、199037 预测、与 R54 共同帧框0差异、替换27768帧。TVSum dev top85=0.256248，官方包未上传，SHA=`c8ba77bfbcd3ac46ea0bcb8578e107f2e1007a8ad31bfa8c23cc35a17385fe27`。报告：`reports/r91_lowparam_resnet_20261019/summary.md`。
# 2026-10-19：R92 传统处理多视图路线复核（设计）

用户提供的 `传统处理接入方案.pdf` 已阅读，文件 SHA256=`04482190c7ba9d90b527175648abfdddaccc9687b26d9309d98f45055635a7ef`。PDF 认为传统图像处理的价值在于保留低频布局、高频结构、运动补偿残差和 ROI 的空间形状，让小模型学习视图联合关系；它不应被实现成更多手工标量。

**与已有实验的边界**：R89 是参数为零的低层标量/候选空间读出；R90 是 32D bank、差分、语义 novelty/速度/加速度的标量代理；二者均未测试空间图上的多视图时序编码。R89 Confirm crop IoU `+0.004104`，R90 低层 bank 低于静态 ResNet，因此不能把 PDF 路线宣布已验证，也不能把旧负结果重复成新实验。

**R92 预注册设计**：冻结轻量 RGB 主干和 R54/R70 空间输出；PHD2/GIF 源级留出；R92-A RGB、R92-E RGB+低频+高频+全局运动补偿残差 gated fusion、R92-F 运动视图源内时间置换。真实 GIF 区间 union 与占位 `gif_hard` 分离。E 相对 A 的源级集合效用 `+0.005` 且 bootstrap 下界为正、F 的收益消失后，才做单一时间变量官方包。尚未运行，无官方分数。

**实际结果**：远端 RTX4080 上完成 5,886 片段四视图缓存（331.45s，162MB）和 RGB/多视图各1,200 steps。RGB 开发源宏固定85% F=`0.415595`，多视图=`0.415335`；448源配对差=`-0.000260`，95% CI=`[-0.003184,+0.002589]`。运动视图源内时间置换=`0.415474`，逐视图置零 `[RGB=0.414683, low=0.417219, high=0.414921, motion=0.413045]`。结论：多视图没有实用增益，运动时间对应未被识别；R92 不生成官方包。指标和逐源结果见 `reports/r92_multiview_traditional_route_20261019/remote_metrics/`。

## 2026-10-19：R93 4fps dense event edge

TimeLens2 generic prompt 从2fps提高到4fps，完整426视频推理覆盖、0错误。基于dense区间构造 `q=2*abs(normalized_phase-.5)` 端点效用，固定 eta=.20 residual，V9/B3空间框与R54逐视频预算不变。产物 `submissions/R93_DENSE_EVENT_EDGE_V9_B3/`，426/426、199037预测、参数8.854531425B、独立checker/ZIP round-trip通过。ZIP SHA=`bc05c7cbd4fa60ea2c01d9a928c55b5704bc7b83748279cf39530c33fd76a28d`，prediction SHA=`00c4e2bd875e7584b16c1f947e4bcf55f733b69aec0d8ffa9a6242f1945e4d26`。相对R60 4fps direct有337视频变化、平均10.59交换帧；这是新观测+端点决策，不是重复包。官方分数待用户上传，未自动提交。

## 2026-10-19：研究1 PDF 路线 H1/H2 判别实验

**R94 U²-Netp H1**：固定 T5/R54 时间与候选窗口，在 LIVE 原生 crop 表上比较冻结 U²-Netp 主体范围/候选关系和同容量零输入控制。U²-Netp 1,131,181 参数、官方权重 SHA `e7567cde013fb64813973ce6e1ecc25a80c05c3ca7adbc5a54f3c3d90991b854`；10,469 个 keyframe forward 无错误。LIVE dev 156 源，U²−control `−0.000463`，95% CI `[−0.003289,+0.002383]`；历史 `live_confirm3` U²−control `+0.000834`，CI `[-0.001906,+0.003808]`。同容量 control−H0 `+0.007883`，说明既有候选读出而非 U² 新信息解释了这部分收益。关闭具体 U² H1，不生成包。报告：`reports/r94_u2_h1_20261019/summary.md`。

**R95 shot structure H2**：在 R92 5,886 个 PHD2/GIF 片段上，把相邻冻结 RGB ResNet18 embedding 的片段内 95 分位变化构造成中心、距切点、shot 长度和边界四通道；控制与结构组同一 516 维头、同一参数 202,953，控制结构通道置零。448 源 dev 结构−control `−0.000364`，95% CI `[−0.002038,+0.001271]`。关闭“embedding 代理 shot 结构”这一具体配方；不能推广为完整原视频的真实 shot 边界无效。报告：`reports/r95_shot_structure_20261019/summary.md`。


## 2026-10-19：R96 事件上下文可观测性

在 R42 V1.3 540 条公开人工诊断（12源）上做源级留一，比较当前帧 InternVideo2 embedding 与前后上下文 logistic probe。static 源宏 AP=.6435，context=.6736，表面 +.0301；11 个有效源的 paired bootstrap 95% CI `[-.0141,+.0723]`，AUC CI 也跨零。差分-only AP=.6460，不能把简单运动差分当上下文。结论是方向性支持完整上下文/事件状态，未形成确认收益或官方包。R42 同账号重复、自由诊断标签和12源规模限制均保留。报告：`reports/r96_event_context_observability_20261019/summary.md`。


## 2026-10-19：R97 PHD/GIF 区间状态辅助监督

从公开 train.json 选择区间构造区间前/中/后状态，空区间保持未标注；在R92冻结RGB TCN上比较状态损失权重0与0.3。448源固定85%集合F：控制=.416244，状态辅助=.415464，配对差=-.000780，95% CI [-.003117,+.001542]。关闭该具体状态辅助配方，不出包；完整上下文/联合效用仍未排除。报告：`reports/r97_event_state_tcn_20261019/summary.md`。
