# 实验历史

本项目所有TVSum数值均按binary proxy、ranking或author-style summary单独命名，不是AIC官方F_video。当前虽已取得RetargetVid crop GT，仍没有AIC联合GT；`official_f_video`和`competition_score`保持null。早期章节保留为历史，最新证据见末节。

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
