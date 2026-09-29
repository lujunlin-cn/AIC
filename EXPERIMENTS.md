# 实验历史

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
