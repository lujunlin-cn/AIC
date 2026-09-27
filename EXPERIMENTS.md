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
