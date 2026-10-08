# 实验历史

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
