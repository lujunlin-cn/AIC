# 2026-09-26 官方分数候选差异核对草稿

本文件记录 `SUB_A/B/C` 冻结发布物的机械核对结果，供
`reports/20260926_official_score_diagnosis.md` 汇总。官方分数由赛事平台反馈，
不是本地 evaluator 计算：`SUB_A=1.01`、`SUB_B=6.08`、`SUB_C=6.64`。

## 冻结变量

| candidate | visual representation | temporal input/head | train videos | threshold | sampling | spatial | loaded bytes |
|---|---|---|---:|---:|---:|---|---:|
| SUB_A | frozen torchvision ResNet18 ImageNet-1K, 512D | Temporal U-Net, BCE, 20 epochs, seed 20260925 | TVSum 27 | 0.40 | 2 FPS | center | 25,685,169 |
| SUB_B | frozen timm DeiT-S/16 ImageNet-1K, 384D | same Temporal U-Net/BCE/20 epochs/seed | TVSum 27 | 0.35 | 2 FPS | center | 46,618,447 |
| SUB_C | same frozen DeiT-S temporal bundle as B | identical to B | TVSum 27 | 0.35 | 2 FPS | YuNet `true_face_smooth` | 46,851,036 |

Both models use the same 27/16/7 TVSum source-group assignment, the same
2-FPS frame indices/timestamps and the same summary-importance labels. The
backbone is frozen in both routes; only the temporal head is trained. The
ResNet18 temporal input adapter is 512→128 and the DeiT adapter is 384→128;
the remaining Temporal U-Net topology is shared. A0 and DeiT are therefore a
near-controlled frozen-representation comparison, but the official A/B score gap
also includes their distinct historical DEV thresholds (`.40` vs `.35`) and
backbone preprocessing/representation, so it cannot be attributed to parameter
count alone.

The two probes also came from different training entry points: A0 used
`aic/train.py` with V100 AMP/GradScaler enabled, while the DeiT probe used
`scripts/train_temporal_probe.py` in ordinary FP32. Both selected checkpoints
by validation video-macro F1 at fixed `.5`, so this is a numerical/trainer
implementation confound rather than a selection-metric difference.

## B→C isolation

The submitted B and C JSONL files have identical selected frame lists for all
174 videos and all 8,133 selected frames. Their bboxes differ on 8,130 of the
8,133 paired predictions. Thus the temporal prediction is held fixed and the
official raw increase `6.08 → 6.64` (`+0.56`, `+9.21%`) is an isolated spatial
differential for YuNet `true_face_smooth`. YuNet contributes 232,589 bytes;
there are no learned temporal differences. The observer remains face-focused
and is not evidence for generic person/object crop quality.

The complete frozen runs also show the selection/calibration shift visible
without inspecting test content: A selected 1,809 frames and had 150 empty
videos; B/C each selected 8,133 frames and had 97 empty videos. A/B runtimes
were 742.213/742.764 seconds, while C took 1,959.029 seconds because dense
YuNet observes every original frame. These counts are engineering diagnostics,
not a reason to change a test threshold after the fact.

## Size economics

All three bundles are below 100 MB, so `k_size=1.00` and weighted scores equal
the reported raw scores. Relative to C=6.64, an M-tier candidate must exceed
`6.64/0.95 = 6.98947` raw (at least +5.263%), and an L-tier candidate must
exceed `6.64/0.90 = 7.37778` raw (at least +11.111%).

## Evidence boundary

The 130-GB official training corpus was not used by these three frozen
releases. Their feature manifests contain only TVSum (27 train, 16 validation,
7 comparison holdout), with `summary_importance_2s` labels. The scores establish
strong official evidence that the frozen DeiT representation plus the current
temporal policy is preferable to the A0 release in this test, and that the
YuNet spatial substitution adds value; they do not separate all possible
effects of backbone, threshold, label semantics, or data scale.
