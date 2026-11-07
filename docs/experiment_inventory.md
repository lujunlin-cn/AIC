# 实验事件台账与证据文件清单

生成日期：2026-11-07。总入口：[index.md](index.md)。由 `scripts/build_research_history_inventory.py` 从工作区原件生成；重建命令：`python scripts/build_research_history_inventory.py`。

## 统计口径

- 两份registry共 **371条非空JSON事件**，来自experiments/registry.jsonl与根registry.jsonl；空行忽略，非空行全部解析成功。
- 一条事件不等于一个独立实验：包含同一run的启动/结束/回分、重复登记、CV折和种子；不去重、不累计成所谓独立试验次数。
- 补充扫描 **253份reports Markdown**，连同根文档、old_docs、ext_data等共 **308份研究/状态Markdown**；另列 **141份JSON配置**。
- 报告数量只统计本节所述目录和扩展名，未声称读取所有1434项reports产物或逐个重算全部checkpoint。JSON/日志关键结果由13份路线回顾与下列摘录补充。
- 表中的“原事件状态”是当时原文，不是今天的最终状态。最终P2=42.74、rank-1=42.46、R91=33.87等以后续回分为准。
- `当前Git已跟踪`表示本次生成时路径在git索引中；`工作区原件`表示未跟踪的新文件/迁移副本。后者路径是定位信息，未伪造GitHub可点证据链接；路线正文已保留讨论所需结果。
- 分组只有一个主归属，跨路线依赖请读正文。整个台账为现存可追溯记录盘点，不保证恢复未落盘聊天、丢失远程实验或不存在的checkpoint。

## 路线覆盖与registry事件数

部分路线没有登记到这两份registry，但有实际报告；0条事件不表示没有做实验。

| 主归属 | 原事件条数 | reports Markdown数 |
|---|---:|---:|
| [01 轻量基线与传统特征](01_classic_visual_baselines.md) | 131 | 8 |
| [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 26 | 26 |
| [03 32B教师](03_qwen32b_teacher.md) | 37 | 8 |
| [04 LFM空间学生](04_lfm_spatial_students.md) | 4 | 10 |
| [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 106 | 64 |
| [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 0 | 22 |
| [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 17 | 22 |
| [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 1 | 20 |
| [09 新表征与迁移](09_new_representations_and_transfer.md) | 5 | 18 |
| [10 Video-ORA冻结系统](10_videoora_frozen_pipeline.md) | 0 | 5 |
| [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 2 | 11 |
| [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | 12 | 5 |
| [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 30 | 34 |

## 逐条registry事件

| 来源行 | 实验／事件ID | 主归属 | 原事件状态或结果 |
|---|---|---|---|
| `experiments/registry.jsonl:1` | `A0_001` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | accepted_temporal_proxy |
| `experiments/registry.jsonl:2` | `A1_001` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | promising_single_seed_proxy |
| `experiments/registry.jsonl:3` | `A2_001` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | rejected_proxy_regression |
| `experiments/registry.jsonl:4` | `A0_002` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | seed_control |
| `experiments/registry.jsonl:5` | `A0_003` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | seed_control |
| `experiments/registry.jsonl:6` | `A1_002` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | seed_control |
| `experiments/registry.jsonl:7` | `A1_003` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | seed_control |
| `experiments/registry.jsonl:8` | `A0_004` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_repaired_protocol |
| `experiments/registry.jsonl:9` | `A0_005` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_repaired_protocol |
| `experiments/registry.jsonl:10` | `A0_006` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_repaired_protocol |
| `experiments/registry.jsonl:11` | `A1_004` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_repaired_protocol |
| `experiments/registry.jsonl:12` | `A1_005` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_repaired_protocol |
| `experiments/registry.jsonl:13` | `A2_002` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_repaired_protocol |
| `experiments/registry.jsonl:14` | `A2_003` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_repaired_protocol |
| `experiments/registry.jsonl:15` | `A2_004` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_repaired_protocol |
| `experiments/registry.jsonl:16` | `A2_005` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_repaired_protocol |
| `experiments/registry.jsonl:17` | `A0_007` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | split_stability_fixed_threshold |
| `experiments/registry.jsonl:18` | `A0_008` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | split_stability_fixed_threshold |
| `experiments/registry.jsonl:19` | `A0_009` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | split_stability_fixed_threshold |
| `experiments/registry.jsonl:20` | `A0_010` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | split_stability_fixed_threshold |
| `experiments/registry.jsonl:21` | `A0_011` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | split_stability_fixed_threshold |
| `experiments/registry.jsonl:22` | `A1_006` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | split_stability_fixed_threshold |
| `experiments/registry.jsonl:23` | `A1_007` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | split_stability_fixed_threshold |
| `experiments/registry.jsonl:24` | `A1_008` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | split_stability_fixed_threshold |
| `experiments/registry.jsonl:25` | `A1_009` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | split_stability_fixed_threshold |
| `experiments/registry.jsonl:26` | `A1_010` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | split_stability_fixed_threshold |
| `experiments/registry.jsonl:27` | `PROXY_BASELINES_001` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_reference_baselines |
| `experiments/registry.jsonl:28` | `B0_probe_vit_b16` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_probe_raw_jsonl_valid |
| `experiments/registry.jsonl:29` | `A0_012` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_loss_ablation |
| `experiments/registry.jsonl:30` | `LOCKBOX_A0_006_001` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | lockbox_frozen_comparison |
| `experiments/registry.jsonl:31` | `LOCKBOX_A0_006_fold_threshold_001` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | calibration_sensitivity_audit |
| `experiments/registry.jsonl:32` | `LOCKBOX_A1_005_001` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | lockbox_frozen_comparison |
| `experiments/registry.jsonl:33` | `LOCKBOX_B0_vit_b16_001` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | lockbox_frozen_comparison |
| `experiments/registry.jsonl:34` | `Bs0_deit_s_probe_001` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | S_tier_challenger |
| `experiments/registry.jsonl:35` | `TSM_internal_full_001` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | deprioritized_no_stable_gain |
| `experiments/registry.jsonl:36` | `OOD_SUMME_LFS_BLOCKER_001` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | blocked_raw_lfs_no_progress |
| `experiments/registry.jsonl:37` | `Bs0_deit_s_seed20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | multi_seed_confirmation |
| `experiments/registry.jsonl:38` | `Bs0_deit_s_seed20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | multi_seed_confirmation |
| `experiments/registry.jsonl:39` | `RG_NCV_A0_r0_f0_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:40` | `RG_NCV_A0_r0_f0_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:41` | `RG_NCV_A0_r0_f0_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:42` | `RG_NCV_A0_r0_f1_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:43` | `RG_NCV_A0_r0_f1_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:44` | `RG_NCV_A0_r0_f1_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:45` | `RG_NCV_A0_r0_f2_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:46` | `RG_NCV_A0_r0_f2_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:47` | `RG_NCV_A0_r0_f2_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:48` | `RG_NCV_A0_r0_f3_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:49` | `RG_NCV_A0_r0_f3_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:50` | `RG_NCV_A0_r0_f3_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:51` | `RG_NCV_A0_r0_f4_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:52` | `RG_NCV_A0_r0_f4_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:53` | `RG_NCV_A0_r0_f4_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:54` | `RG_NCV_A0_r1_f0_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:55` | `RG_NCV_A0_r1_f0_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:56` | `RG_NCV_A0_r1_f0_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:57` | `RG_NCV_A0_r1_f1_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:58` | `RG_NCV_A0_r1_f1_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:59` | `RG_NCV_A0_r1_f1_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:60` | `RG_NCV_A0_r1_f2_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:61` | `RG_NCV_A0_r1_f2_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:62` | `RG_NCV_A0_r1_f2_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:63` | `RG_NCV_A0_r1_f3_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:64` | `RG_NCV_A0_r1_f3_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:65` | `RG_NCV_A0_r1_f3_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:66` | `RG_NCV_A0_r1_f4_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:67` | `RG_NCV_A0_r1_f4_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:68` | `RG_NCV_A0_r1_f4_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:69` | `RG_NCV_DeiT_S_r0_f0_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:70` | `RG_NCV_DeiT_S_r0_f0_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:71` | `RG_NCV_DeiT_S_r0_f0_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:72` | `RG_NCV_DeiT_S_r0_f1_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:73` | `RG_NCV_DeiT_S_r0_f1_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:74` | `RG_NCV_DeiT_S_r0_f1_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:75` | `RG_NCV_DeiT_S_r0_f2_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:76` | `RG_NCV_DeiT_S_r0_f2_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:77` | `RG_NCV_DeiT_S_r0_f2_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:78` | `RG_NCV_DeiT_S_r0_f3_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:79` | `RG_NCV_DeiT_S_r0_f3_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:80` | `RG_NCV_DeiT_S_r0_f3_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:81` | `RG_NCV_DeiT_S_r0_f4_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:82` | `RG_NCV_DeiT_S_r0_f4_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:83` | `RG_NCV_DeiT_S_r0_f4_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:84` | `RG_NCV_DeiT_S_r1_f0_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:85` | `RG_NCV_DeiT_S_r1_f0_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:86` | `RG_NCV_DeiT_S_r1_f0_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:87` | `RG_NCV_DeiT_S_r1_f1_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:88` | `RG_NCV_DeiT_S_r1_f1_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:89` | `RG_NCV_DeiT_S_r1_f1_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:90` | `RG_NCV_DeiT_S_r1_f2_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:91` | `RG_NCV_DeiT_S_r1_f2_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:92` | `RG_NCV_DeiT_S_r1_f2_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:93` | `RG_NCV_DeiT_S_r1_f3_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:94` | `RG_NCV_DeiT_S_r1_f3_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:95` | `RG_NCV_DeiT_S_r1_f3_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:96` | `RG_NCV_DeiT_S_r1_f4_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:97` | `RG_NCV_DeiT_S_r1_f4_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:98` | `RG_NCV_DeiT_S_r1_f4_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:99` | `RG_NCV_ViT_B_r0_f0_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:100` | `RG_NCV_ViT_B_r0_f0_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:101` | `RG_NCV_ViT_B_r0_f0_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:102` | `RG_NCV_ViT_B_r0_f1_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:103` | `RG_NCV_ViT_B_r0_f1_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:104` | `RG_NCV_ViT_B_r0_f1_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:105` | `RG_NCV_ViT_B_r0_f2_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:106` | `RG_NCV_ViT_B_r0_f2_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:107` | `RG_NCV_ViT_B_r0_f2_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:108` | `RG_NCV_ViT_B_r0_f3_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:109` | `RG_NCV_ViT_B_r0_f3_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:110` | `RG_NCV_ViT_B_r0_f3_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:111` | `RG_NCV_ViT_B_r0_f4_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:112` | `RG_NCV_ViT_B_r0_f4_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:113` | `RG_NCV_ViT_B_r0_f4_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:114` | `RG_NCV_ViT_B_r1_f0_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:115` | `RG_NCV_ViT_B_r1_f0_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:116` | `RG_NCV_ViT_B_r1_f0_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:117` | `RG_NCV_ViT_B_r1_f1_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:118` | `RG_NCV_ViT_B_r1_f1_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:119` | `RG_NCV_ViT_B_r1_f1_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:120` | `RG_NCV_ViT_B_r1_f2_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:121` | `RG_NCV_ViT_B_r1_f2_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:122` | `RG_NCV_ViT_B_r1_f2_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:123` | `RG_NCV_ViT_B_r1_f3_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:124` | `RG_NCV_ViT_B_r1_f3_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:125` | `RG_NCV_ViT_B_r1_f3_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:126` | `RG_NCV_ViT_B_r1_f4_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:127` | `RG_NCV_ViT_B_r1_f4_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:128` | `RG_NCV_ViT_B_r1_f4_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:129` | `SUMME_OOD_002_A0` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_zero_shot_ood |
| `experiments/registry.jsonl:130` | `SUMME_OOD_002_DeiT_S` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_zero_shot_ood |
| `experiments/registry.jsonl:131` | `SUMME_OOD_002_ViT_B` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_zero_shot_ood |
| `experiments/registry.jsonl:132` | `SUMME_CV_OOD_001_A0` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_zero_shot_ood |
| `experiments/registry.jsonl:133` | `SUMME_CV_OOD_001_DeiT_S` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_zero_shot_ood |
| `experiments/registry.jsonl:134` | `SUMME_CV_OOD_001_ViT_B` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_zero_shot_ood |
| `experiments/registry.jsonl:135` | `SPATIAL_GT_001` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | completed_dense_crop_benchmark |
| `experiments/registry.jsonl:136` | `SPATIAL_GT_NATIVE_002` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | completed_dense_crop_benchmark |
| `experiments/registry.jsonl:137` | `DENSE_E2E_001` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_raw_jsonl_spatial_equivalence |
| `experiments/registry.jsonl:138` | `SUMME_OOD_001` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | failed_source_nonmonotonic_pts |
| `experiments/registry.jsonl:139` | `SUMME_OOD_003_A0` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_zero_shot_ood |
| `experiments/registry.jsonl:140` | `SUMME_OOD_003_DeiT_S` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_zero_shot_ood |
| `experiments/registry.jsonl:141` | `SUMME_OOD_003_ViT_B` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_zero_shot_ood |
| `experiments/registry.jsonl:142` | `SUMME_CV_OOD_002_A0` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_zero_shot_ood |
| `experiments/registry.jsonl:143` | `SUMME_CV_OOD_002_DeiT_S` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_zero_shot_ood |
| `experiments/registry.jsonl:144` | `SUMME_CV_OOD_002_ViT_B` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_zero_shot_ood |
| `experiments/registry.jsonl:145` | `ENGINEERING_RELEASE_002_A0_center` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_frozen_raw_video_release_verification |
| `experiments/registry.jsonl:146` | `ENGINEERING_RELEASE_002_A0_face_ema` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_frozen_raw_video_release_verification |
| `experiments/registry.jsonl:147` | `ENGINEERING_RELEASE_002_DeiT_center` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_frozen_raw_video_release_verification |
| `experiments/registry.jsonl:148` | `ENGINEERING_RELEASE_003_A0_center` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_frozen_raw_video_release_verification |
| `experiments/registry.jsonl:149` | `ENGINEERING_RELEASE_003_A0_face_ema` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_frozen_raw_video_release_verification |
| `experiments/registry.jsonl:150` | `ENGINEERING_RELEASE_003_DeiT_center` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_frozen_raw_video_release_verification |
| `experiments/registry.jsonl:151` | `ENGINEERING_RELEASE_001` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | failed_cuda_statistics_before_device_initialization |
| `experiments/registry.jsonl:152` | `CROSS_DATASET_DUPLICATE_001` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_sparse_duplicate_screen |
| `experiments/registry.jsonl:153` | `SUMME_BASELINES_001` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_fixed_budget_nonlearning_controls |
| `experiments/registry.jsonl:154` | `RG_LINEAR_A0_r0_f0_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:155` | `RG_LINEAR_A0_r0_f0_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:156` | `RG_LINEAR_A0_r0_f0_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:157` | `RG_LINEAR_A0_r0_f1_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:158` | `RG_LINEAR_A0_r0_f1_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:159` | `RG_LINEAR_A0_r0_f1_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:160` | `RG_LINEAR_A0_r0_f2_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:161` | `RG_LINEAR_A0_r0_f2_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:162` | `RG_LINEAR_A0_r0_f2_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:163` | `RG_LINEAR_A0_r0_f3_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:164` | `RG_LINEAR_A0_r0_f3_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:165` | `RG_LINEAR_A0_r0_f3_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:166` | `RG_LINEAR_A0_r0_f4_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:167` | `RG_LINEAR_A0_r0_f4_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:168` | `RG_LINEAR_A0_r0_f4_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:169` | `RG_LINEAR_A0_r1_f0_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:170` | `RG_LINEAR_A0_r1_f0_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:171` | `RG_LINEAR_A0_r1_f0_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:172` | `RG_LINEAR_A0_r1_f1_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:173` | `RG_LINEAR_A0_r1_f1_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:174` | `RG_LINEAR_A0_r1_f1_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:175` | `RG_LINEAR_A0_r1_f2_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:176` | `RG_LINEAR_A0_r1_f2_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:177` | `RG_LINEAR_A0_r1_f2_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:178` | `RG_LINEAR_A0_r1_f3_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:179` | `RG_LINEAR_A0_r1_f3_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:180` | `RG_LINEAR_A0_r1_f3_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:181` | `RG_LINEAR_A0_r1_f4_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:182` | `RG_LINEAR_A0_r1_f4_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:183` | `RG_LINEAR_A0_r1_f4_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:184` | `RG_LINEAR_DeiT_S_r0_f0_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:185` | `RG_LINEAR_DeiT_S_r0_f0_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:186` | `RG_LINEAR_DeiT_S_r0_f0_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:187` | `RG_LINEAR_DeiT_S_r0_f1_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:188` | `RG_LINEAR_DeiT_S_r0_f1_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:189` | `RG_LINEAR_DeiT_S_r0_f1_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:190` | `RG_LINEAR_DeiT_S_r0_f2_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:191` | `RG_LINEAR_DeiT_S_r0_f2_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:192` | `RG_LINEAR_DeiT_S_r0_f2_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:193` | `RG_LINEAR_DeiT_S_r0_f3_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:194` | `RG_LINEAR_DeiT_S_r0_f3_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:195` | `RG_LINEAR_DeiT_S_r0_f3_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:196` | `RG_LINEAR_DeiT_S_r0_f4_s20260925` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:197` | `RG_LINEAR_DeiT_S_r0_f4_s20260926` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:198` | `RG_LINEAR_DeiT_S_r0_f4_s20260927` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:199` | `RG_LINEAR_DeiT_S_r1_f0_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:200` | `RG_LINEAR_DeiT_S_r1_f0_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:201` | `RG_LINEAR_DeiT_S_r1_f0_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:202` | `RG_LINEAR_DeiT_S_r1_f1_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:203` | `RG_LINEAR_DeiT_S_r1_f1_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:204` | `RG_LINEAR_DeiT_S_r1_f1_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:205` | `RG_LINEAR_DeiT_S_r1_f2_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:206` | `RG_LINEAR_DeiT_S_r1_f2_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:207` | `RG_LINEAR_DeiT_S_r1_f2_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:208` | `RG_LINEAR_DeiT_S_r1_f3_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:209` | `RG_LINEAR_DeiT_S_r1_f3_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:210` | `RG_LINEAR_DeiT_S_r1_f3_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:211` | `RG_LINEAR_DeiT_S_r1_f4_s20260925` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:212` | `RG_LINEAR_DeiT_S_r1_f4_s20260926` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:213` | `RG_LINEAR_DeiT_S_r1_f4_s20260927` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_paired_nested_cv |
| `experiments/registry.jsonl:214` | `SUMME_LINEAR_OOD_001_A0` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_frozen_head_external_comparison |
| `experiments/registry.jsonl:215` | `SUMME_LINEAR_OOD_001_DeiT_S` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_frozen_head_external_comparison |
| `experiments/registry.jsonl:216` | `RELEASE_CLEAN_001` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed |
| `experiments/registry.jsonl:217` | `PROVENANCE_AUDIT_002` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed |
| `experiments/registry.jsonl:218` | `SPATIAL_GROUP_003` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | completed_rejected_fixed_group_observation |
| `experiments/registry.jsonl:219` | `RG_RANK_001` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_exploratory_no_candidate_upgrade |
| `experiments/registry.jsonl:220` | `EVAL_INTAKE_001` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_evaluation_readiness |
| `experiments/registry.jsonl:221` | `OFFICIAL_TEST_20260926_SUB_A_001` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_frozen_official_test_inference |
| `experiments/registry.jsonl:222` | `OFFICIAL_TEST_20260926_SUB_B_001` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_frozen_official_test_inference |
| `experiments/registry.jsonl:223` | `OFFICIAL_TEST_20260926_SUB_C_001` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_frozen_official_test_inference |
| `experiments/registry.jsonl:224` | `OFFICIAL_SCORE_DIAG_20260926` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_external_official_score_record |
| `experiments/registry.jsonl:225` | `DATA_SCALE_20260926` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | completed_exploratory_data_scale |
| `experiments/registry.jsonl:226` | `QVH_DEITS_QUICK_5PCT` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | completed_weak_label_pilot |
| `experiments/registry.jsonl:227` | `QVH_FINETUNE_PROBE_20260926` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | completed_bounded_no_gain |
| `experiments/registry.jsonl:228` | `QVH_SOURCE_AUDIT_20260926` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed |
| `experiments/registry.jsonl:229` | `INTERNVIDEO2_SMOKE_8F_20260926` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | compatibility_smoke_only_not_candidate |
| `experiments/registry.jsonl:230` | `INTERNVIDEO2_SMOKE_16F_20260926` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | compatibility_smoke_only_not_candidate |
| `experiments/registry.jsonl:231` | `VIDEOMAE_QVH_PROBE_V2_20260926` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | completed_bounded_uncertain_no_promotion |
| `experiments/registry.jsonl:232` | `QVH_NATIVE_MEDIA_AUDIT_20260926` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed |
| `experiments/registry.jsonl:233` | `QVH_HUMAN_SANITY_20260926` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | completed_exploratory |
| `experiments/registry.jsonl:234` | `NATIVE_CANDIDATE_deit_V1` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | running |
| `experiments/registry.jsonl:235` | `NATIVE_CANDIDATE_videomae_V1` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | running |
| `experiments/registry.jsonl:236` | `NATIVE_CANDIDATE_internvideo_V1` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | running |
| `experiments/registry.jsonl:237` | `NATIVE_CANDIDATE_videomae_V1` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | rejected_not_ready |
| `experiments/registry.jsonl:238` | `NATIVE_CANDIDATE_internvideo_V1` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | promising_engineering_candidate_release_running |
| `experiments/registry.jsonl:239` | `SUB_E_INTERNVIDEO2_NATIVE_V1_RELEASE` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | READY_TO_UPLOAD_NOT_SUBMITTED |
| `experiments/registry.jsonl:240` | `OFFICIAL_SCORE_SUB_E_20260927` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | completed_external_official_feedback |
| `experiments/registry.jsonl:241` | `VIDEOMAE_LARGE_NATIVE_V1` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | completed_native_dev_promising |
| `experiments/registry.jsonl:242` | `INTERNVIDEO2_STAGE2_NATIVE_V1` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | completed_external_official_feedback |
| `experiments/registry.jsonl:243` | `INTERNVIDEO2_STAGE2_NATIVE_V1_RELEASE` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | READY_TO_UPLOAD_NOT_SUBMITTED |
| `experiments/registry.jsonl:244` | `INTERNVIDEO2_K710_NATIVE_V1` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | completed_dev_gate_rejected_for_release |
| `experiments/registry.jsonl:245` | `SUB_G_INTERNVIDEO2_K710_NATIVE_V1_RELEASE` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | READY_TO_UPLOAD_NOT_SUBMITTED |
| `experiments/registry.jsonl:246` | `OFFICIAL_SCORE_SUB_G_20260927` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | completed_external_feedback_requires_upload_audit |
| `experiments/registry.jsonl:247` | `OFFICIAL_SCORE_E01_20260927` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_external_feedback_requires_identity_audit |
| `experiments/registry.jsonl:248` | `OFFICIAL_SCORE_SUB_F_20260927` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | completed_external_official_feedback |
| `experiments/registry.jsonl:249` | `OFFICIAL_SCORE_SUB_G_FULL_20260927` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | completed_external_official_feedback |
| `experiments/registry.jsonl:250` | `OFFICIAL_SCORE_COMPARISON_20260927` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed_external_feedback |
| `experiments/registry.jsonl:251` | `B0_ALL_SELECT_YUNET_V1` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | completed_ready_not_uploaded |
| `experiments/registry.jsonl:252` | `ALL_SELECT_DIAG_V1` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed |
| `experiments/registry.jsonl:253` | `TIME_MAPPING_CHECK_V1` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed |
| `experiments/registry.jsonl:254` | `SPATIAL_SUBJECT_001` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | completed_negative |
| `experiments/registry.jsonl:255` | `SPATIAL_SUBJECT_002` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | completed |
| `experiments/registry.jsonl:256` | `SUBJECT_SCALE_DEVCHECK_V1` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | completed |
| `experiments/registry.jsonl:257` | `SUBJECT_SCALE_V1` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | completed |
| `experiments/registry.jsonl:258` | `S2_SUBJECT_SCALE_V1` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | ready_exploratory_not_uploaded |
| `experiments/registry.jsonl:259` | `S1_CENTER_MAX_V1` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | ready_exploratory_not_uploaded |
| `experiments/registry.jsonl:260` | `Q2_QWEN_SUBJECT_INTERNAL_PROBE` | [03 32B教师](03_qwen32b_teacher.md) | internal_teacher_probe_not_for_upload |
| `experiments/registry.jsonl:261` | `QWEN_QVH_DEV_AUDIT_V1` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | blocked |
| `experiments/registry.jsonl:262` | `OFFICIAL_SCORE_20260928_SPATIAL_BATCH` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | SCORED |
| `experiments/registry.jsonl:263` | `OBS_CACHE_RETARGET_V1` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | DEV_EVALUATED |
| `experiments/registry.jsonl:264` | `OBS_CACHE_OFFICIAL_V1` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | DEV_EVALUATED |
| `experiments/registry.jsonl:265` | `MAX_WINDOW_PATH_P1A_V1` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | REJECTED |
| `experiments/registry.jsonl:266` | `MAX_WINDOW_CAND_P1B_P2_V1` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | DEV_EVALUATED |
| `experiments/registry.jsonl:267` | `QWEN_SUBJECT_POINT_OFFICIAL_V1` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | DEV_EVALUATED |
| `experiments/registry.jsonl:268` | `MAX_WINDOW_QWEN_POINT_V1` | [03 32B教师](03_qwen32b_teacher.md) | READY_TO_UPLOAD |
| `experiments/registry.jsonl:269` | `MAX_WINDOW_QWEN_NOFACE_V1` | [03 32B教师](03_qwen32b_teacher.md) | READY_TO_UPLOAD |
| `experiments/registry.jsonl:270` | `TEMPORAL_CONTRACT_V1` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | DEV_EVALUATED |
| `experiments/registry.jsonl:271` | `OFFICIAL_SCORE_20260928_QWEN_POINT_BATCH` | [03 32B教师](03_qwen32b_teacher.md) | SCORED |
| `experiments/registry.jsonl:272` | `OFFICIAL_SCORE_20260928_TEACHER_V2_BATCH` | [03 32B教师](03_qwen32b_teacher.md) | SCORED |
| `experiments/registry.jsonl:273` | `PACKAGE_DENSE_TEMP_V3_20260928` | [03 32B教师](03_qwen32b_teacher.md) | packaged_not_uploaded |
| `experiments/registry.jsonl:274` | `T4_TEMP_BOUNDARY_V3` | [03 32B教师](03_qwen32b_teacher.md) | REJECTED_NOT_PACKAGED |
| `experiments/registry.jsonl:275` | `OFFICIAL_SCORE_20260928_DENSE_TEMP_V3` | [03 32B教师](03_qwen32b_teacher.md) | SCORED |
| `experiments/registry.jsonl:276` | `T5_CROPHEAD_V3` | [03 32B教师](03_qwen32b_teacher.md) | PROMOTED_PACKAGED_NOT_UPLOADED |
| `experiments/registry.jsonl:277` | `OFFICIAL_SCORE_20260928_TEMP_CROPHEAD_V3` | [03 32B教师](03_qwen32b_teacher.md) | SCORED |
| `experiments/registry.jsonl:278` | `V4_H1_AUDIT_20260929` | [03 32B教师](03_qwen32b_teacher.md) | DONE |
| `experiments/registry.jsonl:279` | `V4_E1_OBS025` | [03 32B教师](03_qwen32b_teacher.md) | PROMOTED |
| `experiments/registry.jsonl:280` | `V4_E2_CONTEXT_POINT` | [03 32B教师](03_qwen32b_teacher.md) | REJECTED |
| `experiments/registry.jsonl:281` | `V4_S1_KEYFRAME_INTERP` | [03 32B教师](03_qwen32b_teacher.md) | PROMOTED |
| `experiments/registry.jsonl:282` | `V4_E4_VISUAL_RERANK_DEV2` | [03 32B教师](03_qwen32b_teacher.md) | DEV2_PASSED_CONFIRM_RUNNING |
| `experiments/registry.jsonl:283` | `V4_E4_VISUAL_RERANK` | [03 32B教师](03_qwen32b_teacher.md) | REJECTED |
| `experiments/registry.jsonl:284` | `R25_AHA_BF16_K085` | [09 新表征与迁移](09_new_representations_and_transfer.md) | OFFICIAL_SCORE_REPORTED_NOT_PROMOTED |
| `experiments/registry.jsonl:285` | `R26_PUBLIC_POSITION_DIAG` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | COMPLETE_NO_POSITION_ADVANTAGE_ESTABLISHED |
| `experiments/registry.jsonl:286` | `R26_AHA_EXACT085_B3` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | OFFICIAL_SCORE_USER_REPORTED |
| `experiments/registry.jsonl:287` | `R26_POS_CENTER_EXACT085_B3` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | OFFICIAL_SCORE_USER_REPORTED |
| `experiments/registry.jsonl:288` | `R26_R25MASK_CENTER` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | OFFICIAL_SCORE_USER_REPORTED |
| `experiments/registry.jsonl:289` | `R27_CENTER075_B3` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | SCORED_NOT_PROMOTED |
| `experiments/registry.jsonl:290` | `R27_CENTER095_B3` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | SCORED_NOT_PROMOTED |
| `experiments/registry.jsonl:291` | `R27_EARLY085_B3` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | PAUSED_NOT_UPLOADED |
| `experiments/registry.jsonl:292` | `R27_LATE085_B3` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | PAUSED_NOT_UPLOADED |
| `experiments/registry.jsonl:293` | `R28_SPATIAL_AUDIT` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | COMPLETED_CACHE_REPLAY_ONLY |
| `experiments/registry.jsonl:294` | `R28_SPATIAL_PUBLIC_PILOT` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | INTERRUPTED_SSH_RESULTS_NOT_RETRIEVED |
| `experiments/registry.jsonl:295` | `R29_DECISION_THEORY` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | COMPLETED_SYNTHETIC_ENUMERATION_PASS |
| `experiments/registry.jsonl:296` | `R29_PUBLIC_DECISION_PROBE_20261016` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | COMPLETED_PRIMARY_CONTENT_INCREMENT_GATE_FAILED |
| `experiments/registry.jsonl:297` | `R29_R28_SPATIAL_CONTINUATION` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | COMPLETED_EXPOSED_SOURCE_ENGINEERING_PILOT |
| `experiments/registry.jsonl:298` | `R29_SPATIAL_GT_ASSISTED_DIAGNOSTIC` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | COMPLETED_GT_ASSISTED_CANDIDATE_ADEQUACY_ONLY |
| `experiments/registry.jsonl:299` | `R29_AHA_TRANSFER_PRIMARY_SOURCE_CHECK` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | COMPLETED_PRIMARY_SOURCE_MECHANISM_REVIEW |
| `experiments/registry.jsonl:300` | `R30_DYNAMIC_CONTENT_PROBE_20261016` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | COMPLETED_CONTENT_INCREMENT_GATE_FAILED |
| `experiments/registry.jsonl:301` | `R32_VT_K090_B3` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | PACKAGED_NOT_UPLOADED |
| `experiments/registry.jsonl:302` | `R33_VT_REPLAY_B3` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | PACKAGED_NOT_UPLOADED |
| `experiments/registry.jsonl:303` | `R34_VT_EXACTDP_B3` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | PACKAGED_NOT_UPLOADED |
| `experiments/registry.jsonl:304` | `R35_VT_UNION90_B3` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | PACKAGED_NOT_UPLOADED |
| `experiments/registry.jsonl:305` | `R36_VT_CONSENSUS85_B3` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | PACKAGED_NOT_UPLOADED |
| `experiments/registry.jsonl:306` | `R41_CONTENT_GATEDFUSION85_B3` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | OFFICIAL_SCORE_USER_REPORTED |
| `experiments/registry.jsonl:307` | `R40_CONTENT_NOVELTY85_B3` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | OFFICIAL_SCORE_USER_REPORTED |
| `experiments/registry.jsonl:308` | `R39_CONTENT_REDUNDANCY85_B3` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | OFFICIAL_SCORE_USER_REPORTED |
| `experiments/registry.jsonl:309` | `R42_SMOKE_QC_V1` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | STRUCTURE_PASS_CROP_CALIBRATION_REQUIRED |
| `experiments/registry.jsonl:310` | `R42_SMOKE_REVISED_V12` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | STRUCTURE_PASS_SIX_CROP_REPAIRS |
| `experiments/registry.jsonl:311` | `R42_ANNOTATION_V13_EXPANSION` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | DEPLOYED_BROWSER_PASS_AWAITING_HUMAN_LABELS |
| `experiments/registry.jsonl:312` | `R44_HUMAN_CALIBRATED_RESIDUAL_ETA020_B3` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | OFFICIAL_SCORE_USER_REPORTED |
| `experiments/registry.jsonl:313` | `R45_BOUNDARY_GUARDED_RESIDUAL_B3` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | PACKAGED_NOT_UPLOADED |
| `experiments/registry.jsonl:314` | `R67_JOINT_V9_DISAGREEMENT_TIEAVG_G1` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | OFFICIAL_SCORE_USER_REPORTED |
| `experiments/registry.jsonl:316` | `R69_TIMELENS_DIRECT_CENTER_MAXWIN_R54` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | PACKAGED_NOT_UPLOADED |
| `experiments/registry.jsonl:317` | `DINOV3_S_MATCHED_V2` | [09 新表征与迁移](09_new_representations_and_transfer.md) | completed_not_promoted |
| `experiments/registry.jsonl:318` | `DINOV3_PHD2_LABEL_REPAIR_V2` | [09 新表征与迁移](09_new_representations_and_transfer.md) | completed |
| `experiments/registry.jsonl:319` | `NATIVE_RANK1_TEMPORAL_V1_426_SUBMISSION` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | completed |
| `experiments/registry.jsonl:320` | `TP1_SHRINK10_PROBE` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | completed |
| `experiments/registry.jsonl:321` | `SP3E_EXPAND5_PROBE` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | completed |
| `experiments/registry.jsonl:322` | `PUSH_B_DENSE_ANCHOR` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | 原记录未设状态字段 |
| `experiments/registry.jsonl:323` | `LIVEYTVC_FULL_RETRAIN_ARM_A` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | 原记录未设状态字段 |
| `experiments/registry.jsonl:324` | `PUSH_RETRAIN_A` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | packaged_pending_submission |
| `experiments/registry.jsonl:325` | `PUSH_RETRAIN_A_RESULT` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | current_best |
| `experiments/registry.jsonl:326` | `P1_NEWA-NEWB_RESULT` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | completed_no_gain |
| `experiments/registry.jsonl:327` | `P2_NEWA-NEWB_QP06_RESULT` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | completed_tied_best |
| `experiments/registry.jsonl:328` | `P3_RAFT_CANCELLATION` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | cancelled_before_official_candidate |
| `experiments/registry.jsonl:329` | `SP3E_EXPAND5_IDENTITY_RECHECK` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | invalid_identity_probe |
| `experiments/registry.jsonl:330` | `VIDEOORA_3POINT_RESEARCH_20261024` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | plan_ready_pending_GNMC_approval |
| `experiments/registry.jsonl:331` | `VIDEOORA_FULLCROP_A_V1` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | rejected_public_raw_gate |
| `experiments/registry.jsonl:332` | `VIDEOORA_FULLCROP_A_IOU1` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | rejected_learning_gate |
| `experiments/registry.jsonl:333` | `FINAL_P2_OFFICIAL_RESULT_20261024` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | final_score_user_confirmed_pending_material_review |
| `registry.jsonl:1` | `QWEN32B_OFFICIAL_V2_POINT_D2_20260928` | [03 32B教师](03_qwen32b_teacher.md) | completed |
| `registry.jsonl:2` | `QWEN32B_OFFICIAL_V2_REGION_D1_20260928` | [03 32B教师](03_qwen32b_teacher.md) | aborted_falsified |
| `registry.jsonl:3` | `T3_YTH_VAL_PROXY_V1_20260928` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | completed |
| `registry.jsonl:4` | `QWEN32B_OFFICIAL_V2_TEMPORAL_20260928` | [03 32B教师](03_qwen32b_teacher.md) | completed |
| `registry.jsonl:5` | `PACKAGE_DENSE_V2_20260928` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | packaged_not_uploaded |
| `registry.jsonl:6` | `PACKAGE_TEMPORAL_V2_20260928` | [03 32B教师](03_qwen32b_teacher.md) | packaged_hypothesis_not_uploaded |
| `registry.jsonl:7` | `OFFICIAL_SCORE_20260928_TEACHER_V2_BATCH` | [03 32B教师](03_qwen32b_teacher.md) | SCORED |
| `registry.jsonl:8` | `PACKAGE_DENSE_TEMP_V3_20260928` | [03 32B教师](03_qwen32b_teacher.md) | packaged_not_uploaded |
| `registry.jsonl:9` | `T4_TEMP_BOUNDARY_V3` | [03 32B教师](03_qwen32b_teacher.md) | REJECTED_NOT_PACKAGED |
| `registry.jsonl:10` | `OFFICIAL_SCORE_20260928_DENSE_TEMP_V3` | [03 32B教师](03_qwen32b_teacher.md) | SCORED |
| `registry.jsonl:11` | `T5_CROPHEAD_V3` | [03 32B教师](03_qwen32b_teacher.md) | PROMOTED_PACKAGED_NOT_UPLOADED |
| `registry.jsonl:12` | `OFFICIAL_SCORE_20260928_TEMP_CROPHEAD_V3` | [03 32B教师](03_qwen32b_teacher.md) | SCORED |
| `registry.jsonl:13` | `V4_H1_AUDIT_20260929` | [03 32B教师](03_qwen32b_teacher.md) | DONE |
| `registry.jsonl:14` | `V4_E1_OBS025` | [03 32B教师](03_qwen32b_teacher.md) | PROMOTED |
| `registry.jsonl:15` | `V4_E2_CONTEXT_POINT` | [03 32B教师](03_qwen32b_teacher.md) | REJECTED |
| `registry.jsonl:16` | `V4_S1_KEYFRAME_INTERP` | [03 32B教师](03_qwen32b_teacher.md) | PROMOTED |
| `registry.jsonl:17` | `V4_E4_VISUAL_RERANK_DEV2` | [03 32B教师](03_qwen32b_teacher.md) | DEV2_PASSED_CONFIRM_RUNNING |
| `registry.jsonl:18` | `V4_E4_VISUAL_RERANK` | [03 32B教师](03_qwen32b_teacher.md) | REJECTED |
| `registry.jsonl:19` | `OFFICIAL_SCORE_20260929_V4_BATCH` | [03 32B教师](03_qwen32b_teacher.md) | SCORED |
| `registry.jsonl:20` | `V5_P0_XRERANK_OFFICIAL` | [03 32B教师](03_qwen32b_teacher.md) | PACKAGED_PENDING_SCORE |
| `registry.jsonl:21` | `V5_P2_GROUND_PILOT` | [03 32B教师](03_qwen32b_teacher.md) | REJECTED_PILOT_NEGATIVE |
| `registry.jsonl:22` | `V5_H3_VISUAL` | [03 32B教师](03_qwen32b_teacher.md) | NOT_PROMOTED_GATE2_CI |
| `registry.jsonl:23` | `V5_P0_XRERANK_OFFICIAL_SCORE` | [03 32B教师](03_qwen32b_teacher.md) | SCORED |
| `registry.jsonl:24` | `LFM450_RAW_EVAL_V1` | [04 LFM空间学生](04_lfm_spatial_students.md) | PACKAGED_NOT_UPLOADED_STUDENT_ZERO_SHOT_BASELINE |
| `registry.jsonl:25` | `V7_LFM_S_FINAL` | [04 LFM空间学生](04_lfm_spatial_students.md) | PACKAGED_NOT_UPLOADED_TRAINED_STUDENT_SPATIAL_HEAD |
| `registry.jsonl:26` | `V7_LFM_T_DIAG` | [04 LFM空间学生](04_lfm_spatial_students.md) | DIAGNOSTIC_C3_PASS_NO_PACKAGE_CHANNEL |
| `registry.jsonl:27` | `V7_LFM_BLEND_DIAG` | [04 LFM空间学生](04_lfm_spatial_students.md) | DIAGNOSTIC_REPORT_ONLY |
| `registry.jsonl:28` | `R23_RANDPROBE_P250` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | official_scored |
| `registry.jsonl:29` | `R23_RANDPROBE_P500` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | official_scored |
| `registry.jsonl:30` | `R24_vpb_point_s13` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_no_promotion |
| `registry.jsonl:31` | `R24_vpb_point_s17` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_no_promotion |
| `registry.jsonl:32` | `R24_vpb_block_s13` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_no_promotion |
| `registry.jsonl:33` | `R24_vpb_block_s17` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_no_promotion |
| `registry.jsonl:34` | `R24_clip_point_s13` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_no_promotion |
| `registry.jsonl:35` | `R24_clip_point_s17` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_no_promotion |
| `registry.jsonl:36` | `R24_clip_block_s13` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_no_promotion |
| `registry.jsonl:37` | `R24_clip_block_s17` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | completed_no_promotion |
| `registry.jsonl:38` | `R24_AHA_VISUAL_SMOKE` | [09 新表征与迁移](09_new_representations_and_transfer.md) | failed_before_model_load |
| `registry.jsonl:39` | `R24_AHA_REPAIR_20261015` | [09 新表征与迁移](09_new_representations_and_transfer.md) | visual_smoke_pass_quality_unmeasured |

## 两份registry不足以覆盖的末期结果

这些项目由报告/交付JSON补足；不伪造它们在旧registry中的记录。

| 实验／产物 | 最终可确认状态 | 主归属 |
|---|---|---|
| T1/T2 candidate rerank | 实际执行，未形成留出增益，2026-10-22关闭；无新官方提升 | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) |
| native manual/mixed/phase SFT | 训练与fresh reload已做；小源留出不稳，详见路线11 | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) |
| merged temporal T1-base | 官方39.67，结构污染；未当纯时间消融 | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) |
| rank-1 temporal step50 | 已提交，官方42.46，覆盖早期待回分 | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) |
| P3与RAFT-A | P3仅171视频临时输出；RAFT两次启动失败无新权重；均撤销 | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) |
| 连续geometry head/联合LoRA | raw微正，真实40视频轨迹−.000620，未进入426 | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) |
| FULLCROP_B_V2 | raw .872317，但真实路径低于新B，未进入426 | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) |
| FULLCROP_B_MIX_V1 | 240步，3840消费/3456唯一样本；完整426包已交付；无公共质量分、无官方分 | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) |
| P2最终结果 | 42.74，426视频/133921帧/2空；旧42.73事件保留为过程 | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) |

## 关键原件hash与机器结果摘录

下列字段直接从当前工作区JSON取值；是保存结果的摘录，不是本次重跑模型。

### `reports/geometry_head_20261024/gate_a.json`

SHA256：`b0071d04fe82f96d990e61eb0903700942d7674a70f0dca642df169d9f28e3e6`

```json
{
  "raw.full_minus_current.mean_delta": 0.0061313645111129075,
  "final.full_minus_current.mean_delta": -0.0006199635856500867,
  "eligible_for_426": false,
  "official_score": null
}
```

### `reports/geometry_head_20261024/chain_state.json`

SHA256：`671c6569685fe2638b3d4a71f356df9ec21d199150eb795a0a55bfc51a3dc52e`

```json
{
  "event": "finished",
  "official_candidate_ready": false,
  "reason": "real trajectory did not improve current A"
}
```

### `reports/fullcrop_b_v2/deploy_gate.json`

SHA256：`54a6d37d604ebcfb52ed8cee598d66ba6febec3c6939acc3f56c053851c2088b`

```json
{
  "raw_selection.macro.full300": 0.8723171347032747,
  "final.source_macro_iou.new_b": 0.5122160570331376,
  "final.source_macro_iou.full300": 0.47815767840413415,
  "eligible_for_426": false,
  "official_score": null
}
```

### `reports/fullcrop_b_mix_v1/training_config.json`

SHA256：`10da851b136c5d55f1cb9e182e8a1fea887c2a7a462d091a6a1d734a74f47c53`

```json
{
  "steps": 240,
  "batch": 16,
  "consumed_samples": 3840,
  "unique_samples": 3456,
  "official_score": null
}
```

### `reports/fullcrop_b_mix_v1/selection.json`

SHA256：`b38de269080c75e42eea0e5966aefa5000f625548dc7c119af80f0604e8224d5`

```json
{
  "best": "mix240",
  "selection_rule": "final checkpoint; direct 426 inference without metric evaluation per user request",
  "public_metric_evaluation_performed": false
}
```

### `reports/fullcrop_b_mix_v1/delivery_finish_state.json`

SHA256：`b4b7c64d0b847fa7bb02c5640469c168d95a43521d3e2b1dd211487f110b4e7c`

```json
{
  "time": "2026-10-24T19:02:26.683815+08:00",
  "official_candidate_ready": true,
  "public_metric_evaluation": false,
  "predictions_sha256": "a81190df26940c62d127b55e1a80932911623eb2484eab432a1a169f20e2420f",
  "zip_sha256": "b1f3474bc8b95419acd7073d8e774d8236b8eba41b99b5048dcde3a88a5c24b7",
  "parameter_elements": 8999771458
}
```

## 研究与报告文件逐项导航

计划、调查、失败报告与最终报告全部保留标题；列入本表不表示计划已执行。

| 文件 | 主归属 | 原件状态 | 首个标题 | SHA256前16位 |
|---|---|---|---|---|
| `01_赛事规则与评分标准.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 2026 AIC「基于视频大模型的通用视频高光剪辑」赛事规则与评分标准 | `e2e3b6bd7eea0cc5` |
| `02_数据集_模型路线_特征工程研究.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 2026 AIC 视频高光剪辑：数据集、A/B/C模型路线与特征工程研究 | `0311a097f8304475` |
| `03_当前进展审计与继续探索指令.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 2026 AIC：当前进展审计与继续探索指令 | `58a73717e06e626e` |
| `AGENTS.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | AGENTS.md — 2026 AIC 研究与提交工作空间 | `eb16bffdf538213f` |
| `AIC_42.83分之后_空间蒸馏与训练执行指南_20260928.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | AIC：42.83 分之后，先验证 32B 教师能否冲击 60 分（修订版 v2） | `de03a0f1aa9e6268` |
| `AIC_46.84分复盘与32B教师下一轮执行指南_20260928.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | AIC：46.84 分复盘与 32B 教师下一轮执行指南 | `30ecdb87a0809867` |
| `AIC_48.67分复盘与突破60分研究指南_V5_20260929.md` | [03 32B教师](03_qwen32b_teacher.md) | 当前Git已跟踪 | AIC：48.67 分复盘与突破 60 分的 V5 研究指南 | `af79e3ddc2216576` |
| `AIC_下一轮实验执行指南_20260928.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | AIC 下一轮实验执行指南：最大窗口、主体轨迹与可靠时间选择 | `107b3ad83d304f0c` |
| `DECISIONS.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 技术决定 | `65509ccbb09ba46a` |
| `EVAL_HANDOFF.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 评测集接入手册 | `eb2b2fa092039ca5` |
| `EXPERIMENTS.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 实验历史 | `df3a0f5a32bf402a` |
| `Qwen3-VL-32B_教师评测与提交路线.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | Qwen3-VL-32B 教师评测与提交路线 | `882fb04b4b203249` |
| `STATUS.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 2026-10-24：最终 P2=42.74，复赛材料待用户审核 | `8beaf57b0b32ef30` |
| `TODO.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | P0 — 复赛最终材料（当前） | `c8e3f86946358f9a` |
| `aic_valen_highlight_stack_guide_20261020.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | AIC 视频高光：Valen 多模态决策路线技术交接与执行指南 | `4e679aee612fb3de` |
| `dinov3_video_highlight_training_plan.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | DINOv3 ViT-B/16 视频高光检测：基于 AIC 仓库现状的增量训练与验证方案 | `43ce7d00dec06121` |
| `ext_data/README.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | ext_data：外部公开数据采集与预处理 | `0fb443dd2ca9598e` |
| `ext_data/configs/portrait_reframe_pilot_v1/ANNOTATION_SCHEMA.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | portrait_reframe_pilot_v1 — 真实竖屏 → 16:9 补标任务规范 | `b0c161e669200865` |
| `ext_data/configs/portrait_reframe_pilot_v2/ANNOTATION_SCHEMA.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | portrait_reframe_pilot_v2 — 真实竖屏 → 16:9 补标任务规范（v2） | `78a92c88ac79e6fe` |
| `ext_data/reports/20260929_data_releases_v1.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 冻结数据版本 v1（2026-09-29） | `fe53c0d838150570` |
| `ext_data/reports/progress_20260928.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 外部数据采集与预处理进度（2026-09-28 19:30 UTC+8） | `1023af43d80043a7` |
| `ext_data/reports/registry_status.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 外部数据注册表状态（自动生成 2026-09-28T19:26:59+08:00） | `5890a2438cc8b84c` |
| `gdn_npu/README.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | qwen35_gdn_npu — Qwen3.5 GDN / causal_conv1d 的 910A NPU 加速 | `cd827af392882bd1` |
| `night_submit_20261023/README.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 工作区原件 | 夜间提交链(2026-10-23 夜 → 10-24 晨) | `e8037fcdb06b999b` |
| `old_docs/DATASETS.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 工作区原件 | Dataset provenance and preparation | `f438c6a8a9885ab9` |
| `old_docs/OFFICIAL_BASELINE.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 工作区原件 | Public baseline contract reference | `31b7c94ac038e43c` |
| `old_docs/REMOTE_ENV_20260925.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 工作区原件 | 远程训练环境快照 | `957dce18dc68f1ea` |
| `old_docs/aic_qwen35_4b_route_plan.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 工作区原件 | Qwen3.5-4B 后续路线计划书 | `68a03036d244cd70` |
| `old_docs/e2_1b_keep_sweep_note.md` | [04 LFM空间学生](04_lfm_spatial_students.md) | 工作区原件 | E2.1 归档 + E2.1b 低 keep 扫描 | `18dbc845ca5fa82d` |
| `old_docs/keep_scan_closure.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 工作区原件 | keep 扫描结案记录 | `15ce9494ae9cbf22` |
| `old_docs/r15_reward_function_versions.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 工作区原件 | RL 奖励函数版本史（v1 → v6 + 反事实基线 + oracle） | `0265665f482143d8` |
| `old_docs/r15b_experiment_plan_table.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 工作区原件 | R15b 试验计划表 | `03763360667ccffc` |
| `old_docs/r15b_signal_diagnosis_record.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 工作区原件 | R15b 监督信号与官方域诊断记录 | `391e91ab2b5d96a4` |
| `old_docs/r15d_ceiling_and_identifiability.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 工作区原件 | R15d：官方指标天花板前沿 + (d, R̄) 不可识别性 + 一个可识别实验 | `f1fe211f96b16bc6` |
| `old_docs/r21_direction_and_plan_20261015.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 工作区原件 | R21 审计后的方向与执行计划 | `47230ac0594a9015` |
| `old_docs/submission_contract.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 工作区原件 | AIC 输入索引与提交契约（本地校验器） | `7e11e1861e1d6b73` |
| `p0_dataset_supplement_plan_20261023.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 工作区原件 | P0 数据集补充方案：为什么 42.49 "撞上" 官方偏好，以及补什么 | `79aa57d5ddcde2f0` |
| `releases/20260925_v1/README.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 固定工程候选 20260925 v1 | `c3f2cebb725318c4` |
| `reports/20260925_local_validation_phase.md` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | 当前Git已跟踪 | 2026-09-25 local validation phase | `758bb5ca2cd68273` |
| `reports/20260925_long_sprint.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 2026-09-25 自主研究 sprint 记录 | `df109ad673f9ab53` |
| `reports/20260925_pairwise_ranking_and_proxy_ood.md` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | 当前Git已跟踪 | 2026-09-25 pairwise ranking and proxy OOD continuation | `f7f62a5911ed32c5` |
| `reports/20260925_representation_generalization.md` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | 当前Git已跟踪 | 2026-09-25 表示泛化研究记录 | `cc9f8e10429cb566` |
| `reports/20260925_spatial_group_and_ood.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | 2026-09-25 Spatial group observation and OOD acquisition update | `162cb6dbeaabf62c` |
| `reports/20260926_evaluation_readiness.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 2026-09-26 evaluation-set readiness | `b7a355ab0b9cf621` |
| `reports/20260926_foundation_candidate_release.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | Foundation candidate release (2026-09-26) | `15de49fa2697ecd7` |
| `reports/20260926_internvideo2_compatibility.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | InternVideo2 1B bounded compatibility probe — 2026-09-26 | `2ef33610549c5734` |
| `reports/20260926_model_base_inventory.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 2026-09-26 bounded model-base inventory | `0e80b4214b638ec3` |
| `reports/20260926_native_qvh_annotation_acquisition.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | Native QVHighlights annotation acquisition (2026-09-26) | `861101e665e12bca` |
| `reports/20260926_official_score_candidate_diff.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 2026-09-26 官方分数候选差异核对草稿 | `524ac73b790de88c` |
| `reports/20260926_official_score_diagnosis.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | Official score diagnosis (2026-09-26) | `3a85ccf0f813cb64` |
| `reports/20260926_official_test_inference.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 2026-09-26 正式评测集冻结推理与上传包 | `0f9ec77775db35d5` |
| `reports/20260926_official_train_audit.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | Local QVHighlights-derived asset audit (2026-09-26) | `8520dd6b9f74635c` |
| `reports/20260926_qvh_quick_probe.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | QVH native candidate quick probe | `727e695d6291e59e` |
| `reports/20260926_scaling_experiments.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | Scaling experiments (2026-09-26) | `da261e7aff8e002a` |
| `reports/20260926_video_foundation_probe.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | Video foundation probe | `8921167aadea77ea` |
| `reports/20260926_videomae_internvideo_followup.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | VideoMAEv2 / InternVideo2 follow-up (2026-09-26) | `873e908e3b7b6a3e` |
| `reports/20260926_videomae_qvh_probe.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | VideoMAEv2 matched QVHighlights probe (2026-09-26) | `e7b3135fe6256a39` |
| `reports/20260927_all_select_diagnosis.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 全选退化诊断与最小时间对照（P0/P1/P3） | `a89306a5dba29a5c` |
| `reports/20260927_foundation_scaling.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | Foundation scaling continuation — 2026-09-27 | `8f78c3510d4a0120` |
| `reports/20260927_official_score_root_cause.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 官方分数差异根因分析：G / E01 / Stage2 | `ece3ad027b0611e0` |
| `reports/20260927_official_score_root_cause_v2.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | G / E01 / Stage2 / K710 官方分数复核报告（v2） | `e27f5b6aea0ae5cb` |
| `reports/20260927_qwen_temporal_audit.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | Qwen 时间选择审计（P2） | `cda8086eb0f3986e` |
| `reports/20260927_stage2_release.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | InternVideo2 Stage2 release — 2026-09-27 | `fa1a9085201e5315` |
| `reports/20260928_dense_temporal_interaction.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | DENSE × TEMP 2×2 组合（DT_V3） | `52439ca44f3a61cb` |
| `reports/20260928_max_window_path.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | 最大窗口：镜头内轨迹（P1a）与主体观测（P1b/P2） | `b58299400d82b026` |
| `reports/20260928_score_42_83_followup.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 42.83 之后的教师诊断（T0/T1/T2 结果） | `7081f30466fa6619` |
| `reports/20260928_score_46_84_followup.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 46.84 之后的教师实验（指南 v3） | `f37f4e69a29dd976` |
| `reports/20260928_semantic_subject_probe.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | Qwen3-VL-32B 空间主体定位探针（P2） | `8148df39c0af5675` |
| `reports/20260928_spatial_candidates_release.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | 空间裁剪候选包交付（S1 / S2 / B0 / Q2） | `14d0760bc6998c30` |
| `reports/20260928_teacher_60_gap_analysis.md` | [03 32B教师](03_qwen32b_teacher.md) | 当前Git已跟踪 | 教师系统离 60 分参照的差距分析 | `4d7eccf937f55933` |
| `reports/20260928_teacher_crop_decision_probe.md` | [03 32B教师](03_qwen32b_teacher.md) | 当前Git已跟踪 | 教师窗口决策探针：oracle 复核与 T5 候选评分器 | `44409ba5cccceb28` |
| `reports/20260928_temporal_supervision_contract.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 时间监督契约 v1（P3） | `bb6a45dbea7eadd6` |
| `reports/20260929_lfm450_raw_eval.md` | [04 LFM空间学生](04_lfm_spatial_students.md) | 当前Git已跟踪 | LFM2.5-VL-450M 原始模型零样本评估（学生底座准入） | `e972b9acfe3a4db5` |
| `reports/20260929_official_v4_feedback.md` | [03 32B教师](03_qwen32b_teacher.md) | 当前Git已跟踪 | V4 候选包正式成绩反馈（官方分数登记） | `18b9b43295b64486` |
| `reports/20260929_teacher_4788_cycle_v4.md` | [03 32B教师](03_qwen32b_teacher.md) | 当前Git已跟踪 | Teacher cycle V4 (10 h) after 47.88 | `5ac37a5ac12995fe` |
| `reports/20260929_teacher_v5_research.md` | [03 32B教师](03_qwen32b_teacher.md) | 当前Git已跟踪 | V5 研究周期总结（2026-09-29） | `ad9fbea3659013a6` |
| `reports/20260929_v5_p0_xrerank_package.md` | [03 32B教师](03_qwen32b_teacher.md) | 当前Git已跟踪 | V5 P0：INTERP 上的 x 轴视觉复核（官方包 QWEN32B_INTERP_XRERANK_V5_FINAL） | `ae44f69e049847af` |
| `reports/20260929_v5_p2_ground_pilot_negative.md` | [03 32B教师](03_qwen32b_teacher.md) | 当前Git已跟踪 | V5 P2：Qwen 点名主体、GroundingDINO 落位的分工路线（pilot 负结果） | `ef3f861b8cfd9ea6` |
| `reports/20260929_v5_package_and_data_brief.md` | [03 32B教师](03_qwen32b_teacher.md) | 当前Git已跟踪 | V5 交付包改动点 × 数据集盘点简报（2026-09-29 晚） | `7960eb502578c3eb` |
| `reports/20260929_v6_e0_evidence_audit.md` | [04 LFM空间学生](04_lfm_spatial_students.md) | 当前Git已跟踪 | V6 E0 证据与接口审计（2026-09-29） | `0d8b61135b0ee684` |
| `reports/20260930_v6_next10h_research.md` | [04 LFM空间学生](04_lfm_spatial_students.md) | 当前Git已跟踪 | AIC V6 后续 10 小时研究报告（2026-09-29 23:14 → 09-30） | `6e10cd9a6270b24b` |
| `reports/20261001_semifinal_regression_analysis.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 复赛分数滑退归因与激进改造计划（2026-10-01） | `677c76d1a7b56420` |
| `reports/20261001_v9_p0p1_execution.md` | [04 LFM空间学生](04_lfm_spatial_students.md) | 当前Git已跟踪 | V9 P0/P1 执行报告（2026-10-01 夜） | `5ebc66169a434657` |
| `reports/20261002_bottleneck_temporal_denominator.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 复赛瓶颈定位：F1 的时间分母 —— 为什么所有干预都是 ±0.1 | `8a42fcfffcda8d77` |
| `reports/20261002_dataset_request_spec.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 数据集补充需求规格（可直接派给其他 agent 去找） | `2d5c49f9f0b766a6` |
| `reports/20261002_real_score_anchor.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | 复赛得分真相校准（2026-10-02）——所有分都是平台分，不自行换算 | `a179afc07eb41fd9` |
| `reports/20261002_weight_sweep_analysis.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | V9 KD 权重扫描 —— 六臂源级配对 bootstrap（2026-10-02） | `ef3594059166cb81` |
| `reports/20261003_v10_exploration_report.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | V10 exploration report: the ranking wall and one unconfirmed signal | `57385d257323291a` |
| `reports/20261004_round4_report.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Round-4 report: audit gate, the position-prior inversion, and the clean confirm set | `4e92a199f33bf366` |
| `reports/20261004_round5_plan.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Round-5 执行计划：34.95 → 45（分数口径修正版） | `c80dcb99eaf86fc9` |
| `reports/20261004_round5_report.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Round-5 report, phase A: content verdict, spatial oracle, contract rebuild | `30531d70718d2ece` |
| `reports/20261004_v11_day2_report.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | V11 day-2 report: expected-F confirmation and the native-framerate M01 control | `7cac89787436ec16` |
| `reports/20261004_v11_overnight_report.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | V11 overnight execution report (2026-10-03 night -> 10-04 morning) | `e510c1a3a657ec6a` |
| `reports/20261005_r7_rl_disagreement_survey.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R7 addendum survey: can RL recover the annotator-disagreement gap? | `e3ef114ee6595f57` |
| `reports/20261005_round6_plan.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Round-6 执行计划：前沿机制迁移（S-REREAD / T-CONTEXT / T-TOKENS） | `27ec8d9292d68451` |
| `reports/20261005_round6_report.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Round-6 report: R6 plan executed - three verdicts, 9B rule change, V7 | `876f7868245d92e4` |
| `reports/20261005_round7_plan.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Round-7 plan: the 9B era, revised by the S-line verdicts | `b40de627c2b966fe` |
| `reports/20261005_round8_plan.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Round-8 plan: full-information expected utility replaces the decision-layer line; 7B SFT … | `2b3007cf637e642e` |
| `reports/20261007_round11_plan.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R11 研究计划：四层路线与空间主攻（v2） | `7817b3f3a050f4ad` |
| `reports/20261007_round12_frame_ranking_frontier_v3.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R12 计划 v3：帧级排序前沿深挖（TRINITY/ECCV 2026 实测表 + 排序监督新证据） | `7fb9f9880ee1cfc5` |
| `reports/20261007_round12_frontier_addendum.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R12 计划 v2：前沿调查注入（复赛 34→45，帧级排序主攻） | `1bb2cec575a53bfa` |
| `reports/20261007_round12_progress.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R12 进度报告（2026-10-07 深夜）：NPU 重启后探针矩阵 + P0-a 对接审计进行中 | `e73783cbfe3d4a70` |
| `reports/20261007_round12_r45_plan.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R12 计划：复赛 34 → 45（平台差分驱动，单变量快扫） | `0341cfa6b02c0d42` |
| `reports/AIC_F1_bottleneck_research_dossier.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | AIC 视频高光检测 · 复赛阶段研究历程与技术档案 | `c109e2de04764361` |
| `reports/GPT6PRO_PROMPT.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | 给 GPT-6-PRO 的研究提示词 | `1186909be9358241` |
| `reports/GPT6PRO_PROMPT_V10.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Prompt for GPT-6-PRO, round 10: the learnable-signal census — finding alpha over the posi… | `227df29205919060` |
| `reports/GPT6PRO_PROMPT_V2.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Prompt for GPT-6-PRO, round 2: four questions while the LoRA run finishes | `f158d00cc2bcd3e2` |
| `reports/GPT6PRO_PROMPT_V3.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Prompt for GPT-6-PRO, round 3: audit of your V11 plan, with fresh data | `f43774e5af989317` |
| `reports/GPT6PRO_PROMPT_V4.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Prompt for GPT-6-PRO, round 4: strategy under two pending platform scores | `4078ccf6f0588f88` |
| `reports/GPT6PRO_PROMPT_V5.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Prompt for GPT-6-PRO, round 5: the 6.2-raw-gap attack after both scores landed | `32ab8f89b5a35d76` |
| `reports/GPT6PRO_PROMPT_V6.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Prompt for GPT-6-PRO, round 6: track the frontier, compare us to it, then beat it | `35cbf5ec10a89a18` |
| `reports/GPT6PRO_PROMPT_V7.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Prompt for GPT-6-PRO, round 7: the 9B era - correlation, backbone, and what ports | `0c84daf63738212c` |
| `reports/GPT6PRO_PROMPT_V8.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Prompt for GPT-6-PRO, round 8: the annotator-disagreement problem and the RL toolchain | `f8ca6356a9227cfd` |
| `reports/GPT6PRO_PROMPT_V9.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Prompt for GPT-6-PRO, round 9: why every learned head converges to the position prior, an… | `1c5757209794ebaf` |
| `reports/aiku_asset_audit_20261019/summary.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | AIKU × brAIn Video Summarization Challenge 数据审计 | `231778eb8f6b2a18` |
| `reports/annotation_batch_v1_runbook.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 双人标注操作说明（V1.3，2026-10-17） | `efa1d64bd17d12e9` |
| `reports/annotation_protocol_v1.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | AIC 人工先验标注标准 v1.2 | `b92fb2db9843543d` |
| `reports/annotation_protocol_v1_3.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 高光视频打标标准 V1.3 | `b80905eb005d6926` |
| `reports/competition_transfer_review_20261019.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 外部视频竞赛经验的 AIC 迁移裁决（2026-10-19） | `f660a1adb84a5dd7` |
| `reports/dinov3_v1/20261020_dinov3_probe_results.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | DINOv3 v1 matched probes — 2026-10-20 | `0fadf57477e6775c` |
| `reports/dinov3_v1/README.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | DINOv3 v1 asset audit and runtime smoke | `b6bd6229e4733d21` |
| `reports/dinov3_v1/ablation_reassessment_20261020.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | DINOv3 计划消融重评估（2026-10-20） | `97fc5ebb303e34b5` |
| `reports/dinov3_v1/supervision_source_audit_20261020.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | DINOv3 时间监督源审计（2026-10-20） | `c31a40ace7e98657` |
| `reports/dinov3_v2/README.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | DINOv3 匹配对照与标签修复：2026-10-20 | `bf6c8b599aa7c72d` |
| `reports/final_submission_20261024/NOTES.md` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | 工作区原件 | LIVE-YT-VC 全量重训 20261023 — 工作记录 | `9316c504992a13ea` |
| `reports/final_submission_20261024/p2_done/README.md` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | 工作区原件 | P2_NEWA-NEWB_CAP24MS0.15QP0.6 | `0ae60a3138b3589e` |
| `reports/final_submission_20261024/retrain_runtime/README_先读.md` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | 工作区原件 | Video-ORA 42.48版本复现交接包 | `20c7751a3de845a7` |
| `reports/final_submission_20261024/retrain_runtime/licenses/GNMC_PolyForm_Noncommercial_1.0.0.md` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | 工作区原件 | PolyForm Noncommercial License 1.0.0 | `c0ea4a896d2c8c39` |
| `reports/internal_tsm_20260925.md` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | 当前Git已跟踪 | Canonical internal TSM probe (2026-09-25) | `ea1c7129787e95f1` |
| `reports/jev_videoora_transfer_20261021.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | Video-ORA 42.49 对 Jev 路线的迁移结论（2026-10-21） | `0d93c3e886c1e1af` |
| `reports/official_preference_matching_20261020.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 官方偏好匹配研究（2026-10-20） | `bb713f6389695c19` |
| `reports/official_test_inference.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 工作区原件 | 2026-09-26 正式评测集冻结推理与上传包 | `0c6113cb2f448276` |
| `reports/postprocess_A0_006_20260925.md` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | 当前Git已跟踪 | A0_006 zero-parameter temporal post-processing | `4f1efbc3209c6b54` |
| `reports/r10/evidence_ledger.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R10 证据台账（可学习信号普查与策略空间审计） | `7a65449fc70ba997` |
| `reports/r10/gpt6pro/README.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R10 audit bundle | `21699aa14c293707` |
| `reports/r10/gpt6pro/research.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | AIC R10：可学习信号普查与策略空间审计 | `6a5fdddc500a36f6` |
| `reports/r100_spatial_context_utility_20261019/summary.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R100：R70 pooled temporal context 空间代理 | `c119fcde7e0a01fc` |
| `reports/r101_spatial_resnet_context_20261019/summary.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R101：当前帧高分辨率候选 × ResNet18 邻帧视觉上下文 | `680b9387aa47073d` |
| `reports/r102_aiku_transfer_20261020/summary.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | R102 AIKU temporal transfer pilot (2026-10-20) | `22ca83706c31acfc` |
| `reports/r103_v9_token_context_20261020/summary.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R103 genuine V9 token context (2026-10-20) | `fd17c5032115ce89` |
| `reports/r104_avimos_saliency_20261020/summary.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | R104 AViMoS visual saliency pretraining (2026-10-20) | `f35ed51f92146c45` |
| `reports/r105_avimos_aic_utility_20261020/summary.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | R105 AViMoS saliency as AIC crop utility (2026-10-20) | `6a67291dd66ae88b` |
| `reports/r106_breakthrough_temporal_routes_20261020/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R106–R108：跨采样率事件拓扑解码（2026-10-20） | `8c3edd5bc2a07381` |
| `reports/r107_r91_official_postmortem_20261020/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 工作区原件 | R107/R91 官方回分复盘（2026-10-20） | `a1234e84b40c850f` |
| `reports/r107_r91_r108_r93_official_postmortem_20261020/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R107/R91/R108/R93 四包官方回分分析（2026-10-20） | `e57546ebf7ff2def` |
| `reports/r11/20261007_day1_verdicts.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R11 Day-1 执行判定（2026-10-07） | `1835f745edf6dde1` |
| `reports/r11/20261007_s2a_readout_verdict.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R11 S2a 读出判定（readouts a/b/c，live 池） | `9622e327a9a41d92` |
| `reports/r12/20261007_p0a_match_verdict.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R12 P0-a 判定：426 × PHD² 对接审计 —— 结局 B（对接线按预注册关闭） | `dfbf238b676aca4c` |
| `reports/r12/20261008_gatt_vmae_exec_plan.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R12-GATT / R12-VMAE 执行方案（2026-10-08 凌晨批准批） | `c2b89b266b93f75d` |
| `reports/r12/20261009_r12_gatt_verdict_next.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R12-GATT 回分分析与下一步计划（2026-10-09） | `05e2c7a187f3494a` |
| `reports/r13/20261008_r13_frame_ranking_investigation.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R13 调查：底座选型与帧级排序问题（2026-10-08 深夜） | `89f95bd10b361422` |
| `reports/r13/20261008_r13_retrospective_45gap.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R13 复盘：为什么对手到 45 而我们停在 35（2026-10-08） | `8deaf524ab18a233` |
| `reports/r13/20261009_f1_npu8_ablation.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R13 F1 NPU 8 臂消融记录（2026-10-09 晚） | `25026fbfd3915e6a` |
| `reports/r13/20261009_f1f3_official_baseline_audit.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | R13 F1-F3 CPU 窗口审计报告（2026-10-09 上午） | `8add5ae56b9d7343` |
| `reports/r24/summary.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R24：干净源级 point/block、VPB/CLIP 与 AHA 尝试 | `de56b533b38169e5` |
| `reports/r24_aha_repair/summary.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | R24 AHA 依赖修复与真实视觉测试 | `264a8d2147119a37` |
| `reports/r25_aha_submit/summary.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | R25 AHA原生BF16复赛提交包 | `7a949acedf080115` |
| `reports/r25_score_analysis/summary.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | R25 复赛35.18：原因分析与下一轮诊断 | `8e248f7a368887aa` |
| `reports/r26/official_analysis.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R26 复赛回分分析（2026-10-16） | `f0d355b740e05c16` |
| `reports/r26/reproduce.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R26 复现与数据边界 | `356ecdee1672f47c` |
| `reports/r26/summary.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R26 选帧、预算、公共位置与空间对照（2026-10-16） | `fa40470bf4f8c1ec` |
| `reports/r27/summary.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R27：围绕36.04最佳中心时间窗的四项对照 | `4bdf8eb948911ec0` |
| `reports/r28/summary.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R28：停止中心预算扫描，核验空间链并转向主体评分 | `2895b08ebd5e8a31` |
| `reports/r29/aha_transfer_check.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R29 AHA transfer mechanism check | `ee9747d0b9e20642` |
| `reports/r29/decision_theory.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R29：以每视频期望联合 F 为目标的决策推导与精确反例 | `a69a4274665a09f8` |
| `reports/r29/pipeline_contract.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R29：现役信息路径与梯度合同 | `27a6b0132fd6acb2` |
| `reports/r29/reproduction.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R29 复现与证据索引 | `6c02b41b30ea6d57` |
| `reports/r29/spatial_review.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R29 空间路线独立审查与最小诊断 | `5c0fcbe419eb1cfa` |
| `reports/r29/summary.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R29：先辨认目标与信息，再决定训练什么 | `aadd9fee903b83c1` |
| `reports/r29/temporal_evidence.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R29：现有 AHA 标量的决策挽救、监督合同与观测粒度判别 | `33f945beb9b0f91d` |
| `reports/r29/temporal_sources/mrhisum_github_readme.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Mr. HiSum: A Large-scale Dataset for Video Highlight Detection And Summarization | `aceea27fc0832eb3` |
| `reports/r29/temporal_sources/videoxum_github_readme.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | VideoXum: Cross-modal Visual and Textural Summarization of Videos | `073cdd0f5cf5cf4f` |
| `reports/r29/temporal_sources/videoxum_hf_readme.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Dataset Card for VideoXum | `8288ebaa5e311e6a` |
| `reports/r30/summary.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R30_DYNAMIC_CONTENT_PROBE — 2026-10-16 | `bb4c065143b92472` |
| `reports/r31/summary.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R31 低维事件状态探针（2026-10-16） | `c58cb703d0678613` |
| `reports/r37_content_candidates/official_score_analysis_20261017.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R39–R43 官方回分记录与判断 | `1dd8bcacd5c25fc6` |
| `reports/r37_content_candidates_20261016.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R37–R41 内容驱动时间候选 | `59599e127e90e98b` |
| `reports/r42_annotation_v13_20261017/summary.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | R42 V1.3双人标注完整扩展 | `945f45e862c4c0b7` |
| `reports/r42_annotation_v13_full_20261017/summary.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | R42 V1.3 完整标注质量审计 | `a4a7a25f13475e1e` |
| `reports/r42_smoke_quality_20261017/review.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | R42 Smoke 标注质量验收（2026-10-17） | `52441db1032bae87` |
| `reports/r42_smoke_revised_20261017/review.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | R42 Smoke修订复查（2026-10-17） | `f8c2cb72215897d5` |
| `reports/r43_iv2_event_completion_20261017.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R43：InternVideo2 6B 事件完成度路线 | `f9222732e6f40168` |
| `reports/r43_r42_postmortem_and_r44_plan_20261017.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R43回分、人工标签质量与R44路线 | `2c6dc89e45ab86a0` |
| `reports/r44_human_calibrated_residual_20261017/summary.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R44：人工标签校准残差 + R26 B3（2026-10-17） | `e6c4a96110470b97` |
| `reports/r44_score_analysis_20261017/README.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R44/R26 微扰随机交换 null（2026-10-17） | `720ce96357d44bd7` |
| `reports/r44_score_analysis_20261017/analysis.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R44=36.09 回分分析与下一步（2026-10-17） | `994d72447029770a` |
| `reports/r45_boundary_guarded_residual_20261017/summary.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R45 boundary guarded residual (2026-10-17) | `39876b7f0b6ac428` |
| `reports/r45_highfreq_boundary_20261017/summary.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | 高频残差边界诊断（2026-10-17） | `b502fe571f281bc0` |
| `reports/r45_highfreq_state_20261017/summary.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R45 高频局部状态残差（2026-10-17） | `038b854f96898b1e` |
| `reports/r46_geometry_joint_review_20261017/summary.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R46 路线审查：时间—空间联合效用的可行边界 | `7721800cedaa15a9` |
| `reports/r46_local_highfreq_20261018/summary.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R46 D3 local-state residual package | `8eaf0cb0fcfa5a22` |
| `reports/r46_semantic_transition_20261017/summary.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R46 semantic-transition probe (2026-10-17) | `157d95fd8fd7e15b` |
| `reports/r46_spatial_readout_20261017/summary.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R46 空间候选效用审计（2026-10-17） | `eb030e76ecb63c88` |
| `reports/r47_highfreq_consensus_20261018/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R47 cross scale consensus residual (2026-10-18) | `873b1c38263b0cc0` |
| `reports/r47_timelens2_full_20261017.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R47 TimeLens2 full inference and package audit (2026-10-17) | `397844ccc54ef6bd` |
| `reports/r47_timelens2_v9_20261018/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R47 TimeLens2 + V9 vision-only B3 candidate | `37aba3c7edbf0f77` |
| `reports/r47_timelens2_v9b3_package_20261017/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R47 TimeLens2 + VISIONONLY B3（2026-10-17） | `1a0b4a70ae940c27` |
| `reports/r47_timelens2_visiononly_compliance_20261017.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R47 TimeLens2 + VISIONONLY B3 合规核验（2026-10-17） | `1d16fe3decad33f3` |
| `reports/r47_v9_spatial_20261018/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R47 V9 vision-only spatial candidate | `d1b49a41cf4a77b1` |
| `reports/r48_consensus_gated_residual_20261018/summary.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R48 consensus gated temporal residual (2026-10-18) | `cf8b2eeebc5b3a36` |
| `reports/r5/anchor_neutral_protocol_audit.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Anchor vs Neutral slicing protocol audit (round-5 A6) | `b95e29cf188aa3af` |
| `reports/r5/augmentation_backlog_linkage.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | 数据增强候选池 × round-5 裁定的衔接分析 | `e215f4b871074fde` |
| `reports/r5/b3_objective_and_candidate_audit.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | B3 objective and candidate audit (round-5 A5) | `33def7e875f29e46` |
| `reports/r5/claim_ledger.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | Round-5 claim ledger | `67130feb7a7d6239` |
| `reports/r53_quality_conditioned_residual_20261018/summary.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R53：条件残差与人工事件字段判别（2026-10-18） | `39da9843f71f148d` |
| `reports/r53_v9_spatial_audit_20261018/summary.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R53 V9 spatial intervention audit (2026-10-18) | `2dcd407337e702a8` |
| `reports/r54_r46_official_postmortem_20261018/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R54 / R46 官方回分复盘（2026-10-18） | `4e8339a8a1d97c4c` |
| `reports/r54_timelens_center_residual_20261018/package_diff_20261018.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R54/R46 package difference audit (2026-10-18) | `17c6b9890d7408c2` |
| `reports/r54_timelens_center_residual_20261018/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R54：TimeLens2 中心锚定残差（2026-10-18） | `5676d43a8673211c` |
| `reports/r55_timelens_event_phase_20261018/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R55 TimeLens2 event phase residual (2026-10-18) | `cd27cd46a6b6dd14` |
| `reports/r56_timelens_event_edge_20261018/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R56 TimeLens2 event edge residual (2026-10-18) | `b58cb27346e94843` |
| `reports/r56_timelens_peak_public_probe_20261018.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | TimeLens2 peak prompt smoke (2026-10-18) | `063eeb36a914afc3` |
| `reports/r57_fixed_late_window_20261018/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R57：固定后移窗口控制（2026-10-18） | `7c2d36717c7bbce7` |
| `reports/r57_r59_official_postmortem_20261018/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R57/R59 官方回分复盘（2026-10-18） | `3ab1456212c2625e` |
| `reports/r57_r59_r63_r64_last_slot_20261018.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R57/R59 official postmortem and R63/R64 last-slot decision (2026-10-18) | `bd22729e49f1f8df` |
| `reports/r58_timelens_mask_edge_ensemble_20261018/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R58 TimeLens mask + edge multi-view ensemble | `6f2d0d482333c369` |
| `reports/r59_joint_spatial_utility_20261018/summary.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R59 joint spatial utility candidate (2026-10-18) — INVALID DRAFT | `f7adae3b99652ebb` |
| `reports/r59_timelens4fps_ensemble_20261018/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R59 TimeLens2 4fps mask + edge ensemble | `0279c4ea220cfc7c` |
| `reports/r6/claim_ledger.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R6 claim ledger | `52f2a087b474a0dc` |
| `reports/r6/source_ledger.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R6 source ledger — LIVE / RetargetVid exposure accounting | `44a1aa35f864aceb` |
| `reports/r60_joint_spatial_stability_20261018/summary.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R60 joint spatial stability control (2026-10-18) — INVALID DRAFT | `f3bb2e05313d4449` |
| `reports/r60_timelens4fps_direct_20261018/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R60 TimeLens2 4fps direct interval package | `134e150481af7bba` |
| `reports/r61_joint_spatial_utility_20261018/summary.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R61 corrected joint spatial utility candidate (2026-10-18) | `f0b1a87328c3c0c2` |
| `reports/r62_joint_spatial_stability_20261018/summary.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R62 corrected joint spatial stability control (2026-10-18) | `4d1439daa43ee3dc` |
| `reports/r63_joint_spatial_utility_20261018/summary.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R63 corrected joint spatial utility candidate (2026-10-18) | `fcbcdf3e593b2356` |
| `reports/r64_joint_spatial_stability_20261018/summary.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R64 corrected joint spatial stability control (2026-10-18) | `76e46a3987640528` |
| `reports/r67_official_bottleneck_review_20261018/summary.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R67 官方回分后的系统瓶颈复盘（2026-10-18） | `9c5ea65a295bfc1b` |
| `reports/r68_spatial_bottleneck_audit_20261018.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | 空间瓶颈审计与最小突破路线（2026-10-18） | `20f1327181973771` |
| `reports/r69_direct_spatial_grounding_20261018/README.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R69 TimeLens direct spatial grounding (high risk) | `deb7f8511370893c` |
| `reports/r69_direct_spatial_grounding_20261018/summary.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R69 direct spatial grounding package | `249596976623dc35` |
| `reports/r69_r57_r59_final_slot_20261018.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R57/R59 复盘与最后名额判断（2026-10-18） | `715be7f4130d0ec4` |
| `reports/r69_segmented_crop_20261018/README.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R69 segmented crop query | `9ee6b9eb702c31c0` |
| `reports/r69_segmented_stability_gate_20261018/README.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R69 segmented crop stability gate | `6e6a7c89cd9ee2a8` |
| `reports/r70_official_package_20261018/README.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R70 official inference and package (2026-10-18) | `7d796bfba5d65d53` |
| `reports/r70_spatial_utility_pilot_20261018/README.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R70 spatial utility readout pilot | `af3250e4b1a241c2` |
| `reports/r70_topsoft_objective_20261018/README.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R70 topsoft objective and utility driven temporal selection (2026-10-18) | `9be8dd928b07b309` |
| `reports/r70_utility_temporal_reassessment_20261018/summary.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R70 空间效用驱动时间重排复核（2026-10-18） | `c220a01e123f64bd` |
| `reports/r72_b3_asset_recovery_20261018/README.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | B3 original asset recovery audit (2026-10-18) | `37f70399666ec7ee` |
| `reports/r72_spatial_utility_temporal_20261018/README.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R72 spatial-utility temporal residual (2026-10-18) | `4aec1ec0e605ae16` |
| `reports/r73_incremental_spatial_temporal_20261018/README.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R73 incremental spatial utility temporal residual (2026-10-18) | `4e85d75d6203a096` |
| `reports/r78_direct_output_utility_20261018/summary.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R78 direct deployed-box utility readout (2026-10-18) | `687e35e947cb487b` |
| `reports/r78_direct_output_utility_20261019/summary.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R78/R81 直接部署框 utility 路线（2026-10-19） | `0bd688ded6fb378a` |
| `reports/r8/decision_layer_correction.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | DECISION_LAYER correction record (R8) | `8788ad57000b029d` |
| `reports/r82_direct_utility_ensemble_20261019/summary.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R82 direct deployed-box utility ensemble (2026-10-19) | `d79b3d534de66f7c` |
| `reports/r83_interval_policy_20261019/summary.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R83 interval policy probe | `becfd34ad12092c5` |
| `reports/r83_official_local_validity_20261019/summary.md` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | 当前Git已跟踪 | R83：官方回分与本地验证的可识别性审计 | `e1be965454ddc4c3` |
| `reports/r84_coord_consistency_20261019/summary.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R84 coordinate consistency spatial readout | `f70142f6ff44b870` |
| `reports/r85_leaderboard_transfer_review_20261019/summary.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | R85 official leaderboard distribution and highlight-model transfer review | `769d78544c89f329` |
| `reports/r86_r87_observation_and_transfer_20261019/summary.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R86 visual observation path and R87 target transfer pilot | `e275c280b25d69ca` |
| `reports/r89_lowlevel_composition_20261019/summary.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R89/R90：低层构图先验与轻量时序结构 | `607e6d32c3f6c155` |
| `reports/r9/gpt6pro/AIC_20261006_R9_prior_metric_encoder_and_execution.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | AIC R9：位置先验平台、指标灵敏度与编码器重定义 | `f50a6b16f9543fa8` |
| `reports/r9/gpt6pro/README.md` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | 当前Git已跟踪 | R9 metric-audit bundle | `8b67dbb133bdac68` |
| `reports/r91_lowparam_resnet_20261019/summary.md` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | 当前Git已跟踪 | R91：约 98.6M 的轻量语义时间路线 | `571829bd002a9352` |
| `reports/r92_multiview_traditional_route_20261019/summary.md` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | 当前Git已跟踪 | R92 传统处理多视图路线复核（2026-10-19） | `dd7cbfc434dbf297` |
| `reports/r93_dense_event_edge_20261019/summary.md` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | 当前Git已跟踪 | R93：4fps TimeLens2 dense event edge（2026-10-19） | `72c2652ac8a4e040` |
| `reports/r94_u2_h1_20261019/summary.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R94 U²-Netp 主体信息 H1 | `0e61f039ae6024a2` |
| `reports/r95_shot_structure_20261019/summary.md` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | 当前Git已跟踪 | R95 镜头结构 H2 | `70438a3b5e03d191` |
| `reports/r96_event_context_observability_20261019/summary.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R96 事件上下文可观测性判别 | `698bc16b7dda4108` |
| `reports/r97_event_state_tcn_20261019/summary.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R97 GIF 区间事件状态辅助监督 | `193e6dd23775cbfd` |
| `reports/r98_unisal_human_saliency_native_crop_20261019/summary.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | R98：AIM 2024 / UNISAL 视觉注意力迁移 | `687fc90928d8382a` |
| `reports/r99_lowrank_context_tcn_20261019/summary.md` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | 当前Git已跟踪 | R99：局部—全局低秩交互时间头 | `cf7f64ab9bb7d29a` |
| `reports/rank1_426_candidate_audit_20261023.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | rank-1 temporal 候选 426 包审计与三向对照（2026-10-23） | `c414c8e1b8f53d46` |
| `reports/rank1_42_46_postmortem_20261023.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | rank-1 候选官方回分 postmortem：42.46 vs 42.49（2026-10-23） | `f36ebf0ac4cdeb36` |
| `reports/repository_audit_20261106/summary.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 工作区原件 | 本地与 AutoDL 仓库核对记录（2026-11-06） | `9e8db0dfe10161b0` |
| `reports/spatial_benchmark_status.md` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | 当前Git已跟踪 | Spatial benchmark — 2026-09-25 | `e2a5d411462170c4` |
| `reports/v6_next10h/NEXT_ACTIONS.md` | [04 LFM空间学生](04_lfm_spatial_students.md) | 当前Git已跟踪 | V6-NEXT10H 下一步最小实验清单（2026-09-30 冻结版） | `41f1641e467e0618` |
| `reports/v6_next10h/PROGRESS_LOG.md` | [04 LFM空间学生](04_lfm_spatial_students.md) | 当前Git已跟踪 | V6-NEXT10H 进度日志（T0 = 2026-09-29 23:14 CST） | `123e58e4e1ead564` |
| `reports/v7_lfm_24h/PROGRESS_LOG.md` | [04 LFM空间学生](04_lfm_spatial_students.md) | 当前Git已跟踪 | V7 LFM 24h 进度日志（T0 = 2026-09-30 09:15 CST） | `e164119bbbf162a5` |
| `reports/v7_lfm_24h/REPORT.md` | [04 LFM空间学生](04_lfm_spatial_students.md) | 当前Git已跟踪 | V7 LFM 24h 执行报告（T0 = 2026-09-30 09:15 CST，本报告写于 T+5.5h） | `177a19afb2631a47` |
| `reports/v7_lfm_24h/SEMIFINAL_READY.md` | [04 LFM空间学生](04_lfm_spatial_students.md) | 当前Git已跟踪 | 复赛评测集就绪清单（910A 全链，2026-09-30 验证） | `64b41b29e4bf4286` |
| `reports/v8_100_500m/REPORT.md` | [04 LFM空间学生](04_lfm_spatial_students.md) | 当前Git已跟踪 | V8 研究轮报告：100–500M 档位冲分（T0 = 2026-09-30 14:40 CST） | `0c96a5ab3e176d3e` |
| `reports/valen_route/asset_audit.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | Valen route asset audit — 2026-10-20 | `a8f6098bc24bdc3d` |
| `reports/valen_route/temporal_dataset_audit_20261020.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | Valen 时序训练数据审计（2026-10-20） | `86466770eaabb0c6` |
| `reports/valen_route/utility_t006_plan.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | Valen utility target experiment T006 — 2026-10-20 | `c75f4c42c05e1fbe` |
| `reports/valen_route/utility_t006_results_20261020.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | Valen utility target T006 evaluation — 2026-10-20 | `69709e51331b7a0f` |
| `reports/valen_route/valen_sft_verdict_20261020.md` | [09 新表征与迁移](09_new_representations_and_transfer.md) | 当前Git已跟踪 | Valen-2B crop-choice SFT verdict — 2026-10-20 | `6a63685cd1377a47` |
| `reports/videoora_3point_research_20261024/plan.md` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | 工作区原件 | Video-ORA 冲击 +3 分：证据复盘与可执行研究计划 | `6875904e6ccbb4a4` |
| `reports/videoora_42_48_audit_20261020/summary.md` | [10 Video-ORA冻结系统](10_videoora_frozen_pipeline.md) | 当前Git已跟踪 | Video-ORA 42.48 复现包审计（2026-10-20） | `d6c1014083aaa693` |
| `reports/videoora_finetuned_sft_20261022/native_segment_sft_report.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | Native-segment SFT 研究报告（2026-10-22） | `6f1191c0ddd3997c` |
| `reports/videoora_finetuned_sft_20261022/summary.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | Video-ORA 精细化 SFT 阶段报告（2026-10-22） | `55f3c0d3e65dff77` |
| `reports/videoora_next_gain_routes_20261023.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | Video-ORA 42.49 基座后续提分路线研究（2026-10-23） | `7d74e37840259203` |
| `reports/videoora_rank1_temporal_20261023/summary.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | rank-1 temporal step50 候选审计（2026-10-23） | `e0fb2f07c7b7b362` |
| `reports/videoora_reproduction_20261021/portrait_fix_note.md` | [10 Video-ORA冻结系统](10_videoora_frozen_pipeline.md) | 工作区原件 | Portrait route diagnosis and isolated regression | `4d43b8067decaa5c` |
| `reports/videoora_reproduction_20261021/score_42_49_analysis.md` | [10 Video-ORA冻结系统](10_videoora_frozen_pipeline.md) | 当前Git已跟踪 | 42.49 分与输出文件大小分析（2026-10-21） | `572c29768790aad5` |
| `reports/videoora_reproduction_20261021/summary.md` | [10 Video-ORA冻结系统](10_videoora_frozen_pipeline.md) | 当前Git已跟踪 | Video-ORA 42.48 复现验收（2026-10-21） | `122ce7e82a92135f` |
| `reports/videoora_reproduction_20261021/watch_setup.md` | [10 Video-ORA冻结系统](10_videoora_frozen_pipeline.md) | 工作区原件 | 426 rank-1 completion watcher | `40b1009603da4b4f` |
| `reports/videoora_rl_smoke_20261021/summary.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | Video-ORA temporal RL smoke（2026-10-21） | `8caaf4ecda8a58ee` |
| `reports/videoora_rl_training_20261021/failure_analysis.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | TEMPORAL_RL_PUBLIC_CONTEXT_V1 失败原因分析 | `3cd47da90d13b1af` |
| `reports/videoora_rl_training_20261021/next_routes_code.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | T2 后续路线代码（2026-10-21） | `eb1fec969a70adcd` |
| `reports/videoora_rl_training_20261021/summary.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | TEMPORAL_RL_PUBLIC_CONTEXT_V1 质量验收（2026-10-21） | `793fc4fa392bc52f` |
| `reports/videoora_route_deep_analysis_20261022.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | Video-ORA 42.49 路线深度审计与扩展方案（2026-10-22） | `28b4f80e564dcebd` |
| `research_ab_20261023/PLAN.md` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | 工作区原件 | a/b 方向研究记录(2026-10-23 晚) | `f5cc632c648af539` |
| `rtx6000/README.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | R15 RTX PRO 6000 RL 迁移包 | `32742e421b8f0c57` |
| `video_competitions_research_and_agent_handoff.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 工作区原件 | 视频相关竞赛研究发现与 Coding Agent 实践交接 | `464b8a0b18ae98b1` |
| `video_ora_4b_autodl_32gb_codex_plan.md` | [10 Video-ORA冻结系统](10_videoora_frozen_pipeline.md) | 工作区原件 | Codex 研究与实施任务：Video-ORA-4B × AutoDL 单卡 32 GB × 官方视频高光任务 | `3ec395adc4f82a95` |
| `videoora_finetuned_sft_plan_20261022.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | Video-ORA 精细化监督微调计划（2026-10-22） | `9e8b3fcd2793b389` |
| `videoora_post_training_research_plan_20261023.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | VideoORA 42.49 继承后：面向官方偏好的精细后训练研究计划 | `134c85d35ee2bad3` |
| `videoora_rl/README_NEXT_ROUTES.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | Candidate policy follow-up routes | `4e6ad44bdfa23b01` |
| `videoora_rl_next_plan_20261021.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | Video-ORA 下一轮 RL 提分计划（决策级 temporal policy） | `96206d8a16f34320` |
| `videoora_rl_plan_20261020.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | Video-ORA-9B 强化学习增分计划（2026-10-20） | `4a44bdeb7926bfd8` |
| `videoora_rl_training_method_20261021.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | Video-ORA-9B 后续 RL 提分训练方法（2026-10-21） | `9e44b4e1a94e0925` |
| `videoora_rl_training_protocol_20261021.md` | [11 Video-ORA时间后训练](11_videoora_temporal_posttraining.md) | 工作区原件 | Video-ORA RL 训练协议（等待 426 基线验收后执行） | `bada09dacf008cb6` |
| `研究1/aic_transfer_learning_plan.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | AIC 轻量迁移学习与部署方案 | `760eeb6da147012e` |
| `研究1/minimal_falsification_experiments.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | AIC 三项最小判别实验：H1 / H2 / H3 | `2818eb37ef9ea96a` |
| `研究1/open_models_registry.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | AIC 公开模型与算法审计登记 | `e13cb79389b6d8a1` |
| `研究1/research_verdict.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | AIC 轻量视觉路线：研究裁决 | `b7baaa302928a043` |
| `研究1/visual_feature_research.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | AIC 轻量视觉特征研究报告 | `0acdd785f25e32b5` |
| `研究1/执行批判与下一路线_20261019.md` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | 当前Git已跟踪 | 研究1 结论的批判性执行记录 | `39daab66a48f3b5f` |

## JSON配置清单

配置存在不代表该实验完成；需结合上述事件、路线正文与日志。

| 配置文件 | 主归属 | SHA256前16位 |
|---|---|---|
| `configs/A0_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `aad52122bb3d97cc` |
| `configs/A0_002.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `44e33016a281e7ae` |
| `configs/A0_003.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `d67a71a8389a65e5` |
| `configs/A0_004.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `9c35b1107032a25d` |
| `configs/A0_005.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `9ff2f704a5e333b4` |
| `configs/A0_006.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `5ef6e10431b237c4` |
| `configs/A0_007.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `664f9b2175d0bb48` |
| `configs/A0_008.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `db5515a609fd3950` |
| `configs/A0_009.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `47341789a0cfa876` |
| `configs/A0_010.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `0e8047aaeb7c44f0` |
| `configs/A0_011.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `c64c39111750de72` |
| `configs/A0_012.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `56cb5cc2dea56ba9` |
| `configs/A1_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `5e3712132bb078fa` |
| `configs/A1_002.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `1128acd0f92f920f` |
| `configs/A1_003.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `32e8f07d89fe9eb2` |
| `configs/A1_004.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `3caadb36a67e3385` |
| `configs/A1_005.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `3bc164680bc67408` |
| `configs/A1_006.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `566d01c49c7eae70` |
| `configs/A1_007.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `c3d04d55e839a25f` |
| `configs/A1_008.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `5250ab7c07b61e56` |
| `configs/A1_009.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `3d01cde4ee4372dc` |
| `configs/A1_010.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `b7a57072720db5c5` |
| `configs/A2_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `1a0bd6c512818b55` |
| `configs/A2_002.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `0c94a46958250064` |
| `configs/A2_003.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `34a070ea704c824e` |
| `configs/A2_004.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `7d8610468c2c7e88` |
| `configs/A2_005.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `4e605b9f05e74d93` |
| `configs/ALL_SELECT_YUNET_V1.json` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | `5d1eb36b63d66755` |
| `configs/ANNOTATION_ASSIGNMENTS_V13.json` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | `3089f077899e7c50` |
| `configs/ANNOTATION_BATCH_V1.json` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | `b0d4a711e8dfc1ae` |
| `configs/Bs0_deit_s_probe.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `e3f3d66d1fd746ca` |
| `configs/DENSE_E2E_001.json` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | `2e416bc9480e8a63` |
| `configs/DENSE_GROUP_E2E_001.json` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | `511556da47e10589` |
| `configs/DINOV3_S_MATCHED_V1.json` | [09 新表征与迁移](09_new_representations_and_transfer.md) | `91e6bbabfa505f0f` |
| `configs/DINOV3_S_MATCHED_V2.json` | [09 新表征与迁移](09_new_representations_and_transfer.md) | `1bc2b741b1bf8178` |
| `configs/DINOV3_T_MATCHED_V1.json` | [09 新表征与迁移](09_new_representations_and_transfer.md) | `e9401277cbac11c5` |
| `configs/JEV_TEMPORAL_VIDEOORA_TRANSFER_V1.json` | [09 新表征与迁移](09_new_representations_and_transfer.md) | `677fec6c169a79a2` |
| `configs/LFM450_RAW_GROUND_V1.json` | [04 LFM空间学生](04_lfm_spatial_students.md) | `c43ab80a491d4a5d` |
| `configs/LFM450_SPATIAL_DEV_MANIFEST_V1.json` | [04 LFM空间学生](04_lfm_spatial_students.md) | `5595d47af6da4a40` |
| `configs/LFM_QWEN32B_SEMIFINAL.json` | [03 32B教师](03_qwen32b_teacher.md) | `5bfaaa17ed737e9e` |
| `configs/LFM_V7_S_FINAL.json` | [04 LFM空间学生](04_lfm_spatial_students.md) | `c7562441b954292f` |
| `configs/LFM_V7_S_SEMIFINAL.json` | [04 LFM空间学生](04_lfm_spatial_students.md) | `269921ca65461fd6` |
| `configs/LFM_V8_B3_SEMIFINAL.json` | [04 LFM空间学生](04_lfm_spatial_students.md) | `96feb341fd3862f6` |
| `configs/LFM_V9_A1RAW_SEMIFINAL.json` | [04 LFM空间学生](04_lfm_spatial_students.md) | `52a4dbc33cb97ba8` |
| `configs/LFM_V9_B3_VISIONONLY_SEMIFINAL.json` | [04 LFM空间学生](04_lfm_spatial_students.md) | `e86baa5c0590410f` |
| `configs/LFM_V9_VISIONONLY_SEMIFINAL.json` | [04 LFM空间学生](04_lfm_spatial_students.md) | `b5af05ea21a15ef4` |
| `configs/MAX_WINDOW_QWEN_NOFACE_V1.json` | [03 32B教师](03_qwen32b_teacher.md) | `61b84d0fb11b5809` |
| `configs/MAX_WINDOW_QWEN_POINT_V1.json` | [03 32B教师](03_qwen32b_teacher.md) | `4b27126e369b659d` |
| `configs/NATIVE_FOUNDATION_CANDIDATES_V1.json` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | `c8f14d7812c35bff` |
| `configs/QWEN32B_DENSE_EVENT_TEMP_V4.json` | [03 32B教师](03_qwen32b_teacher.md) | `75ea8afaa8e489d9` |
| `configs/QWEN32B_DT_CONTEXT_POINT_V4.json` | [03 32B教师](03_qwen32b_teacher.md) | `909391f50953c0ff` |
| `configs/QWEN32B_DT_INTERP_V4.json` | [03 32B教师](03_qwen32b_teacher.md) | `ca2314cfbba73bd9` |
| `configs/QWEN32B_DT_MEDIAN3_V4.json` | [03 32B教师](03_qwen32b_teacher.md) | `193de57933e2215b` |
| `configs/QWEN32B_DT_OBS025_V4.json` | [03 32B教师](03_qwen32b_teacher.md) | `95d922d756a4b5b9` |
| `configs/QWEN32B_DT_VISUAL_RERANK_V4.json` | [03 32B教师](03_qwen32b_teacher.md) | `33dfa4a9b7f682e4` |
| `configs/QWEN32B_INTERP_XRERANK_V5_FINAL.json` | [03 32B教师](03_qwen32b_teacher.md) | `8c8a391013b021ef` |
| `configs/QWEN32B_NOFACE_CONTEXT_POINT_V4_SPATIAL.json` | [03 32B教师](03_qwen32b_teacher.md) | `7a185bec02c49fe6` |
| `configs/QWEN32B_NOFACE_DENSE_TEMP_V3.json` | [03 32B教师](03_qwen32b_teacher.md) | `7cebeb99b389fb9d` |
| `configs/QWEN32B_NOFACE_DENSE_V2.json` | [03 32B教师](03_qwen32b_teacher.md) | `083cf00bf923e06a` |
| `configs/QWEN32B_NOFACE_INTERP_V4_SPATIAL.json` | [03 32B教师](03_qwen32b_teacher.md) | `55f8a3d2c31c46f3` |
| `configs/QWEN32B_NOFACE_MEDIAN3_V4_SPATIAL.json` | [03 32B教师](03_qwen32b_teacher.md) | `dae86e69910f11b1` |
| `configs/QWEN32B_NOFACE_OBS025_V4_SPATIAL.json` | [03 32B教师](03_qwen32b_teacher.md) | `57ae42527fa81a11` |
| `configs/QWEN32B_NOFACE_TEMPORAL_V2.json` | [03 32B教师](03_qwen32b_teacher.md) | `536692b2c3823726` |
| `configs/QWEN32B_NOFACE_TEMP_CROPHEAD_V3.json` | [03 32B教师](03_qwen32b_teacher.md) | `14e16a1844a5c415` |
| `configs/QWEN32B_NOFACE_VISUAL_RERANK_V4_SPATIAL.json` | [03 32B教师](03_qwen32b_teacher.md) | `51cb9ca0a0af41d9` |
| `configs/QWEN32B_TEMP_H1_XONLY_DIAG_V4.json` | [03 32B教师](03_qwen32b_teacher.md) | `8124d1c45af766d5` |
| `configs/R106_BREAKTHROUGH_TEMPORAL_ROUTES_20261020.json` | [06 TimeLens事件边界](06_timelens_events_and_boundaries.md) | `7499c711968383fd` |
| `configs/R24_AHA_REPAIR_20261015.json` | [09 新表征与迁移](09_new_representations_and_transfer.md) | `c60c0384f95e331e` |
| `configs/R24_EXEC_20261015.json` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | `3fa6000c621be2cb` |
| `configs/R25_AHA_BF16_K085.json` | [09 新表征与迁移](09_new_representations_and_transfer.md) | `429584e4ebb5afbf` |
| `configs/R26_DIAG_20261016.json` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | `a3c8124942e29104` |
| `configs/R27_CENTER_WINDOW_20261016.json` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | `9f232e8bf70cefc7` |
| `configs/R28_SPATIAL_PUBLIC_PILOT.json` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | `52ee988dd66cb9e6` |
| `configs/R29_BREAKTHROUGH_20261016.json` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | `dd1953c196ecc952` |
| `configs/R29_NEXT_SUPERVISION_STAGE.json` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | `5741b1c028c40d4a` |
| `configs/R29_SPATIAL_GT_ASSISTED_DIAGNOSTIC.json` | [05 时间决策与4B后训练](05_temporal_decision_and_qwen4b.md) | `760c095523464c30` |
| `configs/R30_DYNAMIC_CONTENT_PROBE_20261016.json` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | `a6fdffa1d0cb91da` |
| `configs/R31_LOWDIM_EVENT_PROBE_20261016.json` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | `5290daad26fb10c1` |
| `configs/R42_ANNOTATION_QC_V13.json` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | `a7c257c63178a476` |
| `configs/R42_SMOKE_QC_V1.json` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | `4f1bd615ba64d817` |
| `configs/R43_IV2_EVENT_COMPLETION_20261017.json` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | `9251001f46265e8c` |
| `configs/R44_HUMAN_CALIBRATED_RESIDUAL_ETA020_B3.json` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | `874d239946c46347` |
| `configs/R44_OBSERVABILITY_UTILITY_20261017.json` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | `87d3afa2eb9cebe0` |
| `configs/R45_BOUNDARY_GUARDED_RESIDUAL_B3.json` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | `a6895e3f7592b6b1` |
| `configs/R45_HIGHFREQ_SEMANTIC_RESIDUAL_B3.json` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | `148b8c6e677dcff7` |
| `configs/R46_HIGHFREQ_LOCAL_RESIDUAL_B3.json` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | `2d11a10bb5320221` |
| `configs/R48_CONSENSUS_GATED_RESIDUAL_B3.json` | [02 视频表征与时间监督](02_video_encoders_and_temporal_supervision.md) | `6963ede3822fd66e` |
| `configs/R70_SPATIAL_UTILITY_PILOT.json` | [08 时空联合效用](08_joint_spatiotemporal_utility.md) | `1d450c5a233c727c` |
| `configs/R84_COORD_CONSISTENCY_SPATIAL_R54TIME.json` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | `27f40bd04b0d5bd1` |
| `configs/R86_T5_CROPHEAD_V3_R54TIME.json` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | `189e050b0d968ae0` |
| `configs/R87_AHA_PHD_PILOT.json` | [09 新表征与迁移](09_new_representations_and_transfer.md) | `922a4a30ce461e83` |
| `configs/R88_T5_H1_B3POINTS_R54TIME.json` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | `34f8a47bcc697bb0` |
| `configs/R89_LOWLEVEL_COMPOSITION_R54TIME_B3.json` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | `583fe6702b9afc05` |
| `configs/R90_TEMPORAL_STRUCTURE_MOTION_RANK.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `a165de9e14f7b74a` |
| `configs/R91_LOWPARAM_RESNET18_B3_V3.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `dad4f9eaa915697a` |
| `configs/R94_R95_VISUAL_RESEARCH_20261019.json` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | `06e868fb11314c51` |
| `configs/RG_DEV_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `80e466e6d309783f` |
| `configs/RG_FAILURE_FIGURES_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `db224c9fb05581ca` |
| `configs/RG_LINEAR_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `8c794c02e580f092` |
| `configs/RG_NCV_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `a532d9cfcb341bc5` |
| `configs/RG_NCV_SEEDS_002.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `0afd0c971eb26b21` |
| `configs/RG_RANK_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `509759078abe94e2` |
| `configs/SPATIAL_CONFIRM_001.json` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | `e00082d02c3d121a` |
| `configs/SPATIAL_GROUP_003.json` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | `358946514fced99b` |
| `configs/SPATIAL_GROUP_SCORE_003.json` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | `af18bf136817de66` |
| `configs/SUBJECT_SCALE_V1.json` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | `fd7923f26e2f6b66` |
| `configs/SUBJECT_SCALE_V1_Q2.json` | [07 空间构图与路径](07_spatial_grounding_and_composition.md) | `715b09cae874b0c8` |
| `configs/SUMME_BASELINES_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `8e4c53a39062366a` |
| `configs/SUMME_CV_OOD_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `f3a3eacd84bb31f5` |
| `configs/SUMME_CV_OOD_002.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `beefce4dea9c83e1` |
| `configs/SUMME_EXPANSION_ACQUIRE_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `3fc788d9a36bf0a8` |
| `configs/SUMME_LINEAR_OOD_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `163dfaffe1848f82` |
| `configs/SUMME_OOD_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `5a17e2e7a64cea72` |
| `configs/SUMME_OOD_002.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `577ada815c3dae9d` |
| `configs/SUMME_OOD_003.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `06f1e07870c989f3` |
| `configs/SUMME_RANK_OOD_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `746249af59963d29` |
| `configs/T4_TEMP_BOUNDARY_V3_PREREG.json` | [03 32B教师](03_qwen32b_teacher.md) | `3cddfbc554606f55` |
| `configs/T5_CROPHEAD_V3_PREREG.json` | [03 32B教师](03_qwen32b_teacher.md) | `584d751a206a4a12` |
| `configs/TEACHER_V2_PREREG.json` | [03 32B教师](03_qwen32b_teacher.md) | `4a85d8a83c6824d5` |
| `configs/TEMPORAL_RECOVERY_V1.json` | [13 数据标注与评价工程](13_data_annotation_and_evaluation.md) | `166053c836652a9a` |
| `configs/TSM_internal_full_001.json` | [01 轻量基线与传统特征](01_classic_visual_baselines.md) | `fc50b2d35e9db0a3` |
| `configs/V4_BASELINE_MANIFEST.json` | [03 32B教师](03_qwen32b_teacher.md) | `8312a7b845c0c090` |
| `configs/V4_E1_OBS025_PREREG.json` | [03 32B教师](03_qwen32b_teacher.md) | `2b28352c4f9fd169` |
| `configs/V4_E2_CONTEXT_PREREG.json` | [03 32B教师](03_qwen32b_teacher.md) | `2263b13e14aa70eb` |
| `configs/V4_E3B_DROP_ONLY_PREREG.json` | [03 32B教师](03_qwen32b_teacher.md) | `01149e87ce1914c3` |
| `configs/V4_E3_EVENT_PREREG.json` | [03 32B教师](03_qwen32b_teacher.md) | `d3b73136874b0e01` |
| `configs/V4_E4_CROP_RERANK_PREREG.json` | [03 32B教师](03_qwen32b_teacher.md) | `302364839978d614` |
| `configs/V4_S1_INTERP_PREREG.json` | [03 32B教师](03_qwen32b_teacher.md) | `d54b269d76f9abce` |
| `configs/V4_S2_MEDIAN3_PREREG.json` | [03 32B教师](03_qwen32b_teacher.md) | `fc8fdff3fb868e24` |
| `configs/V5_BASELINE_MANIFEST.json` | [03 32B教师](03_qwen32b_teacher.md) | `55cf7eb5a8c34227` |
| `configs/V5_H3_VISUAL_PREREG.json` | [03 32B教师](03_qwen32b_teacher.md) | `eb9ee068f45f3797` |
| `configs/V5_P0_XRERANK_PREREG.json` | [03 32B教师](03_qwen32b_teacher.md) | `ba5f1d1767ebfad2` |
| `configs/V6_NEXT10H_PREREG.json` | [04 LFM空间学生](04_lfm_spatial_students.md) | `439cde2a02e3133c` |
| `configs/V7_LFM_PREREG.json` | [04 LFM空间学生](04_lfm_spatial_students.md) | `d21f5434ff3d772c` |
| `configs/VALEN_2B_SFT_CROP_V1.json` | [09 新表征与迁移](09_new_representations_and_transfer.md) | `a2b08dfd5372c5ae` |
| `configs/VALEN_2B_SFT_UTIL_T006.json` | [09 新表征与迁移](09_new_representations_and_transfer.md) | `6663f1762341650e` |
| `configs/VALEN_HIGHLIGHT_S0.json` | [09 新表征与迁移](09_new_representations_and_transfer.md) | `e50add5a7321f9d6` |
| `configs/VIDEOORA_FULLCROP_IOU1.json` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | `343727e1372824c5` |
| `configs/VIDEOORA_FULLCROP_MULTIRATIO_V1.json` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | `d4cd193d80dcb1ba` |
| `configs/VIDEOORA_GEOMETRY_HEAD_V1.json` | [12 Video-ORA空间完整框](12_videoora_spatial_fullcrop.md) | `85e3701f97d40749` |
| `configs/v9_temporal_components.json` | [04 LFM空间学生](04_lfm_spatial_students.md) | `64a6ed7608748b99` |

## 复核边界

此回顾未启动新训练、GPU推理或比赛上传，也未重新核验官方登录态/全部远程权重。2026-11-06跨主机审计是已保存证据；2026-11-07直接核对的是工作区文档、登记、JSON和产物身份记录。原始实验环境、启动命令、输入清单和checkpoint位置按各报告追溯。
