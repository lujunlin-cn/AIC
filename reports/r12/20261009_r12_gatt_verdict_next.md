# R12-GATT 回分分析与下一步计划（2026-10-09）

回分：**R12_GATT_LFM_SEMIFINAL = 35.17**，对照 B3_TEMPORAL 34.88 → **+0.29**（复赛口径）。
规则备注：比赛已取消 k_size（用户 10-09 明示），后续所有分析不再使用 k_size 口径。

## 1. 归因分析

1. **时间轴首次平台转正**。R12 之前时间轴全部改动从未在平台正过（v1 核实矩阵：
   「时间轴不是被证伪了，是从来没被平台测过」）。本次预测集差异 426/426 视频
   （keep 集全换）仍净 +0.29——不是噪声抖动能解释的持平扰动，是「全面换排序后
   整体不劣且局部更优」。
2. **+0.29 是打包效应**：相对 V9 TCN，打分器内部三个子变量同时变化（架构
   TCN→全局 Transformer、监督 GIF-distance→GIF-union slot keep、损失回归→pairwise）。
   内部拆解已有局部证据：P0-b 4-seed 判定 pairwise 59.17±0.38 > mse 54.00±1.60
   （同架构同特征同协议）。
3. **幅度与缺口**：+0.29 后 35.17，距 45 还差 9.83。keep0.80 掩码质量仅小幅提升
   （此前平台反推掩码质量≈比随机好 7%，45 分需 ~38% 量级）——单换打分器不够，
   监督源是下一级杠杆。
4. **资产沉淀（复利项）**：R12 推理链（temporal_feats 特征缓存→GATT 头→1s 整秒
   keep 掩码→组包脚本 `r12_gatt_package.py`）已打通并验证。后续任何「换监督重训」
   的边际成本 = 只换标签重训 + 一条命令组包。

## 2. 下一步计划（按杠杆排序）

### P1 监督源升级（主攻，预计最大单项增益）
- **R12-QV**：QV Highlights saliency（22 标注者 0-4 连续分，7,100 视频在盘，管线
  现成）换标签重训 GATT 头（pairwise，lfm 特征）。同链单变量=监督源。出 2 seeds。
- **R12-UV**：PHD² user-vote 标签（V10 census 实锤 upstream 有 user 字段被聚合丢弃；
  83.3% 单用户池）同链重训。出 2 seeds。
- 判定：平台差分 vs 35.17。正者进联合训练。

### P2 keep 深度交互（零训练快包）
- 排序质量已变的条件下，keep 0.80→0.60 / 0.70 各出一包（同一 predictions 改阈值，
  组包脚本 --keep-frac 即可）。v1 计划遗留自由度，成本≈0。

### P3 GATT 容量与特征增强
- d3→d6、epochs 50→150、pairwise 难例挖掘；lfm+vmae 特征拼接（中期 lfm+1.8 但
  vmae 含运动通道，互补假设）。
- 仅在 P1 出正后叠加（避免与监督源变量耦合）。

### P4 支线
- S2c 读出判定（今晚 ~20:00 扫描齐，cron 自动）：vs prior null 即关闭。
- AHA 线挂起（资产已落盘，优先级低于监督源轴）。

## 3. 冻结资产清单（压缩对话后凭此恢复）
- GATT best ckpt：`LFM_V11/r12_gatt/lfm_pairwise_npu_s1_best.pt`（dev_ap 67.70）
- 配方：LFM2VL pooled 1s keyframe + Transformer d512×3/RoPE + pairwise
- 组包：`scripts/r12_gatt_package.py`（--ckpt --feat lfm --keep-frac）
- 提交链：`submissions/R12_GATT_LFM_SEMIFINAL/`（35.17）
- P0-b verdict：`r12_p0b/verdict.json`（pairwise 59.17±0.38 vs mse 54.00±1.60）
