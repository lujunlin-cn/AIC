# R12 计划 v3：帧级排序前沿深挖（TRINITY/ECCV 2026 实测表 + 排序监督新证据）

日期：2026-10-07 深夜。前置：`20261007_round12_r45_plan.md`（v1）、`20261007_round12_frontier_addendum.md`（v2）。
触发：用户指令「再检索帧级排序相关的前沿成果」。本报告只收帧级排序（frame-level
importance ranking）直接相关证据，全部给出原文级数字。

---

## 1. 决定性新证据：TRINITY（ECCV 2026，arXiv 2608.29577）

**多视角个人风格高光基准**（Event/Emotion/Nature 三视角，196,070 视频源池，
CLIP 特征+标注已上 HuggingFace：vanilladucky/TRINITY）。其 Table 2 是目前
**Mr.HiSum 帧级排序最全的 fine-tuned SOTA 对比**：

### 1.1 Mr.HiSum 排行榜（mAP@ρ=15% / ρ=50%）

| 方法 | 架构/监督要点 | ρ=15 | ρ=50 |
|---|---|---|---|
| Mr.HiSum 官方基线 | pairwise ranking，YouTube-8M 帧特征 | 50.80 | 67.20 |
| SummDiff | 扩散生成式帧打分 | 33.83 | 65.44 |
| Aha（NeurIPS 2025） | VLM 在线因果排序 | 32.66 | 64.19 |
| PGL-SUM / HierSum | 全局+局部注意力 | 27.45–32.60 | 61.60–63.80 |
| SAVS-Net | 通道注意力编解码 | – | 57.12 |
| **TRINITY (Ours)** | **冻结 CLIP-B/32 + Transformer(d5,h512,1D RoPE) + BCE** | **40.98** | **69.06** |

三个直接可用的结论：
1. **v2 对 AHA 的定位需要修正**：Aha 的 +8.3% 是对「其他 VLM/在线方法」的 SOTA；
   在 Mr.HiSum 上它**低于专用监督基线**（64.19 vs 67.20）。我方是离线提交，不受
   在线约束——**离线排序头应以 TRINITY 式配方为主，AHA 权重降级为 LoRA 初始化
   备选**（R12-AHA 优先级下调）。
2. **最强公开配方与我们的管线同构**：冻结 CLIP 特征（Siglip2 可直接替换）→
   浅 Transformer 全局编码（depth 5、hidden 512、**1D RoPE**）→ 逐帧分数。
   这就是 R12-GATT 的现成参考实现，且**损失用等权 BCE 而非 ranking loss 拿下
   最优**——R12-LOSS（pairwise ranking）仍是独立单变量假设（官方基线用 ranking
   在 ρ=50 上 67.20 仍第二强），两个包都要测。
3. **标签池化保峰**：帧级分数聚合到 5s clip 时「含峰 clip 保留峰值分」（防峰稀释、
   保段间相对排序）——直接适用于我方 in-interval 二值标签 → 帧分/段分的聚合口径。

### 1.2 TRINITY 训练策略的可迁移教训
- 多视角分支 = **4 共享 + 4 私有注意力头**；round-robin 逐任务批次 + 梯度累积；
  单骨干顺序训练灾难性遗忘；**MixBatch（批内混视角）使 Nature mAP15 58.74→42.19
  （暴跌）**——若做辅助视角联合训练，批次必须按视角隔离。
- 三任务等权 BCE 优于 uncertainty 加权与手动调权（简单配方赢）。

### 1.3 Emotion/Nature 视角 = 两个廉价的辅助监督发生器（新增机制包候选）
- **Emotion 视角**：EmotiEffLib 逐帧表情（非中性连续 ≥3 帧才算峰）+ Qwen-VL 复核
  → 二值情感峰值标签。我方有现成人脸管线（YuNet/S2a det cache 10,530 帧），CPU 可产。
- **Nature 视角**：人物占比过滤（YOLO θ=0.3）+ CLIP 风景 prompt 相似度 + 美学回归标签
  → 美学显著段。
- PHD²/GIF 域高光大概率富含「人物/情感」时刻：**情感峰值辅助视角与主任务联合训练
  （共享+私有头、按视角隔离批次）**是 v3 新增的低成本机制包（R12-EMO），单变量=
  「+emotion-peak 辅助监督」。
- Event 视角即 Mr.HiSum replay 统计——证实 Mr.HiSum 帧标签本质是 YouTube
  「Most Replayed」热度曲线，与「用户行为监督」同族，支持 P0-b 激活。

## 2. 排序监督的跨域新证据（谨慎引用）

- **FrameRepeat（arXiv 2603.16256，VideoQA 域，非高光）**：用 MLLM 输出概率作为
  帧重要性监督信号（Add-One-In）训练轻量帧打分网络，配 pairwise ranking loss；
  结论「ranking loss 提供回归之外的成对互补监督」。**注意**：任务是 VQA 推理增强，
  不是高光检测；对本项目的价值是**阶段 3 VLM 判别蒸馏的旁证**（Qwen 判别概率→
  蒸馏 keep 头 + ranking loss 的组合形态）。
- 帧级 pairwise ranking 谱系源头：Yao et al. 2016（第一视角高光成对深排序，
  比 RankSVM +10.5%）→ Mr.HiSum → TRINITY 基线。R12-LOSS 配方沿 Mr.HiSum
  官方实现（在盘 27,877 视频 h5，V6 特征链已复现）。

## 3. QV saliency 臂（R12-QV）的架构参考更新

- **MS-DETR（ACM MM 2025，arXiv 2507.12062，代码 github.com/snailma0229/MS-DETR，
  被引 28）**：运动-语义联合学习做 MR+HD，QVHighlights 强基线——R12-QV 头设计的
  现成参考；其「运动分支」提示我方纯外观特征缺运动通道（VideoMAEv2 端到端可补）。
- **DualGround（NeurIPS）**：短语级+句子级结构化接地，QVHighlights MR+HD 双 SOTA
  ——query 分解思想可用于 CLIPZ 描述符的短语级匹配。
- **SpikingVTG**（SNN+显著性反馈门）：910A 不适配，仅记录。
- Mamba/SSM 线：VSumMamba（ACM MM 2025）、协同 SSM 联合 MR+HD、Video Mamba Suite
  ——线性复杂度全局序列建模，作为 R12-GATT 的变体 2 候选（550 token 量级
  Transformer 已够，SSM 仅在扩到 1Hz 全片 >2k token 时才必要）。

## 4. 个性化监督线（R12-UV 的谱系与边界）

- PR-Net（ICCV 2021，偏好推理）、User-specific highlights（ACM MM 2022，
  arXiv 2207.08352，用户历史高光条件化）、HIPPO-Video（2025，LLM 模拟观看历史）
  ——「按用户条件化排序」是成熟方向；PHD² 83.3% 单用户池的限制下，R12-UV 的
  user-vote 聚合仍是最近的可测形态。
- 社交文本侧 2026 证据（Glasp，2606.09024/2606.10398）：个性化增益主要在
  「选择」而非「显著性」——提示 user-vote 的增益上限可能有限，预期按灰区包管理。

## 5. 对 R12 机制包矩阵的修订（在 v2 八包基础上）

| 包 | 修订 |
|---|---|
| R12-GATT | 参考实现改指 TRINITY 配方：冻结 Siglip2 512d → Transformer d5 h512 + 1D RoPE → 帧分；标签池化用**保峰聚合** |
| R12-LOSS | 维持；与 GATT 正交组合；Mr.HiSum 官方 pairwise 配方 |
| R12-AHA | **优先级下调**：Aha 在 Mr.HiSum 低于监督基线；改为「AHA 权重 LoRA 初始化」备选，主路线换 TRINITY 式离线头 |
| R12-EMO（**新增**） | 情感峰值辅助视角（EmotiEffLib/CPU 人脸管线产标签）+ 共享/私有头联合训练；批次按视角隔离（MixBatch 反例已录） |
| P0-b | 评测协议对齐 TRINITY：Mr.HiSum 官方 split、报 mAP@ρ=15/50，使复现数字可与公开榜直接比对 |
| R12-CLIPZ | 可借 DualGround 短语级分解；其余不变 |
| R12-QV/KEEP/UV/VMAE | 不变 |

## 6. 资源行动清单（增量）
1. 下载：TRINITY 数据/特征（HF vanilladucky/TRINITY）、MS-DETR 代码、
   EmotiEffLib 权重（CPU 推理）、（备选）AHA 权重。
2. P0-b 复现目标：TRINITY 式配方在 Mr.HiSum 官方协议 ρ=15/50 上对齐公开数字
   （40.98/69.06 量级），再迁我方域。
3. R12-EMO 标签生产：先在 live 池 smoke 情感峰值覆盖率，再全量。

## 7. 来源
- TRINITY: arxiv.org/abs/2608.29577（ECCV 2026）；数据 huggingface.co/datasets/vanilladucky/TRINITY
- FrameRepeat: arxiv.org/abs/2603.16256
- MS-DETR: arxiv.org/abs/2507.12062；github.com/snailma0229/MS-DETR
- Aha: arxiv.org/abs/2509.16421；github.com/aiden200/Aha-
- Mr.HiSum: github.com/MRHiSum/MR.HiSum（NeurIPS 2023）
- Yao et al. 2016 pairwise deep ranking（ECCV）；
  PR-Net arxiv.org/abs/2109.01799；User-specific arxiv.org/abs/2207.08352；
  HIPPO-Video（2025）；Glasp 2606.09024 / 2606.10398；VSumMamba（ACM MM 2025）
