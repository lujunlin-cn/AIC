# R12 计划 v2：前沿调查注入（复赛 34→45，帧级排序主攻）

日期：2026-10-07 晚。前置：`reports/20261007_round12_r45_plan.md`（v1 核实矩阵与策略转向仍有效）。
触发：用户指令——环境已选「干净重建」（整机重启中）；调查前沿怎么做，核心解决排序瓶颈。
术语锚点：卡 34 分的瓶颈是**帧级（时间轴）排序/keep 决策**（评分公式分母项）；空间裁剪轴已
饱和/证伪（159/426 无 slide、F8 三连败），本计划不涉及。

---

## 1. 前沿调查结论（2025–2026）

### 1.1 决定性发现：官方任务 = PHD-GIFs
- 复赛 426 集「GIF 高光任务、PHD² 域」的学术源头 = **PHD-GIFs: Personalized Highlight
  Detection for Automatic GIF Creation（ACM MM 2018）**——即我方 427GB 在盘 PHD² 数据集
  本尊。监督形态=**用户行为监督**（用户把哪段做成 GIF）。
- 后继谱系：Video2GIF（源已下线）→ PHD-GIFs → TRINITY（2026 多视角基准）。
- **P0 行动：官方 426 index 与在盘 PHD² 源的 YouTube ID 对接审计**——若命中子集，其原始
  GIF 标注就是同任务合法训练数据（V10 已证初赛 174 集不对接，复赛 426 集未测）；
  同时全盘点 PHD² 原始标注字段（GIF 区间/个性分/多标注者），我方当前只用了 GIF-union 区间。

### 1.2 排序 SOTA 的四个技术要素（对照我方 34 分配方）

| 要素 | 前沿做法 | 我方现状 | 差距判定 |
|---|---|---|---|
| **训练损失** | pairwise/listwise ranking（Mr.HiSum 原生 multiple pairwise ranking，NeurIPS 2023）；negative contrastive（CG-DETR/SHINE）；多实例排序（MINI-Net） | MSE 回归 in-interval 二值标签 | **从未测过 ranking 损失** |
| **时序架构** | 全局 self-attention over 逐帧/clip 特征（Moment-DETR→QD-DETR→CG-DETR→SG-DETR WACV26；HM-Net Mamba 2026） | TCN RF=63 局部因果卷积；V10 十变体全是 pooling 变体，**无全局时序 attention 变体** | 高光=视频内相对排序，全局建模缺口 |
| **主干特征** | VLM 微调（AHA：Qwen-7B+轻量解耦头，NeurIPS 2025，Mr.HiSum **+8.3% mAP SOTA**、TVSum +5.9% 超全部 offline 方法与 VLM）；VideoMAEv2 | SigLIP 图像塔 pooled（R9 slot 读出已证表示墙；LoRA 900 步判负） | AHA 权重 Apache-2.0 开源、推理 24GB 级 |
| **零样本路线** | LLM 生成高光文本描述符 + CLIP 逐帧匹配（2025，TVSum τ=0.217 超人类基线 0.176，零训练） | 未测 | 零成本、当天可出平台分 |

### 1.3 在盘未激活的监督资产（全部零下载成本）
1. **Mr.HiSum 27,877 新闻视频**（V6 时代只做过特征复现验证，**原生 pairwise ranking 用途从未启用**）；
2. **QVH saliency**：22 标注者 0–4 连续分，7,100 视频（v1 计划已列）；
3. **PHD² user-vote**：user 字段被聚合丢弃（V10 census 实锤），user-level 聚合从未测；
4. **PHD² 原始 GIF 标注**（P0-a 对接后）。

## 2. R12 v2 机制包（平台差分归因，单变量×2 变体）

### 阶段 0：本地准备（环境恢复后 Day-1–2，CPU）
- **P0-a PHD-GIFs 对接审计**（新，最高优先）：426 index × PHD2 源 ID 匹配；PHD² 原始标注
  字段盘点；PHD-GIFs 2018 论文配方提取。
- **P0-b Mr.HiSum 排序管线**：激活 27,877 视频 pairwise ranking 监督（特征链 V6 已复现过）。
- P0-c/QV、P0-d/user-vote：沿 v1 计划。

### 阶段 1：平台快扫（第 1–2 周，每包单变量×2）
| 包 | 机制（单变量） | 依据 |
|---|---|---|
| R12-LOSS | **MSE→multiple pairwise ranking loss**（Mr.HiSum 配方），其余同 B3-TCN | 排序损失是前沿标配、我方从未测 |
| R12-GATT | **TCN→全局 Transformer 排序头**（1Hz 550 token self-attn，轻） | 高光=全局相对判断；十变体矩阵无此变体 |
| R12-QV | QV saliency 主监督 keep 头 | v1 已列 |
| R12-KEEP | keep 0.80→0.60 | v1 已列 |
| R12-UV | PHD² user-vote 标签 | v1 已列 |
| R12-VMAE | VideoMAEv2 端到端 | v1 已列 |
| R12-AHA | **AHA 预训练权重**（Apache-2.0）：先在 Mr.HiSum 官方 h5 协议复现 +8.3% 验证权重有效，再 LoRA 对齐我方任务 | 重建后 910A 8 卡可承载（推理 24GB 级、LoRA 可选） |
| R12-CLIPZ | **零样本描述符**：LLM 生成「GIF 精彩时刻」描述符 × Siglip2 帧匹配 | 零训练当天出分；TVSum 已超人类基线 |

### 阶段 2：按平台差分收敛
- 正机制 → 全量扩训；损失+架构正交组合（ranking × global-attn × 最优监督源）；
- 校准/自适应 keep 解锁条件不变（概率型监督到位后兑现 Oracle A +0.116）。

### 阶段 3：VLM 判别蒸馏（环境已裁定重建，随恢复排期）
- Qwen-VL 逐段 pairwise「哪段更精彩」→ 蒸馏 keep 头；与 AHA 结论互证（VLM 主干+轻头是
  排序 SOTA 形态）。

## 3. 风险与止损（沿 v1，新增两条）
- AHA 的 Mr.HiSum 域=新闻长视频，向 14.1s GIF 短视频迁移存在域差——先复现再 LoRA，
  平台单变量验证。
- pairwise ranking 在 27,877 视频上的训练预算（910A CPU/NPU 混合）需先 smoke 计时再全量。
- 其余（噪声 ±0.88、2 变体归因、本地池仅预检）沿 v1。

## 4. 资源行动清单
1. 下载：AHA 权重（Google Drive，Apache-2.0）、PHD-GIFs 2018 论文、TRINITY 基准论文（域画像参考）。
2. 910A 重建完成后：ascend_teacher.env 配方验证 → R12-VMAE/AHA 链路 smoke。
3. 名额预算：阶段 1 约 16 个提交（8 机制×2），按每日 5 个可忽略执行。
