# DINOv3 ViT-B/16 视频高光检测：基于 AIC 仓库现状的增量训练与验证方案

> **版本**：v2.0（替换原《从零构建训练方案》）  
> **修订日期**：2026-10-20  
> **审计对象**：[lujunlin-cn/AIC](https://github.com/lujunlin-cn/AIC)，`main` 分支快照 `d86f0900590842bb170883b52dd3fe4bca150fc9`（2026-10-19 16:52:57 UTC）  
> **状态**：指导性研究与实施方案。**DINOv3 ViT-B/16 在本项目上的训练结果尚未验证。**  
> **范围**：单条模型推理管线，包含共享视觉编码器、时间评分头、裁剪评分头；**不研究集成学习**。  
> **原则**：复用已完成资产，仅训练能够回答新问题的变量；公开代理指标、官方回分、尚未验证的假设严格分开。

## 0. 修订结论：不是重做 E0–E6，而是回答三个尚未回答的问题

选择的视觉骨干仍为 **DINOv3 ViT-B/16**：`facebook/dinov3-vitb16-pretrain-lvd1689m`，约 **86M** 参数，768 维视觉表示，patch size 为 16。这里的“从零训练”指**重新训练本比赛的任务头**，不指随机初始化并预训练 DINOv3 骨干。

**本项目的真实前提与旧版文件不同：**

1. **赛方不提供训练集，也未公开复赛测试标签。**只能使用有明确授权的公开视频、公开视频标注、自建合法训练数据和公开预训练权重。不能写“在官方训练集上精调”“在官方验证集上做 Oracle IoU”等不存在的数据依赖。
2. **当前评分为 `Score = 100 × F_video`，不再乘 `k_size`。**全部实际推理参数不超过 **9B**；新提交记录 `model_params_b`。历史 `model_size_mb` 不能作为新版提交字段。
3. **仓库已经运行了多个时序网络、DINOv2 空间读出、传统多视图、空间候选排序、高频残差和部分骨干解冻实验。**重跑原配方并不能验证 DINOv3 是否提供新的表征增量。
4. 本仓库已经有可工作的完整视频解码、帧号映射、裁剪候选、固定预算选帧、训练与打包链路。**不得新建一个与 `aic/`、`scripts/` 并行的孤立项目，除非现有实现确实不能复用。**
5. 当前已记录的较好复赛包是 **R71 = 36.43、R70 = 36.42、R54 = 36.22**。这些分数在仓库中注明为**用户报告的官方平台反馈**，上传文件与平台回分的逐字节关联未全部独立核验。不同阶段成绩不可混比。参考：[STATUS.md](https://github.com/lujunlin-cn/AIC/blob/main/STATUS.md)。

**DINOv3 后续只优先回答三个问题：**

- **问题 T（时间表征）**：冻结 DINOv3 的单帧特征，是否比现有 ResNet18 / DeiT-S / 现有视觉特征，在**同一 PHD2/GIF 监督和同一时间头**下产生新的源外高光排序信息？
- **问题 S（空间表征）**：冻结 DINOv3 的 patch tokens，是否比当前 R70 空间候选特征，在**相同候选框、相同原生 crop 标注、相同评分头容量**下提高框选择 IoU？
- **问题 F（精确帧）**：只有问题 T 通过，并且能取得覆盖充分的**原始帧级有效监督**时，局部密采样与相邻帧精排是否进一步改善严格原始帧选择？如果没有对应监督，该实验只能是机制诊断，不能宣称精确帧定位能力提升。

因此首轮训练顺序为 **T 与 S 独立并行验证 → 条件式 F → 条件式骨干适配**。时间选择与构图裁剪是同一推理系统的两个任务模块，并不是模型集成。

---

## 1. 现行官方任务和评分契约（必须固定）

项目内经 2026-10-16 核查的规则基线：[`01_赛事规则与评分标准.md`](https://github.com/lujunlin-cn/AIC/blob/main/01_%E8%B5%9B%E4%BA%8B%E8%A7%84%E5%88%99%E4%B8%8E%E8%AF%84%E5%88%86%E6%A0%87%E5%87%86.md)。早期历史说明涉及按文件字节数计档或 `k_size` 的部分，不用于当前复赛。

### 1.1 输入和输出

- 输入为原始视频、`video_id` 以及目标画幅比 `targetRatioWH`，例如 `[16,9]` 或 `[9,16]`。
- 输出为该视频的 `predictions` 列表。每个预测包含原始 `frame` 索引和单个构图框 `bboxes: [x,y,w]`。无高光允许 `predictions: []`。
- `frame` 为**原视频从 0 开始的帧序号**，不是抽帧序号、时间秒数或 GIF 内部序号。
- `x,y,w` 采用原图像素几何；对应框高由目标宽高比推导：

\[
  h=w\frac{\mathrm{target\_h}}{\mathrm{target\_w}}.
\]

必须满足 `x>=0`、`y>=0`、`w>0`、`x+w<=W`、`y+h<=H`，并避免重复帧。数值舍入、VFR 与旋转元数据的严格平台行为仍须以可获得的官方评测实现为准。

### 1.2 当前官方得分

设视频 \(v\) 的预测帧数为 \(N_{p,v}\)、参考帧数为 \(N_{g,v}\)。对**同一个 `video_id` 且精确相同的 `frame`**，计算该帧预测框与参考框的连续 IoU。令这些 IoU 之和为 \(S_v\)。则：

\[
 F_v=\frac{2S_v}{N_{p,v}+N_{g,v}},\qquad
 F_{\mathrm{video}}=\frac{1}{V}\sum_{v=1}^{V}F_v,\qquad
 \mathrm{Score}=100F_{\mathrm{video}}.
\]

当 `N_pred=N_gt=0` 时，按规则 `F_v=1`；仅一者为 0 时，`F_v=0`。没有公布“临近几帧也算匹配”的容忍带。

**重要含义：**

- 时间错误使该帧 IoU 贡献为 0，且错误预测仍增加 `N_pred`。
- 空间错误会直接降低已命中帧的 IoU 贡献。
- 每个视频等权平均；不能改为按帧数微平均。
- **固定 `keep=0.85` 是目前若干既有系统/代理实验的选择策略，不是官方强制规定，也不是已验证的最优选帧预算。**首轮为控制变量可以沿用既有预算，但不得误写为官方规则。
- 只在公开训练/验证数据具有时间和框的**联合真值**时，才能离线计算对应的联合 F。PHD2 的选择区间和 RetargetVid 的 crop GT 通常不在同一组视频，不能拼接后冒充联合真值。

### 1.3 推理模型参数与测试集使用

- **全部学习型推理组件总参数量 ≤ 9,000,000,000。**冻结骨干、辅助检测器和裁剪头均计入；量化不能降低参数个数。共享 DINOv3 权重只计一次。
- 使用 `model_params_b = total_inference_parameters / 1e9`，同时记录权重字节数、SHA256、精度、推理耗时。
- 比赛测试视频只做合规的冻结推理与提交。**不得把测试视频、平台分数或由分数反推出的标签用于训练、伪标注、逐样本调整或公开外传。**
- 仓库中 `aic/evaluation.py` 的历史评测路径仍保留旧 `size_coefficient`/阶段逻辑，不能直接把历史 `final` 分支当作当前复赛评分器。需要基于现行公式编写独立回归测试，并明确记录“本地公式重建，未取得官方同版本可执行 evaluator”。

---

## 2. E0–E6 仓库审计：做过什么，哪些不能重复

下表的 E0–E6 是**讨论中拟定的新实验名称**，不是仓库里已有的 `V6_E0` 等历史编号。审计包含 Git 提交的脚本、报告和实验注册表；没有重新运行远端实验，也没有核验所有远端未提交 checkpoint。

| 设想实验 | 已有接近的仓库实验 | 可核验的结果或事实 | 新方案决策 |
|---|---|---|---|
| **E0 传统画质/低层特征排序** | R31、R89、R90、R92 | R89 低层空间特征在 RetargetVid/LIVE Confirm 上相对中心 `+0.004104` IoU；R90 低维 bank 简单拼接无可靠时间增益；R92 多视图时间模型相对 RGB 为 `−0.000260`，CI 跨 0 | **不重复**低层标量 bank、直接帧差或既有多视图 gated fusion |
| **E1 冻结视觉骨干 + 小头** | A0、ViT-B/16、DeiT-S、R24、V5-H3 | 训练/评估过冻结 CNN/ViT；V5-H3 冻结 **DINOv2 ViT-B/14** 做空间排序，未晋级。**这不是 DINOv3。** | 新增“DINOv3 ViT-B/16 的同条件时间与空间表征对照” |
| **E2 多尺度 TCN** | V7 TCN、V9 PHD2 TCN、R97、R99、R102 | R97 的事件状态辅助头相对 RGB TCN `−0.000780`；R102 的 AIKU→PHD2 时序迁移 `+0.003413`，但不可直接部署 | **复用 TCN**；只把 DINOv3 表征作为主变化量 |
| **E3 独立裁剪排序头** | V5-H3、T5 CropHead、R70、R78/R81、R105 | R70 在公开原生 crop 数据上已有候选读出；topsoft 相对中心宏平均 IoU `+0.07006`；R105 saliency 特征相对控制 `+0.0035215`，但后续校准存在过拟合 | **复用 R70 的 129 候选与标注协议**；只替换为 DINOv3 patch 特征 |
| **E4 高频局部精排** | R45、R46、R47、R93 | 4 FPS 残差、边界和事件区间方法已有完整候选或诊断；多个包没有登记官方回分 | **不重复**4 FPS 差分/端点残差；若重新启动，必须有新帧级监督和明确不同的机制 |
| **E5 骨干部分解冻/迁移** | `QVH_FINETUNE_PROBE_20260926` | DeiT-S 在 37 train / 9 val 视频、2 epoch 条件下，冻结验证 BCE `0.045730`，解冻最后一层 `0.046889`；本次无收益 | 只能推断**该 DeiT 配方**未改善。DINOv3 解冻要等新监督及冻结基线过门后再做 |
| **E6 SigLIP2 语义分支** | V9 vision-only、R24 CLIP | V9 空间系统已经加载 SigLIP2 vision tower；R24 CLIP 对照不等于 SigLIP2 时间专家。R12-GATT 的 pooled 特征后被纠正为 LFM2VL，不能记成 SigLIP2 成功案例 | **本方案不新增语义专家或集成分支**，不重复现有 SigLIP2 使用路径 |

### 2.1 不能跳过的已知负结果

- [R92 多视图](https://github.com/lujunlin-cn/AIC/blob/main/reports/r92_multiview_traditional_route_20261019/summary.md)：公开 PHD2/GIF，5,886 个片段；固定 85% 集合 F：RGB `0.415595`，RGB+低频+高频+运动 `0.415335`。源级 CI 为 `[-0.003184,+0.002589]`。原配方不晋级。
- [R97 事件状态 TCN](https://github.com/lujunlin-cn/AIC/blob/main/reports/r97_event_state_tcn_20261019/summary.md)：控制 `0.416244`，辅助状态 `0.415464`，差 `−0.000780`，不能把区间前/中/后监督直接视作提分机制。
- [V5-H3 / V6-E0 复核](https://github.com/lujunlin-cn/AIC/blob/main/reports/20260929_v6_e0_evidence_audit.md)：DINOv2 候选评分头已真正训练，但在 RetargetVid 开发集的预注册条件未通过，不出包。不能称“DINO 系列已经验证有效”。
- [R101 邻帧视觉上下文](https://github.com/lujunlin-cn/AIC/blob/main/reports/r101_spatial_resnet_context_20261019/summary.md)：ResNet18 邻帧 interaction 对 native crop IoU 为 `+0.000197`，CI 跨 0。不要简单再叠一个邻帧 pooled 分支。
- [R86/R88 映射错误](https://github.com/lujunlin-cn/AIC/blob/main/reports/r86_r87_observation_and_transfer_20261019/summary.md)：R86 官方 34.88 低于 R54 36.22，但其点源错误地由 B3 换成 A1raw，因此**不能**把 R86 下降归因给 T5 CropHead 本身。新实验必须检查父链输入身份。

### 2.2 可以复用的正向线索，但不能跨域折算

- [R70 topsoft 空间候选](https://github.com/lujunlin-cn/AIC/blob/main/reports/r70_topsoft_objective_20261018/README.md)：真实公开 crop 数据上可读出较强的裁剪候选效用。官方用户报告 R70 36.42 较 R54 36.22 高 0.20，但**不是** `+0.07006 IoU × 100` 的直接折算。
- [R102 AIKU 迁移](https://github.com/lujunlin-cn/AIC/blob/main/reports/r102_aiku_transfer_20261020/summary.md)：仅使用 600/6,000 个 AIKU group，时序 trunk 迁移到 PHD2/GIF 的 448 源留出代理 F 产生 `+0.003413`，95% CI `[+0.00080,+0.00610]`。由于 AIKU 发布的是 1024D 预提取特征，不是可重建的原始视频输入，此结果适合作为**时序头初始化的条件实验**。
- [R105 AViMoS 空间 utility](https://github.com/lujunlin-cn/AIC/blob/main/reports/r105_avimos_aic_utility_20261020/summary.md)：原生 crop 开发集相对控制 `+0.0035215`，置换来源后消失；但 tune/hold 的简单混合校准明显过拟合。不得直接作为官方涨分声明。

**审计结论**：需要验证的是 DINOv3 的**新视觉表示是否能够产生额外可泛化信息**，而不是重新证明 TCN、候选框排序和图像滤波可以写出来。

---

## 3. 数据方案：只使用真实存在、可合法取得的监督

| 数据源 | 仓库现状/公开入口 | 可提供的监督 | 本方案用途与限制 |
|---|---|---|---|
| **[PHD2 / 用户 GIF 选择区间](https://github.com/gifs/personalized-highlights-dataset)** | 已有 `ext_data/scripts/ingest_phd2.py`、`scripts/v9_phd2_tcn.py` 及 R92 缓存与训练记录；[原始论文](https://arxiv.org/abs/1804.06604)；数据/媒体许可与版本仍需核对 | 被用户选择的 GIF 时间区间，部分 soft 距离或 union 标签 | **时间主训练源**。未选择不自动等于“确认非高光”；8 帧片段标签不是官方精确帧 GT |
| **YouTube Highlights** | 已有 `ext_data/scripts/ingest_youtube_highlights.py`、公开项目 [DomainSpecificHighlight](https://github.com/aliensunmin/DomainSpecificHighlight) | 类别相关的视频高光片段标注 | 时间跨来源确认或补充训练。先核对原视频下载、精确 PTS、标注转换和同源泄漏 |
| **TVSum** | 已有训练脚本、原视频及源分组协议；[官方项目](https://github.com/yalesong/tvsum) | 50 个视频的分段摘要重要度 | **辅助对照/迁移评估**。摘要重要度不等于 GIF 高光，也不是精确帧标签；使用前再次检查许可证与数据协议 |
| **[AIKU x brAIn 2025](https://www.kaggle.com/competitions/aikubrainvideosummary/data)** | R102 已审计 `6,000` 训练 group；本项目使用 `600` group 做 pilot；Kaggle 页面标记 CC BY-NC-SA 4.0，比赛使用前需核查许可 | 1024D 预提取特征、`gtscore`、`gtsummary`、change points | 只能用于**时序 trunk 预训练/迁移**；不能将这些特征输入 DINOv3 图像编码器，也不能声称 AIKU 源域 DINOv3 已训练 |
| **[RetargetVid](https://github.com/bmezaris/RetargetVid)** | 已有原生裁剪数据接入，R70 在 source-disjoint 数据上训练；`ext_data/` 有数据管线 | 原生空间裁剪框或相应裁剪偏好 | **空间主训练/验证源之一**。注意方向、画幅比、真实标注窗口与比赛差异 |
| **[LIVE-YT-VideoCropping（LIVE）](https://github.com/steven413d/LIVE-YT-VideoCropping)** | R70、R101、R105 已使用公开原生 crop 表；仓库有接入脚本 | 原生 crop 区域监督 | **空间主训练/验证源之一**。部分 LIVE 数据是横屏裁竖屏，官方目标可能是反方向；必须分数据域报告 |
| **GAICv2** | [Grid-Anchor-based Image Cropping](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping)，当前 DINOv3 配方尚未完成本项目实验 | 图像候选裁剪框的 MOS 审美评分；论文记录 3,336 张图 | **可选预训练**；只在原生 IoU 头无法学好时考虑。MOS 不等于 AIC crop IoU，不是首轮必备 |
| **TRINITY** | [论文/数据](https://huggingface.co/datasets/vanilladucky/TRINITY)；公开版本主要是预提取 CLIP 特征 | Event/Emotion/Nature 等不同语义的时间显著性 | 主要借鉴设计和评价协议；不宣称可以直接从其特征训练 DINOv3 骨干 |
| **比赛样例与测试集** | 比赛官方提供索引、样例/测试集及规则；**没有带标签训练集** | 提交格式与未标注测试视频 | 只用于合规推理、完整性校验和平台评测；**不得参与训练或伪标签生成** |

### 3.1 现成数据与新模型的关键兼容检查

旧仓库的 512D ResNet18、768D LFM2VL、DINOv2、SigLIP2 或 1024D AIKU 特征，**不能重命名为 DINOv3 特征**。新提取任务必须从已经获准使用的**原始公开视频帧**开始，记录：

```text
dataset_id / source_group / video_sha256
original_frame_id / pts_seconds / raw_width / raw_height / rotation
backbone_id / backbone_revision / weights_sha256
processor_version / resize_mode / input_size / norm_signature
feature_type(global|patch) / patch_grid_shape / extraction_code_sha256
label_source / label_semantics / label_mask / split_id
```

对没有可下载原始视频的数据，例如当前 AIKU 特征包，不能强行执行 DINOv3 提取。若外部媒体版权/数据协议无法满足使用要求，标记 `license_gate=blocked`，不能通过隐藏或改写来源绕过。

### 3.2 正确的标签语义与数据划分

- `gif_selected`: PHD2/GIF 用户选中区间，可以作为**弱正监督**。
- `unselected_or_unobserved`: 用户未选中的位置；默认**未标注**，不可统一转成可靠负例。
- `summary_importance`: TVSum / AIKU 类连续重要度；单独回归或排序，不转成官方逐帧真值。
- `crop_native_gt`: RetargetVid/LIVE 真实标注窗口，仅用于空间训练。
- `crop_mos`: GAICv2 主观审美分数，仅用于额外预训练或消融。
- `official_test`: **无可训练标签**，不得补造标签。

划分以**原始 `source_group`** 为最小单位。相同原视频的不同 GIF、抽帧、重编码、不同裁剪比例、重复片段必须进入同一侧。优先使用 R92、R70 已冻结的来源划分；任何重新划分都要写入独立版本并标记为新协议，不能把曾参与调参的数据重新称为 fresh holdout。

### 3.3 不应再做的数据假设

1. 不假设有“官方训练帧 + 官方真实框”。
2. 不用 PHD2 片段边界自动制作“唯一正确高光帧”。
3. 不用 GAICv2 的 MOS 代替官方框 IoU。
4. 不用 TVSum 或 AIKU 的摘要得分乘 100 作为 AIC 分数。
5. 不假设官方测试集的无高光比例已知；没有可靠负例池之前，不把空视频检测作为必备训练头。

---

## 4. 模型结构：保持单条管线，先冻结骨干

```text
原始公开视频（只用于训练）/ AIC 视频（只用于冻结推理）
                       │
       原始 frame_id + PTS + 旋转/尺寸映射
                       │
                       ▼
             DINOv3 ViT-B/16
             86M / 768D / patch16
                       │
             ┌─────────┴─────────┐
             │                   │
       全局帧表征           空间 patch tokens
             │                   │
      单帧读出 / TCN       R70 同款合法候选框
             │                   │
     帧效用 + 固定预算      候选框 IoU Ranker
             │                   │
             └─────────┬─────────┘
                       ▼
         合法的原始 frame + [x,y,w]
                       │
              JSONL / 独立校验
```

首轮**不**使用动态专家路由、不加 SigLIP2 独立分支、不做投票或后期融合；这不是对集成学习优劣的判断，而是为了能够识别 DINOv3 的单变量增量。

### 4.1 DINOv3 实现合同

- 权重：[facebook/dinov3-vitb16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)；[官方代码](https://github.com/facebookresearch/dinov3)；[论文](https://arxiv.org/abs/2508.10104)。
- Hugging Face 权重受 **DINOv3 License** 与访问授权约束。首先检查账户是否有权访问、授权用途是否覆盖当前训练与比赛，再记录权重 revision/哈希。
- DINOv3 ViT-B/16 的输出包含 CLS、register tokens 和 patch tokens。按所用实现的官方接口识别 token，不允许把 register tokens 混入空间网格。
- 为时间实验优先保存一组**固定的全局表示**；可选取 CLS 或已声明的 patch pooling，但必须先固定选择策略，不在留出集上反复挑层。
- 空间实验保存**二维 patch 网格**与到原始像素坐标的逆变换。不要用 pooled CLS 代替候选框的内部/外部特征。
- 使用所选官方 processor 对应归一化。初版建议全帧保留画面：可从 224×224 的 aspect-ratio-preserving letterbox 起步；它只是**拟测试设置**，不是 DINOv3 的唯一标准预处理。对裁剪框执行严格坐标回映射验证。
- 原始视频如果用 VFR 或存在旋转元数据，必须以解码器取得的帧序号和 PTS 为准，不能用 `round(t*nominal_fps)` 替代精确帧索引。

### 4.2 时间头：先对照单帧读出，再启用现有 TCN

**必要的两臂：**

- **T0**：冻结 DINOv3；对每个帧特征使用轻量 `Linear/MLP`，不加时序上下文。
- **T1**：同一份 DINOv3 特征、同一源划分和标签，使用仓库已有 dilated TCN 结构训练。

T1 应尽可能沿用 [`scripts/v9_phd2_tcn.py`](https://github.com/lujunlin-cn/AIC/blob/main/scripts/v9_phd2_tcn.py) 或 [`scripts/r97_event_state_tcn.py`](https://github.com/lujunlin-cn/AIC/blob/main/scripts/r97_event_state_tcn.py) 的数据、mask、loss 与评估约定。但 **R97 的状态辅助损失默认关闭**。主比较应当是“相同时间头，不同视觉骨干”；T0/T1 比较是同一 DINOv3 下的时序效果补充。

优先保留项目已有的 1 FPS/8 点观测协议以完成公平比较。粗采样结果可以映射到原始帧轴形成密集时间排序，但插值不是原始帧级监督。比赛当前会保留每视频大量连续帧，**不应该先验地认为高 FPS 精排比改善时间排序和选帧预算更重要**。

### 4.3 空间头：使用 R70 的现有候选体系

**第一版不从零发明 Crop Ranker。**

1. 固定 R70 的 **129 个合法最大窗口候选**、同一裁剪后处理和相同 native crop GT。
2. 使用 DINOv3 patch tokens，为每个候选提取框内、框外、差分或几何特征。新特征应注明维度和归一化；不能直接读 R70 的旧 V9 2,305D 缓存冒充 DINOv3。
3. 固定候选、loss、训练预算、来源分组和同级别头容量，对比 **R70 原视觉特征**与 **DINOv3 特征**。
4. 优先训练预测候选与 native GT 的 IoU，可对照 R70 既有 MSE / topsoft 目标，但首轮只固定一组已公开的目标函数，不再次进行 loss 全排列搜索。
5. **仅在第一轮有稳定源外增量时**，研究可变宽度 `[x,y,w]` 或局部框修正。R70 的固定最大宽度候选不代表官方只能提交最大宽度。

R70 的训练/推理工程入口：[`scripts/r70_spatial_utility_pilot.py`](https://github.com/lujunlin-cn/AIC/blob/main/scripts/r70_spatial_utility_pilot.py)、[R70 源数据与候选协议](https://github.com/lujunlin-cn/AIC/blob/main/reports/r70_spatial_utility_pilot_20261018/README.md)。注意同一视频不同目标画幅比例应按 `source_group` 合并统计，不能按 row 误当独立来源。

### 4.4 条件式精确帧头

当且仅当 T1 具有可复现的源外收益，且已经取得**合法、时间覆盖充分、定义明确的原始帧级标签**时，才进行精确帧头训练：

- 先评估 1 FPS 候选覆盖率，以及局部密采样能否观察到原模型未捕获的真实变化。
- 再比较原头、局部精排头，固定相同每视频输出数量与空间父链。
- 局部采样可试 4 FPS 或 8 FPS，但这是起点，不是最优配置；必须将抽样索引正确映射回原始帧号。
- 若只有 PHD2 的 GIF 区间弱标签，则最多报告**区间内部排序代理**，不得称“精确帧命中率”或“官方时间 Oracle”。

### 4.5 条件式骨干微调

只有当“冻结 DINOv3 + 任务头”在独立来源上表现出真实、可迁移的信号，且有足够可靠的任务监督时，才比较：

- 冻结全部 DINOv3（必须保留）；
- 只训练最后若干 block 或 LoRA（作为单变量对照）。

同一数据、随机种子、训练预算和早停协议；保持推理端完整参数统计。不能因为历史 DeiT-S 的 37/9 弱标签微调失败，就断言 DINOv3 必然失败；也不能预设部分解冻一定涨分。

---

## 5. 需要实际开展的最小实验矩阵

**先跑 T、S 两条相互独立的对照。下面的数值门槛是这份方案的新预注册建议，不是已完成的实验结论；真正执行前写入不可变配置。**

| ID | 唯一改变的实验因素 | 固定控制 | 主验证指标 | 决策 |
|---|---|---|---|---|
| **T-0** | 从冻结 ResNet18/现有视觉表示换为冻结 **DINOv3 ViT-B/16** | 相同 PHD2/GIF source split、标签 mask、TCN 结构、训练步数、预算和 seed | 源级固定预算 F；跨源一致性；位置先验负控制 | 判别 DINOv3 是否产生新增时间信息 |
| **T-1** | DINOv3 上的 `MLP → TCN` | DINOv3 缓存、输出预算、训练监督完全相同 | 源级 F + 时间置换负控制 | 判别是否有真实时序增量 |
| **S-0** | R70 现有表示换成 **DINOv3 patch 特征** | 同 129 候选、相同 native crop 标签/比例、同容量 head、相同训练协议 | RetargetVid 与 LIVE 分域源宏 IoU；候选 oracle gap | 判别空间表示是否比 R70 更好 |
| **S-1（条件）** | 若 S-0 通过，加入更大/更小宽度候选或局部框修正 | 固定 DINOv3、时间集合和 crop 数据划分 | 原生 crop IoU、实际框合法率 | 判别候选几何上限是否限制表现 |
| **F-0（条件）** | 若有真实原帧监督，加入局部高 FPS 评分头 | 固定 DINOv3、候选区间和每视频输出数 | 同一原帧标签定义下的精确帧指标 | 不满足标签条件则取消，不用弱标签冒充 |
| **FT-0（条件）** | 冻结 DINOv3 → 部分解冻/LoRA | 固定数据、输入、任务头、随机种子 | 源外时间 F 或空间 IoU，以及训练/推理成本 | 无稳定增量则保持冻结 |

### 5.1 明确的对照与统计方法

1. T-0 必须使用**同一来源**下重新提取的 DINOv3 与现有骨干特征。若模型输入预处理不同，应标记为“预训练模型 + 官方预处理的整体比较”，不能声称只改变一层权重。
2. 对所有训练臂至少运行 **3 个预先固定的 seed**；每个 seed 记录最终模型，而不是从多个 seed 中只报最高分。
3. 按**原始 source_group** 聚合逐源指标，并对同一组源做 paired bootstrap，输出 `mean_delta` 和 `95% CI`。不要把 8 个相邻帧或同源的两个画幅当独立样本。
4. **T-0 晋级建议门槛**：相对匹配控制 `Δsource_macro_F >= +0.005`，且 95% CI 下界大于 0；同时不被仅位置排序/时间置换控制解释。
5. **S-0 晋级建议门槛**：相对匹配 R70 控制，RetargetVid 与 LIVE 两个数据域的改善方向不冲突；整体 `Δsource_macro_IoU >= +0.005`，且配对 95% CI 下界大于 0。至少保留一个事先冻结、未用于调参的来源组做一次确认。
6. 若任一门失败，不通过追加随机 seed、扫描阈值、挑选验证视频或组合多个弱模型来制造“过门”结果。归档负结果与失败条件。
7. T-0/S-0 通过门槛只表示**公开源外代理**具有可识别增量，不等于 AIC 官方分数提高。

### 5.2 训练损失：优先继承已有配方

**PHD2/GIF 时间监督：**对真实选择区间的可靠弱标签使用 masked BCE；如使用距离到区间的连续监督，则与区间二值标签分开记录，不在不同实验间悄悄切换。可按同源内部构造有意义的 pairwise ranking，但必须保持未观察位置的 mask，并沿用已审计的来源划分。

\[
L_{\mathrm{temporal}}=
\frac{\sum_{t}m_t\,\mathrm{BCEWithLogits}(z_t,y_t)}{\max(1,\sum_t m_t)}.
\]

**空间监督：**在公开原生 crop GT 可用的帧上，将每个合法候选框与 GT 计算 IoU，沿用 R70 已有效的候选质量目标。可用 Huber 回归和/或固定的 topsoft 目标，但首轮避免再做多损失随机搜索。

\[
y_{t,j}=\operatorname{IoU}(b_{t,j},b_t^{GT}),\qquad
L_{\mathrm{crop}}=\operatorname{Huber}(\hat y_{t,j},y_{t,j})
\]

公式用于解释候选质量回归；实际训练需依据原实现对候选、mask、batch 加权和目标函数作逐项复用。空间 GT 不覆盖的帧不计算 crop loss。**不设置不存在的 `L_official`。**

**AIKU 迁移可作为后续 T-2**：先在 AIKU 预提取特征上训练时序 trunk，再对 DINOv3 目标特征**重新初始化输入投影**并只迁移兼容的时序部分。必须比较 `target_only`、`transfer` 两臂。R102 的正向结果不能无条件推广至 DINOv3。

---

## 6. 如何评估：官方目标、公开代理、空间上限分别报告

### 6.1 对本项目当前最现实的评估

**A. 时间源外诊断**

- 数据：PHD2/GIF 源级留出，辅以 YouTube Highlights / TVSum 交叉域诊断（只有通过数据授权与时间对齐检查的样本才纳入）。
- 指标：固定输出预算 `keep=0.85` 下的源宏集合 F、pairwise Δ、source bootstrap CI、Spearman/AP（视标签性质）、平均输出比例、source 分组覆盖。
- 解释：它回答“公开用户选择/摘要偏好是否被模型更好排序”，**不是 AIC 官方 F**。不同数据源指标不能合并成一个无来源权重的总分。

**B. 空间源外诊断**

- 数据：RetargetVid、LIVE 的原生 crop GT 和 R70 对应候选表。
- 指标：source-macro IoU、相对同一候选池中心框和 R70 控制的 IoU 增量、候选 oracle IoU、合法框比例；RetargetVid、LIVE 分开报告。
- 解释：它回答“相同有监督构图目标下，新视觉表示是否提高选框质量”，**不是官方时间+空间联合分数**。

**C. AIC 官方验证**

- 只有当 T-0 或 S-0 达到预注册源外门槛时，才制作冻结的**单变量提交包**。
- 验证时间新方案时，**空间框逐帧继承同一已审计父链**，并将每视频保留帧数固定为父链预算。
- 验证空间新方案时，**时间集合与原帧号逐字节继承同一父链**，只更换空间框。
- 每次提交记录父包 SHA、新包 ZIP SHA、变化视频数、变化帧数、变化框数和公式版本。
- 官方成绩仅能用于判断该冻结候选的整体迁移效果；**不得反向作为测试视频伪标签或逐视频参数优化目标**。

### 6.2 必须避免的错误评估

| 错误做法 | 为什么无效 | 正确替代 |
|---|---|---|
| 在 AIC 测试集计算 `Oracle-frame Crop IoU` | 官方测试 GT 未公开 | 在 RetargetVid/LIVE 的 native GT 上算空间 Oracle |
| 使用 PHD2 选中区间内每一帧作为“官方精确帧命中” | GIF 区间标签与赛事逐帧 GT 不同 | 标明 `PHD2 observed-selection proxy`；无精确帧标签就不算该指标 |
| 把 R70 的公开 `+0.07006` IoU 乘 100 预测官方涨分 | 两域、时间分母及实际框策略不一致 | 仅做冻结空间单变量官方提交后查看回分 |
| 在比较 DINOv3 时同时更换 TCN、loss、keep、候选框和插值规则 | 无法把差异归因到视觉表征 | 一次只动一个因素，保存匹配控制 |
| 用同源片段作普通独立 bootstrap | 会低估方差 | 以原始视频 `source_group` 配对重采样 |
| 沿用旧 `k_size` 或字节档位计算新版最终分 | 当前评分合同已经改变 | `Score=100×F_video`；参数只用于 9B 合规性 |

### 6.3 官方完整分的单位测试

对本地自造的**最小合成 GT**运行测试，而非借助官方隐藏 GT：

1. `video_id` 相同、`frame` 完全相同、空间 IoU=1：贡献 1。
2. 预测帧号错 1：贡献 0，`N_pred` 仍计入。
3. 命中帧但框部分重合：使用连续 IoU，不二值化。
4. `N_pred=N_gt=0`：`F_v=1`；仅一者为 0：`F_v=0`。
5. 两个不同长度的视频：先分别算 `F_v`，再等权平均。
6. 目标宽高比变化、边界框、负坐标/越界框、重复帧、乱序、VFR/旋转：校验器行为可复现。

注意：这是**公式回归测试**，不是已经与官方同版本可执行 evaluator 逐值对齐的声明。

---

## 7. 两篇重点论文：真实内容、链接与明确边界

### 7.1 TRINITY：多视角视频高光显著性

**论文**：*TRINITY: A Multi-Perspective Benchmark for Personal-Style Video Highlight Detection*（ECCV 2026）  
**作者**：Qianqian Chen、Hyun Bin Kim、Denzel Elden Wijaya、Yang Yi、Bo Liu、Yangkai Ding  
**原文**：https://arxiv.org/abs/2608.29577  
**PDF**：https://arxiv.org/pdf/2608.29577  
**代码**：https://github.com/vanilladucky/TRINITY  
**数据**：https://huggingface.co/datasets/vanilladucky/TRINITY

**主要研究内容**：把个人视频中的高光分为 **Event（事件）**、**Emotion（情绪）**、**Nature（自然景观）** 三种显著性维度，采用共享时序表示及专用分支预测。论文报告在 Mr. HiSum 上相对其基线提升 `+7.15/+3.62 mAP`（两种 ρ 设置），在 YouTube Highlights 上提升 `+10.82 mAP`。这些均为**论文作者在其数据与指标下的报告**，不是 AIC 增量。

**本项目可借鉴内容**：区分不同来源的高光标签、明确时序窗口语义、建立严谨的源级对照。尤其可用于检查“通用摘要重要度”与“用户真正剪下来的 GIF”是否属于不同监督目标。

**不能直接照搬**：公开 TRINITY 包主要是预提取 CLIP 特征，并非可复原的 DINOv3 图像训练数据。本文**不复制多专家架构**，也不把 5 秒/片段 mAP 当成 AIC 精确原帧 F。

### 7.2 AesCrop：构图引导的候选生成与质量排序

**论文**：*AesCrop: Aesthetic-driven Cropping Guided by Composition*（ICCV 2025 Workshop, MIPI）  
**作者**：Yen-Hong Wong、Lai-Kuan Wong  
**CVF 原文**：https://openaccess.thecvf.com/content/ICCV2025W/MIPI/html/Wong_AesCrop_Aesthetic-driven_Cropping_Guided_by_Composition_ICCVW_2025_paper.html  
**CVF PDF**：https://openaccess.thecvf.com/content/ICCV2025W/MIPI/papers/Wong_AesCrop_Aesthetic-driven_Cropping_Guided_by_Composition_ICCVW_2025_paper.pdf  
**arXiv**：https://arxiv.org/abs/2510.22528

**主要研究内容**：用 VMamba 图像编码器、Mamba Composition Attention Bias（MCAB）和 Transformer decoder，把构图偏置引入图像裁剪。模型生成多个裁剪候选并给出质量预测，结合回归/排序训练。论文使用 **GAICv2**（3,336 张图，训练/验证/测试 2,636/200/500；主观 MOS）与 **KU-PCP**（构图类型）等数据，报告优于论文所列的裁剪对照方法。

**本项目可借鉴内容**：让候选裁剪位置和质量预测彼此配合；在候选特征中同时考虑框内、框外及主体构图信息。

**不能直接照搬**：AesCrop 的审美 MOS、候选横纵比与 AIC 的 `[x,y,w]` 框 IoU 不同；直接替换 R70 是无控制变量实验。优先沿用 R70 的真实 native crop IoU 目标，把 DINOv3 patch 特征作为唯一新增因素。作者官方可直接运行的完整训练仓库**尚未在本次修订中确认**。

---

## 8. 在现有仓库中的实施路径与验收

### 8.1 优先复用的文件

| 资产 | 职责 | 新增工作 |
|---|---|---|
| [`aic/contract.py`](https://github.com/lujunlin-cn/AIC/blob/main/aic/contract.py) | 提交 schema、原始帧与构图几何 | 对当前 `model_params_b` 合同复核，不破坏旧包兼容 |
| [`aic/temporal_contract.py`](https://github.com/lujunlin-cn/AIC/blob/main/aic/temporal_contract.py) | 时间锚点与标签 mask | 接入 DINOv3 的帧索引、PTS 和公开源标签 |
| [`aic/evaluation.py`](https://github.com/lujunlin-cn/AIC/blob/main/aic/evaluation.py) | 历史本地公式重建 | 单独验证现行 100×macro F；避免旧规模系数影响 |
| [`scripts/v9_phd2_tcn.py`](https://github.com/lujunlin-cn/AIC/blob/main/scripts/v9_phd2_tcn.py) | GIF 弱时间监督与 TCN | 统一 DINOv3 输入、对照头、mask 与 split |
| [`scripts/r97_event_state_tcn.py`](https://github.com/lujunlin-cn/AIC/blob/main/scripts/r97_event_state_tcn.py) | TCN 源级控制/负结果 | 复用 RGB 控制逻辑，不重复失败的状态辅助目标 |
| [`scripts/r70_spatial_utility_pilot.py`](https://github.com/lujunlin-cn/AIC/blob/main/scripts/r70_spatial_utility_pilot.py) | 129 空间候选与 native IoU | 替换空间表征，保留候选与监督协议 |
| [`scripts/v5_h3_dinov2_cache.py`](https://github.com/lujunlin-cn/AIC/blob/main/scripts/v5_h3_dinov2_cache.py) | DINOv2 空间特征提取的历史实现 | 只借鉴缓存/几何身份设计，不混用 DINOv2 权重 |
| [`scripts/r102_aiku_transfer_probe.py`](https://github.com/lujunlin-cn/AIC/blob/main/scripts/r102_aiku_transfer_probe.py) | 已验证的 AIKU 时序 trunk 转移控制 | 仅作条件式时序预训练候选 |
| [`scripts/independent_submission_check.py`](https://github.com/lujunlin-cn/AIC/blob/main/scripts/independent_submission_check.py) | 独立提交结构校验 | 验证新增 DINOv3 包是否按当前参数字段正确标记 |
| `EXPERIMENTS.md` / `DECISIONS.md` / `STATUS.md` | 历史实验与项目决策 | 新实验必须追加，不覆盖旧负结果 |

### 8.2 新增文件建议（按需增量开发）

```text
scripts/dinov3_extract_public_features.py   # 只读授权公开训练视频
scripts/dinov3_temporal_matched_probe.py     # T-0 / T-1
scripts/dinov3_spatial_matched_probe.py      # S-0
scripts/dinov3_candidate_package.py          # 仅在代理晋级后创建冻结提交包
configs/DINOV3_T_MATCHED_V1.json             # 固定来源、seed、loss、budget
configs/DINOV3_S_MATCHED_V1.json
reports/dinov3_v1/data_audit.json
reports/dinov3_v1/representation_contract.json
reports/dinov3_v1/temporal_metrics.json
reports/dinov3_v1/spatial_metrics.json
reports/dinov3_v1/decision.md
```

路径只是**拟建清单**，不代表这些文件已存在。现有命名惯例优先于上面建议名称，不需要为了照抄树形结构而移动旧文件。

### 8.3 Coding Agent 有序执行清单

**任务 A：契约与资产核验**

1. 固定 Git 基准提交与运行环境。
2. 审计 DINOv3 权重许可、可用性、模型参数与特征输出。
3. 列出 R92/PHD2 和 R70/RV/LIVE 的实际本地媒体可用性、来源分组、原始帧索引。
4. 找出当前仓库里的现行复赛参数字段处理路径，运行只依赖合成数据的评分单元测试。
5. 冻结 T-0/S-0 的实验配置与来源留出列表。

**任务 B：时间表征 T-0/T-1**

1. 为已授权且可解码的 PHD2/GIF 视频提取 DINOv3 视觉特征。
2. 在相同训练样本上准备可复核的原骨干匹配控制。
3. 固定 TCN 结构、标签、预算和训练步数，运行 3 个 seed。
4. 跑同特征 MLP 控制、位置先验控制、时间置换控制。
5. 按 `source_group` 生成逐源 F、配对 Δ 与 95% CI。
6. 如果未过预注册门，记录 `STOP_TEMPORAL_DINOV3`，不启动条件 F-0。

**任务 C：空间表征 S-0**

1. 确认 RetargetVid/LIVE 的媒体权限和 source split。
2. 复用 R70 的 129 候选、标签、几何与固定头配方。
3. 提取 DINOv3 patch tokens 并保存逐候选特征来源证明。
4. 运行匹配 R70 控制和 DINOv3 特征两臂，至少 3 seed。
5. 分别报告 RetargetVid、LIVE 的 source-macro IoU 和配对置信区间。
6. 如果未过预注册门，记录 `STOP_SPATIAL_DINOV3`，不启动可变框宽度或复杂 decoder。

**任务 D：有条件的比赛提交**

1. 只接受已过源外门槛且具备完整冻结推理链路的单变量候选。
2. 选择明确的父包并锁定 SHA256、每视频输出数量和空间/时间不变字段。
3. 全量推理，严格复核 `frame`、`[x,y,w]`、`model_params_b`、视频覆盖数、权重账本。
4. 运行项目 validator、独立 checker 和 ZIP 解压回读测试。
5. 输出一份**可由用户自主决定是否上传**的提交包和审计报告。未经明确授权不自动提交。
6. 官方回分仅记为包级验证，不反向生成测试训练目标。

### 8.4 强制验收条件

- **数据**：每条训练样本带来源、许可状态、原视频分组和监督类型；没有把官方测试集变成训练集。
- **模型**：DINOv3 权重 SHA、预处理签名、CLS/patch/register 分离、骨干冻结状态均可复核。
- **公平性**：单次实验只改变预注册变量，保留同数据/训练步数/seed 的对照；指标按原始 source 分组。
- **评测**：公开时间代理、公开 crop IoU、真实官方回分分别存储，不混称。
- **推理**：每个官方输出能追溯到原始帧号；所有框合法；全部模型推理参数 ≤9B。
- **决策**：没有达标结果也算完成实验，但不能写“已找到提分路线”。

---

## 9. 当前最优先的技术判断

**最值得投入的是 DINOv3 表征的两个匹配实验：T-0 和 S-0。**理由是时间 TCN、空间 Ranker、低层滤波、多视图融合均已在仓库存在真实执行证据；DINOv3 ViT-B/16 却尚无对应可核验训练结果。

若 **T-0 不过门、S-0 过门**，将 DINOv3 定位为**空间专用视觉表示**，时间继续用已审计的父链。若 **T-0 过门、S-0 不过门**，则只更换时间表征，保留既有裁剪父链。若两者均过门，也先各自生成独立单变量候选，**不立刻同时切换时间和空间**。如果两个方向都未过门，应冻结负结果，优先研究目标监督质量、域迁移和官方选择策略，而不是扩大 DINOv3 训练参数。

DINOv3 方案不承诺超过 R71 的用户报告分数 **36.43**。对于本比赛，**公开代理上有统计支持的增量，仍需要经过单变量官方提交才有资格讨论真实比赛增益。**

## 10. 核心参考与审计入口

### 论文和模型

1. [DINOv3 论文](https://arxiv.org/abs/2508.10104)；[官方代码](https://github.com/facebookresearch/dinov3)；[ViT-B/16 权重](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m)。
2. [TRINITY（ECCV 2026）](https://arxiv.org/abs/2608.29577)；[代码](https://github.com/vanilladucky/TRINITY)；[数据](https://huggingface.co/datasets/vanilladucky/TRINITY)。
3. [AesCrop（ICCV 2025 Workshop）](https://openaccess.thecvf.com/content/ICCV2025W/MIPI/html/Wong_AesCrop_Aesthetic-driven_Cropping_Guided_by_Composition_ICCVW_2025_paper.html)；[PDF](https://openaccess.thecvf.com/content/ICCV2025W/MIPI/papers/Wong_AesCrop_Aesthetic-driven_Cropping_Guided_by_Composition_ICCVW_2025_paper.pdf)。
4. [PHD2/GIF](https://github.com/gifs/personalized-highlights-dataset)；[YouTube Highlights](https://github.com/aliensunmin/DomainSpecificHighlight)；[TVSum](https://github.com/yalesong/tvsum)；[AIKU x brAIn 2025](https://www.kaggle.com/competitions/aikubrainvideosummary/data)。
5. [RetargetVid](https://github.com/bmezaris/RetargetVid)；[LIVE-YT-VideoCropping](https://github.com/steven413d/LIVE-YT-VideoCropping)；[GAIC 项目](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping)。

### 本项目证据

1. [现行赛事规则基线](https://github.com/lujunlin-cn/AIC/blob/main/01_%E8%B5%9B%E4%BA%8B%E8%A7%84%E5%88%99%E4%B8%8E%E8%AF%84%E5%88%86%E6%A0%87%E5%87%86.md)、[STATUS](https://github.com/lujunlin-cn/AIC/blob/main/STATUS.md)、[EXPERIMENTS](https://github.com/lujunlin-cn/AIC/blob/main/EXPERIMENTS.md)、[DECISIONS](https://github.com/lujunlin-cn/AIC/blob/main/DECISIONS.md)。
2. [R89/R90 低层与时序特征](https://github.com/lujunlin-cn/AIC/blob/main/reports/r89_lowlevel_composition_20261019/summary.md)、[R92 多视图](https://github.com/lujunlin-cn/AIC/blob/main/reports/r92_multiview_traditional_route_20261019/summary.md)、[R97 TCN](https://github.com/lujunlin-cn/AIC/blob/main/reports/r97_event_state_tcn_20261019/summary.md)。
3. [V5-H3 DINOv2 空间评分头证据复核](https://github.com/lujunlin-cn/AIC/blob/main/reports/20260929_v6_e0_evidence_audit.md)、[R70 空间评分](https://github.com/lujunlin-cn/AIC/blob/main/reports/r70_topsoft_objective_20261018/README.md)、[R86/R88 空间身份问题](https://github.com/lujunlin-cn/AIC/blob/main/reports/r86_r87_observation_and_transfer_20261019/summary.md)。
4. [R102 AIKU 时间迁移](https://github.com/lujunlin-cn/AIC/blob/main/reports/r102_aiku_transfer_20261020/summary.md)、[R105 显著性空间增量与过拟合](https://github.com/lujunlin-cn/AIC/blob/main/reports/r105_avimos_aic_utility_20261020/summary.md)。
5. [QVHighlights DeiT-S 冻结/解冻实验的注册表](https://github.com/lujunlin-cn/AIC/blob/main/experiments/registry.jsonl)（查找 `QVH_FINETUNE_PROBE_20260926`）。

**文档结束。**所有训练超参数、晋级门槛和新增文件路径均为待执行的研究方案；截至本次修订，未执行 DINOv3 训练，未取得 DINOv3 的官方回分。
