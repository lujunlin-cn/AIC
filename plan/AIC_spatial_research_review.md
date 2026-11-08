# AIC 大方向二：视频空间裁剪与构图优化研究报告

**研究基线：** P2，半决赛官方 F1 = 44.65

**研究范围：** Position、Scale、候选构图质量、主体条件构图与视频部署稳定性

**决策目标：** 找出能够在真实部署路径上超过 P2 的空间干预；不以训练 loss、Raw IoU 或公开数据代理指标代替官方成绩。

---

## 0. 执行摘要

### 核心结论

1. **空间仍有研究价值，但目前不能认定 Scale 是主要瓶颈。** P2 已有 44.65 官方成绩；常数收缩 C3（A=0.90）得 42.87，C1（A=0.90、B=0.95）得 42.23。相对 P2 分别下降 1.78 和 2.42。公开 LIVE 上的常数尺度收益没有迁移到半决赛。
2. **Position 是目前证据最强、但仍未证明达到上限的能力。** 新 A/B Adapter 在 LIVE 上提高了人工 crop IoU；R25 固定时间集合的空间对照也显示框质量会显著影响联合分数。但这些结果不等于 P2 的 Position 仍可直接提升，也不等于 Position 已饱和。
3. **逐帧单尺度回归不应作为下一轮主路线。** FULLCROP token-CE、IoU1、Geometry Head MLP32 均未把训练或 Raw 改善转化为真实部署增益。C1/C3 又否定了直接迁移全局常数收缩的做法。
4. **最值得重新评估的是“候选构图质量读出”，而不是再训练自由坐标回归器。** R46 在固定 129 个合法候选上发现稳定 Oracle Gap；R70 的 TopSoft 取得过 +0.20 官方增益（但来自旧系统和固定时间集合），说明候选内排序可能比直接回归完整框更有希望。必须重新核验候选特征、模型身份、数据域和部署可复现性。
5. **SACD 和 GAICD 提供的监督类型不同，不能混用。** SACD 是 subject-conditioned 人工排序；GAICD 是候选 crop 的 MOS。LIVE/GNMC 是 crop 框/位置监督。DPO 只有在确实使用独立人工偏好（例如合格的 SACD/GAICD 数据）时才有额外信息价值；对 LIVE/GNMC 同一 IoU 排序换成 chosen/rejected，不构成新的偏好来源。
6. **建议主路线：冻结 P2 几何合同，重新建立可审计的候选 Crop Utility Readout。** 首先恢复或重建当前可用视觉特征与 129 候选；比较冻结 readout、线性校准、小型 MLP、软目标 IoU 预测和真实人工偏好排序。只在两个源外留出域都胜过当前基线后，才考虑完整 890 视频空间替换。

### 推荐优先级

- **主路线：** 固定合法候选池上的候选效用学习（candidate utility readout），以真实 crop IoU 为主监督，以 GAICD MOS / SACD 同主体排序作为独立偏好辅助。
- **备用路线 A：** Position 条件化校准，重点处理多人、遮挡、主体互动和画面边缘场景；不重新训练整个 9B 模型。
- **备用路线 B：** 镜头内空间轨迹约束与多主体覆盖门控，只作为部署层增量，必须证明提升绝对 crop IoU，而不是只降低 jerk。
- **停止路线：** 无差别逐帧 Scale 回归、统一常数收缩、仅以轨迹平滑为目标、无有效偏好对的 DPO、未经原生比例验证的 LIVE 转置 B 迁移。

---

## 1. 评分目标与实验归因

AIC 单视频评分：

\[
F_v=\frac{2\sum_{t\in P_v\cap G_v}IoU_t}{|P_v|+|G_v|}
\]

空间模块只影响命中帧的 \(IoU_t\)，但错误的裁剪框会降低每个命中帧的贡献。时间集合固定时，空间对照可以隔离空间增量；但公开数据的 Source-Macro IoU 不能直接换算为官方 F1。

所有空间实验必须遵守：

- 固定时间帧集合、原始 frame ID、输出帧数量和视频路由；
- 固定 P2 父包及模型权重身份；
- 明确 A（横屏转 9:16）、B（竖屏/转 16:9）和 Native 路由；
- 使用最终部署几何、QP、整数坐标和回退逻辑评价；
- 按独立 source 做 Macro 与 Bootstrap，不按帧 Micro 冒充 source 泛化；
- 官方回分是唯一提交晋级依据。公开 GT 只用于机制筛选。

**基线口径：** 当前研究以用户确认的 P2 半决赛 44.65 为基线。项目历史文档曾记录 42.74，这是更早阶段或不同评测轮次的分数，不得与半决赛分数混用。

---

## 2. 项目历史证据审计

### 2.1 Position：有正向证据，但域边界必须保留

`docs/12_videoora_spatial_fullcrop.md` 记录：

- LIVE-YT-VC：1800 个视频源，1600 train / 200 dev，按 source 划分；每源约 30 个标注帧。
- 新 A Adapter：LIVE 公开验证 IoU 从 0.607452 到 0.627236，增量 +0.019784，源级 CI 为正。
- 新 B Adapter：转置 LIVE IoU 从 0.460591 到 0.586134，增量 +0.125542，源级 CI 为正。
- **B 的转置 LIVE 结果不代表原生 upright 竖屏视频域。**
- 半决赛历史包：PUSH_RETRAIN_A 42.73；P1 新 A+新 B 42.72；P2 新 A+新 B、cap24、QP floor 0.6 等组合 42.74（早期阶段）。当前半决赛基线另为 44.65。

解释：Position Adapter 在公开 LIVE 上学习有效，但不能据此断言当前 P2 的全部视频类型都还有相同增益。下一轮应优先做误差分层与局部校准，不应无条件重训 Position。

### 2.2 完整框与 Scale：连续三类证据没有通过部署门

| 实验 | 已知结果 | 结论边界 |
|---|---|---|
| FULLCROP_A_V1 | Full IoU .625878，matched center .628279，差 -0.002401，CI 跨零 | 完整框未超过中心基线 |
| token-CE | Train scale 仅 4 个不同值，Pearson .0395；Dev 8 个值，Pearson .0517 | 数值/尺度学习塌缩，未证明单帧尺度可预测 |
| FULLCROP_A_IOU1 | Train scale Pearson -0.01753，仅 6 个不同值 | 可微 IoU 目标未修复 greedy 尺度预测 |
| Geometry Head MLP32 | Raw 增益 +.006131；真实 40 视频轨迹 -0.000620 | Raw 改善没有转成部署收益 |
| FULLCROP_B_V2 | Raw .872317；真实路径 .478158，低于新 B .512216，差 -.034058 | 部署路径失败 |
| FULLCROP_B_MIX_V1 | GNMC+LIVE 混合训练，完整 426 包；未做公开质量评价、无官方回分 | 质量未知，不得说已证实失败或成功 |
| C3 | A=.90、B=1.0，半决赛 42.87 | 相对 P2 44.65，下降 1.78 |
| C1 | A=.90、B=.95，半决赛 42.23 | 相对 P2 44.65，下降 2.42 |

关键判断：

- 离散 Scale bins 只能改变动作表达，不能创造输入中不存在的可预测信息。
- C1/C3 的负回分说明公开 LIVE 的尺度统计不能直接迁移到 AIC 半决赛分布。
- FULLCROP_B_MIX_V1 仍是重要未决资产，应先做公开静态与真实轨迹质量评价；它有真实尺度干预（相对旧母包 33,882 框变化，其中 32,678 框宽度变化），但无质量回分。
- 不应再把“训练步数不够”作为完整框失败的默认解释。需要区分标签噪声、数值表达、域迁移和部署后处理。

### 2.3 候选效用学习：目前最值得重新验证的正向机制

**R46：129 候选 Oracle**

- LIVE confirmation 227 源：B3 head IoU .53495，center .46876，Oracle .78188；Oracle-B3 gap .24694，95% CI [.23131,.26295]，227/227 source Oracle 更高。
- LIVE dev 124 源：B3 .54020，center .46897，Oracle .77446；gap .23426，CI [.21365,.25517]。
- RV exposed 70 源：B3 .68203，center .61983，Oracle .84390；gap .16187，CI [.14671,.17841]。

解释：候选池包含更好 crop，但 Oracle 使用 GT，不能证明模型能预测该候选。R46 的原 B3 head / feature 身份曾不可恢复，不能用 V7/V9 权重冒充 B3。

**R70：TopSoft Utility**

旧 R54 时间集合固定，空间 utility pilot 官方结果 36.22 → 36.42（+0.20）。这是实际官方正向证据，但来自旧系统与旧评测阶段，不可直接加到 P2=44.65。应复现其候选效用学习思想，而不是照搬旧权重。

**R82：Utility 重排时间**

外部排序指标有小幅正向，但官方 36.22 → 36.20。说明空间 utility 简单乘入时间排名不可靠；本报告不建议将其作为空间主路线。

### 2.4 上下文与轨迹：不能把“更平滑”当作“更准确”

- R69 Direct Spatial Grounding：时间集合固定，但整视频/分段统一 crop 导致官方 24.95，相对 36.22 大幅下降；轨迹总变差下降 46.5% 仍不能挽救绝对位置错误。
- R100 pooled context：增量 +.000747，95% CI 跨零；置换上下文后仍有类似读数，没有证明 source-specific 上下文价值。
- R101 ResNet18 邻帧上下文：增量 +.000197，CI 跨零；置换上下文反而更高，关闭该配方。
- R103 spatial tokens：增量 -.0006575，修复全 mask NaN 后完整重跑仍为负。
- RetargetVid 20 源基准中，proxy+EMA 相对 center 的增量 +.00528，CI [-.00381,.01651]；face+EMA +.00854，CI [-.01054,.02855]，均跨零。多主体 face 聚合未改善。

结论：上下文和稳定性可作为条件化特征或部署约束，但不能独立作为提分目标。

---

## 3. 数据集与监督语义

### 3.1 数据资产及可承担任务

| 数据集 | 标注类型 | 合适用途 | 不应做什么 |
|---|---|---|---|
| LIVE-YT-VC | 视频逐帧人工 crop / portrait-region 标注；1800 源 | 视频域 Position、轨迹、局部 crop 质量；source-disjoint 验证 | 不把转置 B 当原生 B；不将同一 GT IoU 排序包装成独立人工偏好 |
| GNMC | 多画幅静态图像 crop | 多画幅几何、静态尺度和画幅条件泛化 | 不把静态图像结果当视频轨迹结果；遵守其许可条件 |
| GAICD | 每图多个候选 crop 与 MOS | 同图候选排序、MOS 质量回归、人工偏好监督 | 不能只取最大 MOS 候选做单框 SFT；不能忽略固定 P1 中心过滤造成的样本流失 |
| SACD | 主体条件 crop、主体 ID、人工排序对 | 多主体/指定主体条件下的构图偏好 | 不得将 subject-conditioned 偏好当无条件通用偏好；必须先修复图像级 split 泄漏 |
| RetargetVid | 视频裁剪框、多标注者轨迹 | 视频轨迹稳定性、跨帧 crop 评价 | 不能把最大合法窗标注解释成可变尺度偏好 |

### 3.2 SACD：有价值的人工偏好，不等于当前 Adapter 已正确读入

官方数据页报告：2,777 张图像、超过 24,000 个 crop、超过 5.2 million ranking pairs；目标是“给定指定 subject 的构图推荐”，不是任意 crop 的通用审美排序。原论文描述了多阶段标注流程，并提供多个 subject 的 crop 与排序关系。

项目当前本地适配审计曾报告：图像 2,777、subject annotations 2,906、masks 2,906；旧 split 有 Train/Val 22 张、Train/Test 20 张、Val/Test 4 张图像重叠；当前 Adapter 未识别显式 ranked pairs（0）。

因此：

- 原始 SACD 确实有大量 ranking pairs；项目当前解析器返回 0，首先是 schema/adapter 审计问题，不能直接得出 SACD 没有偏好数据。
- 必须按原始 image ID 重建互斥 split，同一图像所有 subject annotation 必须留在同一 split。
- 必须恢复原始 pair 定义、subject 绑定和候选 crop ID；不能仅凭 best_box 自行伪造 pair。
- SACD 最适合训练 subject-conditioned candidate ranker，不应直接训练通用 Scale 回归。

### 3.3 GAICD：MOS 是与 IoU 不同的监督

GAICD 提供同一图像多个裁剪候选及 MOS。项目已有审计曾发现，在严格比例与中心约束下，候选保留量大幅下降（此前统计 9:16 仅 54 个、16:9 为 2,082 个；仍需在当前冻结 manifest 上复算）。

建议保留两种用途：

1. **不固定 P1 中心的通用构图质量排序**：训练/验证同图候选 MOS 排序。
2. **Scale-only 子集**：固定中心后，只保留中心匹配且比例合法的候选，明确报告过滤率。若数量不足，不将其作为 Scale DPO 主来源。

MOS 是人工审美评分，不是 AIC GT IoU。训练时应保留 `label_type=mos`，不可改名为 IoU。

### 3.4 LIVE/GNMC：有几何监督，但偏好并非独立

对每个固定候选 crop 可计算 IoU。用同一 IoU 生成 hard best-scale、softmax IoU target、pairwise chosen/rejected，本质上是同一 GT 几何监督的不同损失形式。它们可比较训练目标，但不能宣称 DPO 获得了新的人类偏好信息。

---

## 4. 外部研究与可迁移性

### 4.1 SACD / Subject-aware Composition

**Yang et al., “Focusing on your subject: Deep subject-aware image composition recommendation networks,” Computational Visual Media, 2023.**

- 论文：<https://doi.org/10.1007/s41095-021-0263-3>
- 数据主页：<https://cg.cs.tsinghua.edu.cn/SACD/>
- 方法：粗候选窗口评分，筛选后回归窗口角点，再做最终评分；设计 stability losses；构造 subject-aware 排序对。
- 迁移价值：主体不是抽象 saliency peak；给定 subject 后，crop 质量可能变化。对 AIC 多主体、互动、人物与物体共现很有价值。
- 限制：SACD 是静态图像；subject-conditioned policy 与 AIC 通用视频构图不同。需在 subject ID 可恢复时使用。

### 4.2 GAICD / Dense Crop MOS

- 数据/基线索引：<https://github.com/bcmi/Awesome-Aesthetic-Evaluation-and-Cropping>
- 参考实现：<https://github.com/bcmi/Human-Centric-Image-Cropping>
- 另一实现：<https://github.com/AIVRC/GAFIC>
- 方法价值：密集候选 MOS 可用于同图排序；GAFIC 强调全局特征与 crop 边界区域的对齐。
- 迁移价值：可训练候选质量 readout，而不是直接回归一个 crop。
- 限制：MOS 的审美语义与 AIC 标注 crop IoU 不相同；固定 P1 中心筛选会丢失大量候选。

### 4.3 ProCrop（AAAI 2026）

- 论文：<https://arxiv.org/abs/2505.22490>
- 项目页：<https://bwgzk-keke.github.io/ProCrop/>
- 官方代码/数据入口：项目页提供 GitHub 与 Hugging Face 链接。
- 方法：检索专业摄影图像，融合查询图像与专业构图特征，并生成文本描述引导 crop；论文报告 242K 弱标注图像。
- 迁移价值：专业构图先验、检索相似构图、弱监督候选生成。
- 限制：检索增强与专业摄影审美不等于比赛 crop GT；引入额外检索库和运行成本，当前 9B 参数/推理预算下不应直接部署。可先借鉴“生成候选+评估”的思想。

### 4.4 CROP（2026 arXiv）

- 论文：<https://arxiv.org/abs/2605.12545>
- 方法：将 aesthetic cropping 组织为 analysis–proposal–decision，并以 expert preference DPO 对齐人类审美。
- 迁移价值：如果使用 GAICD/SACD 的真实人类排序，可让 Video-ORA 学习候选间取舍；偏好对必须来自人工标注而非同一 IoU 标签重复编码。
- 限制：论文任务是静态审美裁剪，不是视频逐帧 AIC 官方 F；论文结果不能直接迁移为比赛提分保证。需核验官方代码/数据发布状态，不得仅凭论文摘要声称完整复现代码已可用。

### 4.5 CPIC：连续偏好场

- 论文：<https://arxiv.org/abs/2610.00582>
- 代码：<https://github.com/zzqingz/CPIC>
- 数据入口：代码仓库提供 CPICD 数据说明。
- 方法：从离散 crop 标注重建多峰连续偏好场，包括 peak clustering、off-lattice refinement、negative shaping、field assembly；使用 CPF reward 训练 VLM。
- 迁移价值：比单一 GT 框更能表达“多个 crop 都合理、偏好存在多峰”的现象。适合分析 GAICD/SACD 的多候选标注。
- 限制：截至本次仓库检查，CPIC README 标注训练代码、完整模型和 CPICD 仍有 TODO/分批发布状态；必须先核验可用资产。CPICD 对 benchmark GT 的重新校准也不能未经验证替换 AIC 官方标注合同。

### 4.6 其他值得跟踪的方向

- **GenCrop / Learning Subject-Aware Cropping by Outpainting Professional Photos (AAAI 2024)**：<https://ojs.aaai.org/index.php/AAAI/article/view/27990>。利用专业照片和扩图生成弱监督 crop pair。可借鉴数据构造机制，但当前不建议自建大规模生成数据。
- **HumanCrop-Thinker (2026)**：<https://doi.org/10.1016/j.eswa.2026.131664>。以结构化推理与可验证优化处理 human-centric cropping。对 Video-ORA 可借鉴结构化中间变量，但不能把长 CoT 当成构图质量本身。
- **Awesome Aesthetic Evaluation and Cropping**：<https://github.com/bcmi/Awesome-Aesthetic-Evaluation-and-Cropping>。用于继续发现 CPC、SID、UGCrop5K、FCDB 等资源；新增数据必须先审计许可、标注语义和 split。

---

## 5. 当前空间瓶颈判断

不能从现有证据断言“Position 已解决”或“Scale 是唯一瓶颈”。建议按下表区分。

| 子问题 | 当前证据 | 判断 | 下一步需要的证据 |
|---|---|---|---|
| 主体中心/位置 | 新 A/B LIVE 增益；官方 P1/P2 接近；多主体方法不稳定 | 有一定能力，仍可能在困难场景有误差 | P2 在公开 native crop GT 上按单人/多人/遮挡/边缘分层的误差 |
| Scale | 多次 fullcrop 失败；常数收缩官方负回分 | 逐帧直接回归暂不值得继续；不能否定候选尺度选择 | 固定 P2 中心的候选质量读出、中心扰动鲁棒性、原生域验证 |
| Candidate Utility | R46 Oracle Gap 大；R70 TopSoft 曾有官方正结果 | 最值得重新验证 | 同候选、同特征、源外留出，读出器超过冻结 baseline |
| Subject-aware | SACD 原始有大量人工 ranking pairs；当前 adapter 未解析 | 潜力高，数据解析是阻塞 | 恢复 pair/subject schema，修复 image-level split |
| Video context | R100/R101/R103 未显示增益 | 当前配方关闭 | 只有观察合同或监督改变时才重开 |
| Trajectory smoothness | 平滑降低 jerk，但不保证 IoU；R69 大幅掉分 | 只能作约束，不作目标 | 同时报告绝对 IoU 与轨迹指标 |
| Event-conditioned crop | 缺少同源事件语义+crop GT | 暂缓 | 不能用 R42 小样本弱标签直接做主监督 |

---

## 6. 方法选择：哪些训练目标真正有信息？

| 方法 | 监督来源 | 优点 | 风险 | 本项目定位 |
|---|---|---|---|---|
| Direct IoU Regression | LIVE/GNMC crop GT、候选 IoU | 目标直观 | 标注噪声、多峰偏好被平均 | 小型 readout baseline |
| Soft-Target Classification | 同一候选池各候选 IoU | 利用全候选质量，不只 best box | 温度敏感；仍是 GT 派生监督 | 推荐用于几何候选质量学习 |
| Pairwise Ranking | SACD ranking pairs、GAICD MOS pair | 直接学习相对偏好 | tie/subject/split 必须正确 | SACD/GAICD 合格数据优先 |
| SFT | 人工 crop/scale 目标 | 简单、稳定、可部署 | 单标签监督、多峰目标压缩 | 仅作为动作/格式初始化 |
| DPO | 真正独立 chosen/rejected 偏好 | 可将人工偏好写入生成策略 | 需要可靠 pair 和 reference；同一 IoU 排序不带来新信息 | 条件使用，不设为默认 |
| Candidate Utility Readout | 每个候选的 IoU/MOS/subject preference | 保留候选集合，输出可解释 | 需要原始候选特征和部署一致性 | **首选主路线** |
| Small MLP / Linear Residual | 冻结视觉/候选特征 + crop GT | 参数少、训练快 | 可能过拟合 source 或标签策略 | 先线性，再小 MLP |
| CPF / GRPO | 从离散标注估计连续偏好场 | 表达多峰偏好 | 实现与计算成本较高，标注校准风险 | 先做离线 CPF 诊断，不直接训 9B |

关键区分：

- LIVE/GNMC 的 IoU soft target 与 IoU pairwise 都来自同一几何标签；比较损失函数有意义，但不能宣称获得新偏好数据。
- GAICD MOS、SACD subject ranking 是相对独立的人类偏好来源；必须保留 label semantics。
- SACD 的偏好是“针对某个 subject 的 crop 排序”，不能无条件推广到整帧构图。
- DPO 不是自动优于 soft-target CE。若偏好 pair 数量少、源数少、质量差，DPO 可能比直接候选回归更不稳定。

---

## 7. 至少 8 个可执行实验

### E1. P2 Position Error Stratification（不训练）

- **问题：** Position 是否还有可恢复错误？错误集中在哪类场景？
- **假设：** P2 的中心误差在多人、遮挡、边缘主体、多人互动场景存在系统分层。
- **历史差异：** 之前多数 Position 实验报告总体 IoU，没有完整把失败归因到主体结构。
- **数据：** LIVE Train/Val、RetargetVid、GNMC（静态辅助）；GAICD 人工候选可用于补充场景属性，不作视频 GT。
- **输入/输出：** P2 原图与最终部署框；输出按 source 的中心偏差、crop IoU、主体数量/遮挡/边缘分层。
- **目标：** 无训练；建立误差 taxonomy。
- **Video-ORA 训练：** 否。
- **GPU：** 0–0.5 h，优先复用预测缓存。
- **部署：** 无。
- **评价：** Source-Macro IoU、中心误差、各分层差值、source bootstrap CI。
- **失败判据：** 公开源没有稳定误差分层，或分层标签不可可靠恢复，则不做条件 Position 头。
- **P2 对照：** 同帧 P2 crop 是唯一基线。

### E2. GAICD MOS Candidate Ranker（首个训练实验）

- **问题：** 是否能学到超越几何 GT 的人工审美偏好？
- **假设：** MOS 排序可提供不同于 IoU 的构图偏好信号。
- **历史差异：** 旧 Scale 方案在固定中心后大量过滤 GAICD 候选，导致偏好对近乎归零；本实验先不固定 P1 中心，研究通用候选评分。
- **数据：** GAICD Journal Release；只用 Train 构造偏好，Val 选模型，Test 只作最终一次评估。
- **输入/输出：** 原图+候选 crop；输出候选 utility。
- **目标：** pairwise logistic ranking 或 MOS regression；tie 区间不构造硬 pair。
- **Video-ORA 训练：** 否，先冻结视觉特征训练线性/小 MLP readout。
- **GPU：** 0.5–1 h，取决于特征是否缓存。
- **部署：** 暂不部署；作为审美质量 readout 诊断。
- **评价：** Image-Macro SRCC、pairwise accuracy、source bootstrap；不得与 IoU 混为一项。
- **失败判据：** source-disjoint Val 无正增益或仅在特定 ratio 有效，则不迁移到 AIC。
- **P2 对照：** P2 crop 与候选 MOS 的关系仅作外部校准，不声称官方收益。

### E3. SACD Subject-conditioned Pair Recovery + Ranker

- **问题：** SACD 的原始人工排序是否能恢复，并能否改善指定主体的 crop 决策？
- **假设：** 对多人图像，给定 subject ID 的人工 crop 偏好能减少“选错主体/裁掉互动对象”。
- **历史差异：** 项目 Adapter 当前解析为 0 pair；这可能是 schema 适配失败，而非原始数据无 pair。
- **数据：** SACD 原始 annotation；按 image ID 重建互斥 split，同图像所有 subject 进入同一 split。
- **输入/输出：** 图像+subject mask/ID+候选 crop；输出 subject-conditioned utility。
- **目标：** 原始合法 ranking pairs 的 pairwise loss；禁止从 best_box 伪造 pair。
- **Video-ORA 训练：** 否，先用小型 ranker。
- **GPU：** 0.5–1 h；若特征缓存可 CPU。
- **部署：** 不直接部署；验证 subject 条件是否带来跨源增益。
- **评价：** Subject-Macro pairwise accuracy、top-1 crop quality、按单/多主体分层。
- **失败判据：** 无法恢复 subject-pair 合同、split 仍有图像泄漏或 source-heldout 无增益则停止。
- **P2 对照：** 同一候选池的 P2/center crop 与人工排序比较。

### E4. 固定 129 候选的 Utility Readout 复现（主路线）

- **问题：** R46/R70 的候选 Oracle Gap 能否由当前可恢复特征与小型 readout 关闭？
- **假设：** 129 个合法候选已足够；主要瓶颈是候选效用排序而非候选密度。
- **历史差异：** R46 原 B3 head/feature 身份曾缺失；不能把 V7/V9 冒充 B3。需明确本次使用的 encoder、processor、候选和缓存身份。
- **数据：** LIVE Train/Val、RetargetVid source-disjoint；固定同 129 候选和同几何。
- **输入/输出：** 冻结候选视觉特征、位置/窗口特征、ratio；输出 129 个候选的 IoU utility。
- **目标：** Linear → rank-32 MLP，回归候选 IoU 或 softmax(IoU/τ) soft target；分别报告两种损失，不同时堆叠。
- **Video-ORA 训练：** 否。
- **GPU：** 特征已缓存时 0–0.5 h；重新提特征另估。
- **部署：** 仅在 source-heldout 双域通过后，考虑输出候选 top-1；保留 P2 时间集合。
- **评价：** Source-Macro IoU、Oracle Regret、Top-1 candidate IoU、Oracle@K、跨域 CI。
- **失败判据：** LIVE dev 与 confirmation 两池不能同时超过冻结 P2/B3 对照，或收益只来自某 ratio，则停止。
- **P2 对照：** 固定同帧、同候选池、同 crop 合同的 P2 框。

### E5. 多峰 Crop Preference Field 离线重建

- **问题：** 单一 best crop 是否压缩了多种合理构图？
- **假设：** GAICD/SACD 的多候选标签呈多峰偏好，单一 argmax 监督会丢失合理备选。
- **历史差异：** 旧 Scale 训练用单一最佳尺度，未显式建模多峰。
- **数据：** GAICD MOS、SACD subject-conditioned crops；分别建 field，不跨标签语义合并。
- **输入/输出：** 每图候选 crop 与 MOS/rank；输出离散候选偏好场/峰簇。
- **目标：** MOS 校准、peak clustering、连续插值/核平滑；先离线分析，不做 GRPO。
- **Video-ORA 训练：** 否。
- **GPU：** 0 h（主要 CPU）。
- **部署：** 只在后续 candidate selector 使用，不改变 P2。
- **评价：** 人工标注候选覆盖、偏好场校准误差、peak stability、source bootstrap。
- **失败判据：** 多峰结构不稳定或不同标注者/subject 无法对齐，则不采用 CPF。
- **P2 对照：** P2 crop 落在人工偏好场的位置与质量分位数。

### E6. Position-conditioned Scale 的交叉验证诊断

- **问题：** Scale 是否只是在利用训练帧上过于准确的 P1 中心？
- **假设：** 只有在折外中心或真实部署误差分布下，Scale 目标才可迁移。
- **历史差异：** C1/C3 的公开收益迁移失败；此前 61 源 P1 缓存只支持探索，不能推断全域。
- **数据：** LIVE source split；P1 预测缓存；必要时只对小样本生成折外中心。A/B 分开。
- **输入/输出：** P1 中心、中心误差扰动、Scale 候选 IoU；输出标签稳定性和最佳常数对照。
- **目标：** 无大模型训练；先做 center jitter / cross-fit 分析。
- **Video-ORA 训练：** 否。
- **GPU：** 0–1 h，严格限量。
- **部署：** 不直接部署。
- **评价：** best-scale flip rate、IoU regret、position drift、source-macro。
- **失败判据：** Scale 最优标签在合理中心误差下频繁变化且 IoU regret 不稳定，停止逐帧 Scale 学习。
- **P2 对照：** 使用 P2 实际部署中心，而非人工 GT 中心。

### E7. 多主体覆盖条件门控（弱监督）

- **问题：** 多主体/互动场景是否需要不同候选排序偏好？
- **假设：** subject count、主体框覆盖、互动关系可帮助选择 crop，但通用显著性中心不能替代构图。
- **历史差异：** face top-3 聚合未改善；R98 UNISAL 直接替换负向；R68 指出多人互动与单脸代理不一致。
- **数据：** SACD subject ID/mask、LIVE/GNMC crop GT；Label Studio R42 仅作弱诊断，不能作大规模硬标签。
- **输入/输出：** 候选特征+subject coverage/number/overlap；输出候选 utility 的小型门控残差。
- **目标：** 监督候选 IoU；SACD 只对同 subject ranking；不把 subject count 作为人工真值。
- **Video-ORA 训练：** 否，先小型 readout。
- **GPU：** 0.5–1 h。
- **部署：** 仅在多主体子集提升且单主体不退化时启用条件门控。
- **评价：** 按 subject count、遮挡、互动类别分层的 Source-Macro IoU。
- **失败判据：** 只提高主体完整率却降低 crop IoU，或对单主体明显退化，则停止。
- **P2 对照：** 同源 P2 crop。

### E8. 镜头内稳定性作为受限正则

- **问题：** 当前 crop 轨迹是否存在可见跳动，并且跳动是否损害绝对 IoU？
- **假设：** 在不跨 shot 的条件下，小幅 temporal regularization 可降低视觉跳变且不降低平均 IoU。
- **历史差异：** R69 证明更平滑不等于更准确；不能以 jerk 降低作为成功标准。
- **数据：** LIVE-YT-VC、RetargetVid；保留真实镜头切分与标注轨迹。
- **输入/输出：** P2 每帧/锚点 crop；输出镜头内受限轨迹。
- **目标：** 约束优化：最大化 crop IoU，同时惩罚速度/加速度；shot boundary 强制 reset。
- **Video-ORA 训练：** 否。
- **GPU：** 0 h，CPU DP/优化。
- **部署：** 只作为后处理；不跨镜头中位数。
- **评价：** Source-Macro IoU、速度/加速度/jerk、最大单帧 IoU regret。
- **失败判据：** 轨迹更平滑但 IoU 下降，或关键动作处出现滞后，则停止。
- **P2 对照：** 原 P2 轨迹与同帧 GT。

### E9. 事件/语义条件化构图（长期候选）

- **问题：** 构图是否应随事件阶段、主体关系和叙事角色变化？
- **假设：** 同一视觉主体在互动、动作、反应等语义状态下，最佳 crop policy 不同。
- **历史差异：** R100/R101/R103 的邻帧特征未产生增益，但没有充分测试事件语义条件化 crop。
- **数据：** GAICD/SACD 提供静态偏好；R42 仅作少量事件阶段弱标签；缺少大规模同视频事件语义+crop GT。
- **输入/输出：** image/video event embedding + crop candidates；输出条件化候选 utility。
- **目标：** 先在静态公开数据验证条件化价值；没有同源监督前不做 Video-ORA 联合 SFT。
- **Video-ORA 训练：** 当前否。
- **GPU：** 1–2 h 仅在数据/特征可用时。
- **部署：** 高风险；不改变当前 P2。
- **评价：** 事件类别分层 Source-Macro IoU；条件特征置换负控制。
- **失败判据：** 置换条件与真实条件无差别，或仅训练域提升，则停止。
- **P2 对照：** 同帧 P2 crop。

---

## 8. 推荐主路线与备选路线

### 主路线：E4 候选 Crop Utility Readout

原因：

- R46 已证明固定候选池存在稳定 Oracle Gap；
- R70 已有历史官方正向结果；
- 不需要重训 9B；
- 可使用 LIVE/GNMC 几何监督和 GAICD/SACD 独立偏好进行分层校准；
- 可固定 P2 时间集合，做空间单变量验证。

执行顺序：

1. 恢复当前可核验的视觉 encoder、processor、候选池和 feature identity；不冒充缺失 B3。
2. 先复现冻结基线和 Oracle。
3. 训练线性 readout，再训练小 MLP；保持相同输入、候选和 split。
4. 分别报告 LIVE、RetargetVid、GNMC、GAICD、SACD，不混成一个无解释 Macro。
5. 仅当两个独立视频源池均超过基线、Bootstrap CI 为正、目标比例无明显退化，才重提取 AIC 890/426 视频候选。
6. 最终固定 P2 时间集合，生成一个空间-only 候选包，等待官方回分。

### 备用路线 A：Position Error-conditioned Local Calibration

只在 E1 显示误差集中于特定场景时执行。优先使用现有视觉 embedding + 主体结构弱特征训练小型 residual head；不重训 Video-ORA 全参数。必须证明公开 source-heldout crop IoU 提升，并且多人/单人两组都不发生明显退化。

### 备用路线 B：Shot-constrained Crop Trajectory

只在 E8 显示存在明显轨迹抖动且抖动与 IoU 损失相关时执行。使用镜头内优化，shot boundary reset。优化目标必须包含绝对 IoU，不以 jerk 作为主目标。

### 明确停止路线

- 当前 C1/C3 全局常数收缩策略；官方成绩已分别下降 2.42、1.78。
- 无额外信息的 LIVE/GNMC IoU-DPO；优先 soft-target 或直接 utility regression。
- 逐帧连续 Scale 自由回归；已有 token-CE、IoU1、MLP32 和部署路径反证。
- 单纯扩大候选数量；R46 已显示 129 候选池本身有 Oracle 上限，不支持盲目加密。
- 单纯轨迹平滑、face top-k、saliency center 替换；历史证据未支持。
- 无原生 B 域验证的转置 LIVE 结果迁移。

---

## 9. 资源预算与执行顺序

建议把下一轮空间研究控制在 **不超过 6 GPU 小时** 的首轮验证；若主路线过门，再单独申请完整 9B 重提取或训练预算。

| 阶段 | 工作 | GPU 预算 | 晋级条件 |
|---|---|---:|---|
| P0 | 资产、split、候选和权重身份审计 | 0 h | 所有输入可追溯 |
| P1 | E1 Position error stratification | 0–0.5 h | 有稳定可解释分层 |
| P2 | E2 GAICD / E3 SACD 数据恢复与小 readout | 0.5–1 h | 偏好 schema 正确、无 split 泄漏 |
| P3 | E4 129 候选 utility readout | 1–2 h | 两个源外池超过 baseline |
| P4 | E7 多主体门控 / E8 轨迹约束 | 0–1 h | 只在对应机制证据成立时运行 |
| P5 | 部署候选生成与复核 | 1–2 h | 几何合同、参数预算、时间集合全通过 |

以上是资源上限建议，不是已测耗时。CPU 能完成的几何枚举、候选审计、source bootstrap 均不应占 GPU 预算。

---

## 10. Codex 执行规范

### 10.1 开工前

- 检查当前分支、commit、Git 状态；不得覆盖 P2 或历史包。
- 读取最新 `docs/12`、`docs/07` 和本报告。
- 核验模型、processor、adapter、候选缓存、输入 manifest 的 SHA256。
- 任何缺失的 B3 权重、logits 或特征不得用 V7/V9 冒充。
- 对 SACD 重建 image-level split；同图像所有 subject annotation 必须进入同一 split。
- 对 GAICD 记录原始候选数、有效 MOS 数、ratio 匹配数、位置匹配数和有效 pair 数。

### 10.2 训练约束

- 先线性 readout，再小 MLP；不直接启动 9B 全量 LoRA。
- 先用真实候选特征与 native crop GT；没有真实特征时不得使用随机特征代替。
- LIVE/GNMC IoU soft target 与 pairwise 目标视为同源监督；DPO 仅在 GAICD MOS 或 SACD 人工 ranking 合格时作为额外实验。
- 不得把 SACD subject-conditioned pair 混成通用 pair。
- 不得用验证/测试标签拟合 bin、温度、阈值或偏好权重。
- 固定 P2 时间集合、帧 ID、输出数量和官方坐标格式。

### 10.3 评价规范

每个实验必须报告：

- Dataset 内 Source-Macro IoU；
- Oracle@K、Oracle Regret、Top-1 Candidate IoU；
- Preference Accuracy（仅对人工偏好数据）；
- Position Drift、Scale/Window 合法率；
- A/B/Native 或 1:3/3:1 比例分层；
- Source-level paired Bootstrap 95% CI；
- 参数量、延迟、显存、部署几何一致性。

禁止：

- 用 frame-micro 代替 source-macro；
- 用 Raw IoU 代替最终部署 IoU；
- 用 Oracle 提升声称模型可实现；
- 用公开 IoU 提升换算官方 F1；
- 把 MOS、subject ranking、crop IoU 三种标签合并成一个无类型标量；
- 把官方隐藏测试数据用于调参。

### 10.4 晋级门

候选 readout 只有同时满足以下条件才允许制作完整提交包：

1. 至少两个独立 source-heldout 视频池的 Source-Macro IoU 高于冻结基线；
2. paired Bootstrap 95% CI 不跨 0；
3. A/B 分开评价，不能只在单一方向提升；
4. Position Drift 满足项目几何合同；
5. 原始 frame ID、时间集合和预测数量与 P2 完全一致；
6. 部署后处理复现通过；
7. 参数量仍符合 9B 限制。

未过门则停止，不得通过调参无限扫描验证集。

---

## 11. 最终决策

当前空间方向不应继续押注单帧 Scale 回归，也不应直接恢复大模型 DPO。最有价值的研究路线是**固定候选空间上的效用学习**，因为它已有 R46 Oracle Gap 与 R70 官方正向历史证据，同时可以使用现有 LIVE/GNMC/GAICD/SACD 资产开展可归因验证。

Position 不默认冻结为“已解决”，而是先做困难场景误差分层。Scale 不默认无效，而是从“预测单一人工宽度”改为“在合法候选中估计质量”，并把人工偏好与几何 IoU 分开。时序、事件语义和时空联合不在本报告训练范围内。

**下一项实际工作：先恢复并核验 R70/129 候选特征链和 SACD 原始 ranking pair schema，然后做 E4 小型 readout。未证明源外增益前，不训练 Video-ORA-9B，不制作新官方包。**

---

## 12. 参考文献与资源

1. Yang et al. *Focusing on your subject: Deep subject-aware image composition recommendation networks*. Computational Visual Media, 2023. <https://doi.org/10.1007/s41095-021-0263-3>
2. SACD Dataset. <https://cg.cs.tsinghua.edu.cn/SACD/>
3. Zhang et al. *ProCrop: Learning Aesthetic Image Cropping from Professional Compositions*. AAAI 2026. <https://arxiv.org/abs/2505.22490>
4. ProCrop project/code/data. <https://bwgzk-keke.github.io/ProCrop/>
5. Dong et al. *CROP: Expert-Aligned Image Cropping via Compositional Reasoning and Optimizing Preference*. arXiv 2026. <https://arxiv.org/abs/2605.12545>
6. Zhang et al. *Discrete Annotation, Continuous Preference: Rethinking Supervision for Accurate and Generalizable Aesthetic Image Cropping*. arXiv 2026. <https://arxiv.org/abs/2610.00582>
7. CPIC official repository. <https://github.com/zzqingz/CPIC>
8. GenCrop. *Learning Subject-Aware Cropping by Outpainting Professional Photos*. AAAI 2024. <https://ojs.aaai.org/index.php/AAAI/article/view/27990>
9. HumanCrop-Thinker. *An inference-driven framework with explicit thinking for explainable human-centric image cropping*. Expert Systems with Applications, 2026. <https://doi.org/10.1016/j.eswa.2026.131664>
10. Awesome Aesthetic Evaluation and Cropping. <https://github.com/bcmi/Awesome-Aesthetic-Evaluation-and-Cropping>
11. Human-Centric Image Cropping implementation. <https://github.com/bcmi/Human-Centric-Image-Cropping>
12. GAFIC implementation. <https://github.com/AIVRC/GAFIC>
