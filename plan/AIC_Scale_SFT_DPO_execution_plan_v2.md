# AIC 空间裁剪 Scale：SFT → 候选生成 → 偏好构造 → DPO 执行方案

**版本：** v2.0  
**研究范围：** 全局方向二「空间裁剪」，只研究 Scale，不重新训练时间模型，不开展时空联合训练。  
**GPU 预算：** 单张 RTX 4090 vGPU，48 GB 显存，GPU 时间严格不超过 12 小时。  
**执行对象：** Codex。  
**决策状态：** 待执行；本文不代表实验已运行。

---

## 0. 本轮决策与不可变约束

### 0.1 唯一研究问题

当前 AIC 空间系统已经具备较强的位置定位能力。本轮不重新优化 Position，而是研究：

> 在保持现有位置预测能力的前提下，模型能否学习人工构图偏好的裁剪尺度，并通过 DPO 提高实际部署路径上的构图质量？

将 Scale 定义为最大合法裁剪窗口的线性缩放系数：

\[
 s\in[s_{min},1],\qquad
 w=s\cdot w_{max},\quad h=s\cdot h_{max}
\]

其中 \((w_{max},h_{max})\) 是给定输入尺寸和目标画幅下的最大合法窗口。窗口中心由已验证的位置分支提供，第一轮训练中固定位置，不允许 Scale 实验暗中改变中心。

### 0.2 当前系统事实

1. 用户报告的同一轮半决赛评测中，P2 为 **44.65**，P1 为 **44.63**。映射到交付包 ZIP SHA256：P2 `1489ace9d0ab5108af257800fb7a0cf83d07cfccc2b8c9a0c2f53e16fd557644`；P1 `b2280d10e712c44810cffea6a40a25aabf32d87ad7b6c761c533c003c133f7ee`。分数来源为用户回报，平台页面未独立核验。
2. B_MIX 在同轮评测中为 **42.88**，ZIP SHA256 `5ecd5d3af634ba8981d60ab94aed4529c4a482905f4774dca7bc5b037cb55b6a`。
3. P1/P2 均是本轮需要核对的位置基线。按本执行方案先使用 P1 的冻结位置合同作为默认父分支；P2 是当前分数更高的部署对照（+0.02），不是同分对照。明天启动训练前必须核验两者真实模型/adapter 身份及中心输出，不能仅凭交付 ZIP 或分数推断位置质量。不得将整包分数归因于某一个单独 adapter 或后处理变量。
4. B_MIX 不作为最终部署父包。B_MIX 只允许作为候选生成来源或误差分析对象。它的整包低分不能证明每个单帧候选都差，也不能证明 Scale 方向无效。
5. 旧完整框实验出现过尺度塌缩、Raw 指标与部署路径相反等问题。本轮必须先通过候选 Oracle 和真实路径评价，不能直接复刻旧 FullCrop 训练。
6. 用户已完成 SACD、GAICD 的获取和验证。本轮直接审计现有本地 manifest、hash、标注解析与 split，不重新下载数据。
7. PHD2、YouTube Highlights 媒体已删除。本轮不使用这两个数据集，不开展高光语义 DPO，也不开展时空联合训练。
8. SACD、GAICD 已纳入本轮空间数据候选。SACD/GAICD 不得被误写成视频数据；二者提供静态图像构图监督。
9. LIVE-YT-VC、GNMC 继续保留。LIVE-YT-VC 提供视频帧上的 9:16 人工裁剪框；GNMC 提供多画幅静态裁剪监督。
10. SACD、GAICD 的许可、来源和用途限制必须沿用数据集原始条款。代码不得假定数据集可用于商业发布。

### 0.3 GPU 预算：四项合计 12 小时

| 阶段 | GPU 时间上限 | GPU 任务 |
|---|---:|---|
| Scale-SFT | 4 h | LoRA 训练、必要的 GPU 验证 |
| 多候选生成与评价 | 3 h | 模型候选生成、真实路径解码与 GPU 评价 |
| 偏好数据构造 | 1 h | 以 CPU 为主；GPU 时间只用于必要的模型打分/候选重算 |
| Spatial DPO | 4 h | DPO 训练及训练中验证 |
| **总计** | **12 h** | **不得超预算** |

数据清点、manifest 检查、CPU 几何测试、脚本开发、数据转换、日志整理和打包不计入 GPU 时间，但必须记录墙钟时间。GPU 计时从第一个 CUDA 初始化/模型加载任务开始，直到最后一个 GPU 任务结束；不得通过重启进程、切换 notebook 或分阶段记录规避累计计时。

如果某阶段提前通过或提前失败，剩余 GPU 时间不得自动转化为无计划扫参。只有在预先定义的门槛通过后，才可转入下一阶段。

---

## 1. 研究依据：为什么本轮选择 Scale + SFT-DPO

### 1.1 现有实验给出的边界

项目路线 12 已经记录以下事实：

- LIVE-YT-VC 新 A 的 200-source/1200-frame 验证 IoU 从 0.607452 提升到 0.627236，差值 +0.019784，配对区间为正。
- LIVE 转置构造的 B 验证从 0.460591 提升到 0.586134；这是转置几何结果，不等价于真实 upright 域确认。
- P2/P1 的半决赛用户报告分数分别为 44.65/44.63，构成当前高质量空间部署基线；分数不能代替 adapter 与位置输出身份核验。
- B_MIX 同轮为 42.88。它改变了大量框和宽度，但整包表现没有超过 P1/P2。
- FULLCROP_A_V1 的 full IoU 0.625878，低于匹配 center 的 0.628279。
- FULLCROP_A_IOU1 出现尺度预测塌缩。
- Geometry MLP32 的 Raw 增益没有转化为真实轨迹增益。
- FULLCROP_B_V2 的 Raw 0.872317，但真实路径 IoU 0.478158，低于新 B 的 0.512216。

因此，本轮不能把「自由预测完整框」作为默认方案。我们需要一个更受约束的 Scale 学习问题，并在每一步用真实部署路径检查。

### 1.2 SFT-DPO 的外部研究依据

**CROP: Expert-Aligned Image Cropping via Compositional Reasoning and Optimizing Preference**（arXiv:2605.12545，2026-05-09）提出图像裁剪 VLM 的两阶段训练：先用 GAICD MOS 构造 SFT，再使用同一数据的高 MOS/低 MOS crop 构造 DPO 偏好。论文报告使用 Qwen2.5-VL-7B、LoRA；GAICD 训练数据构造约 4,000 条 SFT 样本和 8,000 条 DPO 偏好样本。

参考：
- 论文：https://arxiv.org/abs/2605.12545
- 论文方法与数据构造：https://bytez.com/docs/arxiv/2605.12545/paper

这项工作支持「图像裁剪可以使用 SFT → 偏好数据 → DPO」的可行性，但不证明相同方法必然适用于 AIC。论文目标是静态图像审美裁剪；AIC 要求固定目标画幅、合法 `[x,y,w]` 输出、视频帧域迁移和严格部署几何。因此，本方案只迁移训练范式，不照搬论文的三阶段推理架构、数据量或超参数。

**GAICD**：Zeng et al., CVPR 2019 / TPAMI 2022。每张图像提供不超过约 90 个候选 crop，候选有至少 7 名评分者的 MOS（1–5）；MOS 表示候选之间的审美质量，不是唯一 GT 框。论文明确提出用候选排名评价裁剪质量，而不只依赖 IoU。
- CVPR 论文：https://openaccess.thecvf.com/content_CVPR_2019/papers/Zeng_Reliable_and_Efficient_Image_Cropping_A_Grid_Anchor_Based_Approach_CVPR_2019_paper.pdf
- TPAMI 论文：https://www4.comp.polyu.edu.hk/~cslzhang/paper/GAIC-PAMI.pdf
- 官方代码与数据说明：https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch

**SACD**：Yang et al., Computational Visual Media 2023。数据集包含 2,777 张图像、超过 24,000 个构图窗口和超过 5.2M 个 ranked pairs。标注以指定主体为条件，包含多主体场景和单主体场景；不能把所有 subject-conditioned 偏好不加区分地当成无条件通用构图标签。
- 官方数据页：https://cg.cs.tsinghua.edu.cn/SACD/
- 论文：https://doi.org/10.1007/s41095-021-0263-3
- 论文 PDF：https://cg.cs.tsinghua.edu.cn/cvmj/papers/CVM0263.pdf

---

## 2. 数据选择与监督合同

### 2.1 数据集职责

| 数据集 | 本轮用途 | 可用监督 | 不允许的误用 |
|---|---|---|---|
| LIVE-YT-VC | A 方向视频帧 Scale SFT、真实视频域验证 | 原始视频帧 + 人工 9:16 crop 框 | 不得随机按帧切分；不得把单一框当作完整审美分布 |
| GNMC | 多画幅静态 Scale SFT、跨域检查 | 多目标画幅 crop 框 | 不得把静态图像验证等同于 AIC 视频验证 |
| GAICD | 主要偏好监督；候选 MOS 排名；SFT 辅助 | 同图像多个候选框及 MOS 1–5 | 不得把 MOS 当 IoU；不得把未评分候选当负例；不得跨 image 形成偏好对 |
| SACD | subject-aware 偏好辅助；单主体/指定主体分层实验 | 同图像候选 crop、subject 条件和 ranked pairs | 不得忽略 subject 条件；不得将多主体 subject-A 的偏好直接混为无条件标签 |
| RetargetVid | 仅作位置/路径回归检查或历史对照 | 多标注者视频裁剪框 | 项目审计发现框基本为最大合法窗口；不作为主要 Scale 偏好来源 |

### 2.2 训练/验证划分原则

**必须以源图像/视频源为最小分组单位。** 同一源的所有帧、crop 候选、MOS、ranked pairs、增强视图必须处于同一个 split。

- LIVE-YT-VC：沿用项目已审计的 source-level manifest。项目后期 A/B 记录为 1600 train / 200 dev；早期发布 manifest 为 1115/124/227/334。两者不可混用。Codex 必须读取本轮实际训练使用的 manifest，并记录 SHA256、源 ID 集合和 split 交集。若无法证明 train/dev source-disjoint，停止训练。
- GNMC：沿用已审计的 7000 train / 1000 validation / 2000 test source 划分。只允许使用 train 和 validation；test 永不用于训练、选模或阈值调整。
- GAICD：只使用 journal release 的官方 train=2636、val=200、test=500。只用 train 训练、val 选模；test 只允许在方案冻结后做一次最终数据集评测。项目导入脚本指出 journal val 是从 journal test 中随机抽出的 200 张，且 conference-test 与 journal-val 有 50 张重合。因此必须只以 journal split 为准，不得把 conference release 的重叠图像重新加入 train。
- SACD：沿用官方 8:1:1 split；如本地验证记录采用不同划分，优先使用已验证的本地 source manifest，但必须确保 image/source group 不跨 split。不要把原始 ranked-pair 行随机拆分。

**禁止：** 先生成所有偏好对再随机拆行划分。必须先划分源，再在每个 split 内构造偏好对。

### 2.3 数据使用优先级

第一优先：GAICD train 的同图像 MOS 排名，直接支持 DPO。  
第二优先：SACD train 的同图像、同 subject ranked pairs，作为 subject-aware 辅助偏好。  
第三优先：LIVE-YT-VC train 的单 GT 框，构造受控候选并按部署路径 IoU 排序。  
第四优先：GNMC train 的多画幅 GT 框，构造受控候选并按部署路径 IoU 排序。

LIVE/GNMC 的 IoU 偏好是「对单个标注框的几何接近度」，GAICD/SACD 是「人类审美/主体条件偏好」。二者语义不同。必须保留 `preference_source` 字段，并分别报告指标。不得把所有偏好对不加权地合并成一个无来源数据池。

### 2.4 目标画幅过滤

AIC 部署不是任意画幅裁剪，而是几何路线对应的固定目标画幅：

- A：横屏输入 → 9:16。
- B：竖屏输入 → 16:9。

GAICD、SACD、GNMC 中的候选框只有在目标画幅与 AIC 目标比例一致、且按 AIC 坐标合同可合法部署时，才可直接进入该几何分支的候选偏好训练。

不得把不同 aspect ratio 的 MOS 或 ranked pair 当成同一尺度任务的直接偏好。若数据集没有足够的精确比例候选：

1. 保留原始比例作为独立辅助任务，不伪造目标比例标签。
2. 可使用图像转置构造倒数比例的几何增强，但必须同步转置图像、候选框、subject box 和所有几何元数据。
3. 转置样本只能作为几何增强，不得替代真实 upright/native 目标域验证。

---

## 3. 模型与动作空间：先解决 Scale，不重新训练 Position

### 3.1 父模型

- 默认父模型：按本执行方案使用 P1 空间配置和冻结位置合同；其半决赛用户报告分数为 44.63。
- P2 的用户报告分数为 44.65，作为更高分部署对照，不与 P1 权重混合；启动训练前核验 P1/P2 权重与中心输出身份。
- 不以 B_MIX 作为父模型。
- 不加载第二个独立视觉/语言模型作为推理组件。
- 本轮训练必须替换/更新现有空间 adapter；不得在 P1 的推理链上额外堆叠一个新的独立模型或 adapter，导致参数总数超过 9B。

Codex 必须在训练前输出：

1. P1 基座、A/B adapter、tokenizer、推理配置的 SHA256。
2. P1/P2 真实输出协议及坐标单位。
3. 当前总推理参数量（包含所有实际激活组件和部署缓冲口径）。
4. 本轮训练后 adapter 的替换方式和最终参数计数。

### 3.2 固定 Position 的 Scale 参数化

第一轮采用受约束参数化：

\[
B(c_x,c_y,s)=\left(x(c_x,s),y(c_y,s),w_{max}s,h_{max}s\right)
\]

- `cx, cy` 使用 P1 现有位置预测，第一轮冻结。
- `s` 是相对于最大合法窗口的线性尺度系数。
- crop 目标比例始终固定。
- crop 窗口必须落在原图边界内。
- 若当前 P1 对某几何路线只预测一个自由轴，保持该几何合同，不得擅自改变非活动轴。

尺度候选建议使用离散有序 bins，而不是直接让模型自由输出任意浮点数。初始候选网格由数据训练集的人工尺度分布确定，覆盖从最大合法窗口到明显缩小的窗口；具体 `s` 值必须通过训练集统计生成并写入配置，禁止 Codex 随手指定一组看起来合理的常数。

离散 scale bin 的优点：

- 避免连续数字 token 回归再次塌缩。
- 使 DPO 的 chosen/rejected 输出简短且 tokenization 稳定。
- 可直接映射为确定性几何变换。
- 能分别统计每个尺度 bin 的使用率和 IoU。

本轮不同时开放中心和尺度两个自由度。只有 Scale 通过晋级门后，才可在后续实验中联合微调中心。

### 3.3 候选输出协议

优先采用离散 `scale_id`，而不是生成完整自由坐标：

```json
{"scale_id": 3}
```

推理程序使用固定的 P1 中心与 `scale_id` 查表，得到最终 `[x,y,w]`。如果现有 Video-ORA 结构不能稳定输出 `scale_id`，Codex 必须先完成 tokenizer round-trip 和解析测试，再决定是否采用定点小数坐标。不得未经测试直接切换为自然语言坐标输出。

候选 `scale_id` 必须在输入中有固定、明确的语义顺序。训练和推理时 scale-id 到实际 `s` 的映射必须使用同一个版本化配置文件。

---

## 4. Stage 0：非 GPU 数据与几何审计

**GPU 预算：0 h。** 这是启动任何训练前的强制门控。

### 4.1 数据审计

Codex 逐数据集生成 `data_audit.json`，至少包含：

- dataset/version/license/source URL。
- media count、unique source count、annotation count。
- train/val/test 数量及 source ID 交集。
- 图片宽高、输入画幅分布。
- crop 目标画幅分布。
- crop 坐标范围、非法框数量、空框数量。
- MOS 缺失/未评分数量及值域。
- SACD subject 数量、每图 subject 数、每 subject crop 数。
- 候选框数量分布、同图偏好对数量分布。
- 所有原始 manifest 和标注文件 SHA256。

必须识别 GAICD `MOS=-2/-1` 未评分项；这些候选不进入 chosen/rejected。

必须核验 SACD ranked pair 两端都属于同一 image、同一 subject、同一 split。若 pair 文件没有明确 subject 关联，先停止 SACD 偏好训练，不得自行猜测。

### 4.2 几何合同测试

对 A/B 两种路线各抽取不少于 1000 个训练框，执行：

1. 原图像素坐标 → 归一化坐标 → 原图像素坐标往返。
2. 原始 GT 框 → AIC `[x,y,w]` 表示 → crop 窗口重建。
3. scale=1 时，结果必须与最大合法窗口逐像素一致。
4. scale 变小时，中心必须保持不变（允许整数 rounding 误差不超过 1 px）。
5. 目标画幅误差不超过 1 个像素引起的 rounding tolerance。
6. crop 不得越界、宽高不得为 0、不得出现负宽高。
7. transpose augmentation 必须满足坐标变换的双向 round-trip。

测试失败时停止，不允许进入训练。

### 4.3 源级隔离测试

构造 split manifest 后，检查：

```text
intersection(train_sources, val_sources) == empty
intersection(train_sources, test_sources) == empty
intersection(val_sources, test_sources) == empty
```

对同图像的全部候选、全部 MOS、全部 ranked pairs 执行 group-level split 检查。重复图像需用 image hash 或原始 source ID 检测，不能只依赖文件名。

### 4.4 训练样本可行性门

完成数据审计后，按目标画幅统计可用于 A/B 的有效样本量。

如果某个几何分支满足任一条件，则该分支本轮不训练：

- 精确目标比例下，有效训练源少于 100。
- 该分支可形成有效偏好对的源少于 100。
- train/val 存在 source overlap。
- 超过 5% 的框无法通过坐标合同验证，且无法追溯修复。

不得用重复采样同一源的帧数来掩盖独立源不足。

---

## 5. Stage 1：Scale-SFT（GPU ≤ 4 h）

### 5.1 目标

让 P1 空间模型在保持位置预测不变的条件下，学会根据图像内容选择一个合理的离散尺度。

本阶段只验证模型是否能学习尺度映射。不要把 DPO 偏好目标混入 SFT。

### 5.2 SFT 样本构造

每个样本包含：

- 原图。
- AIC 目标画幅。
- P1 固定中心（归一化）。
- 训练集 GT crop。
- 由 GT crop 与 P1 中心计算得到的最优可行 scale bin。
- 数据来源与 source ID。

因为 LIVE/GNMC 的单 GT 框可能与 P1 中心不一致，必须先计算在 P1 固定中心条件下的可行最优尺度：

\[
 s^* = \arg\max_{s\in\mathcal S} IoU(B(c_{P1},s),B^*)
\]

这一步非常关键：SFT 标签必须是在「位置固定」约束下求出的尺度标签。不得直接取 GT 框宽度作为尺度标签而忽略中心偏差。

如果固定中心下所有尺度的 IoU 都很低，样本标记为 `position_mismatch`，不进入 Scale-SFT 主损失，但保留在诊断报告中。不得让 Scale 分支通过改变尺度补偿明显的位置错误。

数据来源策略：

- LIVE-YT-VC：主视频域 SFT，A 路线优先。
- GNMC：目标比例匹配的静态图像辅助 SFT。
- GAICD：若目标比例筛选后存在可靠单图 top-M MOS 候选，可用最高 MOS 候选作辅助 SFT；不将 MOS 直接换算成 IoU。
- SACD：仅使用 subject 条件明确、可与当前输入合同兼容的样本；默认不进入第一版主 SFT，避免引入推理时不存在的 subject prompt。

### 5.3 训练策略

- 基座：P1 对应 Video-ORA-9B。
- 只训练现有空间 adapter 的 Scale 输出相关路径；冻结时间分支和视觉主干。
- Position 相关输出/adapter 权重冻结，除非当前实现无法拆分；若无法拆分，必须报告实际可训练参数名和比例，并以中心偏差约束作为保护。
- 不增加新的部署推理模型。
- 不解冻整个 9B 模型。
- 不默认使用 QLoRA；先沿用项目已验证的 BF16/LoRA 训练路径。若 48 GB 显存无法容纳，先缩小 batch/序列长度，不得直接改变模型结构。

损失：

\[
\mathcal L_{SFT}=CE(\hat k,k^*)+\lambda_{pos}\mathcal L_{pos-consistency}
\]

若 Position 分支完全冻结，`L_pos-consistency` 可仅作为验证指标，不必增加训练损失。

### 5.4 训练数据混合

不按原始图片数量直接混合。先计算各来源独立 source 数，再做 source-balanced sampling。

建议首轮以 LIVE-YT-VC 为主、GNMC 为辅助；GAICD/SACD 只在目标比例与标注语义匹配时加入。混合比例由有效 source 数和目标比例覆盖确定，写入配置并固定，不在训练中动态扫比例。

每个 batch 必须记录 `dataset_source`、`source_id`、`target_ratio`、`scale_bin`。

### 5.5 SFT 验证指标

主指标按 source macro：

1. `scale_bin_accuracy`。
2. `scale_mae`。
3. `fixed_center_deploy_iou`：固定 P1 中心，经过真实部署几何投影后计算 IoU。
4. `position_drift_px`：SFT 前后中心变化；若理论上冻结 Position，应为 0 或仅有解析 rounding 误差。
5. `scale_collapse_rate`：模型输出同一 scale bin 的样本比例。
6. 每个目标比例、每个数据源分别报告。

**SFT 晋级门：**

- 真实部署路径的 source-macro IoU 高于 P1 固定最大窗口基线。
- 配对 source bootstrap 95% CI 下界大于 0，或在样本源数不足以形成窄区间时，至少不能显著为负，并明确标记为探索性结果。
- `scale_collapse_rate` 不得超过训练集多数尺度 bin 的自然占比上界；若模型几乎只输出一个 bin，判定失败。
- Position drift 不得超过 1 px rounding tolerance。

未通过 SFT 晋级门：停止，不运行多候选 DPO。

---

## 6. Stage 2：多候选生成与评价（GPU ≤ 3 h）

### 6.1 目标

判断当前策略的候选空间是否包含比当前 greedy 输出更好的尺度。DPO 只有在 Oracle@K 高于 Greedy 时才有可利用的候选改进空间。

### 6.2 候选池来源

每张图像构造候选池：

1. SFT greedy 输出。
2. SFT stochastic 输出（不同随机种子；温度只用于候选多样性，不作为训练超参扫描）。
3. 固定中心下的离散尺度网格候选。
4. 若可合法复用，加入 P1/P2/B_MIX 对应输出作为外部候选，但所有候选都必须通过同一几何投影器。

B_MIX 只作为候选来源，不作为父模型。不得因其整包得分低而直接删除所有候选；也不得预设其候选一定更好。

### 6.3 候选数量

初始 `K=8`，由以下组成：

- 1 个 greedy。
- 3 个 SFT stochastic。
- 4 个固定中心尺度网格候选。

如果 GPU 预算不足，先降到 K=4，不得减少 source-held-out 验证源数来保留 K=8。

候选去重：按最终投影后的整数 `[x,y,w]` 去重。相同几何框不能重复计入 K。

### 6.4 Oracle 与多样性指标

对有 GT 的公开验证数据，计算：

\[
Oracle@K_v=\max_{b\in C_v} IoU(b,B_v^*)
\]

\[
Gap_v=Oracle@K_v-IoU(Greedy_v,B_v^*)
\]

按 source macro 报告：

- `greedy_deploy_iou`。
- `oracle_at_4_deploy_iou`、`oracle_at_8_deploy_iou`。
- `oracle_gain`。
- `unique_candidate_ratio`。
- `scale_bin_coverage`。
- `candidate_pair_valid_rate`。
- 位置中心是否固定。

GAICD：以同图像 MOS 最高候选为 Oracle，另报告候选排序与 MOS 的关系。  
SACD：以同图像、同 subject 的已标注 ranked pairs 为准，不用跨 subject 的 crop 直接算统一 Oracle。  
LIVE/GNMC：以部署路径 IoU 计算 Oracle。

### 6.5 候选门

DPO 启动前必须同时满足：

1. `oracle_gain` 的 source-macro 均值为正。
2. source bootstrap 95% CI 下界大于 0；若 CI 跨 0，DPO 只能作为小规模探索，不得占满 4 h。
3. 至少 20% 的 source 存在超过训练集噪声阈值的有效候选差异。
4. 候选尺度覆盖不少于 3 个有效 scale bins。
5. 优质候选不是全部来自单一数据源或单一几何域。

若候选门失败：停止 DPO。回到候选空间/Scale 动作参数化诊断，不得通过放宽偏好阈值制造大量偏好对。

---

## 7. Stage 3：偏好数据构造（GPU ≤ 1 h）

### 7.1 核心原则

偏好必须来自可追溯的监督：

- GAICD：同一图像、同一目标比例下的 MOS 排名。
- SACD：同一图像、同一 subject 下的人工 ranked pairs。
- LIVE/GNMC：同一图像/帧、固定 P1 中心、真实部署路径 IoU 排名。

禁止：

- 不同图像之间构造 chosen/rejected。
- 不同视频源之间比较 IoU 后构造偏好。
- 把 MOS 直接当 IoU。
- 把未评分 crop 当 rejected。
- 把 SACD 不同 subject 的 crop 排名混在一起。
- 用验证/测试 GT 构造训练偏好。
- 用官方半决赛分数或官方测试预测构造偏好。

### 7.2 GAICD 偏好对

每个图像内，先过滤 `valid=true` 的 MOS 候选，再按 MOS 排序。

- chosen：MOS 高的候选。
- rejected：MOS 低的候选。
- 优先选择分差明显且候选几何有差异的 pair。
- 对相同 MOS 或分差小于预先设定噪声阈值的候选，不构造偏好。
- 每张图像最多采样固定数量的 pair，避免候选多的图像支配训练。

GAICD 的 MOS 是至少 7 名评分者的平均分。它衡量静态图像审美构图，不是 AIC 视频 GT。该偏好只能作为通用构图先验，最终必须由 LIVE/GNMC 源外验证。

### 7.3 SACD 偏好对

SACD 原始 ranked pair 必须保留 image ID 和 subject ID。

- 首轮只使用 single-subject 样本，或确认模型输入合同中存在同等 subject 条件的样本。
- multi-subject 样本若无法向模型提供指定 subject，排除出主训练集。
- 同一 subject 内按人工排序构造 chosen/rejected。
- 不根据 subject bbox 的大小自行推断哪个 crop 更好。

SACD 作为辅助数据源，首轮权重不得超过主偏好样本的一半；若 source-level 诊断显示它损害 LIVE/GNMC 泛化，则关闭 SACD 分支。

### 7.4 LIVE/GNMC 几何偏好对

对每张图像/视频帧：

1. 固定 P1 中心。
2. 从候选 Scale 集合生成合法窗口。
3. 经 AIC 部署几何投影器得到最终框。
4. 与该样本人工 GT 框计算 IoU。
5. 只有 IoU 差大于阈值 `delta_iou` 的候选对才生成偏好。

`delta_iou` 不得随意指定。先在训练集计算同源重复、坐标 rounding 和人工标注误差可解释的 IoU 差分布，再在训练集确定一个固定阈值。阈值写入配置并锁定；验证集不得用于挑选阈值。

如果 LIVE/GNMC 单 GT 监督造成过强的唯一框偏好，GAICD/SACD 人工排名应保留为不同 `preference_source`，不得直接将它们压成同一标尺。

### 7.5 偏好数据 schema

建议统一 schema：

```json
{
  "sample_id": "source_id:frame_id:pair_id",
  "source_id": "source-level-id",
  "dataset_source": "GAICD|SACD|LIVE_YT_VC|GNMC",
  "target_ratio": "9:16",
  "input_image": "relative/path/image.jpg",
  "position_source": "P1_frozen",
  "chosen": {"scale_id": 4, "crop_xywh": [0, 0, 0, 0]},
  "rejected": {"scale_id": 1, "crop_xywh": [0, 0, 0, 0]},
  "preference_source": "mos|ranked_pair|deploy_iou",
  "chosen_score": 4.3,
  "rejected_score": 2.1,
  "score_name": "MOS",
  "score_semantics": "same_image_candidate_mean_opinion_score",
  "score_delta": 2.2,
  "valid_geometry": true,
  "split": "train"
}
```

上面的坐标和分数仅为 schema 示例，不是真实样本。若使用 GAICD/SACD，`score_name` 和 `score_semantics` 必须准确标记，不得把 MOS 改名为 IoU。

### 7.6 偏好数据质量门

输出 `preference_audit.json`，包含：

- 各数据源 pair 数量和 unique source 数。
- chosen/rejected 分数差分布。
- scale bin pair 覆盖矩阵。
- 几何重复率。
- 同图像 pair 数上限是否生效。
- split/source overlap。
- MOS 缺失处理。
- SACD subject 合同通过率。
- LIVE/GNMC 的 `delta_iou` 来源与固定值。

若有效偏好对少于 2,000 对，或 unique source 少于 300，必须缩小 DPO 目标或停止；不得通过复制 pair 扩大数据集。

---

## 8. Stage 4：Spatial DPO（GPU ≤ 4 h）

### 8.1 训练目标

使用 Stage 1 的 Scale-SFT checkpoint 作为 policy 初始化，同时使用同一 SFT checkpoint 作为冻结 reference。

标准 DPO：

\[
\mathcal L_{DPO}=-\mathbb E_{(x,y_w,y_l)}\left[
\log\sigma\left(\beta\left[
\log\frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)}-
\log\frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)}
\right]\right)\right]
\]

其中 `y_w/y_l` 是同一输入下的 chosen/rejected `scale_id`。如果现有模型使用结构化 JSON 输出，DPO 必须对相同 prompt 和相同格式计算 completion log-prob；不得对 chosen/rejected 使用不同模板。

### 8.2 训练范围

- 只更新 Stage 1 的空间 adapter。
- 冻结 Video-ORA 主干、时间模块和 P1 Position 分支。
- 不引入额外 reward model。
- 不引入在线 RL。
- 不把 GAICD/SACD MOS 数值直接作为 DPO reward；DPO 只使用构造好的 chosen/rejected pair。
- 不允许把验证/测试偏好对混入训练。

### 8.3 防止 DPO 退化

每个 checkpoint 至少记录：

- train/eval DPO loss。
- chosen/rejected log-prob。
- chosen-rejected margin。
- reference KL 或等价 log-ratio 统计。
- grad norm。
- scale-bin 输出分布。
- 位置中心漂移。
- 真实部署路径 IoU。

如果 DPO loss 下降，但 `chosen-rejected margin` 没有合理变化，或模型输出坍缩到单一尺度 bin，应停止训练并回退到 SFT。

### 8.4 选模规则

不得仅按 DPO eval loss 选择 checkpoint。

主选模依据：LIVE-YT-VC source-macro deploy IoU；次级检查 GNMC、GAICD、SACD 各自独立指标。若 LIVE 提升但 GAICD/SACD 排名严重退化，必须报告域冲突，不得用混合平均分掩盖。

最终只允许一个 DPO checkpoint 进入候选部署。不得扫描多个 beta、学习率、温度后挑选最优验证组合。本轮使用一个预注册配置；如果训练不稳定，停止并保留 SFT。

---

## 9. 评价指标合同：禁止编造、禁止指标漂移

### 9.1 不存在可直接训练的官方 F

公开数据没有 AIC 半决赛逐视频完整 GT 高光帧与官方 crop GT，因此本轮不能计算真实官方 F，也不能声称优化了官方 F。

本轮只报告明确命名的公开代理指标。所有结果必须标记为 `proxy` 或数据集原生指标。

### 9.2 主指标：LIVE/GNMC 的真实部署路径 IoU

对每个 source 先聚合其帧级 IoU，再对 source 等权平均：

\[
IoU_{macro} = \frac{1}{|S|}\sum_{s\in S}\frac{1}{|F_s|}\sum_{t\in F_s}IoU_{s,t}
\]

同时报告 micro IoU，但不得用 micro 取代 source-macro 主指标。

所有 IoU 都必须在以下流程之后计算：

1. 模型输出 scale action。
2. 使用冻结 P1 中心。
3. scale bin 映射为 crop 窗口。
4. 执行 AIC 实际部署投影、整数 rounding、边界 clamp。
5. 转回原图坐标。
6. 与原图 GT 框计算 IoU。

Raw 坐标 IoU 只能作为诊断指标，不能作为主指标。

### 9.3 GAICD 原生指标

GAICD 的主要评价应尊重其候选 MOS 任务：

- Per-image Spearman Rank Correlation（SRCC）。
- Per-image Pairwise Accuracy（仅在 MOS 差超过预注册阈值的 pair 上）。
- Top-1 MOS。
- Regret-to-best-MOS：同图像最佳可评分候选 MOS 减模型选择候选 MOS。

先按 image 计算，再对 image macro-average。不得将 288,069 个 crop 当成独立样本计算置信区间。

### 9.4 SACD 原生指标

- Same-image/same-subject pairwise accuracy。
- Subject-macro ranking accuracy。
- 若候选 crop 是已标注窗口，报告 chosen crop 是否命中标注的 good-window 集合。

不得把 SACD 的 subject-conditioned preference 指标与 GAICD 的无 subject MOS 直接平均成一个总分。

### 9.5 候选 Oracle 指标

- `Greedy@1`：当前模型默认输出质量。
- `Oracle@K`：候选集合中最高真实监督质量。
- `OracleGain = Oracle@K - Greedy@1`。
- `DPO@1`：DPO 最终选择的候选质量。
- `SelectionEfficiency = (DPO@1 - Greedy@1) / (Oracle@K - Greedy@1)`，仅当分母大于预注册最小值时计算。

Oracle@K 是候选空间上限，不是模型成绩，不得当成实际可部署结果。

### 9.6 置信区间与显著性

- 以 source/image 为 bootstrap 单位，不以帧、crop 或 pair 为独立单位。
- 使用 paired bootstrap 比较同一 source 上的 SFT 与 DPO。
- 报告 95% CI、source 数量、样本帧数。
- 同时报告正向 source 比例、负向 source 比例。
- 若 CI 跨 0，结论必须写「尚无稳定增益证据」，不能写「提升」。

### 9.7 禁止指标

不得自行创造或混用：

- `official F`（公开数据没有官方 GT 时禁止）。
- 将 GAICD MOS 伪装成 IoU。
- 将 SACD ranked pair 数量当作独立图片数。
- 将所有帧混合计算的 micro 指标称为 source-macro。
- 用 Raw IoU 替代部署路径 IoU。
- 用 Oracle@K 替代模型实际选择质量。
- 将训练 loss 下降写成模型质量提升。

---

## 10. 12 小时执行编排与停止规则

| 阶段 | 预算 | 具体工作 | 成功条件 | 失败动作 |
|---|---:|---|---|---|
| Scale-SFT | 4 h | 固定中心、离散 scale action、LoRA SFT | deploy source-macro IoU 超过 P1；无尺度塌缩 | 停止，不进入 DPO |
| Candidate | 3 h | K=8 候选、几何投影、Oracle@K | OracleGain 正向且跨 source 稳定 | 停止 DPO，检查候选空间 |
| Preference | 1 h | GAICD MOS、SACD ranked pairs、LIVE/GNMC deploy IoU 构造偏好 | 偏好源级隔离、有效 pair 和源数量过门 | 缩小来源或停止，不复制 pair |
| Spatial DPO | 4 h | SFT policy/reference，固定配置 DPO | DPO 超过 SFT 的部署 source-macro IoU，CI 支持 | 回退 SFT |
| **总计** | **12 h** |  |  |  |

### 10.1 GPU 时间硬计数

Codex 必须维护 `gpu_budget.json`：

```json
{
  "budget_hours": 12.0,
  "stages": {
    "scale_sft": {"limit_hours": 4.0, "used_hours": 0.0},
    "candidate_generation": {"limit_hours": 3.0, "used_hours": 0.0},
    "preference_build": {"limit_hours": 1.0, "used_hours": 0.0},
    "spatial_dpo": {"limit_hours": 4.0, "used_hours": 0.0}
  }
}
```

每次 CUDA 任务记录开始/结束时间、GPU 型号、显存峰值、进程、任务、checkpoint 和日志路径。总 GPU 时间超过 12 h 即违反预算。

### 10.2 48 GB 显存预检

训练前必须执行单 batch forward/backward smoke：

- 记录模型加载后 allocated/reserved。
- 记录 forward、backward、optimizer step 峰值显存。
- 记录实际 trainable parameter 数量。
- 检查 BF16 是否可用。
- 检查 gradient checkpointing 是否生效。
- 检查保存 adapter 后可重新加载。

若 OOM，按顺序调整：per-device batch → gradient accumulation → 图像 token/分辨率合同（必须保持与部署兼容）→ checkpointing。不得先改变模型结构、减少视觉输入而不记录其对 Scale 的影响。

---

## 11. Codex 实施任务拆分

Codex 必须按以下顺序提交代码和产物。不得跳过门控直接开训练。

### Task 1：仓库与资产核验（CPU）

- 读取当前 P1/P2/B_MIX 的 manifest、adapter、推理代码。
- 读取 LIVE/GNMC/GAICD/SACD 的本地数据 manifest。
- 生成 `audit/asset_inventory.md` 和 `audit/data_audit.json`。
- 记录每个数据集的实际路径、版本、SHA256、split、license。
- 不得根据文件名推测数据存在；必须检查文件和 hash。

### Task 2：Scale 几何模块（CPU）

实现独立、可测试的函数：

```python
def scale_crop_from_center(
    image_width: int,
    image_height: int,
    target_ratio: tuple[int, int],
    center_xy: tuple[float, float],
    scale: float,
) -> tuple[int, int, int, int]:
    ...
```

必须与 AIC 实际 `[x,y,w]` 合同一致。实现后对 1000+ 样本做往返、边界、ratio、transpose 测试。

### Task 3：数据适配器（CPU）

分别实现：

- LIVE 单框 → 固定中心下最优 scale bin。
- GNMC 多画幅 GT → 目标画幅匹配的 Scale 标签。
- GAICD MOS candidates → 同图像 MOS 偏好对。
- SACD ranked pairs → 同图像、同 subject 偏好对。

每条样本保留 `source_id`、`dataset_source`、`split`、`target_ratio`、`preference_source`。

### Task 4：Scale-SFT（GPU ≤ 4 h）

- 从 P1 初始化。
- 冻结 Position 与时间路径。
- 训练离散 scale action。
- 保存每个 checkpoint 的训练配置、数据 manifest SHA、模型 SHA、GPU 时间。
- 运行 source-macro 部署路径验证。

### Task 5：候选生成（GPU ≤ 3 h）

- K=8，候选去重。
- 输出每个候选的原始 scale、投影后 crop、监督分数、来源。
- 计算 Oracle@K、Greedy@1、OracleGain。
- 生成候选多样性报告。

### Task 6：偏好构造（GPU ≤ 1 h）

- 先按 source split，再构造 pair。
- 使用 GAICD/SACD 原生偏好和 LIVE/GNMC 几何偏好。
- 生成偏好数据与 `preference_audit.json`。
- 检查 chosen/rejected 是否同源、同图、同画幅、同位置合同。

### Task 7：Spatial DPO（GPU ≤ 4 h）

- SFT policy 初始化。
- 冻结 SFT reference。
- 只训练同一空间 adapter。
- 使用固定 DPO 配置。
- 每个 eval checkpoint 计算真实部署路径指标。
- 若出现 scale collapse、position drift 或验证退化，立即停止。

### Task 8：最终报告与可部署性检查（CPU 为主）

输出：

```text
artifacts/scale_sft_dpo/
  config/
    run_config.yaml
    scale_bins.json
    gpu_budget.json
    dataset_manifest_hashes.json
  audit/
    asset_inventory.md
    data_audit.json
    geometry_contract.json
    preference_audit.json
  candidates/
    candidate_manifest.jsonl
    oracle_metrics.json
  checkpoints/
    scale_sft/
    spatial_dpo/
  evaluation/
    metrics_by_dataset.json
    source_bootstrap.json
    qualitative_cases/
  report.md
  final_adapter/
    adapter_config.json
    adapter_model.safetensors
    manifest.json
```

---

## 12. Codex 必须执行的防错清单

### 数据

- [ ] SACD、GAICD 本地数据已存在，SHA256 与验证记录一致。
- [ ] GAICD 只使用 journal release 主划分，未混入 conference 重叠图像。
- [ ] GAICD MOS=-2/-1 未评分项已过滤。
- [ ] SACD subject 条件被保留，未跨 subject 构造 pair。
- [ ] LIVE split 使用本轮真实 manifest，不混用历史划分。
- [ ] GNMC train/val/test source 不重叠。
- [ ] 同源所有 crop/pair/augmentation 都留在同一 split。
- [ ] 数据许可和来源信息写入报告。

### 几何

- [ ] Scale=1 与最大合法窗口一致。
- [ ] Scale 变化不改变 Position。
- [ ] A/B 目标画幅分别测试。
- [ ] 转置图像和框使用同一变换。
- [ ] Raw crop 和部署 crop 分开记录。
- [ ] 最终评价使用真实部署投影、rounding 和 clamp。
- [ ] 所有输出合法，无越界、零面积和比例错误。

### 训练

- [ ] SFT 从 P1 正确初始化。
- [ ] 时间路径冻结。
- [ ] Position 路径冻结或有明确一致性保护。
- [ ] 没有增加独立推理模型。
- [ ] 训练后总推理参数仍满足 9B 限制。
- [ ] DPO reference 是冻结的 SFT checkpoint。
- [ ] chosen/rejected 使用完全相同 prompt/template。
- [ ] DPO 数据只来自训练 split。
- [ ] 训练没有使用半决赛分数或隐藏测试信息。

### 指标

- [ ] 主指标为 source-macro deployment IoU。
- [ ] GAICD 报告 per-image SRCC、pairwise accuracy、Top-1 MOS、regret。
- [ ] SACD 报告 same-image/same-subject pairwise accuracy。
- [ ] Oracle@K 与实际 DPO@1 分开。
- [ ] 所有置信区间按 source/image bootstrap。
- [ ] 不报告虚构 official F。
- [ ] 不用 micro 指标替代 macro 指标。
- [ ] 不把 Raw IoU 当最终部署质量。

### GPU 预算

- [ ] Scale-SFT ≤ 4 h。
- [ ] Candidate ≤ 3 h。
- [ ] Preference ≤ 1 h GPU。
- [ ] DPO ≤ 4 h。
- [ ] 总 GPU 时间 ≤ 12 h。

---

## 13. 实验结果如何改变研究判断

| 观察结果 | 解释 | 下一步 |
|---|---|---|
| SFT 部署 IoU 无提升 | 当前 scale 标签/参数化/数据域不能让模型学到有效尺度 | 停止 DPO；检查固定中心 Oracle 与数据尺度分布 |
| SFT 提升，但 Oracle@K≈Greedy | 候选生成没有可利用的额外质量 | 停止 DPO；扩展候选动作空间前先做 CPU 几何诊断 |
| Oracle@K 高，DPO 低于 SFT | 候选质量存在，但模型无法稳定学习偏好或 DPO 训练不稳 | 回退 SFT；检查 token/action 表达与偏好噪声 |
| DPO 提升 GAICD/SACD，但 LIVE/GNMC 下降 | 静态审美偏好向视频域迁移失败 | 降低/移除静态偏好源；保留视频域监督 |
| DPO 提升 LIVE，但 GAICD/SACD 下降 | AIC 单框 GT 几何接近度与审美偏好不同 | 不宣称通用审美提升；按 AIC 视频域目标评估 |
| DPO source-macro deploy IoU 稳定超过 SFT | 有证据支持偏好优化改善 Scale 决策 | 再做固定时间集合的 AIC 提交候选；不自动推断官方分数提升 |
| 只有 Raw IoU 提升，部署路径不提升 | 部署几何吞掉了模型增益 | 判定失败，不打包 |
| 位置漂移明显 | Scale 训练污染 Position | 回退，进一步隔离可训练参数 |

---

## 14. 参考文献与实现

1. Dong et al. **CROP: Expert-Aligned Image Cropping via Compositional Reasoning and Optimizing Preference**, arXiv:2605.12545, 2026. https://arxiv.org/abs/2605.12545  
   价值：GAICD MOS → SFT → chosen/rejected → DPO 的直接先例。限制：静态图像审美裁剪，不是 AIC 视频任务。
2. Zeng et al. **Reliable and Efficient Image Cropping: A Grid Anchor Based Approach**, CVPR 2019. https://openaccess.thecvf.com/content_CVPR_2019/papers/Zeng_Reliable_and_Efficient_Image_Cropping_A_Grid_Anchor_Based_Approach_CVPR_2019_paper.pdf  
   价值：GAICD 候选 crop、MOS、多评分者和候选排名评价。
3. Zeng et al. **Grid Anchor Based Image Cropping: A New Benchmark and An Efficient Model**, IEEE TPAMI 2022. https://doi.org/10.1109/TPAMI.2020.3024207  
   价值：解释为什么单一 crop GT/IoU 无法完整表达审美裁剪偏好。
4. Yang et al. **Focusing on Your Subject: Deep Subject-Aware Image Composition Recommendation Networks**, Computational Visual Media 2023. https://doi.org/10.1007/s41095-021-0263-3  
   价值：SACD 多 subject、多窗口和大规模人工 ranked pairs。限制：subject-conditioned 标签合同。
5. AIC 项目路线 12：`docs/12_videoora_spatial_fullcrop.md`。  
   价值：P1/P2/B_MIX、FULLCROP_A/B_V2、GNMC/LIVE 历史实验及失败门。
6. AIC 项目路线 07：`docs/07_spatial_grounding_and_composition.md`。  
   价值：空间候选、位置、显著性、轨迹和上下文实验的边界。
7. AIC 项目 GAICD 导入脚本：`ext_data/scripts/ingest_gaicd.py`。  
   价值：GAICD 坐标字段、MOS 缺失、journal/conference 重叠和 split 规则。

---

## 15. 最终执行指令

Codex 不得将本文理解为「立即跑满 12 小时」。必须严格按 Stage 0 → Stage 1 → Stage 2 → Stage 3 → Stage 4 的顺序执行，每个阶段先报告门控结果，再进入下一阶段。

本轮唯一目标是验证：

> 在冻结 P1 Position 的条件下，Scale-SFT 能否学会人工尺度监督；候选池是否包含 Greedy 之外的更优合法尺度；GAICD/SACD 人工偏好与 LIVE/GNMC 部署 IoU 偏好能否被 DPO 学习，并最终提高 source-macro 真实部署路径 IoU。

**成功标准不是 DPO loss 下降，也不是候选 Oracle 很高，而是 DPO 最终选择的 crop 在独立 source-held-out 数据上，经过真实 AIC 部署几何后，稳定超过 SFT 和 P1 固定尺度基线。**
