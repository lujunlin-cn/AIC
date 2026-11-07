# AIC 轻量视觉与时序迁移学习方案

- 固定研究快照：5251193a9e07fd682008bb7b2a49aef60ee0cf3f。
- 核验日期：2026-10-19，Asia/Shanghai。
- 本轮范围：源码、规则、公开论文与权重端点审计；合成数值核验与静态参数计数。没有训练新模型、运行真实媒体新实验、制作新 ZIP 或上传官方平台。
- 与本文件配套：[特征机制研究](visual_feature_research.md)、[公开模型登记表](open_models_registry.md)、[最小判别实验](minimal_falsification_experiments.md)、[研究结论](research_verdict.md)。后续实施应固定这些文件的版本与数据清单 SHA。

## 1. 主攻方向及它改变的路径

**主攻 H3：在相同视频、相同输出预算与相同空间模块下，加入相机补偿后的低维前景残差运动，再用合法公开时间偏好监督学习小型时序读出。** 首先检验这种新增观察能否提供现有位置与静态特征之外的时间信息；达到源外门槛后，才考虑冻结视觉特征、部分微调和共享多任务。

这一路线改变三条可以分别检查的路径：

| 路径 | 现有不足 | 拟改变的部分 | 必须固定的部分 |
|---|---|---|---|
| 信息路径 | 单帧视觉 token 及相邻 embedding 差分不直接给出像素位移；全局流量又混合了相机与主体运动 | 连续帧的有效像素位移 → 全局相机估计 → 局部残差、运动位置、起止和转折 | 相同原视频、时间戳、采样帧、解码与缩放合同 |
| 监督路径 | 通用事件理解、回看热度、用户 GIF 选择和距离衰减 soft 曾被混用 | 分清标签来源，在源级隔离的公开 GIF 时间选择上学习相对效用；真实空间框只监督空间任务 | 不加入官方隐藏标签，不把未知槽或占位零当负例 |
| 决策路径 | 预算、排序与构图一起改变时，无法知道增益来自哪里 | 第一项实验只改变同一 K 下的帧排序，保持空间输出和扩展到原生帧轴的方法不变 | K、空预测策略、框、合法性规则及主要评价指标 |

**40 分是远期研究目标；当前没有证据可以把公开时间效用增加 0.01 换算为官方增加 1 分。** 时间集合和裁切的共同质量仍决定最终分数，公开时间、空间标签没有联合标注时不能相乘、拼接或校准出“代理官方分”。

首轮 H3 的共同主对比为 Q_res 相对同头位置控制、Q_res 相对 Q_raw；Q_res=P+残差统计+相机字段，Q_raw=P+总流统计+相同相机字段。**主 Q_res 不含动作阶段特征**；起落、转向、峰后状态只在事前登记的机制诊断臂中评估，不能随主方案晋级而自动宣称阶段贡献有效。

### 1.1 为什么现在先检验 H3

仓库的 R30 高维动态表示失败和 R52 原始光流探针，都没有有效否定“相机补偿后、低维、位置受控的局部运动”这一处理。特别是 R52 源码把灰度图缩放到 float32 [0,1] 后传入 Farnebäck。本轮合成平移核验中，已知位移为 (3,2)：uint8 合法数值范围得到平均端点误差 0.000853；R52 的 [0,1] 输入得到约 3.605549、流幅值中位数约 1.47×10⁻⁶。两者都有限，因此“没有 NaN”并不足以证明运动输入有效。该核验只暴露数值问题，没有测出真实视频收益。[源码](../../scripts/r52_optical_flow_probe.py)、[可复算合成脚本](evidence/aic_r52_farneback_synthetic_audit.py)、[原始输出](evidence/aic_r52_farneback_synthetic_audit.json)。

最新 main 还更新了附件中的两个状态：R86 已回分 **34.88**，相对 B3/R54 父链同时把点源改成 A1raw，并应用 T5 H1 的 point→window 映射，因而是两个处理混杂。H1 原本在 Qwen 主体点合同上训练，不能把“A1raw 与 H1 训练输入不符”当作已证实原因；真正未被干净检验的是 H1 应用于 B3 候选窗口中心的迁移。R87 的 NaN 已由 BF16 路径修复，有限值比例恢复为 1.0；记录规模为 96 条训练、48 条验证、32 个验证源。但其 rank_ap 将连续 soft 直接代入二值 AP 公式，全零片段产生 NaN，普通源内均值再接 nanmean 还可能丢掉整个源；32 是源记录数，不是已确认的 AP 有效分母。因此它仅证明存档输出数值已有限，不能据该 AP 裁决目标域迁移。不能继续沿用旧 NaN 状态，也不能把指标问题只归为小样本。[最新 R86/R87 报告](../r86_r87_observation_and_transfer_20261019/summary.md)、[实际指标实现](../../scripts/r87_aha_phd_pilot.py)。

### 1.2 先保持现役空间父链，再争取完整轻系统

第一步是**研究桥接**：固定 B3/R54 的空间处理与预算，对时间信息做单因素实验。只要候选推理仍使用 TimeLens 的官方测试特征、分数、逐视频决策缓存或其他派生产物，参数账本仍须包含 TimeLens；不能把离线预提取称为免费输入。

后续的**独立轻系统目标**是：V9 视觉塔与既有空间 Head/YuNet + 传统 motion/shot + 不超过 1M 参数的小时间头，完整移除 TimeLens 依赖后，规划不超过 **88,407,729 参数，约 88.41M**。这里混合了历史清单、此次源码静态计数和拟新增上限，尚不是实载部署测量；第 7 节给出条件与逐项账本。B3 原始 checkpoint 身份尚未在本轮恢复，复用缓存不等于能够在新公共源上完整重跑 B3。[B3 资产恢复边界](../r72_b3_asset_recovery_20261018/README.md)。

## 2. 六项能力：哪些可复用，哪些必须学习

| 能力 | 当前可复用部分 | 必须学习或校准的部分 | 监督与依赖 | 本阶段结论 |
|---|---|---|---|---|
| 显著主体定位 | YuNet 的人脸位置；候选 U²-NetP 的前景概率；V9 空间 token | 视觉显著物与事件主体的区分、多主体保留、置信度和域偏移 | 原生图像、可靠主体/显著性监督；DHF1K/SALICON 只能提供注意区域辅助目标 | 检测能力可直接复用，置信度需轻量校准；“应该保留谁”需目标监督 |
| 构图窗口决策 | 现役 B3/R54 已生成框、合法框参数化；T5 的公开人工多框 | 目标 ratio、主体留白和尺度；源外构图偏好；跨镜头轨迹选择 | T5 原生坐标 GT、同一帧同一 ratio、可靠新源；B3 新源复现依赖权重身份 | 缓存回放可复用；新读出需目标监督，不能把 saliency argmax 当最优裁切 |
| 镜头结构识别 | 传统帧差、直方图变化、KTS/已有冻结特征；公开 shot 模型 | 渐变、闪光、遮挡的误检；镜头内相对位置的域差异 | 无标签算法可运行，少量合法人工边界用于校准；公开 GIF 用于后续效用读出 | 结构提取可复用、阈值需轻量校准；镜头边界不是高光标签 |
| 运动模式表征 | 合法数值输入的传统 flow、稳健相机拟合、局部残差统计 | 前景与背景分离、失败置信度、Δt 可比性；动作阶段到偏好的映射 | 连续帧、原生时间轴、同镜头有效 correspondence；时间偏好监督 | 几何运动可直接提取并校准；“运动何时重要”需目标监督 |
| 高光偏好排序 | 中心/位置策略；现役时间基线；公共 GIF 选择 | 公共用户偏好到 AIC 的差异；静态结果展示与动态过程的选择 | 合法 source-disjoint GIF/PHD；可信槽位与目标类型 mask；最终未暴露确认源 | 需目标监督；目前没有足够证据可靠拟合 AIC 绝对逐帧概率 |
| 视频级预算与空预测 | 当前固定预算可作为实验控制 | 人类愿意保留的量、整视频无高光概率、预算与 IoU 的联合权衡 | 需有全视频覆盖、包括可信空高光样本的联合任务标注 | 当前无法可靠训练；需新公开标签。先冻结 K 和空预测策略 |

这些模块有明确依赖：主体定位支持构图与前景运动统计，镜头边界约束 flow 和时序窗口；它们的输出必须经过偏好监督才成为高光排序。预算又依赖整段视频的标签覆盖程度。不能从“有关注热图”跳过后两层，直接产出校准后的高光概率和空预测。

## 3. 数据与监督合同

### 3.1 先承认实际资产状态

当前容器没有 /data/aic，未发现 CUDA/NPU 设备节点，nvidia-smi 与 npu-smi 不在路径中。本轮没有探测历史远程训练主机，因此只能说**当前执行器没有取得那些资产**，不能声称远程文件已丢失。T5 四表、PHD 原始索引/片段标签、R42 对应图像及 V9/B3 实际权重都需要在实施 Phase 0 时重新核文件、清单和 SHA。[当前环境原始核验](evidence/current_environment_check.json)。

公开作者页面可访问、历史报告曾使用过、权重 URL 能取前缀，是三种不同证据。训练启动条件是数据文件与标签生成器实际可核验，且确有未暴露确认源；不能直接把历史 train/dev 文件名当成完整复现。

### 3.2 现有项目数据映射

| 数据或字段 | 可训练/验证目标 | 必须使用的有效性条件 | 划分与暴露风险 |
|---|---|---|---|
| PHD/PHD2 的公开用户 GIF 区间 | 该用户的时间选择、视频内弱相对偏好 | 区间对齐与所见片段有效；缺注释为未知。未选中仅能解释为该次用户选择的弱负例，不能称全体人类不感兴趣 | 必须按原始 YouTube/source 隔离，不能只按 user 或片段隔离；作者 is_last 是用户历史协议 |
| 距离衰减 gif_soft | 距离特定 GIF 区间的平滑回归目标 | 明记公式、tau 和单位；它不是人群选择频率，也不是校准概率 | v9_phd2_tcn.py 的 exp(-distance/3) 在阈值 0.5 下向外扩约 2.079 秒，不能冒充原始严格区间 |
| MrHiSum/回看热度等 gif_soft | 对应原始来源的独立弱目标 | 保留 label_origin、量纲及各自归一化；不能与 PHD 距离 soft 直接混成同一概率 | 原 source 去重，记录历史重复使用；按数据集分别报告 |
| 已知占位 gif_hard | 不能提供 hard 负监督 | 对已知占位来源置 hard_valid=false，只屏蔽 hard loss；有效 soft 或图像特征仍可用 | R22 归档 1,979 条 MrHiSum hard 全零占位，不能以“字段存在”判有效，也不能把所有全零合法负例一律删除 |
| T5 RV/LIVE 四表 | 原生多标注者空间裁切，候选 mean IoU | 每个框有限、正面积、界内；native W/H、ratio、kf/frame_id 与图像一致；无 GT 的目标 mask=false | rv_train/live_train/rv_dev/live_dev 白名单；按源拆分，多个 ratio/帧必须同组；历史 dev 与确认集已多次消费 |
| R42 公开人工诊断 | 高光偏好歧义、主体选择、需上下文/音频的案例诊断 | 保留标注协议、同账号重复、置信度与每个目标的 mask；低置信/缺框不补零 | 540 份记录实际只有 360 张唯一图、12 个 QVH 源；不能当 540 个独立样本或独立标注者投票 |
| 现役模型在合法公共训练集上的输出 | 可作为有 provenance 的蒸馏目标 | 明记 teacher、checkpoint SHA、输入、生成时间；教师目标与人工目标分开 loss | 教师不得消费确认集反馈来选择输出。官方测试 teacher cache 不可作为免费学生输入 |

本地依据：[PHD 采样器](../../scripts/v9_phd2_fragments.py)、[PHD 软目标与 TCN](../../scripts/v9_phd2_tcn.py)、[PHD 构建器](../../scripts/v9_phd2_build_samples.py)、[R22 仪器报告](../r22/r22_instrument_report.json)、[T5 配置](../../configs/T5_CROPHEAD_V3_PREREG.json)、[T5 读出代码](../../scripts/t5_crophead_probe.py)、[R70 构建/监督代码](../../scripts/r70_spatial_utility_pilot.py)、[R42 重复性报告](../r42_annotation_v13_full_20261017/summary.md)。

R42 的同账号重标应当作为可靠性限制：V1.3 四个源的 120 张重复图，等级完全一致 34/120（0.2833）、差值不超过 1 为 0.625、平均绝对差约 1.208；线性/二次加权 kappa 约 0.117/0.185。重标不是两个独立标注者，不能靠平均两份记录获得虚假的独立性；该数据目前适合发现监督歧义，不适合训练高置信整视频预算。[原始汇总](../r42_annotation_v13_full_20261017/repeatability_v13.json)。

### 3.3 必须先修复的采样、几何和连接问题

PHD 旧采样器中约 65% 的片段围绕已知 GIF 区间取样，8 槽短窗口容易让位置模式代替内容。新的确认样本必须使用**与标签无关的原视频窗口起点**，保留原始帧号和实际 PTS；不能训练/评估都使用正例锚定后再声称源外事件理解。旧脚本的 sources-file 分支还需独立验证 banned/exposure 排除，不能根据打印文字认定所有路径都执行相同过滤。[采样源码](../../scripts/v9_phd2_fragments.py)。

T5 到 R70 的几何合同也不同：旧 T5 使用过整数取整、带 +1 的像素面积，R70 使用连续框 IoU；候选数和主体点条件也有变化。新空间实验必须从四张原表和原图重建 manifest，在统一连续几何下重算 GT 目标。R70 shard 没有足够的原图身份字段，**禁止按 shard 行号猜测原图连接**。每行应至少有 source、table/split、kf/frame_id、ratio、native W/H、table SHA、image SHA、唯一 row_id 和合法人工框。129 个最大窗口是现有模型选择的候选族，不是官方强制的空间尺度范围。[R70 空间探针](../r70_spatial_utility_pilot_20261018/README.md)。

历史 T5/R70 的 11,276 条 train、1,546 条 dev、20 个 RV dev 源及 148 个 LIVE dev 源，仍然只是历史清单数字；本轮未重新取得四表。此前 LIVE 确认/储备样本的曝光不能在新文件名下清零。不能用反复查看的开发集合替代新确认集。

### 3.4 外部公共监督的用途与许可边界

| 原始数据 | 监督定义 | 本方案用途 | 与 AIC 的缺口及许可 |
|---|---|---|---|
| PHD | 用户制作 GIF 时的时间选择；作者训练用户 12,972、测试用户 850 | 弱时间偏好与采样审计 | 无原生 crop GT；用户划分不等于源隔离。作者 README 未确认独立数据 LICENSE，源视频权利另核 |
| Video2GIF | 超过 100k GIF—YouTube 对及视觉匹配时间 | 可选弱时间来源，先与 PHD 去重 | 与 PHD 不是同一库；数据声明 BSD，357 个可分发测试视频的 CC-BY 不外推所有源；旧模型代码 GPL-3.0 |
| TVSum | 50 视频、20 人/视频，每 2 秒 1–5 重要性 | 视频内相对排序/覆盖对照 | 标题相关摘要，不是严格原生帧二元 GT。作者称原视频 CC-BY 3.0，但项目历史包 WebscopeReadMe 另有 Yahoo 授权/非商业/禁止再分发条款，冲突须按具体资产保留；摘要预算不能移植为 AIC 规则 |
| SumMe | 25 视频的人类静音视觉摘要，15–18 人/视频，摘要长度 5%–15% | 小规模外部诊断与摘要迁移对照 | 不含 crop；原作者 README 对视频限 non-commercial research、标注用于 research 并引用。源数不足以承诺目标域校准 |
| QVHighlights | 查询相关区间及 2 秒片段的三级标注者 saliency | 有查询条件的外部语义迁移对照；R42 来源审计 | AIC 无相应 query；不能把未标区间一律当通用高光负例。代码 MIT，标注 CC-BY-NC-SA 4.0 |
| DHF1K | 真眼动 fixation/密度，动态显著性 | 主体关注区域或动态注意的辅助监督 | 注视不等于高光/最佳框；RV 相关来源需统一去重。作者 dataset/code CC-BY 4.0 |
| SALICON 2017 | COCO 静态图鼠标轨迹派生的 fixation/密度 | 静态关注先验对照 | 不是真眼动仪视频或动作监督；标注 CC-BY 4.0、图像依 Flickr 条件；旧/新 fixation 生成版本不能混用 |

一手来源：[PHD 作者](https://github.com/gifs/personalized-highlights-dataset)、[Video2GIF 数据](https://github.com/gyglim/video2gif_dataset)、[Video2GIF 代码](https://github.com/gyglim/video2gif_code)、[TVSum 作者](https://github.com/yalesong/tvsum)、[SumMe 原作者 README](https://data.vision.ee.ethz.ch/cvl/SumMe/README.txt)、[QVHighlights 数据 schema](https://github.com/jayleicn/moment_detr/blob/main/data/README.md)与[标注许可](https://github.com/jayleicn/moment_detr/blob/main/data/LICENSE)、[DHF1K 作者](https://github.com/wenguanwang/DHF1K)、[SALICON 2017](https://www.salicon.net/challenge/)。这里只转述必要许可事实，完整原网页不作为仓库附件转载；访问 URL/SHA 见 [source metadata](evidence/semantic_source_metadata_20261019.json)。

TVSum 具体资产另有直接历史依据：[DECISIONS.md 的 2026-09-25 许可证记录](../../DECISIONS.md)保存包内 WebscopeReadMe.txt 的 SHA256 407d340bcd06fdc6d17374ebe6760b4a96816bcace228559c8283d9fb2520dea，记录其要求 Yahoo Data Sharing Agreement、获批非商业学术研究及禁止再分发/网络存储。本轮未重新取得该包，不能用今日作者 README 的 CC-BY 表述覆盖这一已知冲突。同条决定已记录用户对本项目实验使用的授权，无须为已有实验权限重复请求；来源、条款与再分发条件仍保留追踪，授权使用不等于第三方许可事实已经统一。

训练、开发、最终确认按原视频源划分；YouTube clip、多个 query、不同 GIF、重复 ratio 和重编码版本均随原源一起分组。先做文件 SHA 去重，再做可审计的近重复候选核查，保留原始父视频 ID。预训练来源不能完全核实的模型应标注“重叠未知”，不能把未知说成无泄漏。本轮没有取得足量新公共源，因此**没有已落实的 100–200 源确认池**。

## 4. 三种架构按证据晋级

### 4.1 路线 A：低维观察 + 单一轻量读出

这是首个真实媒体实验，也是更复杂架构必须击败的控制。与配套 H3 方案一致，起始设计固定每个时间点 48 个输入槽，预留质量/mask 指示；完整槽位及缺失编码在实现配置中冻结。初版位置控制只保留位置/长度与各臂共同的工程质量信息，不借任何高光标签选择窗口。

新增通道按组定义，而不是拼接任意高维 embedding：

- 位置和镜头：整视频相对位置、距两端时间、镜头内位置和长度、距最近边界、边界置信度。
- 相机：稳健全局平移、旋转/缩放代理、模型拟合残差、有效内点比例；首个实现固定 affine，不同时扫 affine/homography/多个 flow。
- 前景残差：扣除相机模型后的流幅值分位数、方向熵、空间集中度、前景/背景对比、不同网格区域的差异。
- 变化阶段：有效同镜头内的速度变化、加速/减速、方向转折、局部峰前/峰后状态。这些是运动阶段代理，不能命名为有人类 GT 的“事件完成度”。
- 主体交互：仅在可靠主体框或前景 mask 已有时加入框内残差与主体尺度变化；缺失时显式标未知，不默认增加一个分割模型。

首轮主读出固定为低容量 TCN：起始设计为 48 输入槽、32 通道、三层 kernel=3、dilation=1/2/4，再接逐时刻线性输出。各观察臂的槽位、通道、层数、初始化与训练预算相同；激活、归一化、边界 halo 和精确参数计数由实现配置冻结。Ridge 可作为事前登记的低维诊断控制，不在各臂分别挑 Ridge/MLP/TCN 最优者。标准化只在训练源拟合，训练按源均衡；主目标采用同源已记录选择相对未记录选择的固定 pairwise logistic，保留弱偏好的含义。[完整实验臂与门槛](minimal_falsification_experiments.md)。

传统 flow 没有神经网络参数，但解码与 CPU 计算不是零成本。初版选择一个已有可核验实现，数值有效性通过后才处理真实视频；不从 Farnebäck、DIS、TV-L1、RAFT 中用代理最高分选赢家。RAFT-small 的价值是未来检查传统算法在特定困难场景是否丢失运动，不是默认依赖。

### 4.2 路线 B：冻结视觉编码器 + 小型时序适配器

只有 A 的新增信息可迁移、或竞争解释需要明确静态内容控制时，才做一项冻结特征读出。首选**复用现有 V9**，避免同时换 encoder、空间框和监督。保持同一源划分、采样和头容量，对比低维运动输入、冻结视觉输入以及受控组合。

读取哪层先固定为现有部署能稳定导出的末层：全局池化或已有空间 ROI 表示，再加相邻差分和真实 Δt。中间层、不同窗口和不同 pooling 不作为搜索轴。静态 embedding 差分仍不等于真实 flow；冻结表征的时间重排、同视频错位和静态均值对照用于判断是否只认出了视频类别。

DINOv2-S 是**一个独立的冻结语义竞争对照**，不是默认叠加模块。其作者权重有 W2 证据，HF tensor header 合计 22,056,576 参数、384 维空间 token；它能提供静态结构/主体语义，没有原生跨帧动作建模。MobileNetV3-Small 是极低成本对照候选，完整分类版 2,542,856 参数；不在本轮与 DINO、DeiT、VideoMAE 并行搜索。公开 ImageNet/K400 指标不能决定 AIC 的候选优先级。[DINO 作者](https://github.com/facebookresearch/dinov2)、[MobileNetV3-Small 官方文档](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)、[权重 header 计数](evidence/semantic_tensor_header_counts_20261019.json)。

VideoMAE-Base 真正观察连续视频，其 K400 分类版是 86,534,800 参数；但论文 180G/clip、5 个 temporal × 3 个 spatial view 的评估条件远高于一个小读出。若以后有充分理由启用，必须做中心帧重复、帧顺序置换与匹配静态平均控制，且先解决 CC-BY-NC 4.0 条件和真实端到端成本。当前不把它加入主链。[作者模型表](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)、[作者 HF](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)。

### 4.3 路线 C：共享 backbone 的多任务轻系统

C 是证据充分后降低重复编码、改善共享表示的备选实现。一个视觉塔输出空间 token，分别接构图头与时间适配器；flow/shot 仍保持可独立检查的观察模块。它减少重复模型，但增加负迁移、目标缺失和梯度冲突风险。与模块化路线比较时必须使用相同真实目标和训练曝光，不能给 C 更多公共数据后把收益归因于共享 backbone。

在当前数据规模与可靠性下，模块化链更便于定位错误，因此作为首选。只有 B 的冻结表示已显示稳定增量、错误审计明确指向表征不足，才固定一种末层适配方式：更新末 1–2 个 block，或一种固定 rank 的 LoRA，二选一。相同目标下对照“只增加 head 容量”后，才能将改善解释为表征更新。冻结时无信号、数据功效不足或标签混乱，都不构成启动全量微调的理由。

蒸馏仅使用合法公共训练源。教师分数、特征与不确定性分别保留来源；人工选择目标仍单独计量。只有学生在完全没有教师推理输入、没有官方测试教师缓存时，才能把教师排除出部署参数。HL-CLIP 值得借鉴的是目标域语义适配，不是直接下载一个已确认的高光权重：本轮未找到其微调权重；作者训练路径使用 query-saliency MSE，继承的 Moment-DETR criterion 不是该路径实际损失。[HL-CLIP 作者实现](https://github.com/dhk1349/HL-CLIP)、[论文](https://arxiv.org/html/2404.01745v1)。

## 5. 输入、缓存与输出 schema

以下是**待实现的接口合同**，不是声明 main 已具有这些模块。所有入口先验证 manifest，再读取数组；manifest 变更后必须重新核验文件身份与源隔离。

### 5.1 来源与时间轴

| 字段 | 类型/约束 | 作用 |
|---|---|---|
| manifest_version、schema_sha、code_commit | 固定字符串与 SHA | 同一实验全部使用同一解释 |
| dataset、table、split、source_uid、parent_source_uid | 白名单；train/dev/confirm 互斥 | 源级去重与跨数据集泄漏检查 |
| source_path、video_sha256、image_sha256、table_sha256 | 实际字节 SHA；不存在时失败 | 文件身份与原表/原图连接 |
| frame_id、pts_num、pts_den、time_base | frame_id 为原生整数帧号，PTS 有理数 | 不凭四舍五入时间秒数猜帧；明确恒帧率/变帧率处理 |
| native_w、native_h、rotation、ratio | 有限正值；明确解码朝向与 ratio 顺序 | 原生框合法性、缩放反变换 |
| row_id、label_origin、label_version、exposure_history | row_id 全局唯一 | 禁止重复行、目标混同与确认集重新命名 |
| annotation_available、objective_validity | 各目标独立布尔 mask | 未知、无效、可信空标签三者分开 |

时间采样需保存请求时刻和实际解码到的帧，不只保存名义 FPS。研究可固定 1 FPS 语义与 4 FPS 运动作为首次成本合同，但它们是设计值，不是实测最优值。运动单位要除实际 Δt；空间分量通过已知缩放矩阵还原，再按原生宽高或对角线归一化。不同 FPS 的支持时间长度不同，不能只除 Δt 就声称观察完全相同。

光流定义在两个真实帧之间。每条特征保存 frame_id_a/frame_id_b、PTS_a/PTS_b、锚定时间和支持区间；与 H3 方案一致，初版统一锚定到帧对中点，再按同一冻结规则映射原生索引，禁止在不同臂偷偷把峰值移半个窗口。跨镜头帧对直接 mask，不插值跨越 cut；输出仍是严格原生帧号。若低频特征要扩展为稠密候选帧分数，必须冻结同一映射规则、单独记录量化误差，并承认低频观察没有恢复未见到的瞬时事件。

### 5.2 观察与目标数组

拟定接口：

~~~python
validate_public_manifest(rows) -> ValidatedManifest
extract_motion_pair(frame_a, frame_b, geometry, dt, config) -> MotionObservation
fit_camera_motion(flow, valid_correspondence, config) -> CameraEstimate
build_temporal_features(observations, shot_ids, config) -> values, masks
build_native_candidates(width, height, ratio, count=129) -> boxes_xyxy
load_objective_targets(manifest, target_spec) -> targets, target_masks
fit_train_only_transform(train_features, train_masks) -> FrozenTransform
predict_temporal(features, masks, checkpoint) -> ranking_scores
select_fixed_budget(scores, original_frame_ids, K, frozen_policy) -> selected_ids
format_native_crops(selected_ids, fixed_spatial_outputs, ratio) -> predictions
~~~

相机拟合输出 affine 2×3、inlier/support 比例、残差分布与 camera_valid。fit 失败时将相关 residual 特征标未知，保留 raw-flow 有效性；**不能用零残差表示失败**。全画面大面积独立运动、强视差、变焦和遮挡可能破坏单 affine 模型，需按预定质量分层报告，不能把差样本事后删掉。

每个候选时间点的数值向量与 mask 分开存储，模型输入维度固定；标准化只对训练集合内该维有效值计算。均值、方差、PCA、缺失填充值或任何 calibration 都不得看 dev/confirm。归一化后用零填缺失必须同时提供 mask，保留“测得为零”和“没有测量”的差别。

空间内部统一连续 xyxy；外部转换为 [x,y,w]，高度由目标 ratio 得到。候选连续坐标不先 round，GT IoU 不加 1。真实 crop 重新编码若以后实施，必须使用明确 native extent 的重采样，默认按候选宽高比例输出，再交冻结 processor；不能给 ROI 分支原生长宽比、给 crop 分支方形拉伸，从而混入几何处理差异。

缓存至少保存 manifest/rows SHA、数组 shape/dtype、全文件 SHA、每行完成状态与行级 hash、encoder 权重 SHA、processor 配置和观察参数。恢复提取时核行顺序、已完成行、shape/dtype 和最终 inventory；SMOKE 产物不能充当完整训练缓存。这里描述的是应实现的合同，当前 main 并没有之前未合并 PR 中的全部验证器。

### 5.3 模型输出与决策含义

时间头首先输出未校准 ranking_score，而非 p(highlight)。只有在有明确标签覆盖与校准集时，才另存概率校准器。相机置信度、主体置信度与高光不确定性应是不同字段，不能相互替代。

H3/H2 的输出预算 K 由同一冻结规则产生，不能根据新分数偷偷变化。候选帧数、空间框、空预测策略也相同。对同一 source 汇总 selected_frame_ids、score、bbox_native、model_dependency_manifest、inference_config_sha 和输入 hash，保证差异可以归到一次处理。

## 6. 损失函数与没有联合 GT 时的边界

对任务 j，使用独立有效性 mask mᵢʲ，按源先平均再汇总。没有有效样本时该项损失不进入 batch，不能用全零 target 补足：

\[
L_j=\frac{1}{|\mathcal V_j|}
\sum_{v\in\mathcal V_j}
\frac{\sum_i m_{vi}^{j}\ell_j(\hat y_{vi},y_{vi})}
{\sum_i m_{vi}^{j}}.
\]

| 损失 | 可用真实/明确弱目标 | 推荐初始定义 | 禁止的解释 |
|---|---|---|---|
| L_temporal | 同源合法 GIF 区间 union 的已记录选择；或单独来源的 soft | 首轮 TCN 用同源固定 pairwise logistic，只在可比较的已记录/未记录选择之间计算；距离 soft 仅可用独立回归项 | 未记录选择仅为弱负；不能将 distance-soft 的 0.5 当人类二元边界，不能把未知当可靠空高光 |
| L_spatial | T5 同帧同 ratio 的人工多框 | 每个候选 c 的目标为所有有效人工框的连续 mean IoU；固定一次回归或 soft ranking 目标 | 不能把教师主体点、saliency 峰或非对应帧的框当联合 GT |
| L_motion | 同镜头有效位移/残差或独立公开运动标注 | A 阶段 flow 不训练，故 λ_m=0；以后若学习运动表征，可重建有效残差统计或做明确自监督一致性 | 不能凭运动峰值创造“事件完成”“高光”真标签 |
| L_consistency | 可证明的几何变换、同一对象的可靠对应关系 | 先用镜像/坐标等变；时序项只在同镜头、可见性与匹配可靠时启用 | 不能对所有帧强制平滑；不能跨 cut 或把构图变化压回 EMA 父链 |
| L_distill | 合法公共训练源上的可追溯 teacher 输出 | 单独 mask、温度/量纲固定；与人工目标分别报告 | 蒸馏目标不是人工偏好，不能借 teacher 测试缓存规避参数预算 |

多任务候选可写成 L=λ_t L_temporal+λ_s L_spatial+λ_m L_motion+λ_c L_consistency，但各项仅在对应标签存在时计算。权重在训练/开发协议内预先固定，不根据确认或官方分改变。先记录不同任务的梯度方向/尺度及源内性能；若共享更新使既有空间目标变差，应回到冻结/模块化，不以时间代理小幅改善掩盖构图损失。

**本项目已核验的现有公共数据尚未形成能直接优化完整 AIC 联合 F 的训练集。** 同一网络交替看 PHD 时间标签和 T5 空间框，是多任务训练；不等于同一帧拥有联合人工选择与裁切偏好。仅凭同名视频、类似场景或时间插值不能创建联合 GT。

## 7. 完整参数账本与预训练依赖

计数方法是对实际推理图所依赖的唯一参数张量求和，共享同一权重计一次，不同 checkpoint 不能因为架构相同就合并。冻结、量化和缓存不会自动减少参数个数。导出中确实删除、不参与任何推理的头应以实际图证明后重计；不能只报告可训练参数或磁盘 MB。

### 7.1 本轮实际完成的 Head 静态计数

脚本仅解析本地 Python AST，对 Linear、LayerNorm、ModuleList/Sequential 的声明做整数求和，没有导入 torch、实例化、forward 或加载 checkpoint：

| 层组 | 参数 |
|---|---:|
| 2305→512→128 投影 | 1,246,336 |
| 两组 QKV | 99,072 |
| 两组 attention 输出投影 | 33,024 |
| 两组 ln1 | 512 |
| 两组 ln2 | 512 |
| 两组 FF 128→256 | 66,048 |
| 两组 FF 256→128 | 65,792 |
| 标量输出 | 129 |
| 合计 | **1,511,425** |

2305 维来自 3×768 池化向量与一个位置量，依据源码 feature concat，不是从 9B 剩余额度反推。V9 Head 与 R70.make 两份声明在此条件下同计数；相同形状不能证明 R70 权重等于历史 B3。历史 B3 checkpoint 身份仍须单独恢复。[脚本](evidence/aic_head_parameter_audit.py)、[带源 SHA 的原始结果](evidence/aic_head_parameter_audit.json)。

### 7.2 三种完整图的条件总数

| 系统 | 完整依赖与算术 | 证据层级与含义 |
|---|---|---|
| 当前 TimeLens + V9/B3/YuNet | 历史 TimeLens 8,767,123,696 + 85,843,200 + 1,511,425 + 53,104 = **8,854,531,425** | 历史申报/资产报告与本轮 source count 的一致性检查；本轮没有完整 load |
| H3 研究桥接 | 上述链 + 新增时间头 ≤1,000,000，即规划 **≤8,855,531,425** | 只要使用现役 teacher 依赖就按此计；传统 flow/shot 0 神经参数，仍需测时间与内存 |
| 首选独立完整轻链 | V9 85,843,200 + Head 1,511,425 + YuNet 53,104 + 新时间头 ≤1,000,000 = **≤88,407,729** | **移除全部 TimeLens 推理与测试派生输入后才成立**；历史 encoder/YuNet 未本轮实载，小头上限为设计 |
| H1 通过后的可选主体模块 | 上项 + U²-NetP 1,131,181 = **≤89,538,910** | 不默认叠加；U²-NetP 此轮为作者源码静态计数与 W2，非完整 load |
| 独立重建的 DINO 小链 | DINO-S 22,056,576 + U²-NetP 1,131,181 + 所有新适配器/任务头 ≤1,000,000 = **≤24,187,757** | 另一图设计，**不含现役 B3 空间父链**；必须重新证明空间能力，不能与首选链混称同一系统 |
| 上项再加可选 RAFT-small | 24,187,757 + 990,162 = **≤25,177,919** | 仅预算比较；当前 H3 不加 RAFT，也未实测端到端速度 |

V9 85,843,200 与 YuNet 53,104 来自历史项目声明，本轮没有独立加载对应资产；U²-NetP 1,131,181 来自作者代码静态声明，RAFT-small 990,162 为 Torchvision 指定权重元数据。所有最终部署必须达到 W3 的完整 hash 与 load，并另外完成唯一参数去重账本、固定输入前向与运行成本核验。[R54 记录](../r54_timelens_center_residual_20261018/summary.md)、[U²-NetP 静态计数](evidence/spatial_static_params/parameter_counts.json)、[空间权重访问审计](evidence/spatial_weight_access_audit.json)。

TimeLens 8,767,123,696 也有直接历史组件声明，并非本报告用总数减空间系统倒推：[R47 合规核验组件表](../r47_timelens2_visiononly_compliance_20261017.md)逐项列出此数；[STATUS 的 R47 记录](../../STATUS.md)称当时核四个 safetensors 分片，[EXPERIMENTS 的 R47_PARAMETER_AUDIT](../../EXPERIMENTS.md)记为 shape metadata 计数。本轮未取得这四个原始分片的完整张量清单或加载权重，因此只引用为历史计数记录，不升格为本轮独立复算。

HL-CLIP 不能直接加入当前 8.854B 链。完整基础 CLIP ViT-B/32 为 151,277,313，直接相加得到 **9,005,808,738 >9B**；作者类还注册未用层，保守完整类 151,564,278，对应 **9,006,095,703**。冻结 text encoder 也不会免计。若删除组件或改为只部署学生，应依据真实新图重新算；不能只算高光 head。HL 微调权重本轮是 W0，基础 CLIP 是 W2，两者不能互换。[作者类](https://github.com/dhk1349/HL-CLIP/blob/main/moment_detr/model.py)、[基础 CLIP 作者](https://github.com/openai/CLIP)。

权重层级统一为 W0 未发现、W1 作者链接、W2 实际 HEAD/小 Range 可取非 HTML 权重端点、W3 完整 SHA + load。此次 7 个语义端点的 32-byte probe 均为 W2，没有把它们升级为推理复现。[可运行脚本](evidence/semantic_weight_access_probe.py)、[该脚本实际运行输出](evidence/semantic_weight_access_probe_result.json)、[原始 header 与访问记录](evidence/semantic_model_access_20261019.json)。

许可也属于完整依赖：V9 上游 LFM2-VL 权重适用 LFM Open License v1.0，不能因视觉初始模型名为 SigLIP2 就改写成 Apache；DINOv2 作者代码/权重为 Apache-2.0；VideoMAE 为 CC-BY-NC 4.0；U²-NetP 代码 Apache-2.0 不自动证明 Drive 权重有同样独立许可。保留使用场景核对，不把“能下载”写成“可以任意提交/再分发”。[LFM 权重许可](https://huggingface.co/LiquidAI/LFM2-VL-450M/blob/main/LICENSE)、[DINO 作者](https://github.com/facebookresearch/dinov2)、[VideoMAE 作者](https://github.com/MCG-NJU/VideoMAE)。

## 8. 六阶段实施、资源和存储

以下数量是**预算示例，不是已取得数据或实测吞吐**。统一规划单位为 100 个合法独立源 × 每源 60 秒，总 6,000 秒；4 FPS 运动产生约 24,000 个采样点，1 FPS 视觉产生 6,000 张图。源数最终由功效和资产审计决定。

| 阶段 | 输入与输出 | 依赖/计算资源 | 示例存储与风险 | 通过/停止条件 |
|---|---|---|---|---|
| Phase 0：合同与资产恢复 | 输入四表/索引/媒体/权重/曝光历史；输出带 SHA 的源清单、时间轴检查、有效标签 mask、参数依赖图 | CPU；只读文件核验和小范围解码；不需训练 GPU | 若原视频平均 1–4 Mb/s，6,000 秒约 0.75–3.0 GB；标签/metadata 通常远小于媒体，实际按清单统计 | source 泄漏、标签来源不明、原图不可连或没有新确认源时不训练；B3 资产不能恢复则明确降级基线身份 |
| Phase 1：特征与数值诊断 | 输入有效视频；输出 raw/camera/residual 低维数组、mask、shot、分布和时间错位审计 | CPU OpenCV；先 10–20 帧对验证几何，再测真实小批 wall time；无需新权重 | 24,000×48×float32 为 **4.608 MB**，另有索引/mask；160×90×2 的 float32 全流单向约 **2.7648 GB**、双向约 **5.5296 GB**，默认流式只留统计/抽样诊断 | 已知平移/相机-only/局部运动/shot 反例通过；无法解释的近零流、fit 失败和错位先修复 |
| Phase 2：单一轻读出 | 输入冻结低维特征和合法 train 目标；输出 transform、checkpoint、源级配对结果 | 主 TCN 需要可用 torch 环境，可 CPU 小烟测/单张 CUDA 训练；Ridge 仅诊断控制，没有必要先安装大模型栈 | ≤1M 参数的 FP32 checkpoint ≤4 MB，仅权重；保留 config、seed、训练日志及 predictions | 主对比达到预注册效应与 CI/类别条件；功效不足只诊断，不把换模型当解决办法 |
| Phase 3：冻结视觉迁移 | 仅在 A/竞争控制有依据时，输入同一源/帧；输出冻结 V9 或一个 DINO 控制的缓存与等容量读出 | 单张 CUDA 先实际 load/smoke；量精度、batch、解码和峰值显存；CPU 仅小规模语义核验 | V9 6,000×768 FP16 global 为 **9.216 MB**；256 token grid 为 **2.359 GB**。DINO 6,000×384 FP16 global 为 **4.608 MB**；16×16 grid 为 **1.180 GB** | 冻结表示必须有源外增量，静态/置换控制不能解释全部收益；额外成本超预算即停 |
| Phase 4：部分微调或蒸馏 | 输入固定 train/dev、已证明有信号的 encoder；输出单一 adapter/学生与空间回归测试 | 一张真实 CUDA 设备先测 memory/time；8–16 GB 冻结、16–24 GB 局部训练仅为试配假设，非保证；不先启动多卡 | Adam 两个 FP32 矩约 8 bytes/训练参数，连 FP32 权重/梯度约 16 bytes/参数，未含 activation/master copy；LoRA 不能免计基础 encoder | 只head增容对照不足以解释改善、空间无不可接受退化、全源确认过门；否则回到冻结模块 |
| Phase 5：联合部署 | 输入已确认的时间模块与可复现空间链；输出完整依赖清单、当前合同 checker 结果、可重建 ZIP | 实际目标硬件逐模块与端到端回归；CPU/CUDA/910A 分别记录，不能互相替代 | 保留单一批准版本；可按 source 流式推理，避免永久存储全部129候选特征 | 当前规则版本、参数≤9B、原生帧/框、依赖/hash、回放均通过后才具备上传候选资格；本轮不执行 |

存储算术均使用十进制 MB/GB，未含 filesystem、压缩比和额外副本。224×224 RGB uint8 的 6,000 张未压缩图约 0.903 GB；若把所有 129 候选、2305 维 FP16 特征常驻，则为 6,000×129×2305×2 ≈ **3.568 GB**，所以不应默认落盘所有候选。真正保留哪些缓存取决于复现需求和重新提取成本。

计算时间不从论文 FPS 推断。先在目标机器记录总媒体时长、分辨率、解码实现、batch、precision、线程数、暖机、是否含 I/O，再把同条件小批的 wall time 按实际规模估算，并加上失败重试和输出校验。不能将已经缓存特征的 Head 耗时当成完整视频系统速度。

## 9. CPU、CUDA 与 Ascend 910A 的实际边界

**当前可确认的是 CPU 环境及传统 flow 合成数值检查。** 本轮 Head 计数是整数 AST 审计；没有神经模型 CPU forward，没有新 CUDA 推理/训练，也没有 910A load 或吞吐测量。当前容器与历史远程主机不是同一设备，不能沿用“曾有 GPU”来启动未核实的任务。[当前环境](evidence/current_environment_check.json)、[合成 flow 环境与输出](evidence/aic_r52_farneback_synthetic_audit.json)。

CUDA 后续首先核验精度稳定性和真实输入。R87 在 FP16/配置组合出现 NaN、BF16 修复的记录说明 dtype 是运行合同的一部分，不能把非有限特征的验证分数用于否定语义迁移，也不能从 BF16 finite 推断目标匹配已经解决。传统 OpenCV 特征可留 CPU，与 GPU 视觉提取异步衔接，但要记录传输和 decode 是否成为瓶颈。[R87 BF16 记录](../r86_r87_observation_and_transfer_20261019/r87_trace_bf16.json)。

910A 迁移不以“PyTorch 模型很小”为可用保证。需要逐项验证：

| 模块 | 潜在算子/运行风险 | 最小验证 |
|---|---|---|
| V9 SigLIP2/NaFlex | 动态 spatial_shapes、position embedding 初始化、attention 实现、dtype 与 CPU/NPU 迁移 | 固定实际 processor 输入，核 finite、shape、候选排序及 CPU/CUDA 对齐；保留历史 patch，不盲用 meta/to_empty |
| 小 TCN/MLP/LoRA | Conv1d、LayerNorm、动态长度、混合精度及 adapter 导出 | 先固定长度合成前后向和保存加载，再少量合法媒体；不能只检查 import |
| U²-NetP | 池化、bilinear upsample、多尺度 concat、动态输入 | 相同 resize/输出尺寸下核概率、主体中心和 crop 排序，非仅均方误差 |
| RAFT-small（非默认） | correlation、grid_sample、迭代更新和精度累积 | 小已知运动与遮挡反例后才测真实 flow；必要 CPU/CUDA 保留该模块 |
| 传统 flow/shot | 多数实现是 CPU OpenCV；NPU 不会自动加速 | CPU 数值合同稳定后测线程与 I/O，作为异构链一部分记录总耗时 |

仓库已有 NPU patch 是历史实现线索，不是本轮成功部署证据。[SigLIP2 NPU patch](../../scripts/v11_npu_siglip2_patch.py)、[CPU/NPU parity 脚本](../../scripts/v11_round5_cpu_npu_parity.py)。不因移植困难换掉模型而把输入/表示变化隐藏在“兼容修复”里。

## 10. 训练、确认与停止规则

### 10.1 评价对象先冻结

H3/H2 的主指标正式命名为**已记录 GIF 选择的集合覆盖代理**。在同一原始源上，以绝对 PTS 合并全部合法已记录 GIF 区间的 union，去除重复/重叠，然后在统一原生采样轴上定义二元 Y；每源只计算一次 J，最后源等权，不让用户数或片段数重复加权。固定 K 时，J=2Σ_selected Y/(K+M)，其中 M 是同一可评估轴上落入该 union 的帧数。若有**同帧真实空间 GT 的合法交集**才可另报 q=IoU 的真实联合效用；通常没有时设 q=1。不能把 T5 平均 IoU 或预测置信度乘进去模拟 AIC 联合分。

首轮 K 使用配套实验文件的唯一 Python3.12 函数：K=min(N,max(0,int(round(0.85*N))))；N 为各臂共同的合法 native 候选数、拒绝 bool/负数，N=0 不参加源指标。round 为 ties-to-even；逐源 K 输出为文件并绑定 SHA，排序平局按原生帧号稳定打破。无效 flow 不改变候选全集 N、K 或评估分母；各学习臂共享质量 mask/fallback。若选择现役父包的逐源预算作为替代，必须在开跑前绑定其 SHA，不能中途切换。

PHD 的区间是用户选择，未选中不等于普遍不感兴趣；该指标只估计公开观察到的选择偏好。标签覆盖未知的整视频不能作为可信 M=0 样本。空预测与预算头因此暂不学习。区间端点、源内多 GIF 聚合、重复选择和多用户分母必须事先写入 label spec。

训练按源均衡，开发集只用于预定的模型选择和有效性诊断；最终确认源一次性打开，按源配对 bootstrap，不能按帧伪造样本量。关键帧上的类别、时长、运动强弱、镜头风格和背景补偿成功率事先分层，报告每层 n 与差异，不从结果挑最好类别。

### 10.2 晋级与否定

沿用配套实验预注册的规划效应 δ_t=0.01（公共源宏集合覆盖代理）、δ_s=0.01（公共源宏空间 IoU），它们不是官方分数换算。根据 train-only 源配对方差和实际独立源数做功效规划；H3 两个共同主对比，以及 H1 事前登记替换现役时相对 B3/R70 的双主门，除展示 95% 描述性区间外，晋级与否定各使用 Bonferroni 校正后的 **97.5% 单项决策 CI**（1.25%–98.75% 分位数），不能重复使用两个普通95%区间。H1 保留原四臂，额外冻结 R70 topsoft 作现役参考；缺少匹配权重、公共同帧输出或版本身份时，只能诊断。样本量不足时不降低门槛“凑成功”。

| 结果 | 下一步 |
|---|---|
| 主效应点估计达到 δ，预注册决策 CI 下界 >0（H3 及 H1 部署双主门各97.5%），必要机制对比成立，类别/成本/失败案例条件满足 | 进入一次部署复现；仍不宣布官方可达 40 |
| 输入与设计有效、功效足够，确认 CI 上界 <δ | 停止“当前实现有值得继续投入的增量”这一主张；不永久否定所有相关视觉方法 |
| CI 同时覆盖 0 与 δ，或未取得足够未暴露独立源 | 暂不可判定；补合法新源或停止本轮，不上传碰运气 |
| 源泄漏、mask 错误、K/空间变化、帧轴错位、flow 数值退化、B3 身份不一致 | 实验无效；先修合同/输入，不把该结果记作模型正负证据 |

竞争解释必须有对应反例：时间位置分层/同源错位排除位置代理；去掉 cut 邻域并控制边界密度排除运动只认转场；source/category 分层与静态语义控制排除仅分类；严格固定 K/框排除预算或空间变化。反例若保留全部收益，应放弃该机制解释，不凭总 AP 上升晋级。[逐项设计](minimal_falsification_experiments.md)。

稳定性门同时采用配套实验的预设：源中位差非负，D<−3δ 的源比例不超过10%；有足够独立源的 leave-one-category 结果不翻负。类别未知或样本不足须限制结论。实际 CPU 秒数、GPU 小时、峰值内存/显存、磁盘和外部费用上限，在 TRAIN 小样本 profile 后写入冻结成本合同；确认源开封后不得为保留收益而放宽。

### 10.3 唯一备用方向

只有 **H3、H1、H2 三项都已实际完成，输入有效、源级功效足够，并按预注册门槛否定有意义增量**，才触发备用：

**获取新的合法公开、目标比例明确的联合时间—裁切偏好监督，并补充标注所需的上下文、音频或主体意图。** 这改变监督和必要信息来源，回应现有弱 GIF/T5 分离目标的缺口。应保留完整原生帧轴、同一选择上的 ratio crop、多标注者/一致性、可信空高光与整视频预算信息，按原视频源保留新确认集。

不是把 R42 同账号重标扩充为伪多人 GT，不是把 T5 框按 PHD 时间拼接，也不是让 VLM 为官方测试伪标后回灌。若只是资产未取到、输入无效或功效不足，备用条件尚未满足。当前不并行启动标注工程或更大模型训练。

## 11. 官方协议与部署集成门

当前官方页面与公式图支持严格同视频原生帧匹配、连续 IoU 累加、每视频 F=2S/(K+M) 后等权宏平均 ×100；FP 增加 K、漏检保留 M，错帧不能靠邻近容忍补 TP，空间不准降低匹配帧的贡献。双空为 1、单空为 0。没有把 85% 或 15% 预算、129 个最大窗口写成官方硬约束；这些属于本项目策略。

上述简洁集合式针对规范、无重复输出。公式图明确错误帧、重复预测或格式无效条目没有 IoU 贡献但计入预测数量；不能先去重再缩小分母。图中没有完整说明重复条目的次序规则，first-valid 是本地实现选择，平台具体计分或提前拒包细节仍待同版本执行器确认。

**同站材料存在版本冲突。** 2026-09-24 复赛通知 PDF 仍写按参数量分档 ≤100M/100–500M/500M–9B，对应 1/.95/.90；当前 HTML 和公式图未显示该系数。本研究按当前页面及用户既有确认采用 100F 工作口径，保留冲突，不能声称全部官方材料一致，更不能从 36.43÷0.9 推导“100M 已等价 40 分”。上传前仍需取得当前规则版本确认或执行 checker。[当前官方页](https://www.aicomp.cn/tracks/tracks-6/4267.html)、[公式图](https://www.aicomp.cn/wp-content/uploads/2026/09/1-1.png)、[同站复赛通知](https://www.aicomp.cn/wp-content/uploads/2026/09/关于举办产业命题赛道基于视频大模型的通用视频高光剪辑复赛的通知.pdf)。

本固定 main 的 aic/contract.py 仍把 semifinal/replay 归到旧 final，并要求旧 model_size_mb 口径；aic/evaluation.py 的 final 仍应用历史 size coefficient；独立 checker 也保留旧协议。之前研究分支的 current 修复尚未合入此 SHA，不能引用未合并文件宣布 main 已修好，更不能把本地实现称为官方 binary。[合同源码](../../aic/contract.py)、[本地 evaluator](../../aic/evaluation.py)、[独立 checker](../../scripts/independent_submission_check.py)。

未来集成先明确 current 参数协议与历史 replay 的分离，再增加合成完美匹配/空集/错误帧/连续小数框/预算/9B 边界与跨视频一致性测试。推理入口需要显式组合新时间读出、空间模块、native frame mapping 和 dependency manifest；现有 aic/inference.py 不是可直接代替所有 R54/R70 脚本的统一端到端部署。[当前通用入口](../../aic/inference.py)。

只有参数账本、合法性、源身份、逐组件回放和当前规则确认都通过，才可能形成下一次单一可审阅的提交候选。本轮没有新包，也没有“已通过官方 checker”的结果。

## 12. 可复现命令与后续实施交接

### 12.1 本轮已实际运行的证据脚本

以下脚本已在本报告附带结果中执行过；为避免覆盖原始结果，复跑输出应使用新路径：

~~~bash
python reports/visual_feature_research_20261019/evidence/aic_head_parameter_audit.py \
  --repo . \
  --output /tmp/aic_head_parameter_audit_rerun.json

python reports/visual_feature_research_20261019/evidence/semantic_weight_access_probe.py \
  --output /tmp/aic_semantic_weight_access_rerun.json
~~~

第二条需要能访问公开作者/官方权重端点的网络，仅每个端点读取 32 bytes，不下载完整模型。运行报告明确 W2、full_hash_verified=false、model_loaded=false。Head 脚本仅 stdlib AST 与整数计算，不要求 torch；它不会打开视频或标签。

### 12.2 待实现命令设计

以下文件名和 CLI 是交接设计，**当前未实现、未运行**。实施者先在隔离分支实现输入验证和合成反例，审阅后再运行真实数据；不可复制命令后假定 main 已包含对应脚本。

~~~bash
# DESIGN ONLY — same proposed entry points as the experiment plan.
# These scripts/configs do not yet exist in this fixed main.
python scripts/visual_probe_manifest.py freeze --spec configs/visual_probe_sources.json
python scripts/visual_probe_spatial.py identity --spec configs/H1_SPATIAL_FROZEN.json
python scripts/visual_probe_motion.py extract --spec configs/H3_MOTION_FROZEN.json
python scripts/visual_probe_motion.py train --spec configs/H3_MOTION_FROZEN.json
python scripts/visual_probe_motion.py confirm --spec configs/H3_MOTION_FROZEN.json
python scripts/visual_probe_structure.py run --spec configs/H2_STRUCTURE_FROZEN.json
~~~

第一项实施工单应产出：真实资产与曝光清单、输入有效性反例、固定观察数组、单一低容量训练合同、源级功效计划。达到这些条件之前不启动 Phase 3/4，不消耗官方上传尝试来补足公开证据。
