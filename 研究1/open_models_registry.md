# AIC 公开模型与算法审计登记

审计日期：2026-10-19。项目仓库锁定 `3b28ffc6bf61f0799d91c9ecf6113e83a98223fc`；第三方原论文、官方源码、模型卡与权重入口按本日取证。**28 个登记条目，每项统一 19 字段**，覆盖传统算法、显著前景、动态注视、主体检测、视频构图、镜头结构、图像语义与短时视频编码。

## 1. 如何使用本表

本表按“提供什么新增信息、能否核验、能否接入固定合同”选择模型，不按不同 benchmark 数值横向排序。公开的像素 F、gaze NSS、光流 EPE、动作 top-1、文本条件高光 mAP 都不能换算成 AIC 连续 IoU 加权 F。

权重状态分四层：官方入口可见；文件实际取得并核 SHA；结构或 checkpoint 元数据核验；真实模型加载与 forward。**本轮只有部分空间模型达到第二或第三层，没有任何候选完成目标视频推理、训练或 NPU 性能测试。** 每条详细记录说明状态。权重许可未单列不等于已证明禁止使用，也不等于所有训练数据和再分发权利已经解决。

参数默认指该独立模块的完整推理参数。接入 AIC 还须加时间、空间、检测及生成测试特征缓存的全部上游。传统算法的 0 学习参数不等于 0 计算成本。R54 固定时间回放与 R91 轻量时间链是不同完整系统，不能借缓存将前者重新声明为 100M。

## 2. 五个优先候选

| 顺序 | 候选 | 首次要验证的新增信息 | 为什么优先 | 进入部署前还缺什么 |
|---|---|---|---|---|
| 1 | **U²-Netp** | 原生 crop 上的主体范围、覆盖和截断信息 | 1,131,181 参数；官方权重已取得，易作冻结空间通路 | 同容量无主体消融；真实 crop 增量；外部权重分发条款与真实 forward |
| 2 | **DIS + LK/RANSAC** | 把相机运动与局部主体运动分开 | 无学习权重，可先做 CPU 控制；直接针对 R30/R90 未检验的机制 | 正确坐标/PTS、补偿失败掩码、完整吞吐；不拿论文最快点当管线速度 |
| 3 | **PySceneDetect** | shot 内位置、切点距离和结构控制 | 无训练，集成成本低，能先排除转场伪运动 | 抽样边界质量；确认源上的结构增量；版本与采样合同 |
| 4 | **ResNet18 冻结特征** | 保留轻量静态语义竞争解释 | 与现有 R91 接近，headless 11,176,512 参数，避免换 backbone 混杂 | 冻结读出资产、完整空间依赖、目标时间指标；未取得官方效果 |
| 5 | **UNISAL 固定域分支的时间上下文对照** | 真时序注视是否超出静态输入 | 权重已取得，元数据计 3,715,640 参数；可固定视频分支比较真实邻帧与重复当前帧 | 同 source、域适配参数、预处理、输出映射；gaze 不能直接当 crop |

YuNet 已在当前 V9 参数账本内，应首先复用而非再加一个人脸模型。RTMDet-tiny 只在全身/多人/器械漏检有明确证据时增加。TinyHD-M rc4 只有 515,281 参数，是值得保留的真时序备选，但官方仓库未见明确 LICENSE，当前不作为已就绪部署项。RAFT-small 仅在传统流质量确实限制 H3 时启动；DINOv2-S 用作空间 token/语义强度对照。以上优先级是研究判断，各自的事实证据见下面对应条目。

UNISAL 的 source 会控制 BN、Gaussian prior、融合/平滑和输入设置。SALICON/image 与 DHF1K/video 直接比较同时改变数据域适配，只能作为完整配置对比；纯时序消融须固定同一 source 与所有域参数，例如只将真实相邻帧替换为重复当前帧。

## 3. 候选索引与规模

| ID | 候选 | 独立模块参数口径 | 权重状态 | 研究用途 |
|---|---|---|---|---|
| S01 | U²-Net / p | 44.010M / 1.131M，源码计数 | p 已下载并核元数据 | 显著前景 |
| S02 | BASNet | 87.060M，源码计数 | 官方入口，未下载 | 边界/较重教师 |
| S03 | IS-Net DIS | 44.047M；训练GT encoder另记 | 版本与下载未核清 | 高分辨率前景 |
| S04 | UNISAL | 3.715640M，checkpoint元数据 | 已下载，未forward | 图像/视频注视 |
| S05 | TinyHD-M rc4 | .515281M，checkpoint元数据 | 已下载，许可未知 | 真时序注视 |
| S06 | YuNet 2023mar | .053104M导出学习初始化器；原论文训练网不同 | ONNX已下载 | 人脸位置 cue |
| S07 | RTMDet-tiny | 作者4.8M | 已下载，元数据解析未完成 | 人体/对象范围 |
| S08 | AutoFlip | 全链未核；不等于0 | 依赖资产未逐项核实 | 路径优化机制 |
| S09 | SmartVidCrop | 全链含TransNet v1+UNISAL，未核 | 全链未运行 | 原生构图与规则控制 |
| S10 | DiffSal | 76.57M音视频 / 70.54M视频版，论文 | 入口可见，未下载 | 多模态注视机制 |
| T01 | PySceneDetect | 0 | 不需要 | 镜头控制 |
| T02 | TransNetV2 | 7.616258M，静态结构推导 | 官方入口，未下载 | 学习式镜头识别 |
| T03 | KTS | 算法0，编码器与读出另加 | 原始代码包未取得 | 核时序分段 |
| T04 | TSM-R50 K400 | 24.327632M，静态结构推导 | 官方入口，未下载 | 短时视频控制 |
| T05 | ResNet18 | 去分类头11.176512M；原分类整网11.689512M | 官方入口，未下载 | 静态语义锚点 |
| T06 | MobileNetV3-Large | 分类整网5.483032M | 官方入口，未下载 | 更低成本图像控制 |
| T07 | DeiT-S | 官方约22M，分类整网 | 官方入口，未下载 | 图像Transformer控制 |
| T08 | DINOv2-S/14 | 官方约21M，编码器 | 官方入口，未下载 | 空间token/自监督语义 |
| T09 | VideoMAE-B K400 | 86.534800M，指定下游结构 | 本轮未取得权重文件 | 真时序表示/教师 |
| T10 | HL-CLIP | 原CLIP双塔151.277313M；当前注册类151.564278M | 高光微调权重未找到 | 查询高光适配证据 |
| M01 | 帧差+全局配准 | 0 | 不需要 | 基础变化控制 |
| M02 | Farnebäck | 0 | 不需要 | 稠密光流参考 |
| M03 | DIS | 0 | 不需要 | 首轮运动主通路 |
| M04 | TV-L1 | 0 | 不需要 | 正则化参考 |
| M05 | 金字塔LK | 0 | 不需要 | 稀疏轨迹/相机估计 |
| M06 | RAFT-small | 论文约1.0M | 官方入口，未下载 | 流质量控制 |
| M07 | GMFlow基础单尺度 | 论文约4.7M | 官方入口，未下载 | 全局匹配控制 |
| M08 | NeuFlow-V2 | v2约9M；当前权重待核 | 作者文件存在，未下载 | 近期边缘设备备选 |

规模档位按完整系统重新计算：≤20M、20M–120M、120M–500M。上表的模块属于某档不保证组合后仍在该档。特别是 VideoMAE-B 加当前87.408M空间链已超过100M；HL-CLIP只训练末层也仍含完整图文基座。未核实官方命名的“GMFlow-small”，本表只登记可追溯的基础单尺度配置。

## 4. 空间模型详细登记

### S01 U²-Net / U²-Netp

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | U²-Net 完整版及 U²-Netp 小型版；仅通用 DUTS 模型，排除 portrait/human 专项变体。 |
| 02 论文、年份与正式来源 | [U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://arxiv.org/abs/2005.09007)，Pattern Recognition 2020；[DOI](https://doi.org/10.1016/j.patcog.2020.107404)。 |
| 03 官方代码 / 模型地址 | [xuebinqin/U-2-Net](https://github.com/xuebinqin/U-2-Net)；[官方架构](https://github.com/xuebinqin/U-2-Net/blob/master/model/u2net.py)。 |
| 04 权重取得状态 | U²-Netp 官方 Drive 权重已下载 4,683,258 B，SHA256 见文末。完整 U²-Net 官方链接可见，未下载；这不等同于完整版权重已验证可用。 |
| 05 编码器 / 头 / 整体参数 | 按官方构造器静态计数：完整 25,676,928 encoder + 18,319,104 decoder + 13,837 side/fusion = 44,009,869；p 为 494,784 + 632,928 + 3,469 = 1,131,181。p 又经 checkpoint 元数据复核。排除 BN 运行缓冲；无 forward。 |
| 06 训练数据、标签与目标函数 | DUTS-TR 10,553 图像及前景掩码；从头训练嵌套 RSU 网络，多尺度输出 BCE 深监督；无 ImageNet 编码器。 |
| 07 输入协议 | 单帧 RGB，论文推理 320×320；坐标回映必须恢复原始宽高。 |
| 08 输出 | 多尺度二值前景软掩码及融合图，不直接输出裁切框。 |
| 09 真实时间信息 | 无；逐帧调用与后处理平滑不会变成学习式时序编码。 |
| 10 空间定位能力 | 分割显著前景的范围、重心、连通结构；不会判断哪个事件值得入选。 |
| 11 预训练能力 | 前景／背景分离及多尺度形状；属于 SOD，并非人眼注视概率预测。 |
| 12 计算成本与测量条件 | 论文 320²、GTX1080Ti：完整 30 FPS，p 40 FPS；存储约 176.3/4.7 MB。原论文 FLOPs、推理峰值显存未报告。MB 不能当参数 M。 |
| 13 部署依赖及 CPU / CUDA / NPU | 原仓库 PyTorch 0.4、torchvision 0.2.1；主要卷积、ceil-mode pooling、双线性插值。CPU 缓存提取可行性需测；910A / torch_npu 未验证。 |
| 14 代码许可证 | [Apache-2.0](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE)。 |
| 15 权重许可证 | 外部 Drive 文件未发现独立权重条款；仓库 Apache 代码许可证不能自动补足未声明的外部权重条款。 |
| 16 官方 benchmark 与协议 | 论文 Table 3，DUTS-TE：完整 maxFβ .873 / MAE .044；p .852 / .054。是像素前景 benchmark，不是裁切 IoU 或 AIC 分数。 |
| 17 AIC 适配方式 | 第一优先级冻结特征：掩码质量、重心、范围、候选 crop 留存质量输入小裁切头；固定 B3/R70 时间点与输出预算。 |
| 18 主要风险与证据边界 | 多显著物体、镜头中心偏置、运动模糊、前景语义和裁切主体不一致；细边界进步可能不改变矩形框 IoU。 |
| 19 本轮核验 / 下一验收 | 下载、静态构造器与checkpoint元数据状态分别见第4/5项及后文审计凭证；本轮无训练、forward或NPU实测。 |

### S02 BASNet

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | BASNet CVPR 2019，固定早期 basnet.pth；不混用 README 后续 2021 移动／网页扩展模型。 |
| 02 论文、年份与正式来源 | [BASNet: Boundary-Aware Salient Object Detection](https://openaccess.thecvf.com/content_CVPR_2019/html/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.html)，CVPR 2019。 |
| 03 官方代码 / 模型地址 | [xuebinqin/BASNet](https://github.com/xuebinqin/BASNet)；官方 model/BASNet.py。 |
| 04 权重取得状态 | 官方 Google Drive 链接存在，ID 1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu；未下载，未完成权重加载。 |
| 05 编码器 / 头 / 整体参数 | 官方构造器静态计数整体 87,060,360；encoder（ResNet34 阶段及额外残差阶段）49,600,896；bridge / decoder / refinement / side heads 合计 37,459,464。不能用 ResNet34 分类器参数替代全网。 |
| 06 训练数据、标签与目标函数 | DUTS-TR；部分 ResNet34 使用 ImageNet 初始化。预测网络加残差细化；BCE + SSIM + IoU 多尺度监督。 |
| 07 输入协议 | 单帧 RGB，论文推理 256×256；训练 resize 256 后随机 crop 224。 |
| 08 输出 | 前景概率掩码，残差分支细化边界。 |
| 09 真实时间信息 | 无。 |
| 10 空间定位能力 | 物体前景及边界定位；无裁切主体选择、跟踪身份或事件定位。 |
| 11 预训练能力 | ImageNet 视觉表征加显著前景／边界分割能力。 |
| 12 计算成本与测量条件 | 论文 PyTorch 0.4、GTX1080Ti、256²：0.040 s/帧，即 25 FPS。FLOPs、峰值显存未报告；显卡容量不是实际峰值。 |
| 13 部署依赖及 CPU / CUDA / NPU | 旧 PyTorch，卷积、残差、插值，训练 SSIM 算子需检查；910A 未验证。计算量与高分辨率显存不能只看 87M 参数。 |
| 14 代码许可证 | [MIT](https://github.com/xuebinqin/BASNet/blob/master/LICENSE)，已查实际 LICENSE。 |
| 15 权重许可证 | 外部权重单独条款未找到。 |
| 16 官方 benchmark 与协议 | 原论文 DUTS-TE maxFβ .860、boundary-relax F .758、MAE .047；须保持各指标定义，不能和裁切 mIoU 横比。 |
| 17 AIC 适配方式 | 作为边界 loss / 空间教师参考；只有轻型冻结掩码先显示原生 crop 增量后，再考虑较重教师。 |
| 18 主要风险与证据边界 | 精确发丝／孔洞的分割收益未必影响裁切框；不能据高 SOD 指标预期提高 AIC。 |
| 19 本轮核验 / 下一验收 | 下载、静态构造器与checkpoint元数据状态分别见第4/5项及后文审计凭证；本轮无训练、forward或NPU实测。 |

### S03 IS-Net / DIS

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | ISNetDIS（ECCV 2022 学术版）与 isnet-general-use 需分开固定；当前官方 DIS V2 仍未完整发布。 |
| 02 论文、年份与正式来源 | [Highly Accurate Dichotomous Image Segmentation](https://arxiv.org/abs/2203.03041)，ECCV 2022；[项目](https://xuebinqin.github.io/dis/index.html)。 |
| 03 官方代码 / 模型地址 | [xuebinqin/DIS](https://github.com/xuebinqin/DIS)；IS-Net/models/isnet.py。 |
| 04 权重取得状态 | README 有学术版／general-use 的 Drive 链接，但描述与链接存在歧义／格式问题；本轮未下载，不能确认特定版本权重。DIS V2 模型未公开。 |
| 05 编码器 / 头 / 整体参数 | ISNetDIS：25,713,856 encoder + 18,319,104 decoder + 13,830 side heads = 44,046,790。训练 GT encoder 6,877,734 单列，不计入部署推理网。官方源码静态计数。 |
| 06 训练数据、标签与目标函数 | DIS5K 5,470：训练 3,000、验证 470、测试 2,000；精细二值掩码。GT mask encoder 先 BCE 训练，再用特征 MSE 加 mask BCE 指导图像模型。 |
| 07 输入协议 | 单帧 RGB，论文主要 1024×1024；高分辨率要求显著高于 320²模型。 |
| 08 输出 | 高精度二值前景软掩码。 |
| 09 真实时间信息 | 无。 |
| 10 空间定位能力 | 高分辨率前景形状与边界；无实例身份、裁切偏好或高光时刻判断。 |
| 11 预训练能力 | 从 mask 域编码器学习结构监督；DIS 不是人眼注视预测的同义词。 |
| 12 计算成本与测量条件 | 原论文 Table 2：模型文件约 176.6 MB，1024² 时间 19.49 ms。对应行硬件／batch／精度无法完整绑定，不能推出本地 FPS。FLOPs、部署峰值显存未核得。 |
| 13 部署依赖及 CPU / CUDA / NPU | PyTorch 1.8 时代代码，卷积、池化、插值；大尺寸中间特征显存可能成为主成本；NPU 未验证。 |
| 14 代码许可证 | 官方明确代码及评价指标采用 [Apache-2.0](https://github.com/xuebinqin/DIS/blob/main/LICENSE.md)。 |
| 15 权重许可证 | 未找到独立、明确权重授权；数据另有 DIS5K Terms of Use，不等于代码许可证。 |
| 16 官方 benchmark 与协议 | 原论文 DIS-VD：maxFβ .791、weightedF .717、MAE .074、S-measure .813；只归属于该数据及版本。 |
| 17 AIC 适配方式 | 较强冻结前景／空间教师候选，适合验证低精度掩码是否为瓶颈；先固定权重版本与原图坐标。 |
| 18 主要风险与证据边界 | 原作者注明早期 DIS 训练分布对人、动物、车辆等覆盖有限；general-use 与论文成绩不可互换；细分割≠正确裁切。 |
| 19 本轮核验 / 下一验收 | 下载、静态构造器与checkpoint元数据状态分别见第4/5项及后文审计凭证；本轮无训练、forward或NPU实测。 |

### S04 UNISAL

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | UNISAL ECCV 2020 官方 pretrained_unisal；视频分支与 SALICON image 分支分别登记。仓库于 2026-10-07 归档。 |
| 02 论文、年份与正式来源 | [Unified Image and Video Saliency Modeling](https://arxiv.org/abs/2003.05477)，ECCV 2020。 |
| 03 官方代码 / 模型地址 | [rdroste/unisal](https://github.com/rdroste/unisal)；[官方权重](https://github.com/rdroste/unisal/blob/master/training_runs/pretrained_unisal/weights_best.pth)。 |
| 04 权重取得状态 | 仓库内官方 weights_best.pth 已实际下载 15,368,928 B，SHA256 见下；仅下载和元数据验证，未 forward。 |
| 05 编码器 / 头 / 整体参数 | checkpoint 元数据排除 BN 缓冲后 3,715,640；cnn 2,223,872，其余时序／解码／域适配 1,491,768。论文 15.5 是 MB，不是 15.5M 参数。 |
| 06 训练数据、标签与目标函数 | ImageNet MobileNetV2；联合 DHF1K、Hollywood-2、UCF Sports 注视数据及 SALICON 图像注意数据。官方配置 KL −0.1 NSS −0.1 CC。 |
| 07 输入协议 | 视频训练 12 帧，重采样约 6 FPS；DHF1K 224×384、Hollywood-2 224×416、UCF 256×384；SALICON 288×384。推理采用交织输出恢复时间覆盖。 |
| 08 输出 | 归一化注视概率热图；域自适应 Gaussian prior、BN、融合和平滑。 |
| 09 真实时间信息 | 视频模式有 ConvGRU；图像模式绕过时序模块。逐帧 image/SALICON 调用不能宣称使用了真正视频时序。 |
| 10 空间定位能力 | 预测人眼可能注视位置，不给目标框／身份／人工裁切范围。 |
| 11 预训练能力 | 静态视觉语义加运动上下文与数据集注视偏置；仍未学过 AIC 高光判断。 |
| 12 计算成本与测量条件 | 原论文约 .009 s/frame GPU；CPU Xeon W-2123 3.6GHz 约 .43 s/frame。不是本轮性能实测；不同论文速度配置不完全匹配。FLOPs、峰值显存未核得。 |
| 13 部署依赖及 CPU / CUDA / NPU | PyTorch、MobileNetV2、depthwise ConvGRU、域 BN／padding／插值；NPU 时序循环与算子覆盖未验证；旧代码维护状态需纳入适配预算。 |
| 14 代码许可证 | [Apache-2.0](https://github.com/rdroste/unisal/blob/master/LICENSE)。 |
| 15 权重许可证 | 已下载文件直接在 Apache 仓库内分发，未见单独排除条款；没有额外独立模型许可证文件。 |
| 16 官方 benchmark 与协议 | 原论文 Table 2 最终联合设置(vi)，DHF1K：AUC-J .901、SIM .390、sAUC .691、CC .490、NSS 2.776。 |
| 17 AIC 适配方式 | 纯时序对照固定同一 source、BN/prior、预处理及 crop mapper，只改变真实邻帧/重复当前帧；跨 SALICON image 与 DHF1K video 的差额只能称完整配置差额。 |
| 18 主要风险与证据边界 | 中心先验可能伪装为内容定位；眼动集中点常比人工裁切范围窄；训练采样率与 1 FPS 稀疏输入不一致。 |
| 19 本轮核验 / 下一验收 | 下载、静态构造器与checkpoint元数据状态分别见第4/5项及后文审计凭证；本轮无训练、forward或NPU实测。 |

### S05 TinyHD

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | TinyHD-M 与 TinyHD-M rc4 分开；优先低成本候选 d123m_rc4_rc1T.pth。不是 TASED-Net，也不是 CVPR 论文。 |
| 02 论文、年份与正式来源 | [TinyHD: Efficient Video Saliency Prediction with Heterogeneous Decoders using Hierarchical Maps Distillation](https://arxiv.org/abs/2301.04619)，WACV 2023；[CVF 原文](https://openaccess.thecvf.com/content/WACV2023/papers/Hu_TinyHD_Efficient_Video_Saliency_Prediction_With_Heterogeneous_Decoders_Using_Hierarchical_WACV_2023_paper.pdf)。第一作者 Feiyan Hu。 |
| 03 官方代码 / 模型地址 | [feiyanhu/tinyHD](https://github.com/feiyanhu/tinyHD)，大小写按官方；作者 Feiyan Hu 等。 |
| 04 权重取得状态 | 官方仓库 rc4 权重已下载 2,481,689 B。完整 M/S 及 rc2 权重链接可见，但未下载；不能宣称全部已加载。 |
| 05 编码器 / 头 / 整体参数 | rc4 checkpoint：encoder 175,880；aggregator 339,394；最终 decoder 7；整体 515,281，排除 BN buffers。论文完整 M 3.92M、S 3.94M；HD2S teacher 的 29.8M 只属训练教师。 |
| 06 训练数据、标签与目标函数 | DHF1K／Hollywood-2／UCF Sports 眼动监督；蒸馏 HD2S 的最终和层级热图，并用无标签 Kinetics-400 扩展教师监督。轻量 MobileNetV2 派生 3D 编码器；KL 蒸馏／真值损失。 |
| 07 输入协议 | 16 帧 RGB，192×256。S 为 16→1，M 为 16→16；rc4 属 M 版本，输出对齐需保留真实帧索引。 |
| 08 输出 | 逐帧人眼注视热图，不直接给裁切框。 |
| 09 真实时间信息 | 有，3D 时空卷积加异构解码器；不是只对独立帧打分后平滑。 |
| 10 空间定位能力 | 注视定位，含运动上下文；无身份或事件完成判断。 |
| 11 预训练能力 | 训练教师和 Kinetics 视频注入的视觉信息可压缩到极小学生；小参数量不意味着没有预训练语义。 |
| 12 计算成本与测量条件 | 论文 Table 5：完整 M 7.95 GMACs、rc4 M 4.06 GMACs，均按 16 张输出图；不是每帧 GFLOPs。完整 S 89.12 GMACs 是 16 次单输出。FPS、峰值显存未报告。 |
| 13 部署依赖及 CPU / CUDA / NPU | PyTorch、3D 卷积／depthwise 路径、插值。参数很小不保证 NPU 延迟低；910A、CPU 吞吐及视频解码成本未验证。 |
| 14 代码许可证 | 官方仓库未发现 LICENSE，API license:null；公开源码不等于明确可再分发许可。 |
| 15 权重许可证 | 仓库权重未发现独立许可证；研究可获取性已确认，部署授权就绪性未确认。 |
| 16 官方 benchmark 与协议 | 论文 DHF1K test 完整 M：AUC-J .905、SIM .387、sAUC .707、CC .493、NSS 2.819。rc4 蒸馏 Table 5 为 validation：AUC-J .8999、CC .4564、NSS 2.5581；不可混报为完整模型 test。 |
| 17 AIC 适配方式 | 真时序空间对照首选之一：冻结 rc4，先与 UNISAL image／中心先验在同一原生框集合比较；不立即迁移到时间排名。 |
| 18 主要风险与证据边界 | 极小模型也可能依赖教师域偏置；16 帧跨 shot 或低帧率误用会偏移热图；许可与真实设备速度待确认。 |
| 19 本轮核验 / 下一验收 | 下载、静态构造器与checkpoint元数据状态分别见第4/5项及后文审计凭证；本轮无训练、forward或NPU实测。 |

### S06 YuNet

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | OpenCV Zoo face_detection_yunet_2023mar.onnx 固定导出；2026may 是支持动态 H/W 的重导出，不能当新的训练模型。 |
| 02 论文、年份与正式来源 | [YuNet: A Tiny Millisecond-level Face Detector](https://link.springer.com/article/10.1007/s11633-023-1423-y)，Machine Intelligence Research 2023。 |
| 03 官方代码 / 模型地址 | [ShiqiYu/libfacedetection](https://github.com/ShiqiYu/libfacedetection)；[训练仓库](https://github.com/ShiqiYu/libfacedetection.train)；[OpenCV Zoo](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)；[官方 HF](https://huggingface.co/opencv/face_detection_yunet)。 |
| 04 权重取得状态 | 2023mar 官方 ONNX 已下载 232,589 B；protobuf 元数据与 SHA256 已核。未进行 ONNXRuntime / OpenCV forward。 |
| 05 编码器 / 头 / 整体参数 | 原论文网络 75,856；所下载导出图 Conv 等学习初始化量 53,104，总 initializer 元素 53,121（含形状常量）。这是融合导出图口径，不替代论文 trainable count；编码器／头未逐一对应。 |
| 06 训练数据、标签与目标函数 | WIDER FACE 框与 5 点关键点；simOTA 分配，分类及 objectness 交叉熵、EIoU box loss、SmoothL1 landmark；论文权重系数 obj/box/landmark 为 1/5/.1。 |
| 07 输入协议 | 单帧，OpenCV 示例默认 320×320，尺寸回映及 BGR/RGB 接口需按导出包装固定。 |
| 08 输出 | 人脸框、5 点关键点、置信度。 |
| 09 真实时间信息 | 无；不进行人物身份识别或跟踪。 |
| 10 空间定位能力 | 人脸位置、姿态近似与大小；不覆盖人体全身或非人类事件主体。 |
| 11 预训练能力 | 窄领域人脸检测监督；可作为明确可解释的空间 cue。 |
| 12 计算成本与测量条件 | 原论文 i7-12700K、320² 1.6 ms；WIDER hard 单尺度 AP .811。不是本下载导出版本的性能复测。峰值显存未报告；NPU 延迟未知。 |
| 13 部署依赖及 CPU / CUDA / NPU | OpenCV DNN + ONNX 可走 CPU，无必需 CUDA；导出包含 Conv/Resize/Reshape/Sigmoid 等，NMS 在外部包装。2026 动态维度对 OpenCV 版本有要求；NPU 未验证。 |
| 14 代码许可证 | 作者项目及所选 Zoo 文件夹 [MIT](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE)。 |
| 15 权重许可证 | Zoo 明示该目录所有文件 MIT，包含所选 ONNX；比只有外部权重链接的授权证据更清楚。 |
| 16 官方 benchmark 与协议 | Zoo 当前 README 概述 AP .834/.824/.708 与评估表 .8844/.8656/.7503 并存，版本／评估口径未完全一致；应钉住导出版本复测，不把 .811 套给 2023mar。 |
| 17 AIC 适配方式 | 第一批低成本 cue：脸的中心、面积、数量、相对边缘距离，配合人体／前景范围构造 crop 候选。 |
| 18 主要风险与证据边界 | 只有脸时会裁掉肢体／球／器械；小脸、侧脸、遮挡易漏；多数脸或最大脸不必然是人工高光主体。 |
| 19 本轮核验 / 下一验收 | 下载、静态构造器与checkpoint元数据状态分别见第4/5项及后文审计凭证；本轮无训练、forward或NPU实测。 |

### S07 RTMDet-tiny

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | RTMDet-tiny，rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc；不是 RTMDet-Ins 或 rotated 变体。 |
| 02 论文、年份与正式来源 | [RTMDet: An Empirical Study of Designing Real-Time Object Detectors](https://arxiv.org/abs/2212.07784)，2022 arXiv 原文。 |
| 03 官方代码 / 模型地址 | [open-mmlab/mmdetection/configs/rtmdet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet)；固定 tiny 配置。 |
| 04 权重取得状态 | 官方 OpenMMLab checkpoint 已下载 57,532,893 B，SHA256 见下；元数据解析遇到 MMEngine 类后安全停止，未 forward。 |
| 05 编码器 / 头 / 整体参数 | 官方 whole detector 4.8M；encoder / neck / head 未从下载包完整分拆。CSPNeXt-tiny 分类器含分类头，不能从其参数简单相减推算检测器部分。 |
| 06 训练数据、标签与目标函数 | ImageNet CSPNeXt 初始化，COCO 2017 检测；Quality Focal Loss + GIoU，动态软标签分配，Mosaic / MixUp。 |
| 07 输入协议 | 单帧，固定配置 640×640；应准确逆转 resize/padding 到原图。 |
| 08 输出 | 80 类对象框、置信度；可筛 person，也应保留球／器械等事件物体。 |
| 09 真实时间信息 | 无。 |
| 10 空间定位能力 | 人体与常见对象范围；不是实例跟踪，也不是内容重要性排序。 |
| 11 预训练能力 | 通用目标检测语义，能提供前景面积之外的对象关系线索。 |
| 12 计算成本与测量条件 | 官方表：8.1 GFLOPs、4.8M、COCO box AP 41.1；640²，TRT8.4.3/cuDNN8.2 FP16/batch1，不含 NMS：RTX3090 .98 ms，T4 2.34 ms。2.34 不是 PyTorch 速度；峰值显存未报告。 |
| 13 部署依赖及 CPU / CUDA / NPU | PyTorch/MMEngine/MMCV；NMS 等扩展及导出后处理需检查，不能照抄 CUDA 安装到910A。ONNX、CPU NMS 为可研究路径，NPU 未验证。 |
| 14 代码许可证 | [Apache-2.0](https://github.com/open-mmlab/mmdetection/blob/main/LICENSE)。 |
| 15 权重许可证 | 官方模型库可获取，未见独立权重许可证；不能用代码 Apache 条款自动推断训练数据授权范围。 |
| 16 官方 benchmark 与协议 | 上述 AP 41.1 属 COCO object detection，官方 postprocess 更新可能带来 .1–.3 AP 差异；不是 person-only AP 或 crop mIoU。 |
| 17 AIC 适配方式 | YuNet 的补充：全身、多人范围、人物与器械相对位置；冻结检测框构建少量候选后，用原生 crop GT 学选择与缩放。 |
| 18 主要风险与证据边界 | 最大 person box 未必是主体；漏检／相邻人框合并可使 crop 过宽；额外模型只有在固定时间点裁切增益验证后才值得部署。 |
| 19 本轮核验 / 下一验收 | 下载、静态构造器与checkpoint元数据状态分别见第4/5项及后文审计凭证；本轮无训练、forward或NPU实测。 |

### S08 AutoFlip

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | Google Research AutoFlip / MediaPipe 桌面视频重构图 pipeline，2020；固定 graph 后审计全依赖。 |
| 02 论文、年份与正式来源 | [AutoFlip: An Open Source Framework for Intelligent Video Reframing](https://research.google/blog/autoflip-an-open-source-framework-for-intelligent-video-reframing/)，Google Research 2020 官方机制介绍，非正式量化 benchmark 论文。 |
| 03 官方代码 / 模型地址 | [google-ai-edge/mediapipe AutoFlip](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/examples/desktop/autoflip)；autoflip_graph.pbtxt。 |
| 04 权重取得状态 | 图与处理代码可读，所依赖 face/object TFLite assets 尚未逐一下载／版本核验；完整可运行链未验证。 |
| 05 编码器 / 头 / 整体参数 | 路径优化器本身 0 学习参数；完整系统还包含 face / object detectors，总参数未核得，不能登记全链为 0。 |
| 06 训练数据、标签与目标函数 | 路径优化采用检测到的显著区和规则权重，不是人工 crop 监督训练；感知模型预训练数据依各模型资产。 |
| 07 输入协议 | 完整视频及目标纵横比；需要 shot、候选区域和逐帧几何。 |
| 08 输出 | 连续裁切轨迹和重构图视频；特定不可行情况会 padding。 |
| 09 真实时间信息 | 有跨帧 shot / trajectory 优化；这不是学习式视频 encoder，也不提供高光时刻选择。 |
| 10 空间定位能力 | 对象／人脸区域加轨迹约束；选择 stationary、panning、tracking 模式。 |
| 11 预训练能力 | 借用预训练感知信号，路径层优化视觉连续性。 |
| 12 计算成本与测量条件 | 官方博客无可复核统一 FLOPs、FPS、峰值显存或 crop benchmark 表；全部未报告。 |
| 13 部署依赖及 CPU / CUDA / NPU | C++/Bazel/MediaPipe/TFLite/FFmpeg；CPU 图可用性需实装验证，移植整套比提取轨迹算法成本高；910A 不自动兼容。 |
| 14 代码许可证 | MediaPipe 主仓库 [Apache-2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)，第三方组件另核。 |
| 15 权重许可证 | 依赖 detector 各模型文件需单独核；本轮未完成，不宣称全链权重许可已清。 |
| 16 官方 benchmark 与协议 | 未找到可与 AIC 对照的官方量化结果；不引用产品演示作为 IoU 增益。 |
| 17 AIC 适配方式 | 借用 shot 内候选约束／低阶路径优化；必须禁用不符合 AIC box 契约的 padding 路径，并保留原始帧编号。 |
| 18 主要风险与证据边界 | 轨迹更稳不直接加 AIC 分；平滑可能把本来正确的单帧框拖离 GT；输出重构图视频不是 AIC 所需提交物。 |
| 19 本轮核验 / 下一验收 | 下载、静态构造器与checkpoint元数据状态分别见第4/5项及后文审计凭证；本轮无训练、forward或NPU实测。 |

### S09 SmartVidCrop / RetargetVid

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | SmartVidCrop：RetargetVid 仓库 ICIP 2021 默认设置及 ISM 2021 use_best_settings=True 分开。 |
| 02 论文、年份与正式来源 | [A fast smart-cropping method and dataset for video retargeting](https://doi.org/10.1109/ICIP42928.2021.9506390)（ICIP 2021）；A Web Service for Video Smart-Cropping（ISM 2021），[作者原文列表](https://github.com/bmezaris/RetargetVid)。 |
| 03 官方代码 / 模型地址 | [bmezaris/RetargetVid](https://github.com/bmezaris/RetargetVid)，包含数据、smartVidCrop.py 及第三方包装。 |
| 04 权重取得状态 | 代码与标注可取；完整链需另取 TransNet v1 权重。UNISAL 权重在 S04 已下载，但 SmartVidCrop 全链未运行。 |
| 05 编码器 / 头 / 整体参数 | 几何后处理 0 学习参数；完整系统含 TransNet v1 + UNISAL，合计未核。不能借用 TransNetV2 参数，也不能称全模型 0。 |
| 06 训练数据、标签与目标函数 | 200 个 DHF1K 视频（001–100、601–700），6 名标注者、1:3 和 3:1 两种比例，2,400 注释文件。算法非端到端人工 crop 训练。 |
| 07 输入协议 | 视频、目标纵横比、采样与显著性配置。 |
| 08 输出 | 逐帧裁切坐标与裁切视频；输入视频时刻不作高光选择。 |
| 09 真实时间信息 | shot 检测、焦点聚类、轨迹平滑构成时序处理。但原始入口 init_unisal_for_images() 明确取 SALICON image 路径，不能据 UNISAL 名称称其热图用了 ConvGRU。 |
| 10 空间定位能力 | 注视热图聚类得到 focus area，再以几何／平滑裁切；没有显式事件主体标签。 |
| 11 预训练能力 | 感知侧继承图像注视监督；裁切侧靠显著区覆盖和轨迹规则。 |
| 12 计算成本与测量条件 | README ratio 运行时间/视频时长：默认 1:3 .19，3:1 .20；best .13/.14。硬件／分辨率绑定不完整，不能转称某设备 FPS。FLOPs、峰值显存未报告。 |
| 13 部署依赖及 CPU / CUDA / NPU | Python、TensorFlow 1 图会话、PyTorch、FFmpeg、聚类／滤波依赖；910A 多框架迁移摩擦较大，建议借方法及数据而非移植全部系统。 |
| 14 代码许可证 | [MIT](https://github.com/bmezaris/RetargetVid/blob/main/LICENSE.md)；第三方组件保留各自许可。 |
| 15 权重许可证 | 主算法无专属权重；第三方 TransNet/UNISAL 独立条款需保留，完整权重链未核完。 |
| 16 官方 benchmark 与协议 | 原 README：默认 ICIP meanIoU 49.9%（1:3）、71.4%（3:1）；best ISM 52.9% / 75.3%。设置、目标比例不同，不能把差额归因于某单一平滑改动。 |
| 17 AIC 适配方式 | 价值主要是有人类框的 crop 监督与规则对照；先审计 DHF1K train / val 视频重叠，再用同一 AIC 原生框评价。 |
| 18 主要风险与证据边界 | 极端纵横比、仅200视频、DHF1K来源域与训练重叠、meanIoU聚合定义和AIC不同；高分不能外推。 |
| 19 本轮核验 / 下一验收 | 下载、静态构造器与checkpoint元数据状态分别见第4/5项及后文审计凭证；本轮无训练、forward或NPU实测。 |

### S10 DiffSal

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | DiffSal CVPR 2024 音视频版及 video-only ablation 分开；固定官方 inference timesteps 配置。 |
| 02 论文、年份与正式来源 | [DiffSal: Joint Audio and Video Learning for Diffusion Saliency Prediction](https://openaccess.thecvf.com/content/CVPR2024/html/Xiong_DiffSal_Joint_Audio_and_Video_Learning_for_Diffusion_Saliency_Prediction_CVPR_2024_paper.html)，CVPR 2024；[arXiv](https://arxiv.org/abs/2403.01226)。 |
| 03 官方代码 / 模型地址 | [junwenxiong/diff_sal](https://github.com/junwenxiong/diff_sal)。当前 README 已提供 Google Drive checkpoints 文件夹。 |
| 04 权重取得状态 | 官方 Drive folder 链接存在，未实际下载；README release 标记的旧缓存与当前源不同，以当前源为准。 |
| 05 编码器 / 头 / 整体参数 | 论文完整音视频 76.57M，video-only 70.54M；encoder/head 未独立核数。教师／预训练任务模型不等于额外部署模型。 |
| 06 训练数据、标签与目标函数 | MViTv2 视频、VGGish 音频预训练；DHF1K 视觉预训练及 AVAD、DIEM、Coutrot1/2、ETMD、SumMe 注视集；预测清晰热图的 MSE 扩散目标。这里 SumMe 注视任务不是高光标签。 |
| 07 输入协议 | 16 帧 224×384 RGB；音频 log-Mel 表征，配置约 9 片×112×192；必须核对精确同步。 |
| 08 输出 | 条件扩散生成注视热图。 |
| 09 真实时间信息 | 有视频时空编码和音视频融合；diffusion step 是生成迭代，不是视频时间长度。 |
| 10 空间定位能力 | 音视频注视空间分布，不直接回归 crop 框或判断哪段高光。 |
| 11 预训练能力 | 视频与音频预训练加注视监督，能提供发声主体／运动上下文，但并非语言语义判据。 |
| 12 计算成本与测量条件 | 论文音视频 187.31 GFLOPs、video-only 161.06 GFLOPs；训练使用4×RTX4090。文中模块内存1.20G不能当全pipeline峰值；FPS／统一峰值未报告。论文主实验推理4步，当前配置可为1步，二者成绩／成本不能混配；训练1000扩散步也不是推理次数。 |
| 13 部署依赖及 CPU / CUDA / NPU | PyTorch、MViTv2 attention、VGGish、UNet及DDIM；音频前处理和视频同步增加成本，910A算子／精度未验证。 |
| 14 代码许可证 | 官方仓库未发现 LICENSE，API license:null。 |
| 15 权重许可证 | 外部 checkpoint 未找到独立权重许可证，未验证下载。 |
| 16 官方 benchmark 与协议 | 论文 Table 5 AVAD 音视频 CC .738、NSS4.22、AUC-J .935、SIM .571；公开注视指标不对应 AIC。 |
| 17 AIC 适配方式 | 近年机制参考与较重空间教师备选，优先级低于小型模型；只有明确需要声源辅助定位时，再固定时间点做单独增量。 |
| 18 主要风险与证据边界 | 额外音频可能聚焦配音／离屏声源；算力、同步、许可成本较高；不能用 CVPR 新近发表替代本任务验证。 |
| 19 本轮核验 / 下一验收 | 下载、静态构造器与checkpoint元数据状态分别见第4/5项及后文审计凭证；本轮无训练、forward或NPU实测。 |

## 5. 时序结构与轻量语义详细登记

### T01 — PySceneDetect：ContentDetector / AdaptiveDetector

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | PySceneDetect 的 ContentDetector / AdaptiveDetector；本日审查官方 latest API 文档，未安装，未锁定本机包版本。它是算法工具，非预训练神经网络。 |
| 02 论文、年份与正式来源 | 不冒充某一篇论文模型；以作者项目和算法 API 为规范来源：[官方文档][PSD-doc]。 |
| 03 官方代码 / 模型地址 | [Breakthrough/PySceneDetect][PSD-repo]。 |
| 04 权重取得状态 | N/A，不需要学习权重；软件源码入口已核实，未安装执行。 |
| 05 编码器 / 头 / 整体参数 | 本算法三者均0个学习参数；阈值、最短镜头长、窗口大小是超参数。若前后接图像编码器，则另计。 |
| 06 训练数据、标签与目标函数 | 无训练集、无监督训练。Content比较相邻图像变化，Adaptive使用局部变化基线。 |
| 07 输入协议 | 按顺序解码的RGB视频帧；分辨率/降采样与帧号映射必须固定。 |
| 08 输出 | 镜头起止、逐帧content score / adaptive ratio；不输出高光置信度、主体框或动作类别。 |
| 09 真实时间信息 | 有真实相邻帧比较；Adaptive用滚动邻域，是有未来上下文的两遍算法。没有学习的时序语义。 |
| 10 空间定位能力 | 颜色/亮度/边缘变化经全帧汇总，不能提供主体位置。 |
| 11 预训练能力 | 无预训练；提供剪辑边界与视觉变化证据。 |
| 12 计算成本与测量条件 | 像素变化的基本成本随帧数与图像像素增长，属算法分析；本轮FPS/内存未测。可CPU运行，但完整解码成本必须计入。 |
| 13 部署依赖及 CPU / CUDA / NPU | Python/OpenCV及视频后端；不需要CUDA。NPU未测，通常可放CPU侧；解码帧号对齐是首要集成风险。 |
| 14 代码许可证 | 官方代码 BSD-3-Clause；依赖按各自许可。 |
| 15 权重许可证 | N/A，不需要学习权重。 |
| 16 官方 benchmark 与协议 | 不填未经核实的跨数据集F1/FPS。API默认Content阈值27；Adaptive阈值3、窗口宽度2、最短镜头15帧等只是默认值，不是AIC最佳值。 |
| 17 AIC 适配方式 | 先冻结算法，生成距切点、镜头长度、镜头内相位与跨镜头禁止平滑标记；仅作为附加结构特征和控制。 |
| 18 主要风险与证据边界 | 快速摇摄、闪光、主体运动可能造成假切点；切镜不必然是事件高潮；低采样率会丢短转场。不得把“变化大”直接指定为正例。 |
| 19 本轮核验 / 下一验收 | 官方文档、仓库和许可已读；软件未执行；权重N/A；NPU未测。 |

### T02 — TransNet V2

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | TransNet V2；登记官方 PyTorch 默认配置F=16、L=3、S=2、D=1024，开启frame similarity、color histograms和双头。 |
| 02 论文、年份与正式来源 | Souček、Lokoč，TransNet V2: An effective deep network architecture for fast shot transition detection，2020：[arXiv原文][TN-paper]；不将其误报为某年CVPR/ICPR论文。 |
| 03 官方代码 / 模型地址 | [soCzech/TransNetV2][TN-repo]；[官方PyTorch实现][TN-code]。 |
| 04 权重取得状态 | 官方仓库列TF预训练权重与PyTorch转换/推理路径；存在Git LFS获取步骤。本轮只核实代码/入口，没有下载LFS对象、转换或加载。 |
| 05 编码器 / 头 / 整体参数 | 官方PyTorch默认结构静态计数7,616,258：卷积+BN 2,548,864，相似特征83,584，共享FC4,981,760，单帧/辅助头各1,025。AIC额外头/空间链另计。 |
| 06 训练数据、标签与目标函数 | IACC.3/ClipShots镜头；15%真实转场、35%合成hard cut、50%合成dissolve。双头交叉熵分别学习转场中点和全部转场帧。 |
| 07 输入协议 | RGB短视频，官方PyTorch要求uint8 [B,T,27,48,3]；推理以100帧窗口、保留中间50帧预测形成上下文。 |
| 08 输出 | 每帧转场概率、镜头区间；示例镜头端点为0起始且两端inclusive。 |
| 09 真实时间信息 | 因式分解时空卷积、时间膨胀、前后50帧相似向量，是真正短时序模型；所学是转场识别。 |
| 10 空间定位能力 | 全帧低分辨率证据，不输出可用主体框/显著图；不应代替crop模型。 |
| 11 预训练能力 | 学习分辨剪辑转换与连续镜头的外观变化；不直接学习“最精彩时刻”或动作成功/失败。 |
| 12 计算成本与测量条件 | 本轮未测FLOPs/FPS/显存；所审4页论文未提供可直接采用的推理FPS。论文17小时V100 16GB为训练耗时，不能当推理成本。 |
| 13 部署依赖及 CPU / CUDA / NPU | 原版TensorFlow，作者提供PyTorch实现；Conv3d、BatchNorm、相似矩阵、gather及整数histogram scatter。非纯卷积图；NPU导出与这些算子均未验证。 |
| 14 代码许可证 | 官方代码 MIT。 |
| 15 权重许可证 | 预训练权重随官方项目发布；独立权重许可未核清，采用时保存具体分发条款。 |
| 16 官方 benchmark 与协议 | 论文表2转场F1：ClipShots77.9、BBC96.2、RAI93.9；TransNetV2阈值0.5，协议允许至多2帧偏差。这不是AIC零时间容忍的连续IoU F。 |
| 17 AIC 适配方式 | 冻结输出作镜头结构、shot reset/平滑边界；仅在PySceneDetect假切/漏切造成可观察误差后升级。保留原始frame索引，禁止秒数重取整。 |
| 18 主要风险与证据边界 | 转场F1向高光迁移未经验证；剪辑风格偏移、短镜头、decode吞吐；全链加约7.6M可能占用严格100M实验预算。 |
| 19 本轮核验 / 下一验收 | 论文、源码、许可证入口和权重路径已查；参数仅静态推导；权重未下载/运行；NPU未测。 |

### T03 — Kernel Temporal Segmentation（KTS）

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | Potapov等的KTS时序分段算法；区分KTS分段本身与原论文的category-specific summarization完整系统。 |
| 02 论文、年份与正式来源 | Category-specific video summarization，ECCV 2014；[DOI][KTS-paper]；[原作者项目][KTS-project]。 |
| 03 官方代码 / 模型地址 | 原作者项目有Temporal segmentation code入口，不强行指认后续第三方Python移植为原始实现。 |
| 04 权重取得状态 | KTS自身无权重。原作者代码下载在本轮工具中返回错误，未得到压缩包；所需视觉描述子的权重需单列。 |
| 05 编码器 / 头 / 整体参数 | KTS算法学习参数0；上游描述子encoder不为0。完整推理=所选encoder+实际时间分数头；原论文事件SVM也不能忽略。 |
| 06 训练数据、标签与目标函数 | KTS通过核空间内分段一致性与复杂度惩罚确定变化点，不使用高光GT训练。原论文category importance使用事件类别监督，属于额外系统。 |
| 07 输入协议 | 有序描述子及其Gram核矩阵；可以来自现有图像语义编码器，不一定需要原论文整套特征。 |
| 08 输出 | 变化点/连续分段；原生不产生逐帧高光概率或空间框。 |
| 09 真实时间信息 | 强制连续时间分段，利用区间特征分布一致性；不显式学动作方向/因果关系。 |
| 10 空间定位能力 | 取决于描述子，KTS本身没有主体定位。 |
| 11 预训练能力 | KTS无预训练；其“语义”能力主要来自输入表示，不能把核分段当成语义编码器。 |
| 12 计算成本与测量条件 | 稠密Gram矩阵存储约O(T²)；给定最大切点数m的朴素动态规划约O(mT²)，这是算法量级而非实测。长视频需固定下采样及上限。 |
| 13 部署依赖及 CPU / CUDA / NPU | 核计算与CPU动态规划可实现；原作者压缩包本轮未取到，具体依赖未确认。NPU未测，也不能因为学习参数0而记总成本0。 |
| 14 代码许可证 | 未知：原作者页面未给出已核验 LICENSE，代码包未取得；不能沿用第三方 fork 许可。 |
| 15 权重许可证 | KTS 本身无权重；上游视觉描述子、原论文分类器需单独审计。 |
| 16 官方 benchmark 与协议 | 原项目MED Summaries为类别相关摘要评估，不能抽取为AIC高光成绩；本轮未核实可独立归因于KTS的数值，留待原包/论文表核验。 |
| 17 AIC 适配方式 | 在现有冻结embedding上产生稳定段、段长和段内相位，作为与shot cut不同的结构控制；先限制候选数和时域分辨率。 |
| 18 主要风险与证据边界 | 表示/核选择、切点数与下采样会改变结果；动作在单镜头内发生时有价值，但分段质量不等同于重要性。许可与原始实现仍阻塞直接纳入。 |
| 19 本轮核验 / 下一验收 | 作者页面、正式论文身份和代码入口已核实；原始包获取失败；License未知；没有运行、参数加载或NPU结果。 |

### T04 — TSM with ResNet50

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | Temporal Shift Module，ResNet50 RGB、8 segments、blockres、shift_div=8、无non-local的Kinetics-400版本。 |
| 02 论文、年份与正式来源 | Lin、Gan、Han，TSM: Temporal Shift Module for Efficient Video Understanding，ICCV 2019；[原文][TSM-paper]。 |
| 03 官方代码 / 模型地址 | [mit-han-lab/temporal-shift-module][TSM-repo]；ops/models.py明确替换分类头。 |
| 04 权重取得状态 | 作者README有K400的dense/uniform两个checkpoint链接；[dense 8帧模型][TSM-weights]。本轮未实际下载、加载，不能将两种训练采样版本混用。 |
| 05 编码器 / 头 / 整体参数 | TSM移位本身新增0。按TorchVision ResNet50基座和K400头静态推导：encoder23,508,032，head819,600，完整24,327,632。非non-local/非flow条件；不得只报告0。 |
| 06 训练数据、标签与目标函数 | ImageNet初始化的2D CNN，Kinetics-400视频动作分类训练；官方另有Something-Something权重。不是AIC高光标签训练。 |
| 07 输入协议 | 按固定策略采样的多帧RGB序列；此条登记8帧。Flow与RGBDiff是另两种模式，不能宣称RGB版本调用了光流。 |
| 08 输出 | 视频/clip动作logits、时序混合后的特征；无crop框或高光分数。 |
| 09 真实时间信息 | 将部分通道移到相邻时间位置，使后续2D卷积利用邻帧；offline双向、另有online因果版本。比独立帧平均具有顺序结构。 |
| 10 空间定位能力 | CNN保留空间网格，可接定位头，但动作模型本身不提供目标框。 |
| 11 预训练能力 | 对短时动作和外观演化的判别，仍可能利用场景/物体静态捷径；需ordered/repeat/shuffle对照量化。 |
| 12 计算成本与测量条件 | 官方“0额外FLOPs”只指shift没有乘加，不表示内存搬运0成本。整个8帧ResNet50约8倍单帧骨干计算；本轮FPS/显存未测，不套其他设备报告。 |
| 13 部署依赖及 CPU / CUDA / NPU | PyTorch/TorchVision；普通shift含reshape/slice/赋值，in-place优化实现与目标后端另审。CUDA示例可用不等于NPU支持；本轮无导出测试。 |
| 14 代码许可证 | 项目 MIT；TorchVision 等依赖按各自许可。 |
| 15 权重许可证 | 实际 checkpoint 分发许可未独立核清；不把项目代码许可等同于全部训练数据授权。 |
| 16 官方 benchmark 与协议 | 作者README：K400 dense 8帧×10clips为74.1% top-1；uniform 8帧×1clip center71.2%，full-res72.8%。三者不是同一成本设置。 |
| 17 AIC 适配方式 | 先冻结正确预训练权重，用短窗读出作真实时序参照；保持同解码帧与同预算，比较同backbone独立帧池化。LoRA/微调仅列后续选项，本轮不执行。 |
| 18 主要风险与证据边界 | AIC早期TSM失败须核对是否正确加载/采样，不能以错误工程反证架构无效；均匀抽帧跨度变化导致任务变质；约24M还需加空间链。 |
| 19 本轮核验 / 下一验收 | 原文、README、模型构造与官方权重链接已核实；参数是静态条件推导；未加载权重/未跑NPU。 |

### T05 — ResNet18 / IMAGENET1K_V1

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | TorchVision ResNet18，ResNet18_Weights.IMAGENET1K_V1；明确预处理与权重枚举。 |
| 02 论文、年份与正式来源 | He等，Deep Residual Learning for Image Recognition，CVPR 2016，2015预印本；[原文][RN-paper]。 |
| 03 官方代码 / 模型地址 | [TorchVision官方实现与模型页][RN-doc]，为该登记权重的维护者来源。 |
| 04 权重取得状态 | [官方.pth入口][RN-weights]已确认；本轮未下载/校验hash/加载。AIC R91另有历史特征抽取证据，不是本轮权重实测。 |
| 05 编码器 / 头 / 整体参数 | 官方完整11,689,512；去1000类FC后11,176,512；FC513,000。替换时间头/保留空间链后另计；冻结不扣除。 |
| 06 训练数据、标签与目标函数 | ImageNet-1K有监督类别分类。包含物体与类别语义，不因没有语言输入就变成“无语义”。 |
| 07 输入协议 | 单张RGB；枚举规定resize256→center crop224、ImageNet归一化。全帧warp或foreground crop是不同输入合同。 |
| 08 输出 | 1000类logits、pooled512D embedding或卷积网格；不原生输出高光/主体框。 |
| 09 真实时间信息 | 原模型没有跨帧交互；一串帧独立编码也不自动获得动作顺序。 |
| 10 空间定位能力 | 中间卷积网格包含位置结构，全局平均特征损失显式坐标；CAM不是有框监督的crop模型。 |
| 11 预训练能力 | 局部纹理、物体形状、场景类别等层级判别，提供轻量隐式语义控制。 |
| 12 计算成本与测量条件 | 官方224输入1.81 GFLOPS；FPS/VRAM未在本轮测。批量、帧率、解码、精度会影响实际吞吐。 |
| 13 部署依赖及 CPU / CUDA / NPU | PyTorch/TorchVision；Conv/BN/ReLU/pool/Linear，结构较常规；NPU仍未测试，不能承诺可直接部署。 |
| 14 代码许可证 | TorchVision 代码 BSD-3-Clause。 |
| 15 权重许可证 | 具体官方权重分发与训练数据条款需分别保存；本轮未完成独立权重授权核验。 |
| 16 官方 benchmark 与协议 | ImageNet-1K，官方V1 top-1 69.758%、top-5 89.078%，对应上述224预处理；不是高光性能。 |
| 17 AIC 适配方式 | 最便宜且已有仓库路径的冻结语义对照；在同frame/PTS/crop下比较linear/ridge小头及结构残差，先不启动新backbone训练。 |
| 18 主要风险与证据边界 | 背景捷径、静态物体替代动作；R91使用该骨干只证明某条工程/代理结果，不证明100M模型达到官方40分。 |
| 19 本轮核验 / 下一验收 | 原文、官方metadata/权重入口和R91源码记录已核实；参数分解静态；本轮无权重/推理/NPU。 |

### T06 — MobileNetV3-Large / IMAGENET1K_V2

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | TorchVision MobileNetV3-Large，width_mult=1.0，IMAGENET1K_V2；不与Small或V1混称。 |
| 02 论文、年份与正式来源 | Howard等，Searching for MobileNetV3，ICCV 2019；[原文][MN-paper]。 |
| 03 官方代码 / 模型地址 | [TorchVision官方模型页][MN-doc]及models/mobilenetv3.py。 |
| 04 权重取得状态 | [官方V2权重][MN-weights]入口已确认；没有实际下载或加载。 |
| 05 编码器 / 头 / 整体参数 | 完整官方5,483,032；移除整个classifier为2,971,952；classifier两层共2,511,080。若仅移除最后一层，不能报前一种encoder计数。 |
| 06 训练数据、标签与目标函数 | ImageNet-1K分类，V2采用TorchVision改进训练recipe；原论文架构与V2权重训练不是同一实验。 |
| 07 输入协议 | 单帧RGB；V2 resize232→center crop224及ImageNet归一化，不能套V1 resize256。 |
| 08 输出 | 分类logits；features+avgpool为960D表示，可取空间特征但没有定位头。 |
| 09 真实时间信息 | 无跨帧时序。连续帧特征差分/时间头属于新增方法。 |
| 10 空间定位能力 | 卷积网格保留局部线索；SE全局通道门控不等同于主体空间定位。 |
| 11 预训练能力 | 面向分类的轻量视觉语义；不原生识别动作终点、故事转折或高光政策。 |
| 12 计算成本与测量条件 | 官方约0.217 GFLOPS/224图；本轮未测FPS/显存。低FLOPs与NPU延迟不能直接等同。 |
| 13 部署依赖及 CPU / CUDA / NPU | PyTorch/TorchVision；depthwise convolution、SE、Hardswish/Hardsigmoid。量化/融合和目标芯片支持需实际验证；NPU未测。 |
| 14 代码许可证 | TorchVision 代码 BSD-3-Clause。 |
| 15 权重许可证 | 具体官方权重分发与训练数据条款需分别保存；本轮未完成独立权重授权核验。 |
| 16 官方 benchmark 与协议 | ImageNet-1K V2 top-1 75.274%、top-5 92.566%，224 crop；V1的74.042%是另一权重。 |
| 17 AIC 适配方式 | 在固定输入下替换ResNet18作成本控制，冻结特征+同一小头；只有端到端吞吐验证后才有部署优先级。 |
| 18 主要风险与证据边界 | 减小参数会改变表征与归纳偏置，不保证高光更优；复杂小算子可能受内存和内核调度限制。 |
| 19 本轮核验 / 下一验收 | 官方源码、参数/计算/准确率metadata和权重入口已核实；head分解静态；权重未下载/运行，NPU未测。 |

### T07 — DeiT-S / patch16-224

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | deit_small_patch16_224；另登记distilled_small作为明确不同权重，不能把其指标套普通版。 |
| 02 论文、年份与正式来源 | Touvron等，Training data-efficient image transformers & distillation through attention，ICML 2021；[PMLR正式页][DEIT-paper]。 |
| 03 官方代码 / 模型地址 | [facebookresearch/deit][DEIT-repo]，原版配置与结果见README_deit.md。 |
| 04 权重取得状态 | 作者分别提供[普通Small][DEIT-weights]与[distilled Small][DEIT-distilled]下载URL；本轮未下载/加载。 |
| 05 编码器 / 头 / 整体参数 | 官方Small完整约22M；普通分类头384→1000为385,000，encoder约21.7M；distilled有额外token与第二分类头，仍不可只报“22M骨干+0”。精确checkpoint统计待加载。 |
| 06 训练数据、标签与目标函数 | ImageNet-1K监督分类；distilled版使用教师类别输出、蒸馏token，不代表两版共享完全相同目标。 |
| 07 输入协议 | 单帧RGB 224×224，16×16patch；按具体checkpoint的resize/mean/std处理。 |
| 08 输出 | 类别logits、384D类token及patch tokens；不直接给高光或crop框。 |
| 09 真实时间信息 | 原生图像Transformer，无时间轴注意力。帧级token序列需另加时间模型。 |
| 10 空间定位能力 | Patch tokens含位置线索，可接前景读出；CLS全局embedding不等同于定位。 |
| 11 预训练能力 | 图像类别/对象关系的视觉表示，蒸馏版受教师归纳偏置影响；足以构成“轻量语义”假说的控制，不等于高级叙事推理。 |
| 12 计算成本与测量条件 | 本轮未核实该具体checkpoint的硬件FPS/显存；不抄其他ViT/输入分辨率数值。参数不是实时性证据。 |
| 13 部署依赖及 CPU / CUDA / NPU | 原代码PyTorch/timm；attention、Linear、LayerNorm、GELU；目标NPU的attention导出和精度未验证。 |
| 14 代码许可证 | 原项目 Apache-2.0。 |
| 15 权重许可证 | 采用具体 checkpoint 分发条件；未核得额外独立权重条款，训练数据另记。 |
| 16 官方 benchmark 与协议 | 作者表：普通DeiT-S ImageNet224 top-1 79.9%、top-5 95.0%；distilled 81.2%/95.4%。并非AIC的现有旧checkpoint结果。 |
| 17 AIC 适配方式 | 使用冻结embedding或同预算浅层微调作为ResNet对照；仓库旧DeiT结论必须按其数据、输入、head和暴露状态解释。 |
| 18 主要风险与证据边界 | 小样本微调/标签噪声、场景捷径、全局池化丢主体信息；既有失败不能推广到所有DeiT训练合同。 |
| 19 本轮核验 / 下一验收 | 正式论文页、作者源码/许可、两个权重入口和基准表已读；权重未下载，精确全链计数/运行/NPU待核验。 |

### T08 — DINOv2 ViT-S/14（无register版）

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | dinov2_vits14；本条无register版。DINOv2-S与DINOv2-B、DINOv3不能混用。 |
| 02 论文、年份与正式来源 | Oquab等，DINOv2: Learning Robust Visual Features without Supervision，2023预印本；[原文][DINO-paper]。 |
| 03 官方代码 / 模型地址 | [facebookresearch/dinov2][DINO-repo]及[官方Model Card][DINO-card]。 |
| 04 权重取得状态 | [官方vits14权重][DINO-weights]直链已确认；本轮没有下载、hash或载入验证。 |
| 05 编码器 / 头 / 整体参数 | 官方模型卡约21M视觉编码器；预训练projection/分类probe是否保留取决于checkpoint用途。AIC encoder外加实际训练的小头与空间模型，不能以线性头参数替代总量。精确部署计数待实例核验。 |
| 06 训练数据、标签与目标函数 | LVD-142M图像自监督；小模型由大教师蒸馏。无AIC高光/人工crop训练监督。 |
| 07 输入协议 | 单图，patch14，ViT-S token维度384；分辨率和位置插值配置应固定。 |
| 08 输出 | 全局embedding和patch tokens，适合前景池化/语义相似性；无原生时刻分数或crop框。 |
| 09 真实时间信息 | 图像模型，没有真实跨帧交互；时间差分来自外部算法。 |
| 10 空间定位能力 | Patch特征可支持稠密对应和前景读出，仍需额外验证与定位/裁剪头。 |
| 11 预训练能力 | 通过图像级自蒸馏和patch级目标学习通用视觉结构；“没有文本”不等于“没有语义”。 |
| 12 计算成本与测量条件 | 本轮未测FLOPs/FPS/显存；分辨率增大会改变token数和attention成本，不能只凭约21M推测比CNN快。 |
| 13 部署依赖及 CPU / CUDA / NPU | PyTorch；官方部分高效路径使用xFormers/特定GPU依赖，适配时需确认可替代attention路径；本轮未测NPU。 |
| 14 代码许可证 | 官方项目代码 Apache-2.0；模型许可另见下一项。 |
| 15 权重许可证 | 官方 Model Card 明确模型 Apache-2.0；不据此声称数据及其他权利均被覆盖。 |
| 16 官方 benchmark 与协议 | 作者README无register ViT-S/14：ImageNet k-NN79.0%，linear probe81.1%。是冻结表示读出，不是端到端高光、也不是crop IoU。 |
| 17 AIC 适配方式 | 冻结少量已有/新采样源的局部语义特征，比较全局池化与前景池化及段内变化；先证明在同预算/同源split有增量，再考虑微调或蒸馏。 |
| 18 主要风险与证据边界 | 输入裁剪改变主体可见性，attention背景偏置；新特征抽取成本、视频域偏移与缓存合同；大教师预训练能力不能自动归因于部署时参数。 |
| 19 本轮核验 / 下一验收 | 原文、官方model card/README和权重入口已核实；没有加载、性能或NPU实测；精确实例参数待核验。 |

### T09 — VideoMAE-B

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | 原始VideoMAE，ViT-B patch16，16帧；登记K400自监督预训练/动作微调版本，区别VideoMAE V2和更大H/L。 |
| 02 论文、年份与正式来源 | Tong等，VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training，NeurIPS 2022；[论文][VMAE-paper]。 |
| 03 官方代码 / 模型地址 | [MCG-NJU/VideoMAE][VMAE-repo]及[官方Model Zoo][VMAE-zoo]。 |
| 04 权重取得状态 | Model Zoo列[预训练][VMAE-pre]及[微调][VMAE-ft]Google Drive入口；本轮打开仅见Sign in，未得到文件。不得记作“已下载可用”。 |
| 05 编码器 / 头 / 整体参数 | 论文ViT-B约87M；官方fine-tune默认结构静态推导，encoder86,227,200，K400头307,600，完整86,534,800。条件：固定正弦位置、无LayerScale、tubelet2；预训练重建decoder不用于这条下游推理。AIC还须加空间链。 |
| 06 训练数据、标签与目标函数 | K400/SSV2等视频；自监督以90%–95%tube mask重建像素，随后动作分类微调。不同数据/预训练与微调checkpoint需分开。 |
| 07 输入协议 | 短RGB视频16帧、224²；tubelet2×16×16。论文K400 temporal stride4，SSV2 stride2；不能随意用1fps稀疏片段却沿用动作基准解释。 |
| 08 输出 | 时空tokens/clip embedding/动作logits；无原生高光时刻或crop框。 |
| 09 真实时间信息 | Joint space-time attention利用有序时间输入与位置，是真时序模型；需ordered/repeat/shuffle或同帧控制确认本任务利用了运动。 |
| 10 空间定位能力 | 有时空patch token，可为位置读出提供证据；池化表示与现成主体框不同。 |
| 11 预训练能力 | 从残缺视频恢复结构和外观，动作微调进一步学习短时动作判别；不直接建模长篇故事高光。 |
| 12 计算成本与测量条件 | 论文B为180 GFLOPs/16帧224单view；K400测试为180×5clips×3crops。训练mask节省不能套到全token推理；FPS/显存本轮未测。 |
| 13 部署依赖及 CPU / CUDA / NPU | PyTorch/timm，Conv3d tubelet、attention、LayerNorm；训练工具如DeepSpeed不等于推理必需。NPU导出、精度和长token显存未验证。 |
| 14 代码许可证 | 原项目主体 CC BY-NC 4.0，部分依赖另有 Apache/MIT 条款；不是整个项目 MIT。 |
| 15 权重许可证 | 按原项目的 CC BY-NC 4.0 与具体分发条件核对；未取得另外的商业授权，竞赛/再分发范围须明确。 |
| 16 官方 benchmark 与协议 | K400 ViT-B，1600epoch预训练，16×5×3：top-1 81.5%、top-5 95.1%；SSV2 B，16×2×3：70.8%/92.4%。摘要87.4%属于ViT-H更大设置，不能填到B。 |
| 17 AIC 适配方式 | 先审仓库已有ord/rep缓存和原始采样合同，作为短时序教师/冻结读出；全链100M下是否可部署需与空间链重新加总。蒸馏是待验证设计。 |
| 18 主要风险与证据边界 | 权重获取门槛、非商用许可、全链预算、密集抽帧成本；稀疏抽帧或跨镜头窗口破坏训练域；动作准确率不预测AIC增量。 |
| 19 本轮核验 / 下一验收 | 论文/Model Zoo/许可证/构造源码已读；参数静态；权重未取到文件，未运行或测NPU。 |

### T10 — HL-CLIP（2024高光研究）

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | HL-CLIP，论文微调CLIP ViT-B/32最后两层；区分原模型与HL-CLIP+SP。官方repo仍有未清理实验组件。 |
| 02 论文、年份与正式来源 | Han等，Unleash the Potential of CLIP for Video Highlight Detection，CVPR Workshops 2024；[正式论文页][HL-paper]、[全文][HL-full]。 |
| 03 官方代码 / 模型地址 | [dhk1349/HL-CLIP][HL-repo]；核心CLIPFT位于moment_detr/model.py。 |
| 04 权重取得状态 | CLIP基座有官方入口；所审HL-CLIP README未找到可直接取得的高光微调checkpoint，示例引用作者本地路径。不能把能下载CLIP基座记作已得到HL-CLIP权重。 |
| 05 编码器 / 头 / 整体参数 | 视觉87,849,216、文本63,428,096；CLIP含scale共151,277,313。当前CLIPFT类另注册未调用的MLP/Mixture286,965，总151,564,278。只训练最后两层不会降低全推理参数；纸面余弦头/SP无学习参数。 |
| 06 训练数据、标签与目标函数 | QVHighlights的query相关高光等级，以MSE拟合归一化saliency、微调图文encoder末层。论文将未标片段设Very Bad；这一处理不能复制到PHD2的未知未选帧。 |
| 07 输入协议 | RGB帧+文本query，CLIP ViT-B/32预处理；官方类把时间长度固定展开75，变长序列与padding需重新审计。 |
| 08 输出 | 图像与query embedding的余弦相关分数；不输出crop框；不是无query的generic highlight现成方案。 |
| 09 真实时间信息 | 原始CLIP逐帧编码；SP为相邻帧saliency平均，属于时间分数平滑，无学习的跨帧动作编码。 |
| 10 空间定位能力 | 视觉encoder有patch结构，但论文输出全局query匹配，无显式主体定位/裁剪。 |
| 11 预训练能力 | CLIP先以4亿图文对对比学习视觉语言相关性，HL微调使其符合query高光标签；不是从原始像素学到独立的动作理解模块。 |
| 12 计算成本与测量条件 | 本轮未核实可复用的官方FPS/显存/FLOPs；多帧视觉encoder是主要成本，文本可缓存时也须确认申报规则。参数冻结/低可训练参数不等于低运行成本。 |
| 13 部署依赖及 CPU / CUDA / NPU | PyTorch/OpenAI CLIP；官方构造硬编码device=cuda、float32，原项目继承Moment-DETR环境和作者本地数据路径；NPU移植未测。 |
| 14 代码许可证 | HL-CLIP 代码 MIT；CLIP 基座代码按其官方许可。 |
| 15 权重许可证 | CLIP 基座按官方代码/权重分发许可；未取得 HL-CLIP 高光微调权重，其独立许可未知；QVHighlights 数据条款另记。 |
| 16 官方 benchmark 与协议 | QVHighlights val，表2四seed：HL-CLIP2 mAP40.00±0.06、HIT68.99±0.28；+SP42.40±0.05、72.42±0.25。test表1另为三seed均值41.94/70.60；不混成同一实验。 |
| 17 AIC 适配方式 | 借鉴“冻结特征不足时监督适配末层”和“局部平滑需独立验证”；无query部署要另定义任务，不能悄悄给每视频补人工query。可作训练期语义参照，不能直接宣布100M路线。 |
| 18 主要风险与证据边界 | query条件与AIC generic任务不同；未标记负例政策不兼容；SP可能抹平精确帧边界；完整图文链超100M，且高光checkpoint获取未证实。 |
| 19 本轮核验 / 下一验收 | 正式论文/全文、README、CLIPFT实际forward、原CLIP结构与配置已审；有静态参数算术，无权重下载/运行/NPU。 |

## 6. 运动算法与模型详细登记

### M01：帧差 + 全局运动补偿

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | 确定性组合基线：灰度绝对差分；另一路先估计全局相似/仿射变换，再对齐差分。ECC 与 LK+RANSAC 两种变换估计应视为不同配置。 |
| 02 论文、年份与正式来源 | 差分本身是算子；全局配准可依据 Evangelidis & Psarakis，*Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization*，TPAMI 2008，[DOI/IEEE](https://ieeexplore.ieee.org/document/4515873/)。本文的 AIC 组合与读出不是该论文的现成模型。 |
| 03 官方代码 / 模型地址 | [OpenCV 4.13.0 video tracking 文档](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html)：`findTransformECC`；[calib3d 文档](https://docs.opencv.org/4.13.0/d9/d0c/group__calib3d.html)：`estimateAffinePartial2D`、`findHomography`。 |
| 04 权重取得状态 | 不需要离线训练权重；每对视频帧估计的变换矩阵应随特征缓存保存。 |
| 05 编码器 / 头 / 整体参数 | 编码器 0，运动读出 0，运动模块 0；每对输入拟合的变换自由度与窗口、迭代次数、稳健阈值均需记录，但不是预训练网络参数。若学习高光线性头，再加其全部系数。 |
| 06 训练数据、标签与目标函数 | 无训练集与学习标签。ECC 配准优化图像相关性；RANSAC 版本优化稳健几何一致性。任何阈值选择只能用开发侧数据。 |
| 07 输入协议 | 同一 shot 内两帧灰度图，真实 PTS 间隔、原分辨率和缩放映射；允许有效像素掩码。 |
| 08 输出 | 全局变换、配准可靠度、有效域，以及原始/对齐绝对亮度差与其统计量。 |
| 09 真实时间信息 | 两个时间点的外观变化。单次差分没有方向、对应轨迹或事件完成状态。 |
| 10 空间定位能力 | 像素变化位置与全局几何；不是“主体 mask”，也不是深度或真实三维运动。 |
| 11 预训练能力 | 无。 |
| 12 计算成本与测量条件 | 本组合没有原论文的统一性能数；本次未测 FLOPs/内存/延迟。须连同解码、缩放、配准、warp、统计一起计时。 |
| 13 部署依赖及 CPU / CUDA / NPU | OpenCV CPU 即可做参考；对齐变换约定、有效边界、颜色空间、初始化与失败处理必须固定。 |
| 14 代码许可证 | 所选 OpenCV 4.13.0 主项目 Apache-2.0；第三方组件依通知。[官方许可](https://opencv.org/license/)。 |
| 15 权重许可证 | N/A，本算法不需要离线学习权重。 |
| 16 官方 benchmark 与协议 | ECC 论文评估图像配准；没有本组合的高光、AIC 或现代端侧 benchmark。本表不把配准结果换算成高光效果。 |
| 17 AIC 适配方式 | 与 R89 原始差分并列，输入同一原始帧轴；先冻结空间框与每视频 K，只测试残余运动对时间读出的增量。 |
| 18 主要风险与证据边界 | 视差、滚动快门、大主体占满画面、切镜、闪光可能破坏单一全局变换。配准失败应标记缺测，不能编码为“运动为零”。 |
| 19 本轮核验 / 下一验收 | 首轮必做诊断；先人工复查静止、摇摄、主体运动、切镜各类片段与有效域，再做来源隔离的配对增量试验。 本轮未运行本算法的目标视频基准。 |

### M02：Farnebäck 稠密光流

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | Farnebäck 两帧多项式展开光流；部署候选固定为 OpenCV 4.13.0 `calcOpticalFlowFarneback`。 |
| 02 论文、年份与正式来源 | Gunnar Farnebäck，*Two-Frame Motion Estimation Based on Polynomial Expansion*，SCIA 2003，pp. 363–370，[出版社原文](https://link.springer.com/content/pdf/10.1007/3-540-45103-X_50.pdf)。 |
| 03 官方代码 / 模型地址 | [OpenCV 官方 API](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html)；[实现仓库](https://github.com/opencv/opencv)。作者当年的 Matlab/C 实现与现代 OpenCV 移植应区分。 |
| 04 权重取得状态 | 无训练权重、无需下载。 |
| 05 编码器 / 头 / 整体参数 | 均为 0 个离线训练参数；局部多项式系数是每帧估计量。`pyr_scale/levels/winsize/iterations/poly_n/poly_sigma/flags` 是需冻结的算法超参数。 |
| 06 训练数据、标签与目标函数 | 无训练数据；利用局部二次多项式近似与位移关系、邻域加权和多尺度迭代估计稠密位移。 |
| 07 输入协议 | 连续或短间隔两张同尺寸灰度图；OpenCV API 输入格式需统一。 |
| 08 输出 | H×W×2 浮点位移场，单位为输入图像像素/帧对。 |
| 09 真实时间信息 | 两帧对应位移；跨多个帧对聚合才形成轨迹或运动阶段特征。 |
| 10 空间定位能力 | 稠密局部位移；没有预定义的人、物类别或主体标签。 |
| 11 预训练能力 | 无。 |
| 12 计算成本与测量条件 | 原论文没有当前硬件下的 FLOPs、显存或统一 FPS；本次不借用其他论文测得的 Farnebäck 速度当原论文结果。成本随分辨率、尺度和迭代变。 |
| 13 部署依赖及 CPU / CUDA / NPU | OpenCV CPU；选定构建版本、线程数和向量化环境。无需 PyTorch 或学习型加速器。 |
| 14 代码许可证 | 所选 OpenCV 4.13.0 Apache-2.0；保留适用文件及第三方通知。[官方许可](https://opencv.org/license/)。 |
| 15 权重许可证 | N/A，本算法不需要离线学习权重。 |
| 16 官方 benchmark 与协议 | 原文使用 Yosemite 等序列与角度误差，并区分模型假设/是否计入天空；不可与 Sintel EPE 直接比较。未登记现代硬件性能。 |
| 17 AIC 适配方式 | 给原始差分提供低依赖稠密运动参考；与全局补偿组合后产生残余流强度、方向与空间集中度。 |
| 18 主要风险与证据边界 | 纹理不足、遮挡、光照变化、大位移与细小快速主体会影响结果；平滑窗口可能抹掉主体边缘。AIC 缓存和 910A 成本未测。 |
| 19 本轮核验 / 下一验收 | 保留为可复现参考；优先与 DIS 在同一小批视频、同一缩放、同一时间间隔上核验质量和完整耗时。 本轮未运行本算法的目标视频基准。 |

### M03：DIS Optical Flow

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | Dense Inverse Search optical flow；优先试 OpenCV 4.13.0 FAST，MEDIUM 作精度对照，参数显式保存。 |
| 02 论文、年份与正式来源 | Kroeger, Timofte, Dai & Van Gool，*Fast Optical Flow using Dense Inverse Search*，ECCV 2016，[原论文含补充材料](https://arxiv.org/pdf/1603.03590)。 |
| 03 官方代码 / 模型地址 | [作者 OF_DIS](https://github.com/tikroeger/OF_DIS)；[OpenCV DISOpticalFlow API](https://docs.opencv.org/4.13.0/de/d4f/classcv_1_1DISOpticalFlow.html)。两者不是相同二进制与默认配置。 |
| 04 权重取得状态 | 无学习权重。 |
| 05 编码器 / 头 / 整体参数 | 0/0/0 个离线训练参数；patch 尺寸、步长、金字塔、反向搜索迭代、传播及变分细化参数仍需登记。 |
| 06 训练数据、标签与目标函数 | 无训练数据；逆组合 patch 对齐、稠密聚合与可选变分细化。 |
| 07 输入协议 | 两张同尺寸 8-bit 单通道图；可提供初始流。 |
| 08 输出 | H×W×2 稠密位移；OpenCV 提供空间传播等控制项。 |
| 09 真实时间信息 | 默认是一个帧对；使用上一对的初始流才额外引入历史信息，必须在切镜处重置。 |
| 10 空间定位能力 | 局部匹配、传播、平滑后的稠密流，无主体语义。 |
| 11 预训练能力 | 无。 |
| 12 计算成本与测量条件 | 原文补充 Table 5/9：i7-3770K 3.50GHz，Sintel 1024×436，四种工作点分别 1.65/3.32/97.8/1925 ms；最快点峰值内存 35.52 MB。论文基准排除磁盘、缩放、梯度等预处理。不是完整视频管线速度，也不是 OpenCV FAST 的承诺。 |
| 13 部署依赖及 CPU / CUDA / NPU | OpenCV CPU 移植便于首轮使用。作者原版依赖 C++/Eigen/OpenCV/x86 SSE，不能假定在 ARM 环境无修改可运行。 |
| 14 代码许可证 | 作者 OF_DIS 标 GPL-3.0，同时 README 限 scientific/personal use 并要求商业用途联系，文本差异需澄清。所选 OpenCV 移植按 Apache-2.0 及具体文件通知，不能混用两套许可。 |
| 15 权重许可证 | N/A，本算法不需要离线学习权重。 |
| 16 官方 benchmark 与协议 | Sintel、KITTI 流误差/速度权衡。快慢工作点精度不同，不能把最高速度与最高精度拼成一个配置；不登记跨论文重测数。 |
| 17 AIC 适配方式 | 建议首轮主运动通道：短间隔 DIS 流 → 稳健全局补偿 → 低维残余流特征 → 固定的小读出。 |
| 18 主要风险与证据边界 | 快速配置容易漏小主体/大位移，跨帧初始化可能传播错误；OpenCV 有额外传播行为。端侧线程、热启动和解码占比未测。 |
| 19 本轮核验 / 下一验收 | 第一优先级；先比较 FAST/MEDIUM 的轨迹合理性和含解码成本，再选一个配置冻结，避免公开验证集上反复搜索。 本轮未运行本算法的目标视频基准。 |

### M04：Dual TV-L1 光流

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | Zach–Pock–Bischof TV-L1；部署候选 OpenCV contrib `DualTVL1OpticalFlow`。 |
| 02 论文、年份与正式来源 | Zach, Pock & Bischof，*A Duality Based Approach for Realtime TV-L1 Optical Flow*，DAGM 2007，[原论文](https://www.cvlibs.net/projects/autonomous_vision_survey/literature/Zach2007DAGM.pdf)，[出版社](https://link.springer.com/chapter/10.1007/978-3-540-74936-3_22)。 |
| 03 官方代码 / 模型地址 | [OpenCV 4.13.0 contrib API](https://docs.opencv.org/4.13.0/dc/d4d/classcv_1_1optflow_1_1DualTVL1OpticalFlow.html)，[实现仓库](https://github.com/opencv/opencv_contrib)。原作者 GPU 实现与 OpenCV 移植不同。 |
| 04 权重取得状态 | 无学习权重。 |
| 05 编码器 / 头 / 整体参数 | 0/0/0 个离线训练参数。`tau/lambda/theta/nscales/warps/epsilon/iterations` 等是超参数；求解的流场不是预训练参数。 |
| 06 训练数据、标签与目标函数 | 无训练数据；L1 亮度数据项加总变分正则，借助对偶优化、多尺度与 warping 求解。 |
| 07 输入协议 | 同尺寸灰度帧对；数值范围按选定 API 固定。 |
| 08 输出 | 稠密二维光流。 |
| 09 真实时间信息 | 一个帧对的运动对应，不包含长程事件语义。 |
| 10 空间定位能力 | 边缘保持式正则化下的空间位移；不能由此保证物体边界正确。 |
| 11 预训练能力 | 无。 |
| 12 计算成本与测量条件 | 原文 Table 1：GeForce Go 7900 GTX、256×256、每层 50 次迭代为 34.1 FPS；原实现为 OpenGL/Cg，计时含纹理上传与最终显示。不能套用于现代 OpenCV CPU、CUDA 或 NPU。现代成本未测。 |
| 13 部署依赖及 CPU / CUDA / NPU | OpenCV contrib 的 CPU optflow 模块；CUDA 版本若另选应单独登记编译/硬件。无需神经网络权重。 |
| 14 代码许可证 | 按 OpenCV/contrib 的具体 tag 与文件头；主项目4.5+ Apache-2.0，本轮未逐一核完 contrib 依赖。[官方说明](https://opencv.org/license/)。 |
| 15 权重许可证 | N/A，本算法不需要离线学习权重。 |
| 16 官方 benchmark 与协议 | Yosemite 帧 8/9：50 次迭代、天空除外的平均角度误差 2.85°；计入天空为 5.06°。评价协议与现代 EPE 不同。 |
| 17 AIC 适配方式 | 当 DIS 残余运动不稳定时，作为另一种正则化假设的对照；不必同时作为多路大 bank 常驻。 |
| 18 主要风险与证据边界 | 亮度变化、遮挡、纹理不足、大位移仍会出错；多尺度局部线性化也可能错过小而快主体。迭代成本与质量需要本机验证。 |
| 19 本轮核验 / 下一验收 | 二级参考。只在 DIS/RAFT 结果提示流质量可能是瓶颈时增加，避免先堆模型再解释效果。 本轮未运行本算法的目标视频基准。 |

### M05：Lucas–Kanade / 金字塔 LK

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | OpenCV 4.13.0 金字塔稀疏 LK：`calcOpticalFlowPyrLK`；与 1981 原始局部配准方法区分。 |
| 02 论文、年份与正式来源 | Lucas & Kanade，*An Iterative Image Registration Technique with an Application to Stereo Vision*，IJCAI 1981，pp. 674–679，[原文](https://www.ijcai.org/Proceedings/81-2/Papers/017.pdf)。 |
| 03 官方代码 / 模型地址 | [OpenCV tracking API](https://docs.opencv.org/4.13.0/dc/d6b/group__video__track.html)，[OpenCV 仓库](https://github.com/opencv/opencv)。 |
| 04 权重取得状态 | 无训练权重。角点检测方法与补点策略另记。 |
| 05 编码器 / 头 / 整体参数 | 0/0/0 个离线训练参数；点数、窗口、层数、终止准则、最小特征值阈值是超参数。 |
| 06 训练数据、标签与目标函数 | 无训练数据；利用图像梯度与局部最小二乘迭代拟合配准位移。 |
| 07 输入协议 | 两帧及第一帧待跟踪点，可选初始位移；足够纹理点更适合局部估计。 |
| 08 输出 | 对应点坐标、跟踪状态及误差量；跨帧关联后可构造短轨迹。 |
| 09 真实时间信息 | 默认两帧对应；持续维护同一批点才形成多帧轨迹，丢失后重初始化应有明确策略。 |
| 10 空间定位能力 | 稀疏点位置/运动，不能直接代表完整前景区域或对象 ID。 |
| 11 预训练能力 | 无。 |
| 12 计算成本与测量条件 | 1981 原文不是现代金字塔 API 的端侧 benchmark；当前 FLOPs、内存、速度未测，主要受点数、窗口、尺度、迭代影响。 |
| 13 部署依赖及 CPU / CUDA / NPU | OpenCV CPU。可接 `estimateAffinePartial2D` + RANSAC 估全局运动；需保存内点比例及点的空间覆盖度。 |
| 14 代码许可证 | 所选 OpenCV 4.13.0 Apache-2.0 与适用通知。[官方许可](https://opencv.org/license/)。 |
| 15 权重许可证 | N/A，本算法不需要离线学习权重。 |
| 16 官方 benchmark 与协议 | 原文是配准与立体视觉验证，没有可直接登记为当前 OpenCV 高光/稠密光流性能的数值。 |
| 17 AIC 适配方式 | 为低成本相机运动估计提供背景点轨迹；对少量主体轨迹作定性诊断，而不是替换全部稠密流。 |
| 18 主要风险与证据边界 | 孔径问题、弱纹理、遮挡、尺度变化、点聚集在运动主体上会导致错估；拟合出的“主运动”不总等于相机运动。 |
| 19 本轮核验 / 下一验收 | 与 M01 配合，优先验收几何质量。需要稠密主体覆盖时使用 M03/M06 对照。 本轮未运行本算法的目标视频基准。 |

### M06：RAFT-small

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | 作者正式发布的 RAFT small 架构；运行时迭代数必须作为配置固定。 |
| 02 论文、年份与正式来源 | Teed & Deng，*RAFT: Recurrent All-Pairs Field Transforms for Optical Flow*，ECCV 2020，[论文及补充](https://arxiv.org/pdf/2003.12039)。 |
| 03 官方代码 / 模型地址 | [princeton-vl/RAFT](https://github.com/princeton-vl/RAFT)，有 `--small` 配置，不能用 full 模型计数替代。 |
| 04 权重取得状态 | 官方 README/`download_models.sh` 提供模型包入口；本次核实入口，未下载解压、核 small 权重 SHA 或严格加载。应在启用前固定实际 small checkpoint 与训练阶段。 |
| 05 编码器 / 头 / 整体参数 | 原论文补充 Table 4：small 整网约 **1.0M**；编码器/更新头精确分项未报告。本次没有加载计数，不使用 full 的 5.3M 或 full 更新器的 2.7M 作为 small 分项。 |
| 06 训练数据、标签与目标函数 | 光流真值监督；论文训练链涉及 FlyingChairs→FlyingThings3D，并对目标域使用 Sintel/KITTI/HD1K 等阶段。具体 small 权重的训练数据组合须与实际 checkpoint 对应，不能把整个项目所有数据都默认归给该文件。 |
| 07 输入协议 | 两张 RGB 图；保存归一化、padding、尺寸与帧对间隔。 |
| 08 输出 | 与图像对应的稠密二维流，可返回各次更新结果。 |
| 09 真实时间信息 | 两帧对应；GRU 的多次更新是同一个流场的优化展开，并非已学会跨完整视频的事件完成时序。 |
| 10 空间定位能力 | 像素特征、全局相关体和局部查询；没有直接提供主体实例 mask。 |
| 11 预训练能力 | 有监督光流估计，不是高光、动作完成度或审美构图。 |
| 12 计算成本与测量条件 | 补充 Table 4：GTX 1080Ti、10 次更新，small 0.05 秒/帧对。相关体的内存随空间位置组合增长；可选替代相关实现可改变成本。未测当前 910A 延迟、FLOPs 或峰值内存。 |
| 13 部署依赖及 CPU / CUDA / NPU | 作者环境历史测试 PyTorch 1.6/CUDA 10.1；普通 PyTorch 路径与可选 `alt_cuda_corr` 自定义 CUDA 路径应区分。NPU 算子支持、导出和数值一致性未验证。 |
| 14 代码许可证 | 作者仓库代码 BSD-3-Clause。 |
| 15 权重许可证 | 官方权重入口/文件存在；独立权重许可及训练数据权利未进一步核清，不能只凭代码许可认定全部授权。 |
| 16 官方 benchmark 与协议 | 同一补充表 small：Sintel **train final** EPE **3.37**，10 次更新；不是论文 full 模型的 Sintel test 成绩。 |
| 17 AIC 适配方式 | 作为冻结的较高质量光流控制组：若正确时轴/相机补偿后 DIS 无增量，可用它判断“流不准”是否解释阴性结果。 |
| 18 主要风险与证据边界 | 合成训练到真实短视频存在域差；遮挡、运动模糊与超大位移仍困难。精确参数、权重 bytes、strict load、NPU 成本均待验。 |
| 19 本轮核验 / 下一验收 | 第一批学习型对照；先做小批质量与完整耗时验收，不先训练新的高光大网络。 本轮未下载或运行学习型运动权重，NPU未测。 |

### M07：GMFlow 基础单尺度模型

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | GMFlow，单尺度 1/8 分辨率、无细尺度 refinement 的基础版本。**未核实存在官方命名的 GMFlow-small**。 |
| 02 论文、年份与正式来源 | Xu et al.，*GMFlow: Learning Optical Flow via Global Matching*，CVPR 2022，[原论文](https://arxiv.org/pdf/2111.13680)。 |
| 03 官方代码 / 模型地址 | [haofeixu/gmflow](https://github.com/haofeixu/gmflow)；官方基础运行与 `num_scales=1` 配置可查。不要把后续 UniMatch 的规模/速度混入本条。 |
| 04 权重取得状态 | 官方有预训练入口，README 使用 `gmflow_things-e9887eda.pth` 等权重名；本次未下载核 SHA 或严格加载。运行配置和权重必须成对冻结。 |
| 05 编码器 / 头 / 整体参数 | 原文 Table 3 无 refinement 整网约 **4.7M**，完整编码器/Transformer/传播模块分项未报告。共享参数的 1 次 refinement 模型也约 4.7M，但计算量明显不同。 |
| 06 训练数据、标签与目标函数 | Table 3 训练于 FlyingChairs 与 FlyingThings3D，使用有监督光流损失；另有目标域 finetune 权重，不能与 Things 泛化结果互换。 |
| 07 输入协议 | 两帧 RGB；官方缩放/padding 与注意力配置需固定。 |
| 08 输出 | 稠密光流；全局相关矩阵复用支持双向估计及一致性检查。 |
| 09 真实时间信息 | 两帧跨图像匹配；单次前向不是长视频事件建模。 |
| 10 空间定位能力 | CNN 特征、Transformer 特征增强、全局 softmax 匹配与流传播，不直接输出主体类别或 mask。 |
| 11 预训练能力 | 有监督光流估计。 |
| 12 计算成本与测量条件 | 原文 Table 3，436×1024：无 refinement 为 V100 **57 ms**、A100 **26 ms**；1 次 refinement 为 151/66 ms。未测本地/NPU成本；全局匹配矩阵仍可能有明显内存开销。 |
| 13 部署依赖及 CPU / CUDA / NPU | PyTorch 及 attention、矩阵乘法、采样/插值等操作；当前目标环境与模型导出未验证。 |
| 14 代码许可证 | 作者仓库代码 Apache-2.0。 |
| 15 权重许可证 | 官方权重入口/文件存在；独立权重许可及训练数据权利未进一步核清，不能只凭代码许可认定全部授权。 |
| 16 官方 benchmark 与协议 | 同 Table 3 无 refinement、Chairs+Things 训练，Sintel train clean/final EPE **1.50/2.96**；不是目标域 test 成绩。 |
| 17 AIC 适配方式 | 仅当需要检验全局匹配/大位移质量时加入。用同一帧对与同一运动统计读出，避免把模型与头同时更换。 |
| 18 主要风险与证据边界 | 参数少不等于吞吐高；全局匹配容易受重复纹理、遮挡、场景切换影响。正式 small 型号、当前权重精确参数均未证实。 |
| 19 本轮核验 / 下一验收 | 可登记的轻量基础配置，优先级低于先完成 DIS/RAFT-small 的因果对照；不可写成“GMFlow-small 已验证”。 本轮未下载或运行学习型运动权重，NPU未测。 |

### M08：NeuFlow-V2

| 字段 | 审计结果 |
|---|---|
| 01 正式名称与版本 | NeuFlow-V2；论文默认 1 次 1/16 尺度更新 + 8 次 1/8 尺度更新，共 9 次。 |
| 02 论文、年份与正式来源 | *NeuFlow v2: High-Efficiency Optical Flow Estimation on Edge Devices*，[arXiv 2408.10161v2，2024](https://arxiv.org/html/2408.10161v2)；后续 [v3 原文](https://arxiv.org/html/2408.10161v3) 标题为 *NeuFlow-V2: Push High-Efficiency Optical Flow To the Limit*。以下数值逐项标版本。 |
| 03 官方代码 / 模型地址 | [neufieldrobotics/NeuFlow_v2](https://github.com/neufieldrobotics/NeuFlow_v2)。 |
| 04 权重取得状态 | 作者仓库列有 `neuflow_things.pth`、`neuflow_sintel.pth`、`neuflow_mixed.pth`，因此文件存在性比仅发布承诺更强；本次未下载加载或核 SHA。 |
| 05 编码器 / 头 / 整体参数 | **v2 结论段称整网约 9M**；分项未报告。v3 页面没有本次可核的精确计数，不能断言当前所有 checkpoint 都恰为 9,000,000 参数。 |
| 06 训练数据、标签与目标函数 | v3 Table I/实验段的对应泛化实验使用 FlyingThings 光流真值监督；仓库另列 Chairs/Things/Sintel/KITTI/HD1K 的训练流程。具体 mixed 权重的来源组合仍须绑定配置核验。 |
| 07 输入协议 | RGB 帧对；推理缩放、padding、精度及迭代数需固定。 |
| 08 输出 | 稠密光流，粗尺度全局匹配后逐步细化。 |
| 09 真实时间信息 | 两帧对应；RNN 更新是流估计内部过程，不等于完整视频高光时序编码器。 |
| 10 空间定位能力 | 多尺度卷积特征、跨图注意力和位移传播；没有直接的前景语义监督输出。 |
| 11 预训练能力 | 有监督光流估计。 |
| 12 计算成本与测量条件 | v3 Table I：半精度、1024×436、9 次更新，RTX 2080 **0.015 s**、Jetson Orin Nano **0.106 s**。摘要的 >20 FPS 对应较小的 512×384，不能与该行混用。表注硬件/批量说明有文字不一致，本表不延伸其批量内存结论。 |
| 13 部署依赖及 CPU / CUDA / NPU | 官方要求 PyTorch ≥2.0，示例 2.0.1/torchvision 0.15.2/CUDA 11.7；卷积、相关、采样、循环更新须在目标 NPU 上单独核验。Jetson 成绩不能证明 910A 可运行。 |
| 14 代码许可证 | 作者仓库标 Apache-2.0。 |
| 15 权重许可证 | 官方权重入口/文件存在；独立权重许可及训练数据权利未进一步核清，不能只凭代码许可认定全部授权。 |
| 16 官方 benchmark 与协议 | v3 同表：Sintel train clean/final EPE **1.24/2.67**；KITTI train EPE **4.33**、Fl-all **15.3%**。对应 Things 泛化实验，不是 AIC 收益或官方 test 排名。 |
| 17 AIC 适配方式 | 近期边缘推理备选；可在 DIS 与 RAFT-small 后，以固定运动统计检验流精度是否继续改善事件阶段判断。 |
| 18 主要风险与证据边界 | 参数未必最少；v2 明确提到相关计算内存压力。论文版本、权重训练阶段、当前精确参数与目标硬件耗时尚未完成一致性验证。 |
| 19 本轮核验 / 下一验收 | 保留候选，暂不优先替换基线。先取得固定权重 SHA、strict load/全参数计数，再决定是否值得投入适配。 本轮未下载或运行学习型运动权重，NPU未测。 |

## 7. 已实际下载的权重凭证

下列 `evidence/` 路径位于完整证据包 `aic_visual_research_bundle.zip` 的解压目录。

这些文件仅作为临时取证，不随最终研究交付分发二进制。均从作者或官方维护者发布位置读取；未执行模型。完整 URL、字节数、Content-Type 与 SHA256 保存在 `evidence/spatial_sources/download_manifest.json`。

| 权重 | 实际字节数 | SHA256 |
|---|---:|---|
| U²-Netp | 4,683,258 | e7567cde013fb64813973ce6e1ecc25a80c05c3ca7adbc5a54f3c3d90991b854 |
| UNISAL | 15,368,928 | 4a9157411f1741d588b15670d15295e998805648b8b6348599fe447298338481 |
| TinyHD-M rc4 | 2,481,689 | 53e8a306d3e260f6c236e99694bec5852ee208b21d816d93a2918c1828771958 |
| YuNet 2023mar | 232,589 | 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 |
| RTMDet-tiny | 57,532,893 | 78e30dcce0c6f594eaff0d6977b84b4103688b4aff0ad1aa16008a8cc854a7fb |

### 7.1 计数证据及复核范围

- `evidence/spatial_sources/static_parameter_counts.json`：基于官方模块构造器的形状算术，U²-Net/p、BASNet、ISNetDIS 与训练用 GT encoder。计数 Conv 权重／bias 及 BN 可训练参数，排除 BN 均值／方差。无张量 forward。
- `evidence/spatial_sources/checkpoint_tensor_inventory.json`：显式全局白名单的 pickle 元数据读取，统计形状，排除 BN 运行缓冲。U²-Netp 与源码计数相互核对；UNISAL、TinyHD 按实际下载权重核对。不是 `torch.nn.Module.parameters()` 的运行结果；不能检验权重数值、NaN、运算正确性或推理速度。
- `evidence/spatial_sources/yunet_onnx_inventory.json`：只解析 ONNX protobuf 图、初始化器和算子；融合导出图与原训练网络不是同一参数口径。
- RTMDet checkpoint 解析遇到超出白名单的 MMEngine HistoryBuffer 后停止，参数沿用原作者公开 4.8M，而不冒充完整 checkpoint 精确计数。
- 复核脚本与说明另见 `evidence/spatial_sources/audit_spatial_assets.py`。脚本不需要下载权重进入交付包；用 manifest 中固定 URL 取到相同哈希后可复核。

### 7.2 明确排除的直接部署解释

RETVI（TVCG 2024；2023 arXiv）做自适应视频变形，不能直接输出符合 AIC 原生图像坐标的合法裁切框；LIVE-YT VC++ 与深度 reframing 模型在 2026-10-19 官方仓库仍标 Coming soon；LOPOV/FEVR 的发布、训练监督和权重均需单独核实。它们仍有研究机制价值，详见 `visual_feature_research.md`。


## 8. 时序与语义参数的静态算术核验

| 模型/条件 | 编码或特征部分 | 分类/分数部分 | 完整指定模块 |
|---|---:|---:|---:|
| TransNetV2 默认 PyTorch | 卷积+BN 2,548,864；两类相似特征 83,584；共享 FC 4,981,760 | 两个头各1,025 | 7,616,258 |
| TSM ResNet50，K400，无 non-local | 23,508,032；shift新增0 | 819,600 | 24,327,632 |
| ResNet18，ImageNet1K | 去FC编码器11,176,512 | 513,000 | 官方11,689,512 |
| MobileNetV3-Large，宽度1.0 | 移除整个classifier后2,971,952 | 两层classifier共2,511,080 | 官方5,483,032 |
| VideoMAE-B，K400，固定正弦位置、无LayerScale | 去head编码器86,227,200 | 307,600 | 86,534,800；论文约87M |
| CLIP ViT-B/32基座 | 视觉87,849,216；文本63,428,096 | logit scale 1 | 151,277,313 |
| HL-CLIP仓库CLIPFT类 | 上述完整CLIP基座 | 另注册286,965个未被所审forward调用的MLP/Mixture参数 | 注册总数151,564,278 |

上表的静态数值不等于已核验某个 checkpoint、部署图或 AIC 合规申报。特别是 HL-CLIP 原类仍包含未调用模块；若将其移除，应以实际导出产物重新计数并证明输出等价。


## 9. 覆盖范围与尚未通过的门槛

检索覆盖 arXiv、CVPR/ICCV/ECCV/NeurIPS/ICML/ACM Multimedia 原始研究，官方 GitHub 实现、Hugging Face 官方模型页、作者权重以及 benchmark 入口。Hugging Face 的 YuNet 官方维护者页见 S06；Papers With Code 只用于查找论文与任务入口，具体技术数值回到原论文/作者实现核验。2025–2026 的 MS-DETR、SVHighlights 与 LIVE-YT 研究用于机制比较，详见 `visual_feature_research.md`，没有将尚未交付的权重填成可运行候选。

目前不存在“以上28项已安装并测试”的结果。首轮只取五个优先候选中的必要通路：H1先用U²-Netp与现有空间链，H2/H3复用ResNet18、PySceneDetect和DIS/LK；UNISAL属于后续真时序空间控制。每次只增加可证伪的信息，不同时更换主干、监督、解码器与预算。

任何学习型部署还需完成：固定仓库commit、权重URL/文件名/SHA256、完整配置、严格加载及允许key差异、真实样例有限输出、坐标/时间回映、目标设备全流程成本。参数计数或权重下载不能代替这些步骤。新增头与缓存上游的完整账本见 `aic_transfer_learning_plan.md`。

## 原始来源补充索引

以下缩写链接服务于上文条目，全部指向作者、官方维护者或正式论文。文件、参数与复现证据另在随附证据包中。

[PSD-doc]: https://www.scenedetect.com/docs/latest/api/detectors.html
[PSD-repo]: https://github.com/Breakthrough/PySceneDetect
[TN-paper]: https://arxiv.org/abs/2008.04838
[TN-repo]: https://github.com/soCzech/TransNetV2
[TN-code]: https://github.com/soCzech/TransNetV2/blob/master/inference-pytorch/transnetv2_pytorch.py
[KTS-paper]: https://doi.org/10.1007/978-3-319-10599-4_35
[KTS-project]: https://lear.inrialpes.fr/people/potapov/med_summaries.html
[TSM-paper]: https://arxiv.org/abs/1811.08383
[TSM-repo]: https://github.com/mit-han-lab/temporal-shift-module
[TSM-weights]: https://hanlab18.mit.edu/projects/tsm/models/TSM_kinetics_RGB_resnet50_shift8_blockres_avg_segment8_e100_dense.pth
[RN-paper]: https://arxiv.org/abs/1512.03385
[RN-doc]: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet18.html
[RN-weights]: https://download.pytorch.org/models/resnet18-f37072fd.pth
[MN-paper]: https://arxiv.org/abs/1905.02244
[MN-doc]: https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html
[MN-weights]: https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth
[DEIT-paper]: https://proceedings.mlr.press/v139/touvron21a.html
[DEIT-repo]: https://github.com/facebookresearch/deit/blob/main/README_deit.md
[DEIT-weights]: https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth
[DEIT-distilled]: https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth
[DINO-paper]: https://arxiv.org/abs/2304.07193
[DINO-repo]: https://github.com/facebookresearch/dinov2
[DINO-card]: https://github.com/facebookresearch/dinov2/blob/main/MODEL_CARD.md
[DINO-weights]: https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth
[VMAE-paper]: https://arxiv.org/abs/2203.12602
[VMAE-repo]: https://github.com/MCG-NJU/VideoMAE
[VMAE-zoo]: https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md
[VMAE-pre]: https://drive.google.com/file/d/1tEhLyskjb755TJ65ptsrafUG2llSwQE1/view
[VMAE-ft]: https://drive.google.com/file/d/1MzwteHH-1yuMnFb8vRBQDvngV1Zl-d3z/view
[HL-paper]: https://openaccess.thecvf.com/content/CVPR2024W/ELVM/html/Han_Unleash_the_Potential_of_CLIP_for_Video_Highlight_Detection_CVPRW_2024_paper.html
[HL-full]: https://arxiv.org/html/2404.01745v1
[HL-repo]: https://github.com/dhk1349/HL-CLIP
