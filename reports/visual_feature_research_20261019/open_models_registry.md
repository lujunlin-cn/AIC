# 公开视觉模型与算法注册表

日期：2026-10-19（Asia/Shanghai）。AIC固定快照：[5251193a9e07fd682008bb7b2a49aef60ee0cf3f](https://github.com/lujunlin-cn/AIC/commit/5251193a9e07fd682008bb7b2a49aef60ee0cf3f)。

本表登记26个具体算法、模型版本或有必要单列的部署框架，每项覆盖用户要求的17字段。它服务于[单一研究主线](research_verdict.md)：先检验目标域监督下的低维相机补偿残差运动；空间mask、镜头和静态语义分别用于可识别的对照。表中论文benchmark不能换算AIC分数，40分仍是未验证目标。

## 1. 证据和参数口径

**本轮没有任何新增W3模型。** 实际完成的是作者代码/论文审计、部分源码参数算术、权重端点HEAD或小Range访问、safetensors张量header解析，以及独立的Farnebäck合成输入实验。没有完整下载并load新权重，没有对这些神经模型运行CPU/CUDA/Ascend前向或显存/FPS测试，也没有新比赛包。[本轮环境](evidence/current_environment_check.json)、[合成实验](evidence/aic_r52_farneback_synthetic_audit.json)。

| 等级 | 本表严格含义 | 不代表什么 |
|---|---|---|
| W0 | 本轮未发现可确认的该模型权重发布入口 | 不证明世界上不存在该权重 |
| W1 | 作者给出权重链接或仓库文件声明，未验证可取得实际实体 | Drive预览HTML、登录页、失效链接和LFS指针均不能算实体 |
| W2 | 实际HTTP端点经HEAD或小Range访问，响应与权重实体相符 | 未完整下载、未校验整包hash、未load或验证输出；端点未来仍可能改变 |
| W3 | 完整下载、记录文件hash，并以对应模型成功load | 仍需单独验证前向、数值、许可证、运行成本与任务效果 |
| N/A | 无学习权重的算法 | 不等于零计算、零内存或零部署工作 |

**参数分桶用完整部署组件的元素数，不用可训练部分或文件MB。** 本表的0参数只覆盖该传统算法，输入descriptor、检测器、任务head仍要计入。encoder/head/full能拆则拆，不能确认则写未知；源码静态计数、张量header计数、作者约数、历史AIC记录各自注明。safetensors header内的shape与dtype不是完整权重内容验证。0–20M、>20–120M、>120–500M之外，单列身份/完整计数未核定项，不强行给其分桶。

**代码许可不自动证明作者权重、基础预训练数据和下游数据的使用权。** 卡片分开登记；“未确认独立权重条款”是证据缺口，不等于已判定禁止。采购、商用或比赛部署阶段应按具体权重及赛事条件核验。模型卡版本、训练输入尺度、测试裁剪数和硬件条件可能不同，不能跨行比较裸FPS。

## 2. 优先队列：五项工具对应五个问题

| 优先项 | 它改变的观察或处理路径 | 首轮用法 | 晋级/替换条件 |
|---|---|---|---|
| Farnebäck + LK/RANSAC | 从静态/稀疏embedding差分改为像素运动；再分离全局相机与局部残差 | H3先固定流算子、可靠性mask、同头同预算，区分输入修复与相机补偿 | 必须相对raw+camera的同维臂出现源外增量，不能只比位置臂 |
| PySceneDetect | 给出跨镜头无效区与镜头内位置 | H3有效性mask；H2单独测镜头结构 | 若真实质量审计表明误切/漏切破坏机制，才改用TransNetV2 |
| U²-NetP | 新增主体面积、边界、候选窗覆盖信息 | H1用native crop监督，冻结时间与point来源 | 与中心/高斯及B3恒等回放比较；mask好不自动证明高光好 |
| DINOv2-S | 提供新的静态稠密语义表征对照 | 已有V9先冻结；需要判别表征不足时才引入 | 冻结特征有稳定增量后才考虑末层适配；不是初始链必加组件 |
| RAFT-small | 在传统流存在具体质量失败时提供学习型两帧对应 | 不与多个flow版本一起择优；只替换失败的flow质量环节 | 先改善可观测质量，再重新验证同一任务机制，不能靠换流模型绕过负结果 |

UNISAL是值得保留的真实时序注视模型备选，但它预测观看注视而非AIC高光/裁切；不把其作为主线的默认附加网络。更精细的BASNet/IS-Net、更大视频encoder也须由明确的失败条件触发。完整判别臂和停止门见[minimal_falsification_experiments.md](minimal_falsification_experiments.md)。

## 3. 总览

“未知/待核定”不代表不轻量；代表本轮尚未把具体权重、模型构造与完整元素数绑定。GAIC PyTorch论文的<2.5M属于作者系列界限，实际demo/发布包仍待恢复。所有W2仅为本轮已观测的端点访问事实。

| ID | 模型/版本 | 参数桶 | 完整参数或边界 | 权重等级 | 主要观察 | AIC位置 |
|---|---|---|---|---|---|---|
| M01 | Pyramidal Lucas–Kanade | ≤20M | 0 | N/A | 稀疏点位移、全局相机拟合 | H3首轮，与M02配套 |
| M02 | Farnebäck optical flow | ≤20M | 0 | N/A | 稠密像素流、补偿后残差 | H3首轮 |
| M03 | DIS optical flow | ≤20M | 0 | N/A | 快速稠密流 | 速度/质量诊断备用 |
| M04 | Dual TV-L1 optical flow | ≤20M | 0 | N/A | 鲁棒变分流 | 离线质量对照 |
| M05 | RAFT-small | ≤20M | 990,162（官方实现） | W2 | 学习型两帧流 | 传统流质量失败才启用 |
| M06 | GMFlow scale1 | ≤20M | 约4.7M（scale1） | W2 | 全局匹配两帧流 | 大位移替代，非首轮 |
| M07 | PySceneDetect | ≤20M | 0 | N/A | 镜头边界、跨镜头mask | H3预处理 / H2 |
| M08 | TransNetV2 | 待核定 | 未核定；不能由MB换算 | W2 | 神经镜头转场 | M07质量失败才启用 |
| M09 | Kernel Temporal Segmentation | ≤20M | 0；descriptor另计 | N/A | 特征时间核分段 | H2条件性对照 |
| M10 | TSM MobileNetV2 | 待核定 | shift=0新增；full未知 | W2 | 轻量短时动作信息 | 条件性时序迁移 |
| M11 | U²-NetP | ≤20M | 1,131,181（源码静态） | W2 | 前景mask、边界、候选覆盖 | H1首选 |
| M12 | U²-Net full | 20–120M | 44,009,869（源码静态） | W1 | 更大前景mask网络 | M11质量失败的容量对照 |
| M13 | BASNet | 待核定 | 完整计数未知 | W1 | 边界细化mask | 边界质量失败时考虑 |
| M14 | IS-Net / Dichotomous Image Segmentation | 20–120M | 44,046,790（推理类静态） | W1 | 高分辨率前景分割 | 高精度mask条件性对照 |
| M15 | UNISAL | 待核定 | 完整计数未知 | W2 | 真实ConvGRU动态注视密度 | 动态注意力独立机制备用 |
| M16 | GAIC CVPR2019 MatConvNet | 待核定 | 完整计数未知 | W1 | RoI/RoD美学crop分数 | 借区域读出，非主攻 |
| M17 | GAIC author PyTorch / MobileNetV2 | 待核定 | 系列<2.5M；确切artifact待核 | W1 | 多尺度RoI/RoD美学crop | 先恢复构建/权重身份 |
| M18 | AutoFlip | 待核定 | 检测器组合总量未知 | W1（依赖权重） | 检测、镜头、相机路径框架 | 借路径设计，非整体移植 |
| M19 | YOLOX-Nano | ≤20M | 约0.91M（作者） | W2 | 类别目标框 | H1多主体诊断 |
| M20 | MobileNetV3-Small | ≤20M | 2,542,856（含分类器） | W2 | 轻静态语义 | CPU预算紧的替代表征 |
| M21 | MobileNetV3-Large | ≤20M | 5,483,032（含分类器） | W2 | 轻静态语义 | 容量对照 |
| M22 | DeiT-S non-distilled | 20–120M | 约22M（含分类器） | W2 | ImageNet监督ViT | 监督语义对照 |
| M23 | DINOv2-S / ViT-S14 | 20–120M | 22,056,576（张量header） | W2 | 自监督稠密语义token | 条件性语义对照优先 |
| M24 | VideoMAE-Base K400 fine-tuned | 20–120M | 86,534,800（分类包header） | W2 | 真实时空encoder | 低成本路径被排除后考虑 |
| M25 | HL-CLIP / Highlight-CLIP | 120–500M | 151,564,278（完整作者类） | W0；基础CLIP=W2 | query-saliency迁移 | 研究思想；不能直接叠加 |
| M26 | 既有 V9 / SigLIP2 NaFlex tower | 20–120M | 87,407,729（历史+静态） | 本轮未新增W3 | 现有静态crop效用链 | 冻结锚 / 轻链优先复用 |

## 4. 无学习权重算法与运动/镜头模型

### M01 Pyramidal Lucas–Kanade

参数桶：≤20M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | OpenCV 4.10 的 calcOpticalFlowPyrLK；金字塔稀疏 Lucas–Kanade。 |
| 02 论文 | [An Iterative Image Registration Technique with an Application to Stereo Vision, 1981](https://publications.ri.cmu.edu/an-iterative-image-registration-technique-with-an-application-to-stereo-vision-ijcai)。 |
| 03 原始开源地址 | [OpenCV tracking API](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)；[4.10 源码](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/lkpyramid.cpp)。 |
| 04 权重与实际取得等级 | 不适用：无学习权重。已读官方代码/API，未实跑。 |
| 05 参数量：encoder / head / full | encoder=0，head=0，算法 full=0；若用学习 detector/descriptor 或任务读出，另计。 |
| 06 训练数据、标注与目标函数 | 不训练；局部光度配准的迭代优化。 |
| 07 输入 | 相邻同尺寸图像、float32 跟踪点坐标、金字塔参数；使用官方像素格式合同。 |
| 08 输出 | 下一帧点、status、跟踪 error；单位是像素位移。 |
| 09 真实时间信息 | 两帧真实对应；可逐帧串联，但不自动拥有主体身份或长时语义。 |
| 10 空间主体定位 | 只跟踪给定点，不判断哪个主体重要。背景占多数且内点可靠时可用来估计相机仿射。 |
| 11 预训练任务学到了什么 | 无预训练；提供几何运动约束。 |
| 12 计算成本与测量条件 | 本轮未测 FPS/峰值内存；原1981论文与当前 OpenCV 平台不可横比。 |
| 13 部署依赖与 CUDA / NPU | OpenCV CPU 路径；配合前后向跟踪及 RANSAC。无需 CUDA；NPU/910A 尚未验证，也无必要先移植。 |
| 14 代码与权重许可 | [OpenCV 4.10 LICENSE](https://github.com/opencv/opencv/blob/4.10.0/LICENSE) 根许可 Apache-2.0，相关旧源码保留 notices；无权重许可。 |
| 15 作者 / 官方 benchmark | 本轮未找到可直接绑定此版本/配置的统一 F1/EPE/FPS；不填第三方结果。 |
| 16 AIC 适配方式 | 算法集成：为 Farnebäck 残差提供背景变换和可靠性估计，固定空间与时间预算。 |
| 17 主要风险与停止条件 | 主体占画面多数、低纹理、遮挡、滚动快门会破坏背景假设；内点不足时必须标 invalid，不能把失败流归零当静止。 |

### M02 Farnebäck optical flow

参数桶：≤20M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | OpenCV 4.10 calcOpticalFlowFarneback；先沿用 R52 固定参数并修正输入尺度。 |
| 02 论文 | [Two-Frame Motion Estimation Based on Polynomial Expansion, 2003](https://www.diva-portal.org/smash/get/diva2%3A273847/FULLTEXT01.pdf)。 |
| 03 原始开源地址 | [作者论文页](https://www.ida.liu.se/ext/WITAS-ev/Computer_Vision_Technologies/PaperInfo/farneback03.html)；[OpenCV API](https://docs.opencv.org/4.x/dc/d6b/group__video__track.html)；[4.10 实现](https://github.com/opencv/opencv/blob/4.10.0/modules/video/src/optflowgf.cpp)。 |
| 04 权重与实际取得等级 | 不适用：无权重。已运行受控合成 E0，未跑真实合法数据上的任务试验。 |
| 05 参数量：encoder / head / full | encoder=0，head=0，算法 full=0；新目标读出另计。 |
| 06 训练数据、标注与目标函数 | 无训练集；局部多项式展开、金字塔与位移迭代估计。 |
| 07 输入 | 两帧 8-bit 单通道同尺寸图；R52 的 float32/255 合同在本轮合成平移上退化近零。 |
| 08 输出 | H×W×2 流；是像素/帧对位移，需按图像尺度和真实 Δt 归一化。 |
| 09 真实时间信息 | 真实两帧对应；单独的总流没有相机/前景分离，更没有高光含义。 |
| 10 空间主体定位 | 有每像素位移而无主体语义；相机补偿后可构造候选独立运动区域，但不保证它对应真实前景或事件主体。 |
| 11 预训练任务学到了什么 | 无预训练；提供运动方向、位移与局部分布。 |
| 12 计算成本与测量条件 | [本轮 E0](evidence/aic_r52_farneback_synthetic_audit.json)：160×90、已知平移(3,2)，uint8 EPE=0.0008528868；float32/255 EPE=3.6055493355。仅数值合同，不是 AIC 准确率或真实吞吐；显存不适用，CPU内存/墙钟未测。 |
| 13 部署依赖与 CUDA / NPU | OpenCV CPU；本轮独立 OpenCV4.10.0/NumPy2.3.5/Python3.12.14。没有 GPU 或 NPU 推理实验。 |
| 14 代码与权重许可 | OpenCV 对应版本许可及源码 notices；无权重。 |
| 15 作者 / 官方 benchmark | E0 是本轮受控反证，未替代 Sintel/KITTI 或 AIC benchmark；不报告未经复现的跨模型速度。 |
| 16 AIC 适配方式 | 算法集成：可靠总流→LK/RANSAC 相机变换→残差→低维统计→公开目标域轻量读出。固定 K 与空间框。 |
| 17 主要风险与停止条件 | 曝光、纹理少、大位移、遮挡、转场仍会失败。必须区分仅修复总流与真正相机补偿增益；运动峰值不能直接作 highlight。 |

### M03 DIS optical flow

参数桶：≤20M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | Dense Inverse Search；OpenCV ULTRAFAST/FAST/MEDIUM 与作者 OF_DIS 分开标识。 |
| 02 论文 | [Fast Optical Flow using Dense Inverse Search, ECCV2016](https://arxiv.org/abs/1603.03590)。 |
| 03 原始开源地址 | [作者 OF_DIS](https://github.com/tikroeger/OF_DIS)；[OpenCV DIS API](https://docs.opencv.org/4.x/de/d4f/classcv_1_1DISOpticalFlow.html)。 |
| 04 权重与实际取得等级 | 不适用：无权重；作者/官方 API 可读，未实跑。 |
| 05 参数量：encoder / head / full | encoder=0，head=0，算法 full=0。 |
| 06 训练数据、标注与目标函数 | 无神经训练；inverse-compositional patch 匹配、多尺度聚合、可选变分细化。论文开发算子参数用过 Sintel train，不等于端到端神经预训练。 |
| 07 输入 | 相邻灰度图；具体预设与 OpenCV propagation 选项固定记录。 |
| 08 输出 | 稠密二维位移。 |
| 09 真实时间信息 | 真实两帧对应；不含动作或事件状态。 |
| 10 空间主体定位 | 像素对应，无主体身份。与本表 IS-Net 的 DIS 完全不同。 |
| 11 预训练任务学到了什么 | 无预训练；提供可计算的运动场。 |
| 12 计算成本与测量条件 | 论文 1024×436、i7-3770K 3.5GHz 单核有 10–600Hz 不同工作点，主图不含部分预处理，另约20ms预处理；不能把最快点和最准点拼在一起。本机成本未测。 |
| 13 部署依赖与 CUDA / NPU | 作者 C++/OpenCV CPU；采用哪个实现需锁定。NPU 未验证；CPU可独立预处理。 |
| 14 代码与权重许可 | [作者实现 GPL-3.0](https://github.com/tikroeger/OF_DIS/blob/master/LICENSE)；OpenCV 发行根许可 Apache-2.0 与 notices 另核，不能互换。无权重许可。 |
| 15 作者 / 官方 benchmark | 原论文具体 DIS-Fast 工作点：Sintel final test EPE10.13、0.023s；不是600Hz点，也不是本轮 OpenCV 实测。 |
| 16 AIC 适配方式 | 算法集成：仅在 Farnebäck 真实 CPU 成本/质量触发时替换流估计器，补偿、采样、读出全部固定。 |
| 17 主要风险与停止条件 | 与 Farnebäck 并行扫预设会失去因果性；快速运动、纹理与遮挡风险仍在。没有 CPU 需求就不引入第二个流实现。 |

### M04 Dual TV-L1 optical flow

参数桶：≤20M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | OpenCV contrib DualTVL1OpticalFlow，固定对应4.10代码版本。 |
| 02 论文 | [A Duality Based Approach for Realtime TV-L1 Optical Flow, DAGM2007](https://www.vrvis.at/en/publications/pdfs/PB-VRVis-2007-012.pdf)。 |
| 03 原始开源地址 | [官方 API](https://docs.opencv.org/4.x/dc/d4d/classcv_1_1optflow_1_1DualTVL1OpticalFlow.html)；[OpenCV contrib 实现](https://github.com/opencv/opencv_contrib/blob/4.10.0/modules/optflow/src/tvl1flow.cpp)。 |
| 04 权重与实际取得等级 | 不适用：无权重。仅审阅，未实跑。 |
| 05 参数量：encoder / head / full | encoder=0，head=0，算法 full=0。 |
| 06 训练数据、标注与目标函数 | 无训练；L1 光度数据项 + total variation 正则的多尺度 warping/迭代优化。 |
| 07 输入 | 相邻图像及金字塔/迭代设置；具体输入格式按选择的 API 固定。 |
| 08 输出 | 稠密二维位移，倾向保留运动边界。 |
| 09 真实时间信息 | 真实两帧对应；不是长时动作编码。 |
| 10 空间主体定位 | 无主体语义，只有运动边界结构。 |
| 11 预训练任务学到了什么 | 无预训练；相比二次误差目标提供不同的光度鲁棒性。 |
| 12 计算成本与测量条件 | 原论文约30fps为320×240、每层50次迭代、当时OpenGL/Cg实现；另一256²/50次、GeForce Go7900GTX为34.1fps。本轮CPU/NPU成本及内存未测。 |
| 13 部署依赖与 CUDA / NPU | cv.optflow 需要 contrib；普通 opencv-python-headless 不保证包含该算法。原 GPU/Cg 结果不是现有 NPU 实现。 |
| 14 代码与权重许可 | 采用 OpenCV/contrib 对应发行许可；原作者完整实现许可本轮未核定；无权重。 |
| 15 作者 / 官方 benchmark | 未另列未经核验的 benchmark EPE。上述速度只描述原论文具体条件。 |
| 16 AIC 适配方式 | 算法集成：传统流的鲁棒性对照，须保持补偿、特征、帧对、读出不变。 |
| 17 主要风险与停止条件 | 大曝光、透明体、遮挡仍不保证正确；若未定位到匹配鲁棒性瓶颈，停止增加此分支。 |

### M05 RAFT-small

参数桶：≤20M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | 作者 RAFT --small；可获得的精确备选为 Torchvision raft_small C_T_V1。 |
| 02 论文 | [RAFT: Recurrent All-Pairs Field Transforms for Optical Flow, ECCV2020](https://arxiv.org/pdf/2003.12039)。 |
| 03 原始开源地址 | [作者代码](https://github.com/princeton-vl/RAFT)；[作者下载脚本](https://github.com/princeton-vl/RAFT/blob/master/download_models.sh)；[Torchvision 元数据/实现](https://docs.pytorch.org/vision/main/_modules/torchvision/models/optical_flow/raft.html)。 |
| 04 权重与实际取得等级 | [官方移植 C_T_V1](https://download.pytorch.org/models/raft_small_C_T_V1-ad48884c.pth) W2，206/4,006,189 bytes；作者 models.zip 也 W2，但本轮未看完整目录、未确认包内 small 文件。无 load。 |
| 05 参数量：encoder / head / full | 官方 Torchvision full=990,162；encoder/update heads 未单独枚举。原论文 small≈1.0M为另一口径；最终选定实现再实数。 |
| 06 训练数据、标注与目标函数 | FlyingChairs→FlyingThings3D 的 C+T；Sintel/KITTI微调版另列。GT flow 的多步L1监督。 |
| 07 输入 | RGB 两帧；作者路径接收0–255并内部映射[-1,1]，Torchvision transform合同不同，不能双重归一化。 |
| 08 输出 | 多次迭代的像素二维流。 |
| 09 真实时间信息 | 真实两帧匹配+GRU迭代求解；GRU更新不是长视频事件记忆。 |
| 10 空间主体定位 | 密集运动对应，无主体重要性或语义定位。 |
| 11 预训练任务学到了什么 | 学习像素对应及遮挡/复杂纹理的匹配先验，不学习 GIF 选择。 |
| 12 计算成本与测量条件 | 原论文small：GTX1080Ti、约1088×436、10次更新、0.05s。all-pairs相关体随空间位置对增长；小参数不等于低显存。本轮延时/显存未测。 |
| 13 部署依赖与 CUDA / NPU | PyTorch、correlation、grid_sample、迭代；可选 alt_cuda_corr 需CUDA，纯PyTorch路径仍需实际导出。NPU/910A尚未验证。 |
| 14 代码与权重许可 | 作者代码与Torchvision代码为BSD-3-Clause；未确认独立权重许可范围，训练数据条款另行记录。 |
| 15 作者 / 官方 benchmark | 原论文 C+T/Sintel train final：small EPE3.37；Torchvision C_T_V1元数据clean/final=2.1231/3.2790，协议不能混合。 |
| 16 AIC 适配方式 | Frozen flow作为质量fallback；仅在传统流对大位移/模糊等已证明失败后替换，不同时换目标监督和选择政策。 |
| 17 主要风险与停止条件 | 相关体成本、部署算子、域偏移；若更准flow不能改善同容量目标域读出，关闭升级。 |

### M06 GMFlow scale1

参数桶：≤20M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | 作者 GMFlow / UniMatch model zoo 的 gmflow-scale1-things；未核验到官方 GMFlow-small 发布名。 |
| 02 论文 | [GMFlow: Learning Optical Flow via Global Matching, CVPR2022](https://arxiv.org/pdf/2111.13680)。 |
| 03 原始开源地址 | [作者 GMFlow](https://github.com/haofeixu/gmflow)；[作者后继 UniMatch model zoo](https://github.com/autonomousvision/unimatch/blob/master/MODEL_ZOO.md)。 |
| 04 权重与实际取得等级 | [scale1-things 权重](https://s3.eu-central-1.amazonaws.com/avg-projects/unimatch/pretrained/gmflow-scale1-things-e9887eda.pth) W2，206/18,768,907 bytes/ZIP前缀；未 load。 |
| 05 参数量：encoder / head / full | 作者scale1/scale2约4.7M，scale2-regrefine6约7.4M；当前卡片只登记scale1。encoder/head拆分未实数；不能拿4.2M层数消融冒充发布small。 |
| 06 训练数据、标注与目标函数 | Chairs→Things，后续Sintel/KITTI/mixdata不同权重分开；GT flow多级L1。 |
| 07 输入 | 两个RGB图像、固定尺度与padding。 |
| 08 输出 | 由全局匹配、传播、可选细化得到稠密流；可求双向流。 |
| 09 真实时间信息 | 真实两帧关联，双向计算有复用但不是零额外成本。 |
| 10 空间主体定位 | 密集位置对应，不预测主体重要性。 |
| 11 预训练任务学到了什么 | 学习局部特征、self/cross-attention及全局匹配，而非高光偏好。 |
| 12 计算成本与测量条件 | model zoo：A100单GPU、batch1、448×1024、100次均值26ms。原README另有Sintel436×1024、A10026ms/V10057ms；不能混为同设置。显存本轮未测。 |
| 13 部署依赖与 CUDA / NPU | PyTorch、attention/correlation、warp、unfold与padding；NPU算子/数值未验证。 |
| 14 代码与权重许可 | 作者代码Apache-2.0；预训练文件独立权重条款本轮未核定。 |
| 15 作者 / 官方 benchmark | 原论文无refine的4.7M模型，C+T后Sintel train final EPE2.94；不是本轮权重复现结果。 |
| 16 AIC 适配方式 | Frozen flow的后备比较；优先级低于先修正现有Farnebäck，再触发RAFT-small。 |
| 17 主要风险与停止条件 | 全局对应成本与错误匹配仍存在；不创造small版本，也不以flow EPE替代AIC源级收益。 |

### M07 PySceneDetect

参数桶：≤20M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | ContentDetector / AdaptiveDetector / ThresholdDetector；部署时锁版本与阈值，当前调查官方latest API。 |
| 02 论文 | 软件项目，无单一对应模型论文；[官方算法文档](https://www.scenedetect.com/docs/latest/api/detectors.html)。 |
| 03 原始开源地址 | [作者代码](https://github.com/Breakthrough/PySceneDetect)。 |
| 04 权重与实际取得等级 | 不适用：无权重；已读官方API/源码，未实跑。 |
| 05 参数量：encoder / head / full | encoder=0，head=0，算法full=0；视频解码器不带学习参数。 |
| 06 训练数据、标注与目标函数 | 无训练；HSV相邻变化、邻域相对变化、淡入淡出阈值与最短镜头约束。 |
| 07 输入 | 按时间顺序解码的视频帧；需保留原始fps和PTS/帧号映射。 |
| 08 输出 | shot边界、变化分数及统计。 |
| 09 真实时间信息 | 真实邻帧/邻域时序变化；不理解事件结果。 |
| 10 空间主体定位 | 不定位主体，仅提供镜头状态边界。 |
| 11 预训练任务学到了什么 | 无预训练；识别图像变化模式，不产生高光语义。 |
| 12 计算成本与测量条件 | 官方文档未给可与本任务直接对齐的统一F1/FPS；CPU墙钟、峰值内存未测。 |
| 13 部署依赖与 CUDA / NPU | Python/OpenCV或FFmpeg解码；CPU可做，不依赖CUDA。NPU无需先移植；解码与统计仍要计时。 |
| 14 代码与权重许可 | [代码BSD-3-Clause](https://github.com/Breakthrough/PySceneDetect/blob/main/LICENSE)；解码依赖许可/构建另记。无权重。 |
| 15 作者 / 官方 benchmark | 本轮未实测剪切/渐变F1，不引第三方速度数。 |
| 16 AIC 适配方式 | 算法集成：作为光流无效跨镜头掩码及隐藏状态reset。与R47强制P95切点区分；不直接取shot center当高光。 |
| 17 主要风险与停止条件 | 闪光、快速摇移、渐变会误检/漏检；先检查边界质量，只有不足时触发TransNetV2。 |

### M08 TransNetV2

参数桶：待核定。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | 作者TransNetV2 TensorFlow SavedModel；PyTorch目录是转换实现，不能视为已取得作者pth。 |
| 02 论文 | [TransNet V2: An Effective Deep Network Architecture for Fast Shot Transition Detection, 2020](https://arxiv.org/pdf/2008.04838)。 |
| 03 原始开源地址 | [作者仓库](https://github.com/soCzech/TransNetV2)；[TF推理/权重](https://github.com/soCzech/TransNetV2/tree/master/inference)；[PyTorch转换](https://github.com/soCzech/TransNetV2/tree/master/inference-pytorch)。 |
| 04 权重与实际取得等级 | [LFS实际data端点](https://media.githubusercontent.com/media/soCzech/TransNetV2/master/inference/transnetv2-weights/variables/variables.data-00000-of-00001) W2：30,516,656 bytes；index5,526、saved_model.pb5,933,260也W2。raw data仅133-byte LFS pointer，不能算权重。未完整hash/转换/load。 |
| 05 参数量：encoder / head / full | 完整网络精确元素数本轮未核验；encoder/heads未拆。不能用30.5MB除4给出参数档。 |
| 06 训练数据、标注与目标函数 | ClipShots真实转场与IACC.3/ClipShots镜头合成cut/dissolve；约15%真实、35%合成cut、50%合成dissolve；两种转场标签监督。 |
| 07 输入 | 作者PyTorch接口[B,T,27,48,3] uint8 RGB，内部/255；常用100帧窗，中间50帧有效输出。 |
| 08 输出 | single-frame与all-transition概率。 |
| 09 真实时间信息 | 真实多帧(2+1)D膨胀卷积、帧相似与直方图，窗口会看未来帧；不是动作识别。 |
| 10 空间主体定位 | 无主体框或mask。 |
| 11 预训练任务学到了什么 | 学习剪切/渐变等转场结构，不学习精彩程度。 |
| 12 计算成本与测量条件 | 17小时/V100是训练耗时，不是推理FPS。当前CPU/GPU/NPU吞吐、FLOPs、峰值显存尚未验证。 |
| 13 部署依赖与 CUDA / NPU | 旧TF SavedModel或TF→PyTorch转换；dilation、动态gather、直方图位运算/scatter与依赖要核。NPU尚未验证。 |
| 14 代码与权重许可 | 作者代码MIT；TF权重独立许可范围未确认。作者LFS声明hash不是本轮完整hash。 |
| 15 作者 / 官方 benchmark | 论文ClipShots F1=77.9、BBC=96.2、RAI=93.9；匹配允许±2帧，阈值/解码需固定。 |
| 16 AIC 适配方式 | Frozen边界检测fallback：只有PySceneDetect真实质量不足时使用；提供reset/跨镜头无效掩码。 |
| 17 主要风险与停止条件 | 误把LFS指针当下载、把fps变化当同时间窗、混两个头或转换误差；边界F1高不能直接推出AIC收益。 |

### M09 Kernel Temporal Segmentation

参数桶：≤20M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | 作者KTS v1.1归档；不是任意第三方Python移植。 |
| 02 论文 | [Category-specific video summarization, ECCV2014](https://hal.inria.fr/hal-01022967)；[作者slides](https://lear.inrialpes.fr/people/potapov/slides/khronos_summarization.pdf)。 |
| 03 原始开源地址 | [作者项目](https://lear.inrialpes.fr/people/potapov/med_summaries.html)；[原代码归档](http://pascal.inrialpes.fr/data2/potapov/med_summaries/kts_ver1.1.tar.gz)。 |
| 04 权重与实际取得等级 | 无学习权重。代码链接存在，但HTTPS503、HTTP超时，本轮未取得原实现/许可；不能称开箱可用。 |
| 05 参数量：encoder / head / full | KTS算法0；输入descriptor模型与重要性head另计。预提取缓存不免除encoder计数。 |
| 06 训练数据、标注与目标函数 | 不训练神经网络；Gram/kernel上的段内离散度与分段复杂度惩罚，动态规划选变点。 |
| 07 输入 | 时间有序的帧/片段特征及其Gram矩阵。 |
| 08 输出 | 变点与片段，不是重要性分数。 |
| 09 真实时间信息 | 全序列自相似和段内一致性；区别于相邻embedding差分。 |
| 10 空间主体定位 | 不定位主体；空间能力取决于输入descriptor。 |
| 11 预训练任务学到了什么 | 算法本身无预训练；获得时间分段结构。 |
| 12 计算成本与测量条件 | Gram存储O(T²)、常用DP约O(mT²)；实际FPS/内存未测，长视频采样需保留时间映射。 |
| 13 部署依赖与 CUDA / NPU | 原实现获取尚阻塞；NumPy/动态规划移植需验证。上游descriptor依赖另计，NPU未验证。 |
| 14 代码与权重许可 | 原作者代码许可本轮未取得；不能替第三方fork背书。无权重许可。 |
| 15 作者 / 官方 benchmark | 原项目MED Summaries 160视频/60val+100test是事件摘要协议；本轮没取标签或复现实验，不填未经核验F1。 |
| 16 AIC 适配方式 | 算法集成或仅作理论对照；在许可与原实现恢复前不列可立即部署项。 |
| 17 主要风险与停止条件 | O(T²)成本、源descriptor域偏移、切段不是全局去重/高光；不能把零参数等同可用。 |

### M10 TSM MobileNetV2

参数桶：待核定。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | 作者TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense；在线/离线方向必须另记。 |
| 02 论文 | [TSM: Temporal Shift Module for Efficient Video Understanding, ICCV2019](https://arxiv.org/pdf/1811.08383)。 |
| 03 原始开源地址 | [作者代码/model zoo](https://github.com/mit-han-lab/temporal-shift-module)；[shift实现](https://github.com/mit-han-lab/temporal-shift-module/blob/master/ops/temporal_shift.py)。 |
| 04 权重与实际取得等级 | [作者MobileNetV2权重](https://hanlab18.mit.edu/projects/tsm/models/TSM_kinetics_RGB_mobilenetv2_shift8_blockres_avg_segment8_e100_dense.pth) W2，206/11,144,548 bytes，pickle前缀；未反序列化。 |
| 05 参数量：encoder / head / full | shift模块本身0新增参数，但完整MobileNetV2+分类器元素数本轮未核。论文R50≈24.3M是另一模型，不能给本权重套用。 |
| 06 训练数据、标注与目标函数 | 本权重Kinetics动作分类监督；作者还训练Something-Something/Jester等版本，应分开。目标为动作类别CE。 |
| 07 输入 | 8帧RGB clip；dense多clip测试与单clip不同；在线单向需保持状态。 |
| 08 输出 | 动作logit/时序图像feature。 |
| 09 真实时间信息 | 逐层通道沿时间移位，使2D卷积混合顺序；离线双向可看未来，在线单向不能。 |
| 10 空间主体定位 | 保留空间特征图，但无主体/裁剪定位head。 |
| 11 预训练任务学到了什么 | 学习动作相关的时空表征，不学习GIF偏好。 |
| 12 计算成本与测量条件 | MobileNetV2边缘论文13.4ms/Jetson Nano、34.5ms/Note8为TVM编译online特定实现；本轮无本机/NPU/显存验证。shift有内存搬运。 |
| 13 部署依赖与 CUDA / NPU | PyTorch/tensor shift；某些in-place/CUDA路径、TVM编译与移动部署是不同实现。NPU尚未验证。 |
| 14 代码与权重许可 | 作者代码MIT；作者checkpoint独立权重许可未确认，Kinetics素材条款另记。 |
| 15 作者 / 官方 benchmark | 该MNetV2权重Kinetics top1=69.5，8帧×10clips dense；R50 8帧SthV1 center crop top1=45.6/33GFLOPs是另一设置。 |
| 16 AIC 适配方式 | Frozen feature或目标域小范围FT；同backbone有/无shift、同帧数、同监督、顺序扰动才识别时序增量。 |
| 17 主要风险与停止条件 | 动作任务错域，dense benchmark与在线时延混用；在残差运动可归因首测之前不优先引入。 |

## 5. 主体、边界与裁切模型

### M11 U²-NetP

参数桶：≤20M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | U²-Net†的小模型；作者类U2NETP(in_ch=3,out_ch=1)，不是人像专用权重。 |
| 02 论文 | [U²-Net: Going Deeper with Nested U-Structure for Salient Object Detection, Pattern Recognition2020](https://arxiv.org/pdf/2005.09007)。 |
| 03 原始开源地址 | [作者仓库](https://github.com/xuebinqin/U-2-Net)；[结构源码](https://github.com/xuebinqin/U-2-Net/blob/master/model/u2net.py)；[预处理](https://github.com/xuebinqin/U-2-Net/blob/master/data_loader.py)。 |
| 04 权重与实际取得等级 | [作者Drive](https://drive.google.com/file/d/1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy/view?usp=sharing)；标准下载端点HEAD最终200/octet-stream、u2netp.pth、4,683,258 bytes，W2。[原始记录](evidence/spatial_weight_access_audit.json)。 |
| 05 参数量：encoder / head / full | full=1,131,181，含side/fuse heads；BN buffers5,808单列。来自[静态AST计数](evidence/spatial_static_params/parameter_counts.json)，未load。encoder/head未拆账，不能只数训练部分。 |
| 06 训练数据、标注与目标函数 | DUTS-TR 10,553图像/显著对象mask；嵌套RSU从头训练、多输出BCE监督，无ImageNet backbone。 |
| 07 输入 | 单张RGB，论文测试320²。作者ToTensorLab(flag=0)有按图像最大值缩放与RGB标准化，不能挪用YOLOX的0–255合同。 |
| 08 输出 | 单图前景显著概率；融合输出与六个side outputs，缩放回原图。 |
| 09 真实时间信息 | 没有跨帧状态、身份或动作信息；逐帧运行仍是单图模型。 |
| 10 空间主体定位 | 可定位显著主体面积、外轮廓；不判断事件主角或最优crop window。 |
| 11 预训练任务学到了什么 | 学到显著前景/背景与多尺度轮廓，不是人类眼动或GIF偏好。 |
| 12 计算成本与测量条件 | 作者GTX1080Ti、320²：4.7MB、40FPS，表中均值25ms；不等于本机含解码端到端速率。FLOPs/峰值显存本轮未核。 |
| 13 部署依赖与 CUDA / NPU | PyTorch、conv/pooling/bilinear；本轮仅静态算术，无CPU/CUDA模型推理；NPU/910A尚未验证。黑图/恒定输出需有限值保护。 |
| 14 代码与权重许可 | [代码Apache-2.0](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE)；Drive权重未发现单独明确许可，不将代码/data许可自动外推。 |
| 15 作者 / 官方 benchmark | 论文ECSSD maxFβ=.943、MAE=.041；DUT-OMRON .813/.060。是掩码指标，不能換算crop IoU或AIC分。 |
| 16 AIC 适配方式 | Frozen mask→候选内/外主体质量、边界截断风险→原生crop监督轻头；必要时任务微调/KD另设对照。 |
| 17 主要风险与停止条件 | mask centroid不等于crop中心、显著主体不等于事件主体。真mask须优于同容量置乱/几何对照；只改善可视化不降低crop regret则停止。 |

### M12 U²-Net full

参数桶：20–120M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | 作者U2NET(in_ch=3,out_ch=1)，通用DUTS权重。 |
| 02 论文 | [U²-Net, Pattern Recognition2020](https://arxiv.org/pdf/2005.09007)。 |
| 03 原始开源地址 | [作者代码](https://github.com/xuebinqin/U-2-Net)；[结构源码](https://github.com/xuebinqin/U-2-Net/blob/master/model/u2net.py)。 |
| 04 权重与实际取得等级 | [作者大模型Drive](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view?usp=sharing)仅链接核验，W1；未HTTP模型端点核验/下载。 |
| 05 参数量：encoder / head / full | full=44,009,869，含side/fuse heads；BN buffers28,912。[静态AST证据](evidence/spatial_static_params/parameter_counts.json)，未load；encoder/head未拆。 |
| 06 训练数据、标注与目标函数 | 同U²-NetP，DUTS-TR显著mask，多输出BCE，从头训练。 |
| 07 输入 | 单张RGB、测试320²；预处理同作者实现。 |
| 08 输出 | 前景显著概率图。 |
| 09 真实时间信息 | 无真实时序。 |
| 10 空间主体定位 | 显著主体区域和更丰富轮廓，仍不决定正确裁剪。 |
| 11 预训练任务学到了什么 | 同监督更大容量；没有新增训练目标或时间机制。 |
| 12 计算成本与测量条件 | 作者GTX1080Ti、320²：176.3MB、30FPS/表中33ms。FLOPs和峰值显存未核。 |
| 13 部署依赖与 CUDA / NPU | PyTorch；未本机推理/导出/910A验证。 |
| 14 代码与权重许可 | [代码Apache-2.0](https://github.com/xuebinqin/U-2-Net/blob/master/LICENSE)；权重独立条款未确认。 |
| 15 作者 / 官方 benchmark | ECSSD maxFβ=.951、MAE=.033；DUT-OMRON .823/.054，作者同表条件。 |
| 16 AIC 适配方式 | Frozen容量对照：只有U2P确因轮廓漏分割而失败时，用同候选、同头、同监督比较P/full。 |
| 17 主要风险与停止条件 | 将高容量、更多输入或不同监督一起改不能识别容量原因；若P已给足够主体外轮廓则不升级。 |

### M13 BASNet

参数桶：待核定。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | CVPR2019 BASNet；作者README还含2021-05-02扩展版，二者成绩/速度分开。 |
| 02 论文 | [BASNet: Boundary-Aware Salient Object Detection, CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Qin_BASNet_Boundary-Aware_Salient_Object_Detection_CVPR_2019_paper.html)；[2021扩展论文](https://arxiv.org/abs/2101.04704)。 |
| 03 原始开源地址 | [作者代码](https://github.com/xuebinqin/BASNet)；[模型](https://github.com/xuebinqin/BASNet/blob/master/model/BASNet.py)。 |
| 04 权重与实际取得等级 | [当前作者basnet.pth链接](https://drive.google.com/open?id=1s52ek_4YTDRt_EOkx1FS53u-vJa0c4nu) W1；未证明与2019实验checkpoint一致。 |
| 05 参数量：encoder / head / full | ResNet34 encoder+预测/残差细化网络，不能只数ResNet34。本轮未枚举full/head参数；不由348.6MB反算。 |
| 06 训练数据、标注与目标函数 | DUTS-TR显著对象mask；BCE+SSIM+IoU分别监督像素、局部结构与全图mask。 |
| 07 输入 | 单图RGB；原论文测试256²。 |
| 08 输出 | 显著对象mask与边界细化结果。 |
| 09 真实时间信息 | 无时序。 |
| 10 空间主体定位 | 主体区域及细边界；不是主体身份/意图或最佳窗口。 |
| 11 预训练任务学到了什么 | 学习前景分割与边界细化。 |
| 12 计算成本与测量条件 | CVPR摘要单GPU>25FPS；2021扩展报告>70FPS，不能合并。作者[IS-Net论文](https://arxiv.org/html/2203.03041v4)重训对照列348.6MB；本轮FLOPs/显存/平台推理未核。 |
| 13 部署依赖与 CUDA / NPU | 旧PyTorch/torchvision；模型constructor会请求ResNet34预训练，必须锁依赖/权重。未CPU/CUDA/NPU复现。 |
| 14 代码与权重许可 | [代码MIT](https://github.com/xuebinqin/BASNet/blob/master/LICENSE)；Drive权重独立许可未确认。 |
| 15 作者 / 官方 benchmark | 原论文六个公共SOD集与区域/边界指标；本轮未重核各表具体数，不移用其他版本性能。 |
| 16 AIC 适配方式 | Frozen边界风险特征，或原生crop监督下的辅助分割/蒸馏。先比较更廉价U2P。 |
| 17 主要风险与停止条件 | 细毛发改善未必影响最大合法窗口；只有实际边界瓶颈才进入，若效用不增或身份不清则停止。 |

### M14 IS-Net / Dichotomous Image Segmentation

参数桶：20–120M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | ECCV2022 ISNetDIS；academic isnet.pth 与 general-use分开；不等于DIS光流、也不是DIS V2。 |
| 02 论文 | [Highly Accurate Dichotomous Image Segmentation, ECCV2022](https://arxiv.org/html/2203.03041v4)。 |
| 03 原始开源地址 | [作者仓库](https://github.com/xuebinqin/DIS)；[结构](https://github.com/xuebinqin/DIS/blob/main/IS-Net/models/isnet.py)；[Inference.py](https://github.com/xuebinqin/DIS/blob/main/IS-Net/Inference.py)。 |
| 04 权重与实际取得等级 | [作者Drive ID](https://drive.google.com/file/d/1XHIzgTzY5BQHw140EDIgwIb53K659ENH/view?usp=sharing)被README同时列在academic/general段，另有坏嵌套链接；只到W1，权重身份待核。 |
| 05 参数量：encoder / head / full | 推理ISNetDIS=44,046,790；训练GTEncoder=6,877,734，推理不调用才不计。[静态声明证据](evidence/spatial_static_params/parameter_counts.json)；未load，推理head未单拆。 |
| 06 训练数据、标注与目标函数 | DIS5K 5,470图：3000train/470val/4×500test；高精度二值mask与GTEncoder中间特征监督，论文默认L2特征匹配。 |
| 07 输入 | 单RGB，general-use入口resize1024²、/255后减0.5；回到原图输出。 |
| 08 输出 | 细致前景mask与多层特征。 |
| 09 真实时间信息 | 无跨帧状态。 |
| 10 空间主体定位 | 高精度轮廓、孔洞/细结构；不含事件主体选择或crop utility。 |
| 11 预训练任务学到了什么 | 通过mask结构教师提供中间监督，学高分辨率二值分割。 |
| 12 计算成本与测量条件 | 作者RTX A6000、1024²、19.49ms、176.6MB；与U2P的320²/1080Ti不同。FLOPs/峰值显存本轮未核。 |
| 13 部署依赖与 CUDA / NPU | PyTorch、1024²卷积/bilinear；输出min–max缺恒定值保护。本轮未推理，NPU尚未验证。 |
| 14 代码与权重许可 | [代码/metric Apache-2.0](https://github.com/xuebinqin/DIS/blob/main/LICENSE.md)；权重独立许可未确认。[DIS5K数据terms](https://github.com/xuebinqin/DIS/blob/main/DIS5K-Dataset-Terms-of-Use.pdf)另限非商业研究/教育与分发，不是权重许可。 |
| 15 作者 / 官方 benchmark | 论文DIS-VD maxFβ=.791；各模型均DIS-TR重训，不能比较成DUTS原权重零样本成绩。 |
| 16 AIC 适配方式 | Frozen精细mask或空间辅助监督；只有U2P明确因边界分辨率失败才升级。 |
| 17 主要风险与停止条件 | 初版人物/动物/车覆盖不足，权重链接身份混乱；精细mask未转成原生crop效用则停止。 |

### M15 UNISAL

参数桶：待核定。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | 作者pretrained_unisal weights_best；MIT1003 fine-tuned文件另计。 |
| 02 论文 | [Unified Image and Video Saliency Modeling, ECCV2020](https://arxiv.org/pdf/2003.05477)。 |
| 03 原始开源地址 | [作者仓库](https://github.com/rdroste/unisal)；[模型](https://github.com/rdroste/unisal/blob/master/unisal/model.py)；[发布配置](https://github.com/rdroste/unisal/blob/master/training_runs/pretrained_unisal/UNISAL.json)。 |
| 04 权重与实际取得等级 | [作者weights_best.pth](https://raw.githubusercontent.com/rdroste/unisal/master/training_runs/pretrained_unisal/weights_best.pth) W2，200/octet-stream/15,368,928 bytes；无load。[HEAD证据](evidence/spatial_weight_access_audit.json)。 |
| 05 参数量：encoder / head / full | MobileNetV2 encoder+ConvGRU+多域decoder；full及分支元素数未实数。论文15.5MB不能除4作申报。 |
| 06 训练数据、标注与目标函数 | DHF1K/Hollywood2/UCF Sports/SALICON眼动/注意密度；[Trainer.json](https://github.com/rdroste/unisal/blob/master/training_runs/pretrained_unisal/Trainer.json) loss=KLD−0.1NSS−0.1CC。 |
| 07 输入 | [B,T,C,H,W]；DHF1K发布配置224×384、12帧、每5帧采样、输出目标360×640，ImageNet RGB mean/std。 |
| 08 输出 | 空间log-softmax注意密度及可选hidden state；不是sigmoid前景mask。 |
| 09 真实时间信息 | 真实ConvGRU历史；单帧默认static=True会bypass RNN。显式static=False/h0才可区分reset与延续。 |
| 10 空间主体定位 | 眼动关注位置/分布，不保证完整主体面积或裁剪偏好。 |
| 11 预训练任务学到了什么 | 跨图像与视频域的视觉注意及其时间转移。 |
| 12 计算成本与测量条件 | 论文15.5MB、GPU0.009s/frame；CPU Xeon W-2123@3.60GHz=0.43s。训练用1080Ti，运行表没逐项完整复述硬件；本机显存/吞吐未测。 |
| 13 部署依赖与 CUDA / NPU | PyTorch+separable ConvGRU、BN域分支、smoothing；仓库不再维护。CPU/GPU实际复现与910A均未验证。 |
| 14 代码与权重许可 | [代码Apache-2.0](https://github.com/rdroste/unisal/blob/master/LICENSE)；同仓库权重未见另述独立范围，不自动外推数据权利。 |
| 15 作者 / 官方 benchmark | 四域联合训练DHF1K测试AUC-J=.901、SIM=.390、NSS=2.776；注意密度指标，非高光。 |
| 16 AIC 适配方式 | Frozen视频注意密度→crop候选质量；相同source/frames/checkpoint下static=False且h0延续vs重置，验证时序贡献。 |
| 17 主要风险与停止条件 | 切镜reset、log-density误读、source域切换和DHF1K/RetargetVid重源；注视脸/字幕可能破坏完整主体，效用不增则停止。 |

### M16 GAIC CVPR2019 MatConvNet

参数桶：待核定。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | Grid Anchor based Image Cropping原MatConvNet版，最终截断VGG16；不是PyTorch扩展。 |
| 02 论文 | [Reliable and Efficient Image Cropping: A Grid Anchor based Approach, CVPR2019](https://www4.comp.polyu.edu.hk/~cslzhang/paper/CVPR19-GAIC.pdf)。 |
| 03 原始开源地址 | [作者原仓库](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping)。 |
| 04 权重与实际取得等级 | [作者原模型](https://drive.google.com/open?id=1OvLT_ul17zCK4ljAi4myGAgA50PmLy3Y) W1；作者声明发布版重新训练、与论文略有差别。 |
| 05 参数量：encoder / head / full | 截断VGG16+RoI/RoD/MLP；full未实数。论文参数占用用Mbit，不能直接当百万参数或移用全VGG分类器数。 |
| 06 训练数据、标注与目标函数 | GAICD人工MOS；留200图测试，其余1036图共89519候选train/val；Huber回归，MOS训练集归一化。 |
| 07 输入 | 单图（短边256）+候选crop boxes。 |
| 08 输出 | 每候选美学MOS预测/排序。 |
| 09 真实时间信息 | 无时序。 |
| 10 空间主体定位 | RoI与被丢弃区域RoD共同描述内容保留；不独立输出主体mask。 |
| 11 预训练任务学到了什么 | 学习人工裁剪美学偏好，而非AIC固定最大窗口native IoU。 |
| 12 计算成本与测量条件 | 作者i7-6800K/64GB/GTX1080Ti约125FPS，README明示不含预处理；FLOPs/显存未核。 |
| 13 部署依赖与 CUDA / NPU | MatConvNet/MATLAB与旧编译环境；未实跑或NPU部署。 |
| 14 代码与权重许可 | 原repo根目录未见LICENSE，未确认一般源码授权；原模型与数据独立许可也未确认。不能套用下一条PyTorch repo的MIT。 |
| 15 作者 / 官方 benchmark | 原论文SRCC=.735；Acc实现后来修复，应固定修订版本，不同时选旧Acc与新代码。 |
| 16 AIC 适配方式 | 借RoI/RoD读出思想；在冻结backbone/candidates下以AIC可用native crop IoU重新监督。 |
| 17 主要风险与停止条件 | MOS与IoU目标、候选形状/面积不同；与R70同属全图特征池化，不夸为新增像素观察；许可/构建未清不作为即时依赖。 |

### M17 GAIC author PyTorch / MobileNetV2

参数桶：待核定。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | 作者TPAMI扩展PyTorch实现；demo为multi-scale、MobileNetV2、alignsize9、reddim8。 |
| 02 论文 | [Grid Anchor Based Image Cropping: A New Benchmark and An Efficient Model](https://doi.org/10.1109/TPAMI.2020.3024207)，2020-09-15在线；TPAMI 44(3):1304–1319，2022卷期。[作者全文](https://www4.comp.polyu.edu.hk/~cslzhang/paper/GAIC-PAMI.pdf)、[IEEE元数据](https://ieeexplore.ieee.org/abstract/document/9197621)。 |
| 03 原始开源地址 | [作者PyTorch仓库](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch)；[demo_eval.py](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch/blob/master/demo_eval.py)；[croppingModel.py](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch/blob/master/croppingModel.py)。 |
| 04 权重与实际取得等级 | [作者journal模型包](https://drive.google.com/file/d/1KWYQdL6R5hmOC9toTymbMORZDThpiEW4/view?usp=sharing) W1；未确认包内完整文件、hash或load。 |
| 05 参数量：encoder / head / full | 作者journal全文报告轻量MobileNetV2/ShuffleNetV2系列完整模型<2.5M；本登记demo的encoder/head/full精确元素数待核，权重未load；不借旧VGG数字。 |
| 06 训练数据、标注与目标函数 | 扩展GAICD共3,336源图/288,069候选，每crop至少7人评分；MOS与区域审美回归。具体发布checkpoint划分/损失需随权重恢复，不混原版89519训练候选。 |
| 07 输入 | 单图+候选框；源码多层feature缩放拼接，不是每个真实crop重编码。 |
| 08 输出 | 候选美学score。 |
| 09 真实时间信息 | 无时序。 |
| 10 空间主体定位 | 有裁剪内容/丢弃内容表征，没有独立主体分割或追踪。 |
| 11 预训练任务学到了什么 | 美学构图监督；迁到AIC需要native crop目标。 |
| 12 计算成本与测量条件 | 作者journal系列报告最高200FPS（GTX1080Ti）/12FPS（i7-6800K）；未确认与所链MobileNetV2 artifact、batch及预处理完全一致，因此不当本卡实测速度。FLOPs/显存未知。 |
| 13 部署依赖与 CUDA / NPU | Python2.7/PyTorch0.4.1及自定义RoI/RoD算子；fc_layers()引用被注释掉的dropout，源码会NameError。NPU未验证。 |
| 14 代码与权重许可 | [此repo代码MIT](https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch/blob/master/LICENSE)；权重/数据独立条款未确认。 |
| 15 作者 / 官方 benchmark | journal具体指标未逐表核验，登记unknown；不转用conference分数证明该checkpoint。 |
| 16 AIC 适配方式 | 主要借共享多尺度/区域读出；先修复与权重对齐，再做Frozen feature+native crop head；不优先全量FT。 |
| 17 主要风险与停止条件 | 构建错误、版本/权重身份、旧算子与目标错域；到AIC候选族后无效即关闭，不靠扫变体保留。 |

### M18 AutoFlip

参数桶：待核定。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | MediaPipe AutoFlip视频重构图系统；固定当前autoflip_graph及face/SSDLite子图。 |
| 02 论文 | 无单一论文；[Google作者技术说明：AutoFlip, 2020](https://research.google/blog/autoflip-an-open-source-framework-for-intelligent-video-reframing/)。 |
| 03 原始开源地址 | [官方代码](https://github.com/google-ai-edge/mediapipe/tree/master/mediapipe/examples/desktop/autoflip)；[原官方文档](https://chuoling.github.io/mediapipe/solutions/autoflip.html)。 |
| 04 权重与实际取得等级 | [官方external_files.bzl](https://github.com/google-ai-edge/mediapipe/blob/master/third_party/external_files.bzl)列face_detection_full_range_sparse与ssdlite_object_detection实体URL和SHA；仅W1，未probe/download。 |
| 05 参数量：encoder / head / full | 框架无单一参数量；路径优化0，但运行的face detector+SSDLite及任何新增observer全计。总元素数本轮待核。 |
| 06 训练数据、标注与目标函数 | 框架本身无统一训练集/损失；所配置detectors各有训练谱系，本轮未核其精确数据，不由标签名推断。 |
| 07 输入 | 完整视频；图先宽480，200000μs约5Hz取检测帧，full-rate做shot；face192²/object320²。 |
| 08 输出 | 重构尺寸视频与镜头相机路径；默认可padding，须改为AIC合法原图bbox。 |
| 09 真实时间信息 | 镜头缓冲/未来上下文，静止/摇移/跟踪选择和路径拟合；不是严格低延迟逐帧tracker。 |
| 10 空间主体定位 | 检测主体/人脸区域后优化窗口；检测与主体选择权重仍不同于native crop GT。 |
| 11 预训练任务学到了什么 | 组合已有检测能力与手工相机策略，不提供高光偏好预训练。 |
| 12 计算成本与测量条件 | 未报告可直接对齐AIC的统一FLOPs/FPS/显存；默认max_scene_size600是缓冲参数。不得把路径求解0参数当0运行成本。 |
| 13 部署依赖与 CUDA / NPU | Bazel/OpenCV/TFLite CPU/MediaPipe calculators；本轮未构建。默认padding和resize需要合同适配，NPU未验证。 |
| 14 代码与权重许可 | [MediaPipe代码Apache-2.0](https://github.com/google-ai-edge/mediapipe/blob/master/LICENSE)；具体detector权重/数据许可未独立核定。 |
| 15 作者 / 官方 benchmark | 作者展示和设计解释，无本轮可复现的统一AIC类crop GT benchmark；不填想象分数。 |
| 16 AIC 适配方式 | 算法集成：冻结点来源、镜头与时间mask，仅比较decoder。先做identity replay，避免R86式点/映射混杂。 |
| 17 主要风险与停止条件 | 平滑不代表中心正确、letterbox不合AICcrop合同；只改善稳定视觉不改善crop效用则停止。 |

### M19 YOLOX-Nano

参数桶：≤20M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | YOLOX-Nano，作者release0.1.1rc0；当前COCO val25.8版本，旧25.3版本另列。 |
| 02 论文 | [YOLOX: Exceeding YOLO Series in 2021, 2021](https://arxiv.org/abs/2107.08430)。 |
| 03 原始开源地址 | [作者代码/model table](https://github.com/Megvii-BaseDetection/YOLOX)；[nano config](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/exps/default/yolox_nano.py)；[预处理](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/data/data_augment.py)。 |
| 04 权重与实际取得等级 | [作者release权重](https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.pth) W2，200/octet-stream/7,694,953 bytes；未load。[HEAD证据](evidence/spatial_weight_access_audit.json)。 |
| 05 参数量：encoder / head / full | 作者full约0.91M；backbone/decoupled head未单独实数。文件含多少其他state未load，不能由bytes反算。 |
| 06 训练数据、标注与目标函数 | COCO80类bbox/class监督；anchor-free、decoupled分类/回归/objectness、SimOTA分配。 |
| 07 输入 | 416² letterbox。当前ValTransform(legacy=False)：OpenCV BGR、float32 0–255、padding114；legacy才RGB、/255和ImageNet normalize。 |
| 08 输出 | bbox、类别、置信度。 |
| 09 真实时间信息 | 无跨帧状态；video demo仍逐帧检测。 |
| 10 空间主体定位 | 多类对象范围，支持人/宠物/车等覆盖；不自动选事件主体或crop中心。 |
| 11 预训练任务学到了什么 | 类别与对象几何检测，不是高光价值。 |
| 12 计算成本与测量条件 | 官方Nano1.08GFLOPs/416²；没有Nano独立V100时延，不能借YOLOX-S速度。显存/端到端本轮未测。 |
| 13 部署依赖与 CUDA / NPU | PyTorch，NMS及导出后处理；CPU/CUDA/NPU尚未本轮验证。 |
| 14 代码与权重许可 | [代码Apache-2.0](https://github.com/Megvii-BaseDetection/YOLOX/blob/main/LICENSE)；release权重独立许可范围未确认，COCO素材条款另记。 |
| 15 作者 / 官方 benchmark | 当前作者表COCOval AP25.8；原论文/旧权重25.3，不能混用。 |
| 16 AIC 适配方式 | Frozen检测框→候选对象覆盖/裁断/多主体冲突低维特征→native crop head。比直接取最大置信框中心可归因。 |
| 17 主要风险与停止条件 | 未覆盖类、遮挡拥挤、主角选择、输入尺度错误；真框特征不胜置乱/几何对照即停止；首次接入不同时换tracker。 |

## 6. 轻语义、视频表示与现有AIC视觉链

### M20 MobileNetV3-Small

参数桶：≤20M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | Torchvision mobilenet_v3_small，IMAGENET1K_V1，含1000类分类器。 |
| 02 论文 | [Searching for MobileNetV3, ICCV2019](https://arxiv.org/abs/1905.02244)。 |
| 03 原始开源地址 | [Torchvision官方模型页](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_small.html)；[源码](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py)。 |
| 04 权重与实际取得等级 | [官方V1权重](https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth) W2：32-byte Range206、PK头、10,306,551 bytes；无load。 |
| 05 参数量：encoder / head / full | 官方full=2,542,856，含分类器；encoder/head未独立普查，移除分类器后必须按实际导出重计。 |
| 06 训练数据、标注与目标函数 | ImageNet1K类别标注，分类监督；当前官方V1训练recipe与论文搜索过程区分。 |
| 07 输入 | 单RGB、224 crop；使用该Weights枚举的固定resize/normalize。 |
| 08 输出 | 1000类logits，或去掉分类器的图像/空间feature。 |
| 09 真实时间信息 | 无时序；多帧独立平均不是动作encoder。 |
| 10 空间主体定位 | 有空间feature，无主体定位/分割/crop head。 |
| 11 预训练任务学到了什么 | 学类别、物体外观和纹理等隐式语义，不是无语义低层算子。 |
| 12 计算成本与测量条件 | 官方0.06GFLOPs/224；本轮FPS、峰值显存、解码端到端未测。 |
| 13 部署依赖与 CUDA / NPU | PyTorch/Torchvision，depthwise conv/SE/hard-swish；本轮无实际CPU/CUDA/NPU运行。 |
| 14 代码与权重许可 | [Torchvision代码BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE)；[权重总说明](https://docs.pytorch.org/vision/stable/models.html)要求考虑训练数据条款，不能把代码许可当所有来源图像授权。 |
| 15 作者 / 官方 benchmark | 官方ImageNet1K top1=67.668；分类准确率不是AIC得分。 |
| 16 AIC 适配方式 | Frozen小骨干/真实crop像素编码容量控制；需要native crop或合法时间监督的新头；KD可作为后续。 |
| 17 主要风险与停止条件 | 便宜不等于懂GIF偏好；全图/真实crop与候选次数会改变成本。与DINO不并行扫所有变体，先固定机制。 |

### M21 MobileNetV3-Large

参数桶：≤20M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | Torchvision mobilenet_v3_large，IMAGENET1K_V2；与V1权重分开。 |
| 02 论文 | [Searching for MobileNetV3, ICCV2019](https://arxiv.org/abs/1905.02244)；V2为后续官方recipe。 |
| 03 原始开源地址 | [官方Large模型页](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.mobilenet_v3_large.html)；[源码](https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py)。 |
| 04 权重与实际取得等级 | [官方V2权重](https://download.pytorch.org/models/mobilenet_v3_large-5c1a4163.pth) W2，32-byte Range206、PK头、22,132,113 bytes；content-type不标准但非HTML，未load。 |
| 05 参数量：encoder / head / full | 官方full=5,483,032，含分类器；encoder/head未分别复算。 |
| 06 训练数据、标注与目标函数 | ImageNet1K分类监督，改进训练recipe；不是时序模型。 |
| 07 输入 | 单RGB，官方V2 resize232→crop224与配套normalize。 |
| 08 输出 | 分类logit/可提取空间feature。 |
| 09 真实时间信息 | 无真实时序。 |
| 10 空间主体定位 | 隐式对象表征，无定位head。 |
| 11 预训练任务学到了什么 | 单图类别与外观语义。 |
| 12 计算成本与测量条件 | 官方0.22GFLOPs；FPS/峰值显存与本机端到端未测。 |
| 13 部署依赖与 CUDA / NPU | PyTorch/Torchvision、depthwise/SE/hard-swish；NPU尚未验证。 |
| 14 代码与权重许可 | [代码BSD-3-Clause](https://github.com/pytorch/vision/blob/main/LICENSE)；权重/训练数据范围按官方总说明另核。 |
| 15 作者 / 官方 benchmark | V2 ImageNet top1=75.274；V1为74.042，不能换用数据。 |
| 16 AIC 适配方式 | Frozen feature或小范围FT；作为Small容量升级的条件性对照，而非默认第二条主攻。 |
| 17 主要风险与停止条件 | Small/large差异可由容量与recipe解释，不能叫新观察路径；真实任务收益不稳定即停止。 |

### M22 DeiT-S non-distilled

参数桶：20–120M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | deit_small_patch16_224，非distilled；不混distilled token/head版本。 |
| 02 论文 | [Training data-efficient image transformers & distillation through attention, ICML2021](https://proceedings.mlr.press/v139/touvron21a/touvron21a.pdf)。 |
| 03 原始开源地址 | [作者模型表](https://github.com/facebookresearch/deit/blob/main/README_deit.md)；[作者仓库](https://github.com/facebookresearch/deit)。 |
| 04 权重与实际取得等级 | [作者DeiT-S权重](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) W2，32-byte206/PK头/88,251,143 bytes；未load。 |
| 05 参数量：encoder / head / full | 作者约22M full、含分类head；本轮未逐参数枚举。384维/12层/6heads/patch16配置不等于精确参数清单。 |
| 06 训练数据、标注与目标函数 | ImageNet1K监督分类；当前非distilled权重不能借蒸馏版81.2成绩。 |
| 07 输入 | 单RGB224²/16²patch，固定ImageNet预处理。 |
| 08 输出 | 图像token/分类logit。 |
| 09 真实时间信息 | 无跨帧时序。 |
| 10 空间主体定位 | 空间patch表征，无显式主体mask或crop目标。 |
| 11 预训练任务学到了什么 | 对象类别与全图关系的静态语义。 |
| 12 计算成本与测量条件 | 论文V10016GB、224²、最大可容纳batch、30次平均940.4images/s；不是batch1或视频端到端。显存峰值非本轮实测。 |
| 13 部署依赖与 CUDA / NPU | PyTorch/timm，attention；旧作者入口要求timm0.3.2，移植要验证权重与预处理。NPU未验证。 |
| 14 代码与权重许可 | [作者repo Apache-2.0](https://github.com/facebookresearch/deit/blob/main/LICENSE)；权重随项目发布，训练图像权利不自动同许可。 |
| 15 作者 / 官方 benchmark | 作者具体checkpoint ImageNet top1=79.854（表约79.9）；不混distilled-S。 |
| 16 AIC 适配方式 | Frozen静态语义控制；与DINO/现有V9固定候选读出比表征，必要时目标域FT。 |
| 17 主要风险与停止条件 | 与已有V9同属静态feature，换模型不保证信息增加；若优劣仅由训练头容量/像素预算解释则停止。 |

### M23 DINOv2-S / ViT-S14

参数桶：20–120M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | 作者无registers DINOv2-S；HF facebook/dinov2-small revision ed25f3a31f01632728cabb09d1542f84ab7b0056。 |
| 02 论文 | [DINOv2: Learning Robust Visual Features without Supervision, 2023](https://arxiv.org/abs/2304.07193)。 |
| 03 原始开源地址 | [作者代码](https://github.com/facebookresearch/dinov2)；[作者HF](https://huggingface.co/facebook/dinov2-small)。 |
| 04 权重与实际取得等级 | [作者pth](https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth)与[固定revision safetensors](https://huggingface.co/facebook/dinov2-small/resolve/ed25f3a31f01632728cabb09d1542f84ab7b0056/model.safetensors)均W2；后者仅读23,656-byte头，不读张量。 |
| 05 参数量：encoder / head / full | 完整无分类head encoder=22,056,576：embedding753,024+transformer21,302,784+finalnorm768；head=0。[header审计](evidence/semantic_tensor_header_counts_20261019.json)，未load。作者约称21M不替代精确数。 |
| 06 训练数据、标注与目标函数 | LVD-142M图像，自监督/大teacher蒸馏；学习全局与patch表征，不需要AIC标签预训练。 |
| 07 输入 | 单RGB，patch14；输入分辨率、中心裁剪/resize与patch网格映射必须冻结。 |
| 08 输出 | CLS/patch embedding；可接分类、分割、深度或crop读出。 |
| 09 真实时间信息 | 无原生时间attention；序列需另加轻量头。 |
| 10 空间主体定位 | 稠密空间token有潜力支持主体/结构定位，但没有预训练native crop head。 |
| 11 预训练任务学到了什么 | 从自监督数据学得丰富视觉语义/结构；不能称完全无语义。 |
| 12 计算成本与测量条件 | 本轮未取得可与AIC输入/硬件匹配的FPS/峰值显存，不由22M推速度；真实crop编码次数决定主要成本。 |
| 13 部署依赖与 CUDA / NPU | PyTorch/attention及作者可选优化实现；NPU/910A、FP16/BF16、形状导出尚未验证。 |
| 14 代码与权重许可 | 作者明确[code and weights Apache-2.0](https://github.com/facebookresearch/dinov2)；LVD原图权利不随权重许可转移。 |
| 15 作者 / 官方 benchmark | 作者小模型ImageNet线性探针81.1；这是固定特征线性评估，不与fully supervised top1混写。 |
| 16 AIC 适配方式 | 首选Frozen稠密语义对照；同native crop目标/同候选/同容量读出比较V9。若主张真实crop新信息，应在同一encoder内对全图ROI vs真实crop重编码。 |
| 17 主要风险与停止条件 | 新骨干与新观察同时改会混杂；源重叠、输入缩放、局部token细节和教师训练谱系需记录。无源级独立增益即停止。 |

### M24 VideoMAE-Base K400 fine-tuned

参数桶：20–120M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | 作者MCG-NJU/videomae-base-finetuned-kinetics；revision488eb9a0565f257b32866000305c8178965eb9f6。 |
| 02 论文 | [VideoMAE: Masked Autoencoders Are Data-Efficient Learners for Self-Supervised Video Pre-Training, NeurIPS2022](https://arxiv.org/html/2203.12602v3)。 |
| 03 原始开源地址 | [作者repo](https://github.com/MCG-NJU/VideoMAE)；[model zoo](https://github.com/MCG-NJU/VideoMAE/blob/main/MODEL_ZOO.md)；[作者K400 HF](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics)。 |
| 04 权重与实际取得等级 | [固定revision权重](https://huggingface.co/MCG-NJU/videomae-base-finetuned-kinetics/resolve/488eb9a0565f257b32866000305c8178965eb9f6/model.safetensors) W2，346,161,616 bytes，仅22,416-byte头；未张量load。 |
| 05 参数量：encoder / head / full | full=86,534,800=encoder86,225,664+fc_norm1,536+classifier307,600。[header证据](evidence/semantic_tensor_header_counts_20261019.json)。预训练重建包94,210,944含decoder，不能互换。 |
| 06 训练数据、标注与目标函数 | Kinetics400视频：90% tube masking的像素重建预训练，再动作类别监督fine-tuning。 |
| 07 输入 | 16帧RGB、224²、2×16×16 tubelet；连续帧采样stride与clip时间定位需记录。 |
| 08 输出 | 短clip embedding或400类动作logit。 |
| 09 真实时间信息 | 真实联合时空attention，区别于独立图片平均。 |
| 10 空间主体定位 | 有spatiotemporal token，无主体检测/crop窗口head。 |
| 11 预训练任务学到了什么 | 遮挡重建与动作语义，可能捕捉时序但不直接学GIF偏好。 |
| 12 计算成本与测量条件 | 论文单clip约180GFLOPs；测试180×5temporal×3spatial≈2700G，不能误写180G整视频。FPS/峰值显存本轮未测。 |
| 13 部署依赖与 CUDA / NPU | PyTorch/transformer/tubelet conv，多帧缓存；NPU算子与静态T尚未验证。 |
| 14 代码与权重许可 | [repo](https://github.com/MCG-NJU/VideoMAE)主体CC-BY-NC4.0（部分依赖另许），作者HF权重同CC-BY-NC4.0；不当Apache模型。 |
| 15 作者 / 官方 benchmark | K400 top1=81.5，224²/16帧/5×3测试views；不是单clip或AIC结果。 |
| 16 AIC 适配方式 | Frozen真实时序对照，必要时目标域末层FT/KD。对连续帧、中心帧重复、同组乱序、静态平均做匹配像素预算。 |
| 17 主要风险与停止条件 | NC条款、高算量、动作目标错域、片段采样带入标签位置；若时序优势被更多画面/位置对照复现则停止。 |

### M25 HL-CLIP / Highlight-CLIP

参数桶：120–500M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | 作者HL-CLIP的CLIPFT，基础CLIP ViT-B/32；不能把旧Moment-DETR演示head当该模型。 |
| 02 论文 | [Unleash the Potential of CLIP for Video Highlight Detection, 2024](https://arxiv.org/html/2404.01745v1)。 |
| 03 原始开源地址 | [作者项目](https://dhk1349.github.io/hlclip/)；[作者代码](https://github.com/dhk1349/HL-CLIP)；[模型](https://github.com/dhk1349/HL-CLIP/blob/3794871e9e2499a537e862ba0042ab8a59aa4a2a/moment_detr/clip_ft.py)；[实际训练](https://github.com/dhk1349/HL-CLIP/blob/3794871e9e2499a537e862ba0042ab8a59aa4a2a/moment_detr/train.py)。 |
| 04 权重与实际取得等级 | HL微调权重W0：本轮未发现可确认发布URL。[基础OpenAI CLIP](https://github.com/openai/CLIP) ViT-B/32权重W2，但不能提升HL等级。仓库16.95MB旧Moment-DETR演示ckpt不算HL权重。 |
| 05 参数量：encoder / head / full | 基础完整CLIP=151,277,313：vision87,849,216+text63,428,096+scale1；作者类另注册286,965未用参数，保守full151,564,278。仅末层训练不能只计末层；无load。 |
| 06 训练数据、标注与目标函数 | QVHighlights query-saliency监督；实际train_epoch用MSE(outputs.saliency_scores, labels)，不是继承Moment-DETR criterion。微调图/文末两层等；基础CLIP先做图文对比预训练。 |
| 07 输入 | RGB帧与query文本；作者forward硬编码把文本扩到75帧，需适配AIC无query设置。 |
| 08 输出 | 逐帧图文cosine；评估邻域0.3/0.5/0.3池化，系数和1.1，不是概率校准。 |
| 09 真实时间信息 | encoder没有跨帧神经attention；只有邻帧分数池化，不能称动作时序模型。 |
| 10 空间主体定位 | CLIP全图语义相似度，不输出主体框或native crop head。 |
| 11 预训练任务学到了什么 | 把静态图文表征适配到查询相关高光偏好；不是无查询AIC同目标。 |
| 12 计算成本与测量条件 | 论文未报告可复用吞吐/显存；作者实现CLIP.float()，不能用小head文件bytes当运行成本。 |
| 13 部署依赖与 CUDA / NPU | PyTorch/OpenAI CLIP、CUDA硬编码、75帧shape与未整理代码；NPU未验证。 |
| 14 代码与权重许可 | HL repo MIT，基础CLIP代码MIT；HL微调权重无已确认独立许可，QV标注NC-SA4.0另记。 |
| 15 作者 / 官方 benchmark | 论文四seed QVval mAP42.40±.05/HIT1 72.42±.25；三seed另一表不同，不混汇总。 |
| 16 AIC 适配方式 | 借目标域encoder适配或KD思想；Frozen/head增容/末层FT必须同目标对照。不直接叠加当前TimeLens系统。 |
| 17 主要风险与停止条件 | 完整作者类叠加当前8,854,531,425为9,006,095,703，超9B；仅删未用286,965后仍9,005,808,738超限。须替换实际组件或仅部署学生；无query/未知槽不能照抄QV负标签。 |

### M26 既有 V9 / SigLIP2 NaFlex tower

参数桶：20–120M。

| 字段 | 核验记录 |
|---|---|
| 01 模型名称与版本 | AIC V9：从LiquidAI LFM2-VL-450M提取的SigLIP2 NaFlex base vision tower + B3/R70 head + YuNet；不是完整450M VLM。 |
| 02 论文 | [SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features, 2025](https://arxiv.org/abs/2502.14786)；实际权重来源以[LiquidAI模型卡](https://huggingface.co/LiquidAI/LFM2-VL-450M)为准，不能只按基础论文追溯。 |
| 03 原始开源地址 | [上游HF](https://huggingface.co/LiquidAI/LFM2-VL-450M)；[实际config](https://huggingface.co/LiquidAI/LFM2-VL-450M/resolve/main/config.json)；[AIC R70入口](https://github.com/lujunlin-cn/AIC/blob/5251193a9e07fd682008bb7b2a49aef60ee0cf3f/scripts/r70_spatial_utility_pilot.py)。 |
| 04 权重与实际取得等级 | 历史AIC有清单与运行报告；本轮未重取当前V9导出权重、不是新增W3。上游HF可读不证明当前导出字节身份。 |
| 05 参数量：encoder / head / full | 历史tower85,843,200；B3/R70 head静态1,511,425；历史YuNet53,104；空间full87,407,729。[head源码审计](evidence/aic_head_parameter_audit.json)。历史/静态/实装三种证据不混。 |
| 06 训练数据、标注与目标函数 | 上游图文预训练/蒸馏等形成视觉语义；AIC B3/R70用公共native crop IoU效用学习。上游具体继续训练谱系按LiquidAI卡，不假定原始SigLIP权重完全相同。 |
| 07 输入 | 独立RGB图，768维/12层/patch16；R70单tile、min=max1、max_image_tokens256。 |
| 08 输出 | 全图grid与129候选2305维读出；最终crop point/window。 |
| 09 真实时间信息 | V9 encoder无时间模型；后续hold/EMA/shot reset是另外的坐标决策路径。 |
| 10 空间主体定位 | 空间token/candidate utility有直接crop能力；全图ROI池化没有重新观察crop原始像素。 |
| 11 预训练任务学到了什么 | 先学静态视觉语义，再用native crop监督学候选决策；不是纯手工低层视觉。 |
| 12 计算成本与测量条件 | 上游完整VLM广告不能当V9成本；本轮未重新计时/测显存或实际权重。 |
| 13 部署依赖与 CUDA / NPU | 自定义视觉塔提取、候选池化与PyTorch head；已有历史GPU运行，本轮未复现实装或910A。 |
| 14 代码与权重许可 | [上游权重LFM Open License v1.0](https://huggingface.co/LiquidAI/LFM2-VL-450M/blob/main/LICENSE)，不能因SigLIP名称自动视作Apache；自定义AIC代码许可未独立单列。 |
| 15 作者 / 官方 benchmark | AIC已报告R70=36.42（固定R54时间），R71=36.43为另一个预算链路；没有把空间模块单独官方评分。 |
| 16 AIC 适配方式 | 首选冻结已有空间锚；轻量完整系统可去TimeLens后保留V9+新≤1M时间头，但删组件后的质量须重新验证。DINO是条件性替代表征对照。 |
| 17 主要风险与停止条件 | 历史权重/point身份缺失、已暴露T5源、decoder差异可制造假增益；先恢复identity/manifest，不因上游450M名称错误记账。 |

## 7. AIC参数账本：替换组件和附加组件必须分开

下表是研究预算推算。现役数值来自历史[AIC R86/R87报告](../r86_r87_observation_and_transfer_20261019/summary.md)及部署清单；V9/R70 head来自本轮[源码构造算术](evidence/aic_head_parameter_audit.json)。新模型数值来自上面的特定来源。**这些组合尚未完整load、去重和导出，因此不是新的合规验收。**

| 实际/拟议组件组合 | 总参数 | 证据与含义 |
|---|---:|---|
| 历史TimeLens2 8,767,123,696 + V9空间链87,407,729 | 8,854,531,425 | 现役依赖链；冻结和缓存不免除生成组件 |
| 9B减现役链 | 145,468,575 | 可用参数余量，不是GPU显存余量 |
| 现役 + U²-NetP 1,131,181 | 8,855,662,606 | 条件性附加预算；空间收益未证明 |
| 现役 + DINOv2-S 22,056,576 | 8,876,588,001 | 条件性附加预算；DINO不能免费复用V9身份 |
| 现役 + VideoMAE分类包86,534,800 | 8,941,066,225 | 尚在9B内，但计算/许可证/时序任务均需独立验证 |
| 现役 + 完整HL-CLIP作者类151,564,278 | **9,006,095,703** | **超过9B**；只微调末层不降低完整计数 |
| 现役 + 删去286,965未用参数后的完整CLIP151,277,313 | **9,005,808,738** | **仍超过9B**；须实际替换组件或仅部署学生 |
| 删除TimeLens2；保留V9 tower85,843,200 + head1,511,425 + YuNet53,104 | 87,407,729 | 历史+源码静态清单，未核当前完整资产 |
| 上行 + ≤1M新时间head + 0学习参数的传统运动/镜头算法 | **≤88,407,729** | 首选完整轻链预算；不能继续依赖TimeLens分数缓存 |
| 上行若H1通过，再加U²-NetP | **≤89,538,910** | 条件性轻链；所有模块与实际权重必须重新合规计数 |

同一实例被多个函数调用只计一次；若运行两套不同权重encoder则都计。LoRA是否合并、保留何种分类/重建head、训练辅助网络是否实际在推理执行，都须以导出的推理图为准。不能把训练GTEncoder加在IS-Net上，也不能在仍用它推理时把它从账本删去。轻链是否足够有效，必须由真实目标域留出实验回答。[迁移实施与监督合同](aic_transfer_learning_plan.md)。

## 8. 可复核证据与本轮访问边界

| 文件 | 记录内容 |
|---|---|
| [weight_access_audit.json](evidence/weight_access_audit.json) / [aic_visual_weight_access_probe.py](evidence/aic_visual_weight_access_probe.py) | 运动/镜头模型作者端点的小Range或HEAD访问；含TransNet LFS实体区别 |
| [kts_access_audit.json](evidence/kts_access_audit.json) | KTS作者归档入口的访问边界 |
| [primary_source_fetch_manifest.json](evidence/primary_source_fetch_manifest.json) | 运动/镜头论文与作者来源清单 |
| [spatial_weight_access_audit.json](evidence/spatial_weight_access_audit.json) / [spatial_weight_access_probe.py](evidence/spatial_weight_access_probe.py) | U²-NetP、UNISAL、YOLOX-Nano作者权重HEAD；响应体读取0字节 |
| [spatial_weight_heads](evidence/spatial_weight_heads/) | HEAD响应记录；交付的YOLOX临时签名Location已移除签名URL，仅留重定向host，**不是未经改动的完整响应头原件** |
| [spatial_static_params/parameter_counts.json](evidence/spatial_static_params/parameter_counts.json) / [count_declared_params.py](evidence/spatial_static_params/count_declared_params.py) / [sources.json](evidence/spatial_static_params/sources.json) | U²-Net/U²-NetP/IS-Net精确作者Git blob上的AST算术；BN buffers单列，无torch、无权重、无forward |
| [semantic_model_access_20261019.json](evidence/semantic_model_access_20261019.json) | 语义模型初轮URL及小Range响应记录 |
| [semantic_weight_access_probe.py](evidence/semantic_weight_access_probe.py) / [semantic_weight_access_probe_result.json](evidence/semantic_weight_access_probe_result.json) | 可重执行的7个语义权重端点探测；均为W2，非W3 |
| [semantic_tensor_header_counts_20261019.json](evidence/semantic_tensor_header_counts_20261019.json) | DINOv2/VideoMAE张量header及HL类参数账本来源 |
| [dinov2_small_tensor_header.json](evidence/dinov2_small_tensor_header.json) / [videomae_k400_tensor_header.json](evidence/videomae_k400_tensor_header.json) | 实际读到的safetensors元信息；没有读取全部tensor payload |
| [semantic_source_metadata_20261019.json](evidence/semantic_source_metadata_20261019.json) | 作者代码/模型卡/许可信息的URL、版本或SHA；临时原文路径不是交付依赖 |
| [aic_head_parameter_audit.py](evidence/aic_head_parameter_audit.py) / [aic_head_parameter_audit.json](evidence/aic_head_parameter_audit.json) | 当前V9 Head与R70 make同构，均1,511,425；源码静态验证而非checkpoint验证 |
| [aic_r52_farneback_synthetic_audit.py](evidence/aic_r52_farneback_synthetic_audit.py) / [aic_r52_farneback_synthetic_audit.json](evidence/aic_r52_farneback_synthetic_audit.json) | 只复现已知合成平移上的输入尺度失败，不能据此宣称真实视频收益 |
| [current_environment_check.json](evidence/current_environment_check.json) | 本轮环境检查；历史GPU记录不等于本机可用设备 |

静态空间参数复算所需作者文件以[sources.json](evidence/spatial_static_params/sources.json)中的精确Git blob API URL取得，解码到临时目录后运行：

```bash
python reports/visual_feature_research_20261019/evidence/spatial_static_params/count_declared_params.py \
  --source-dir /tmp/aic_spatial_exact_sources \
  --output /tmp/aic_spatial_parameter_counts_recheck.json
```

脚本校验作者blob身份，再做受限AST构造算术；无法识别的构造不静默估算。交付保留审计脚本、来源SHA与结果，未将完整第三方模型源码或数据README复制入研究报告。

端点访问和轻量静态审计只解决“是否有可追溯候选”及“已知哪部分成本”两个问题。正式采用前仍需取得允许使用的确切权重，完成W3、输入/输出数值核验、CPU/CUDA/目标NPU的必要兼容性与成本检查，再按源隔离的任务实验判别收益。任何公共benchmark、短合成probe或历史checker记录均不能替代这一证据链。
