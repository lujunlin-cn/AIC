# R29 空间路线独立审查与最小诊断

2026-10-16。只读审查已有代码、公开 RetargetVid 031–034 小样本和 R28 输出；未访问远端/GPU、official test 或 protected14。本报告由并行子任务产生，未改 R28 原脚本、历史结果或主文档。

## 会改变行动的判断

完整 24 项 pilot 发现：Qwen 自由坐标系统 20 项输出中心，其中 y 轴 12/12 项全部是中心；公共源 034 两项 x 裁剪为零 IoU，但固定相同最大窗口和 129 格候选的 GT 辅助选择可达 0.948 / 0.967。**这支持“当前坐标系统会漏掉候选可达的、明显离中心的框”，削弱“此样本主要受候选最大窗口限制”的解释。它没有识别视觉编码器、坐标读出、prompt 或训练中哪一项负责，也不构成 B3 的反证。**

现有最直接证据支持“检查候选排序所依赖的信息与监督”，不能支持“换成更大的 Qwen 就能突破”。B3 已是监督候选效用模型，公共候选 oracle 与模型的差距不能区分视觉表示、读出、监督覆盖与迁移。Qwen 直接坐标 pilot 同时改变表示、读出和任务提示，只适合作为空间替代系统的工程诊断，不能单独证明 LFM 编码器不足。

R28 完整缓存回放证明部署后处理沿袭一致，不证明输入图像、processor、权重及训练合同正确。恢复真实 B3 权重是原始复现需要；已有 V7 不能冒充 B3。权重未恢复前，继续合法公共 pilot 与准备诊断数据可以并行进行。

主攻所需的第一项公平信息是：**同一原生几何、同一源帧、同一最大窗口候选上，B3 的候选是否好、模型是否选对、换观察或读出是否帮助。** 缺少此表前，不启动新大训练。

## R28 pilot 实现审查及已复算证据

`scripts/r28_spatial_public_pilot.py` 的两种几何是正确的：640×360 源图 1:3 目标取 120×360、沿 x 滑动；3:1 取 640×213.333…、沿 y 滑动。归一化 center 沿原图轴定位，裁剪中心投影到合法范围。图像 thumbnail 保持比例，因此这里没有 x/y 交换或错误拉伸的可见证据。

采样依据该源的 GT 行数，选择最接近固定 25/50/75% 的缓存命名帧，平局取较早帧；不是连续时间采样，也没有验证缓存 PNG 与原视频的像素/原帧绑定。记录 PNG SHA 能绑定本轮输入，不能替代这项来源验证。若源很短或缓存过稀，三个分位可能落到同帧；continuation 在生成前显式要求 24 个不同 source/frame/ratio key。GT 在这里决定视频长度和评测，不进入模型提示或 crop 决策。

旧 R28 测分使用 raw GT、连续面积；RetargetVid 原生 benchmark 是先 clamp 负 GT 坐标、预测边界坐标取整、再用 inclusive +1 IoU。两者应并列报告，不混成同一指标。主负责人取回的 24 项六人 GT 中，负坐标数为 0，因此缺 clamp 在本小样本不造成偏差。冻结 continuation 已完成：复用原 13 项、补 11 项，24/24 parse valid、exit 0；完整输出由不依赖 NumPy/原评分函数的 scalar 实现重算，保存指标最大差 **2.22e-16**。证据：`spatial_recompute.json`。旧 `spatial_partial_recompute.json` 仅保留为实现检查，不用于完整方法比较。

原 parser 会从 Markdown、数组或解释中的首个合法 JSON 对象提取 `center`，不要求整个回复严格只有一个对象。已有输出常为数组加代码块，被计为 parse valid 与原代码一致。这是确定的宽松解析合同，不应把 parse-valid 当作遵循整个提示或定位正确的证明。本轮 continuation 保留原 parser，避免看结果后改变算法。

新增 `scripts/r29_spatial_continue.py` 保持 R28 prompt、模型、processor、分位取帧、greedy/no-thinking/128 tokens 不变。生成前验证所有输入，再验证旧行的 key、PNG SHA、原 parse、crop 和 raw-GT 连续 IoU；旧行不重新生成。新目录保存完整预测、抽取 GT、资源、运行状态和三种测分口径，外层 20 min timeout 由主负责人执行。`scripts/r29_spatial_recompute.py` 供完整结果独立复算。其独立性是数值实现独立，不是样本独立确认。

实测本次补跑：加载模型 8.17 s、11 项新生成共 14.04 s、进程约 27.29 s，peak allocated 9,358,655,488 bytes；运行模型去重后 Parameter 元素 4,539,265,536。该数只描述本次 Qwen 模型，不代替任何包含其他组件的提交系统参数核算。完整预测 SHA256 为 `6ec82c74f7bfe9466c239dbe59be62dd8b0ede2dd3f56c0a191164eae36c8113`。

## 完整公共结果与 GT 辅助候选诊断

连续 IoU 是对同一行六位原人工标注分别计算后求均值，再每源平均、源间等权；不先平均框坐标。4 源各有 3 帧×2 比例，故这次所有行平均恰与源宏平均相同。RetargetVid native 口径的整体 Qwen/center 为 **0.601083 / 0.580877**，不与连续口径混报。

在看 Qwen 完整输出后、计算 oracle 之前，另用 `configs/R29_SPATIAL_GT_ASSISTED_DIAGNOSTIC.json` 固定候选数 129、最大窗口、合法 offset 等距含两端、六人 mean continuous IoU、源级汇总、tie-break 和可作主张的边界。它明确是 **exploratory GT-assisted**，不伪装成预先未观察 Qwen 结果的确认实验。`scripts/r29_spatial_oracle.py` 做 CPU 穷举；原 Qwen 回复也仅按最近合法 offset 投影到同 129 格，没有新推理或模型选择。

| 同样本 continuous IoU | center | Qwen 自由坐标 | Qwen 投影 129 格 | GT 辅助 129 格 oracle |
|---|---:|---:|---:|---:|
| 全部，4 源 / 24 项 | 0.579117 | 0.599645 | 0.600223 | 0.868315 |
| x 滑动，4 源 / 12 项 | 0.415057 | 0.456114 | 0.457268 | 0.854424 |
| y 滑动，4 源 / 12 项 | 0.743177 | 0.743177 | 0.743177 | 0.882205 |

这次投影没有造成可见的整体损失，反而 +0.000577；因此当前自由坐标系统的弱结果不能用“129 格过粗而连续坐标才够精确”解释。oracle 与 Qwen 的差距说明这些样本存在合法候选选择余地，**不说明**输入足够可观测、对应关系已完成原媒体 QC、收益可学习、B3 低于 Qwen、或能产生官方涨分。y 轴样本仍是横屏→更宽横裁，不是 native portrait→landscape。

| 既有缓存失败/成功例，均 x 滑动 | Qwen | center | oracle | 观察与边界 |
|---|---:|---:|---:|---|
| 034 f240 | 0.000000 | 0.000000 | 0.947777 | Qwen 回复 0.5；已有图像中人物偏左，最大窗可达 |
| 034 f360 | 0.000000 | 0.000000 | 0.966675 | 再次回复 0.5，oracle offset 比中心左移 134.06 px |
| 033 f420 | 0.388962 | 0.425128 | 0.885170 | Qwen 向右移过头，比中心还差 0.036166 |
| 033 f630 | 0.350742 | 0.416134 | 0.851383 | 同一源第二个类似失败；不是第二个独立源 |
| 032 f90 | 0.623048 | 0.210351 | 0.930596 | Qwen 右移有效，但仍没有选到此 GT 政策下的最佳候选 |

可检查 [contact sheet](spatial_failure_cases.png)：红为已有 Qwen 输出、蓝为中心、绿为 GT 辅助 oracle；**绿框不是某位人工标注者的真值**。原图为已有公共缓存，PNG SHA 与 GT 导出记录一致；原媒体断链，原帧名与原视频的绑定仍未重新验证。图像只帮助检查机制，不产生人工 GT。

竞争解释必须保留：prompt 给出的唯一数值示例是 `{"center": 0.5}`，20/24 中心输出可能受示例复制、坐标任务理解或读出接口影响，不能由其频率直接断言“Qwen 看不见主体”。四个非中心输出与 GT 的偏差也可能来自主体政策不同或坐标定位不准。下一小实验应在**另行划定的公共开发源**固定比较已冻结直接坐标接口、候选选择接口和已知目标坐标 probe；若它能定位明确指定的可见对象，却在构图任务仍选错，才更支持效用/政策错配；若两者均错，仍需区分 processor、空间表示和坐标读出。不能在这四源上反复调提示后宣称泛化。

本轮事实因此只改变一项行动：不把冻结通用 Qwen 直接坐标配方升级为提交候选，也不马上扩大模型训练；先恢复真正 B3 同帧输出，或取得有效的新 native 监督并做受限读出/接口对照。它不结束整个空间研究。

复现命令（本地 CPU，均 exit 0；环境及各文件 SHA 见 `spatial_cpu_evidence.json`）：

```bash
python3 scripts/r29_spatial_recompute.py --predictions reports/r29/spatial_completion/spatial_continuation/predictions.jsonl --sampled-gt reports/r29/spatial_completion/spatial_continuation/sampled_gt.json --output reports/r29/spatial_recompute.json
python3 scripts/r29_spatial_oracle.py --protocol configs/R29_SPATIAL_GT_ASSISTED_DIAGNOSTIC.json --predictions reports/r29/spatial_completion/spatial_continuation/predictions.jsonl --sampled-gt reports/r29/spatial_completion/spatial_continuation/sampled_gt.json --output reports/r29/spatial_oracle.json
python3 scripts/r29_spatial_contact_sheet.py --oracle reports/r29/spatial_oracle.json --sampled-gt reports/r29/spatial_completion/spatial_continuation/sampled_gt.json --image-root artifacts/r29/public_evidence/public_rv_images --output reports/r29/spatial_failure_cases.png
```

## 不能由代码审查直接升级为真实缺陷的两项线索

1. `v8_s_official_points.py` 先把 NaFlex grid 截成偶数尺寸，但后面构造 mask 时仍用原始 fh/fw。若任一轴是奇数，mask 的 token 数与 flat 不同，矩阵乘会失败。这是条件性的代码缺陷；尚未核实实际 processor 会产生奇数空间形状，不能声称现役预测受此影响。缓存路径全部成功也不能证明该分支一定可达或不可达。最小核验是读取一组**公共**实际 `spatial_shapes/grid.shape`，不需重跑官方集。
2. B3 的 129 格候选以 floor/ceil 硬 token mask 做 win/out 均值。多个候选可能有相同 mask，候选数不等于独立空间观察数。不过候选的 position 输入仍可与内容交互，token 也可能编码 token 内的位置；相同 mask **不构成**“只能用位置先验”或“无法亚 token 定位”的数学证明。应在实际 grid 上计数，再检验同表征受限读出与额外局部观察。R17 已有 fractional pooling 负结果，不凭这一观察原样重开该配方。

## 能区分解释的最小干预

先固定同一批有人工 crop GT 的公开源/帧、目标 ratio、图像方向与输入长边。主表按 native x/y 分层，先每源平均再宏平均；所有模型用同一帧表，六人标签取期望 IoU，不平均坐标。B3 与其他模型比较前必须恢复权重/processor 身份并对其公共预测留 SHA。

| 对照 | 固定项与干预 | 支持的区分 | 不能推出 |
|---|---|---|---|
| 中心、B3、129 候选 oracle | 相同图像/候选/GT；只换选择器 | 候选覆盖不足，或候选内选择有差距 | oracle gap 是可学收益；LFM 编码器是原因 |
| B3 当前 head 与受限新 readout | 同一冻结 grid、候选、监督与源级划分；只换有限容量读出 | 当前读出/优化是否不足 | readout 无提升证明 grid 无信息 |
| 固定 readout，实际局部观察与原 pooled feature | 同源监督、候选及训练预算；改变可见局部/布局信息 | 原 pooling/分辨率是否限制候选判别 | 任何结果代表更大 VLM 容量作用 |
| 同关键帧 raw points、hold、hold+EMA | 同点、同帧 GT、同 reset；只改变传播 | 运动滞后和传播影响 | 较平滑视频一定更符合官方 IoU |
| Qwen 自由坐标与投影到同 129 格 | 同一原回复、图像/GT；只改变允许决策集合 | 连续坐标精度是否主导差异 | Qwen 比 B3 表示更强 |

本轮已把输出自由坐标按 crop offset 投影到同 129 格，不改 prompt、不新推理，结果见上表；这去掉当前比较中候选连续精度的主要混杂。但 Qwen 与 B3 仍是两个系统的比较，不能隔离编码器。提示 Qwen 选候选索引会改变任务接口，若做必须单独预注册；不能看本轮失败例反复优化同一池提示。

对公共逐帧传播诊断，先从少数源保存全部原点与 reset，再对 raw/hold/EMA 统一重计分。现有 R28 只有孤立 12 帧、没有相邻时序观察，因此**无法**用它量化平滑滞后，也不应再次以旧 V7 插值结果推断现 B3。若没有完整时序 GT，则后处理项保持未知。

若同一候选 oracle 好、B3 差、有限新 head 能跨源改善，优先改读出/监督而非主干。若新 head 训练也不能拟合，而更细观察在同样标签下能拟合并跨源改善，支持信息/表征方向。若训练能拟合但跨源失败，优先检验来源、几何与标注政策。若候选 oracle 已差，则改候选/尺度的必要性优先于换 encoder。以上各项都需要实际对照，不由目前 oracle 表直接裁决。

## 最小 native 人工监督协议

旧 RV 横屏源与旋转增强无法替代真实 portrait 内容。旋转既改变几何，又把人物/物体旋转，不能当作保留自然方向的原生竖屏输入。LIVE 提供真实横屏转竖裁 x 轴监督，但不提供 native portrait→landscape 的 y 轴证据。缺口首先是原始方向、主体关系及标注政策覆盖，不是样本帧数本身。

分阶段推进，以诊断和总体估计区分：

- 第一阶段 12 个合法公开源：6 个 native portrait、6 个 landscape，各 4 帧，2 位独立人工标注者。标注前另冻结每源实际W/H、目标比例及算出的自由轴：landscape→9:16、portrait→16:9，验证6个x轴、6个y轴；原视频方向不能代替自由轴覆盖。源级预先划分开发与确认，历史已暴露源只用于开发；source provenance、许可证/下载入口、原视频 SHA、rotation、解码 frame/PTS、帧 PNG SHA 均记录。这个规模用于发现合同和机制问题，不估计赛事域占比，也不能承诺统计显著。
- 每源均匀取 2 帧，另外 2 帧按**预先定义、不看模型误差**的内容条件取：单主体、多人互动/主体跨边缘/状态变化；没有该内容记 missing，不用教师补造。保存约 2 s 上下文给标注者，记录“只看单帧”和“上下文改变选择”两种标志。若目的是挑战失败机制，应明确这是诊断采样，不估总体分布收益。
- 第一轮只标最大合法窗口及自由轴位置，目标比例按原始方向固定为 9:16 / 16:9；允许“无可辨识主体”“多种合理构图”和 abstain。另一栏允许变尺度 crop，用于检验最大窗口约束，而不是把两者混成同一 GT。
- 两位标注者互盲，不看模型框、教师框或另一人的框；先写应保留的对象/交互，再选 crop。不同答案保留为标注分布，以每个合法候选对两人 IoU 的平均做决策效用。不要把两框均值当唯一真值；这种均值可能落在两主体之间。
- 人工分歧、GT 方向/ratio/合法性和是否看完上下文先 QC。机器教师只做待核建议，永不写 `human_gt=true`。若没有真实人工参与，标注状态为待完成，不能宣称空间确认集已建立。
- 下一阶段是否扩至此前建议的每方向至少 24 源、每源 6 帧，取决于第一阶段发现的误差及所需源级不确定性；重复帧和标注者不会增加独立源数。若需量化联合 F，必须在**相同视频**补完整时间 GT 和命中帧 crop GT；单独空间集不能换算官方涨分。

手工预算按实际试标一源耗时外推；在有试标日志前不把“48 帧×2人”写成已知小时数。没有人工标注通道不阻塞恢复公共模型对照、目标理论或合法公共时间分支。

## 当前能推进与停止的边界

冻结 Qwen pilot、同输入独立数值复算、固定候选 oracle、离散化对照和缓存失败例已完成；这些是开发诊断，没有独立新源确认、B3 同帧比较或官方联合 F。最小 native 人工协议已写明，但当前没有新人工标签，不宣称确认集成立。停止凭历史 oracle gap 直接认定 encoder 容量、继续扫中心 keep、或将教师坐标当人工监督。是否选择新表示应由上述最小干预改变，而非模型已下载或已有训练投入。
