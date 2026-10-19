# AIC R85 接管结果、可复现实验与执行缺口

日期：2026-10-19。最新研究基线：main `41f5e9113b92e22e00367ce266c76527872eecf4`。

**主攻已确定并实现：共享冻结 V9 视觉塔的真实候选裁剪图像重新编码。**
**本轮尚未运行真实数据/GPU训练、原视频推理，也没有形成新的比赛提交包。**
阻碍是当前会话无法访问现有 GPU、媒体和权重，不是 R83/R84 文档缺失。

已完成完整新状态审阅、三臂协议与代码、160 项定向回归、四个历史候选的八次真实预测检查。
这些证据不能替代新模型性能或官方 40 分。

## 1. 本轮采用的主攻及其可识别性

固定 R54 的原始时间集合、每视频预算、129 个合法最大窗口和人工构图监督，
以已评分的 R70 topsoft=36.42 作空间锚。R71=36.43 保留为当前最高分回退；
本轮不同时改变预算。

| 臂 | 每候选输入 | 用途 |
|---|---|---|
| A：冻结 R70 topsoft | 原 2305d 全图候选特征 | 当前已评分空间控制；使用真实 checkpoint/norm，不以同形状替代头冒名 |
| B：旧观察的布局读出 | 原2305d + 原全图网格在同候选ROI内2×2面积池化3072d | 控制新增头容量与旧网格中已有的局部布局 |
| C：真实候选新观察 | 原2305d + 真实crop用共享V9重新编码后2×2池化3072d | 检验新增候选图像观察是否有边际价值 |

B/C 都用 5377d 输入、同一 head 家族、MSE 人工平均 IoU 目标、两个预定 seed、
1200 步、batch16、AdamW(lr=3e-4, wd=1e-3)。最后固定步保存，不按 DEV 选择 checkpoint。
程序记录并核对 B/C 初始模型 SHA 与训练批次序列 SHA。归一化只用 train。

**C−B 是信息路径比较，C−A 是是否值得替换现役空间子系统的比较。**
A 的历史 topsoft 目标不同，不能把 C−A 单独解释成新增观察的因果效应。
当前仓库没有找到完整 topsoft 训练入口，因此 A 必须恢复并核验既有冻结权重，
不补造历史 loss。加入局部分支置换诊断时，候选位置、旧2305d和输出框不变；
置换也造成分布变化，结果只作为机制支持。

R70/R84 的路径是“整图编码一次 → 同一网格的候选聚合”。
C 改为“候选对应的原生图像区域 → 视觉塔重新编码 → 局部布局读出”，
让候选图像的局部采样、边界和上下文进入表示。C 能否胜出仍未知。

这是对现有 R70/R84 的明确改变。整个仓库更早做过 Qwen 大模型的实际 crop
生成式 rerank；不能泛称项目从未用过真实 crop。本轮的新问题是共享小视觉塔、
同一 native IoU 监督与强匹配控制下的判别增量。

依据与完整竞争解释见 [latest_state_audit.md](latest_state_audit.md)。

## 2. R83/R84 审阅改变了哪些判断

- R83 实际只使用五个哈希桶中的最后一桶留出，没有完整训练五次交叉验证。
  8槽仅有两个合法7/8窗口；平值标签固定选窗口0，归档验证片段约88.9%没有正例。
  BCE 的 pos_weight 是正例/负例，不能作为充分补偿稀疏正例的强控制。
  因此结果仅关闭这次受限配方，不能推出所有小模型政策无效。
- R84 仍读 R70 同一 [129,2305] 特征。它换了623,361参数MLP与坐标一致性损失，
  没有新增图像观察，且公共指标未胜过 R70。
- R84 的 +0.02547 是合并168源宏平均；R70 的 +0.07006 是两域等权平均。
  将 R70 改成相同168源权重，增量约 +0.06919467，仍明显较高。
- 本轮查到 R70 topsoft 原 manifest 尾部有字面量反斜杠+n，严格 JSON 解析失败。
  只在审计内存中修正读取，保留原件；这不等于历史预测或官方 ZIP 无效。

R83/R84 原始报告、代码和每项推导的来源链接均在最新审计中。
官方分数继续标为用户报告，未重新登录平台核验上传字节。

## 3. 现有数据与身份保护

正式输入沿用 R70/R84 的 T5_CROPHEAD_V3 四张表：
rv_train、live_train、rv_dev、live_dev。

| 域 | 当前代码划分 | 已归档规模与限制 |
|---|---|---|
| RetargetVid | 001–080 train；081–100 dev | RV dev 20源；标签经过T5处理，保留原数值并用连续IoU |
| LIVE | live_train_index中的source按SHA1余数分train/dev | LIVE dev 148源；raw_ltrb曾按W−1/H−1裁剪 |
| 合计 | 只用公开人工native crop监督 | 11276 train行、1546 dev行；当前实际表/图像仍需恢复 |

这些划分不同于旧 B3 的 RV110 或早期 spatial_crop_v1，不能混写。
公共 DEV 已用于 R70/R84，所有配对置信区间均为已暴露开发证据，不称新的盲确认。
不使用官方测试标签训练、伪标注或反推 GT。

新 manifest 从 T5 的 keys[J] 与 gf 直接连接 source、frame、ratio，
记录 row_id、图像/table SHA、原生 W/H 和人工框。R70 旧 shard 没有 frame_id/image SHA，
因此不以 shard 行号猜图片。源泄漏、重复身份、图像尺寸不符、非白名单表、
错误坐标或改变的图像直接停止；没有隐式缩放、坐标尺度猜测或EXIF旋转。

## 4. 已实现的代码

- [冻结配置](../../configs/R85_REAL_CROP_OBSERVATION.json)：唯一 B/C 配方、两 seed、
  数据版本、A复现门、晋级门和参数账本。
- [crop_observation.py](../../aic/crop_observation.py)：原生连续几何、真实 EXTENT 裁剪、
  ROI 2×2 精确面积池化、IoU与manifest验证。
- [r85_build_public_manifest.py](../../scripts/r85_build_public_manifest.py)：只读四张公共表，
  构建有图像与标签来源摘要的逐帧输入。
- [r85_crop_observation_probe.py](../../scripts/r85_crop_observation_probe.py)：共享冻结塔提取、
  同步B/C缓存、固定配对训练、源级bootstrap、候选局部分支置换。
- [check_current_package.py](../../scripts/check_current_package.py)：只读ZIP双checker，
  输出范围、SHA和未验证事项均显式记录。

提取恢复验证有序行、所有数组shape/dtype、已完成行摘要及完成缓存全文件SHA；
代码与关键运行库版本也参与身份比较。恢复不会把被重排或改写的缓存重新认证。
训练本身不支持中途恢复；中断时保留旧目录，并按相同冻结配置在新目录重跑，
不得把换 seed 或改日程称为恢复。

正式训练前必须复现 A：真实 R70 checkpoint SHA、11276/1546行、
RV20/LIVE148源，以及归档 A/center 的域级宏均值（绝对容差1e-4）。
这是身份/输入复现条件，不是用官方分训练或调门。

## 5. 本轮实际运行的验证

环境：Python3.12.14，pytest8.4.2，临时 CPU-only PyTorch2.8.0+cpu。
没有 CUDA GPU，尚未安装/加载本轮真实视觉模型。

```bash
python -m pytest -q \
  tests/test_contract.py tests/test_evaluation.py \
  tests/test_current_contract.py tests/test_independent_submission_check.py \
  tests/test_current_package.py tests/test_crop_observation.py \
  tests/test_crop_observation_probe.py
```

实际结果：**160 passed in 7.19s**。git diff --check 通过。

回归包含当前9B参数边界与评分、历史回放、ZIP结构、连续几何与镜像协变、
图像/表SHA与源隔离、缓存身份/恢复、train-only统计、局部置换，以及正式head的
CPU前向/反向/优化/保存加载。六行合成缓存运行了两seed×B/C，
每臂1步；这只是程序结构验证，正式配置的1200步没有在真实数据执行。

无信息合成数据的所有候选IoU相同，程序正确保持
engineering_gate_pass=false、candidate_content_alignment_supported=false、
new_competition_package=null。四个临时测试权重不是候选产物。

### 已存在真实预测的联合检查

| 版本 | 视频数 | 预测数 | 相对R54删帧 | 共同帧改变的框 | 两套current checker |
|---|---:|---:|---:|---:|---|
| R54 | 426 | 199037 | 0 | 0 | PASS/PASS |
| R70 topsoft | 426 | 199037 | 0 | 101797 | PASS/PASS |
| R71 | 426 | 195214 | 3823 | 0 | PASS/PASS |
| R84 | 426 | 199037 | 0 | 112141 | PASS/PASS |

这8次是真实归档预测上的检查，不是新模型推理。
参数数只和历史声明比对，没有加载权重证明实际账本。
当前可访问 JSONL 与索引，历史 ZIP 字节没有恢复。
[artifact_identity_audit.json](artifact_identity_audit.json) 保存命令、完整checker输出和哈希，
[artifact_identity_audit.md](artifact_identity_audit.md) 解释其范围。

当前合同修复也用父版本合成样本复现：旧semifinal拒绝只有合法9B字段的记录，
独立旧路径可接受99B声明，旧大小系数令完美匹配得90而非100。
修复不是官方模型涨分；[contract_reproduction.json](contract_reproduction.json)
保留原始结果。最新main的相关代码与该父版本一致，所以修复已移植到新状态。

## 6. 恢复计算后可直接执行的入口

下列是**尚未执行的真实数据命令模板**。路径与权重SHA先在现有授权主机核验；
不填造缺失的 A checkpoint。R85_RUN 等变量是本实验专用。

```bash
python scripts/r85_build_public_manifest.py \
  --t5-dir "$R85_T5_DIR" \
  --rv-frames "$R85_RV_FRAMES" \
  --output "$R85_RUN/public_manifest.jsonl"

python scripts/r85_crop_observation_probe.py extract \
  --manifest "$R85_RUN/public_manifest.jsonl" \
  --vision-dir "$R85_VISION_DIR" \
  --processor-dir "$R85_PROCESSOR_DIR" \
  --output "$R85_RUN/smoke" --limit 16 --crop-batch 32 --device cuda
```

16行按manifest位置确定性分布抽取，不按标签选择。smoke缓存不能用于正式train。
入口实际比较单图和混合形状batch的valid grid与FP16数值，记录吞吐和峰值显存；
通过后用同一config在另一个features目录提取完整集合。

```bash
python scripts/r85_crop_observation_probe.py extract \
  --manifest "$R85_RUN/public_manifest.jsonl" \
  --vision-dir "$R85_VISION_DIR" \
  --processor-dir "$R85_PROCESSOR_DIR" \
  --output "$R85_RUN/features" --crop-batch 32 --device cuda

python scripts/r85_crop_observation_probe.py train \
  --features "$R85_RUN/features" \
  --baseline-head "$R85_VERIFIED_R70_HEAD" \
  --baseline-head-sha256 "$R85_VERIFIED_R70_SHA256" \
  --output "$R85_RUN/paired" --device cuda
```

完整公共提取约1,654,038次crop编码，另加全图；三个FP16特征数组共约27.95GB
（26.03GiB），不含模型、目标和临时数据。这是数量估算，不是已测GPU时间。
先检查可用设备与磁盘，依据smoke估计墙钟并记录运行预算；旧12小时硬限制已由
新AGENTS撤销，不能再当永久规则。单配置不按DEV结果延长或换seed。

## 7. 晋级、40分与停止条件

配置已冻结：每个seed、每个DEV域的C−A和C−B都需平均IoU至少+0.01、
源级配对95%区间下界>0、至少2/3源严格改善。任一失败记录负结果，
不降门、不换seed、不做扫参式提交。局部分支打乱不降低C均值时，
停止“对应候选内容产生收益”的解释；不能只看engineering_gate_pass。

| 竞争解释/风险 | 最小判别或停止条件 |
|---|---|
| 只因多了参数或读取旧布局 | B与C同维度/头；C不胜B就关闭本次重编码主张 |
| 新观察有信号但弱于现役 | C还必须胜冻结R70 A，否则不生成晋级包 |
| 重复源、错帧、缩放或标签错误 | source/row/image/table身份失败就停止并作废该次结果 |
| 只拟合TRAIN或少数来源 | 分域、逐源、train/dev差值检查；不靠扩head或换seed救结果 |
| 公共域与官方域不一致 | 通过本地门也不预测官方分，记录原生方向/比例覆盖空缺 |
| 模型账本或运行成本不合规 | 实际总参数>9B、额外fallback依赖、资源不满足即停止 |
| 原视频无法重现、ZIP不一致 | 不形成可上传结论，不能以缓存重新命名替代新模型 |

固定R54时间时，R70的36.42若要到40，需要相同分母下的宏平均加权IoU贡献
相对提高约 **40/36.42−1=9.83%**。这是代数要求，不是公开IoU与官方分的校准，
没有识别任何官方GT或标签密度。没有证据保证R85会达到这个增幅。

C的head在CPU实例化实测 **3,084,289参数**。若实际部署仅保留报告中的
TimeLens2 8,767,123,696、共享V9 85,843,200和该head，总计
**8,856,051,185=8.856051185B**，距9B余量143,948,815。
全套预训练权重尚未加载；该合计仍须实际loader、权重张量与fallback清单核验。
共享塔多次前向只计一份参数，但增加计算。A/B对照头不得混入最终C执行图。

候选过门后，才进入原视频推理与导出实现：逐视频逐帧核对R54时间mask不变，
完成全部426视频、原生坐标和ratio、实际参数清单、预测/ZIP SHA、
只读ZIP双checker及原视频复现记录。本轮没有执行这些新候选步骤，也不自动上传。

## 8. 当前唯一外部依赖

执行中 main 从c84bfec更新到41f5e91，R83/R84已取得且读完；
早期报告缺失判断已撤销。初始合同修复保留于独立草稿分支，
最新研究代码与报告一并保存到该草稿分支，main 保持不变。

当前会话没有GPU、媒体、T5 PKL/NPZ或预训练权重；AutoDL最新记录入口
connect.westc.seetacloud.com:22527在提升网络权限后仍DNS失败，
旧学校119.62.14.24:22022拒绝TCP连接。不能据DNS失败断言实例停机。
当前没有可用SSH agent/key，未发送凭据、修改远端或租用额外资源。

**恢复所需：能访问现有媒体、T5表、V9/processor和R70 topsoft权重的授权计算会话。**
优先把任务接续到已经配置认证的AIC主机，或在当前会话配置可用SSH入口；
无需再次提供R83/R84文档，也不需要官方测试标签或在聊天里粘贴私钥。

实际执行状态见 [execution_status.json](execution_status.json)。
