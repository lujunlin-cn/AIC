# 路线 13：数据、人工标注、多模态探索与评估工程

整理日期：2026-11-07。总入口：[index.md](index.md)。这条支撑路线解释哪些实验可以信、哪些需要作废或降级；没有把工程校验折算为比赛分数。

## 数据与监督实际用到了什么

| 数据／资产 | 实际角色与使用状态 | 主要边界 |
|---|---|---|
| TVSum / SumMe | 早期时间训练、作者evaluator对照、跨域测试；SumMe先LFS阻塞后恢复14源 | summary、binary proxy与AIC高光不同；已读外部池不能反复称新lockbox |
| QVHighlights | 弱标签pilot、原生query saliency训练与视频主干对比 | 完整800/89曾有23原源交叉，purge成775/89；后续native96/24/40重新冻结 |
| RetargetVid / DHF1K | 真实多标注员crop、显著性、旋转几何、候选utility | 同原源关系必须记入exposure；显著性不是高光GT |
| LIVE-YT-VC | 早期1115/124/227冻结池；后期原始1800源空间重训 | 两批划分/数据版本不同；转置B验证不等于upright或native域 |
| PHD2 / GIF | 用户区间、union/soft、时间模型、teacher缓存、后期observed-positive辅助 | 未选不等于负例；同用户重叠不能重复投票；上游test家族排除 |
| MrHiSum / TRINITY | replay连续分、pairwise复现和迁移；TRINITY有文献/方法与资产调查 | 文献讨论或下载不等于每种辅助视角已训练；gif_hard占位符问题必须修正 |
| YouTube Highlights | 多域相对高光、迁移诊断、后续候选时间协议 | 小源集仅先导；新协议计划不自动代表完成训练 |
| VideoXum | R24–R31源绑定、十人时间GT、位置与内容对照 | 已暴露dev，不能声称fresh独立确认 |
| GNMC | 原B小训练集、后期完整B/多比例、混合B | 取得真实split与标签后才做训练；native16:9验证缺口保留 |
| AIKU / AViMoS | 已下载审计并有真实训练probe，见路线09 | AIKU缺原视频/提取器；眼动标签不等于crop GT |
| GAICD / DAVSOD等 | 资产登记、下载或备用研究 | 未发现直接可引用的独立官方增益；不能把候选清单当实际成功路线 |

## 人工标注 R42 与后续时间监督

R42从60条smoke正式导出开始，先做结构/几何QC，再修V1.2、扩V1.3事件阶段与构图标注。最初crop需要校准，不能把“JSON合法”当人工标签正确；旧60条也没有自动升级为V1.3重标。

项目建立了标注协议、浏览器界面、源视频上下文、真实PTS、highlight等级/event_phase/crop字段和质检记录。后续R44使用人工等级做局部残差，官方36.09；Video-ORA RL/SFT也使用manual上下文，但dev最少只有3个源，rank-1公开门也仅7源。人工监督已经尝试，缺口在规模、源独立性、多事件结构及任务政策，而非“从未标注”。

标注必须限定合规公共训练/验证源；官方测试媒体、测试预测、官方回分及其反推信息不进入人工标注、伪标签或训练。

## 多模态与低成本信号不是空白

| 方向 | 已尝试范围 | 可支持结论 |
|---|---|---|
| 音频embedding | R10全池音频特征及时间头、不同真实音频/控制读出 | 当前音频表征没有实用增量；早期audio-zero bank不能代替这批真实实验 |
| ASR | 语音提取、形态/密度/时间统计，长链诊断 | 已测形态层不增益；词典/编码器语义层有未测部分，不能说全部ASR都被验证失败 |
| OCR | 文字状态/覆盖调查 | 当时数字文字覆盖不足，未形成可用完整候选；不代表未来所有文本场景无用 |
| active speaker | speech-dominated前提、Light-ASD/人脸轨迹等准备与诊断 | 部分为前提或工程线，不能当已完成官方收益 |
| motion/flow | embedding差分、Farneback、全局运动补偿、多视图 | 具体残差/融合配方多为负；光流与后期RAFT-A拒绝采样微调是两个不同概念 |
| R53/R58音频 | 低维DSP、20Hz瞬态与实际帧交换 | R58公共soft/binary约−.0059，反证“音频峰直接换帧”配方 |

## 测量与部署中真正修过的问题

1. **原始帧/PTS与解码**：floor锚点约+.99秒偏差；非单调PTS与ordinal协议；seek单位错误导致24–1608秒偏离和15/16重复帧；旋转/转置/首帧非零PTS。均需明确输入合同。
2. **标签语义与身份**：unknown误写negative、MrHiSum硬标签占位符、同用户重叠soft>1、812行标签读2508行缓存、源身份与row index错绑；旧值不能靠换标题修复。
3. **评价目标**：binary时间F、query AP、summary F、crop IoU、轨迹稳定性和官方联合F不是同一个量；oracle、暴露dev与独立confirm分级。
4. **计算栈**：V100标准kernel，910A的FP32/eager fallback、SigLIP位置插值全零与parity、fork×编译线程锁死、GDN/attention训练图与内存；PRO6000/4090迁移。环境修复不是模型涨分。
5. **打包与参数**：单shard近半分事故、完整174/426覆盖、原始帧范围、合法crop、空输出、ZIP解压回读、全部实际推理参数清单。量化不减参数，缓存不免生成模型参数。
6. **后期特殊纠错**：SP3E恒等探针、R63/R64 stable-rank、temporal merge污染全部阶段、raw与真实轨迹评价相反、P2旧分数被最终42.74覆盖。

## 当前规则与复现边界

项目规则文件最新独立官网核查记为2026-10-16，核心是总推理参数≤9B、`model_params_b`、视频宏平均连续IoU加权F、无现行k_size。此回顾没有再宣称2026-11-07已在线刷新全部赛事规则；未来训练或新阶段提交仍需核对最新正式公告。

官方`F_i=2ΣIoU/(N_pred+N_gt)`；同video_id、同原始frame才匹配。官方可执行同版本evaluator和逐视频GT未取得，项目本地evaluator只复现公开公式与合同。

2026-10-17自主权修订已覆盖旧固定路线、GPU列表、12小时时限、禁止扫参和永久停止门。回顾中“关闭”仅指当时决策，不能阻止有新机制和新证据的探索。

## 与 GPT 讨论的问题

1. 应优先补什么能改变路线判断的数据：独立时间源、真实B几何、native多比例，还是同源时间+空间联合标注？
2. 怎样把同一协议下的机制验证与官方迁移确认分开，同时避免把有限公开dev用尽？
3. 哪些现有产物应先做可复现恢复和真实路径验证，才能有效使用下一轮GPU预算？

## 证据入口

- `EXPERIMENTS.md`、`DECISIONS.md`、`experiments/registry.jsonl`、根`registry.jsonl`。
- `ext_data/reports/registry_status.md`、`ext_data/reports/20260929_data_releases_v1.md`。
- `reports/annotation_protocol_v1_3.md`、`reports/r42_annotation_v13_full_20261017/summary.md`。
- `reports/r10/evidence_ledger.md`、`reports/r11/20261007_day1_verdicts.md`。
- `reports/repository_audit_20261106/summary.md`；本次逐条盘点与源文件hash见[实验台账](experiment_inventory.md)。
