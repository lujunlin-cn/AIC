# 路线 12：Video-ORA 空间重训、完整构图框与连续几何

整理日期：2026-11-07。总入口：[index.md](index.md)。本页是当前42.74最佳系统与末期未提交候选的重点说明；回分来自项目保存的用户记录。

## 主路线

第一阶段扩大公共空间监督重训A/B坐标adapter；第二阶段突破“最大合法窗+单轴中心”，尝试`[cx,cy,scale]`完整框、可微IoU、连续head和GNMC/LIVE跨域训练。二者不是同一种改动。

部署的A是横向画面转9:16，B是需要上下裁到16:9；LIVE横屏图经转置构造B监督，不等于普通upright域或原生16:9完整验证。

## 一、位置adapter重训与当前最好结果

LIVE-YT-VC实际1800源×30标注帧，1600train/200dev按源分开；候选池每臂48000训练帧、6000dev帧。新A/B均训练1400 updates、batch8，**实际消费11200个不重复样本**，覆盖1600训练源，没有把48000帧全部遍历。旧222帧图像和response均逐字节对齐，保留翻转与光度增强的确定性种子。

公开评价已统一原图像素几何，旧1×1归一化IoU不再引用：A的200源1200帧IoU .607452→.627236，Δ+.019784、CI正；B转置LIVE的200源2000帧 .460591→.586134，Δ+.125542、CI正。B结果不代表upright域的同等收益。

| 完整候选 | 空间与部署变化 | 官方回分 | 最终解释 |
|---|---|---:|---|
| 固定基线 | 旧A+旧B，原管线 | 42.49 | 冻结回退 |
| PUSH_B_DENSE_ANCHOR | 更密主体锚点 | 42.50 | 单次+.01，小幅正 |
| PUSH_RETRAIN_A | 新A+旧B+dense，QP floor .15 | **42.73** | 安全母包 |
| P1_NEWA-NEWB | 新A+新B+dense，QP floor .15 | **42.72** | 新B没有带来整体增益 |
| P2_NEWA-NEWB_CAP24MS0.15QP0.6 | 新A+新B，4fps/cap24，速度上限.15，QP floor .6 | **42.74** | 最新最终确认，覆盖早期P2=42.73过程值 |

后三包都是426唯一视频、133921帧、2空，帧集合相同。P1相对安全母包改变132视频28504框；P2相对P1只改50视频2906框，窗口宽度不变。不能把P2的+.02说成一个全新模型。

最终保守申报8,999,771,458元素（含缓冲），逻辑参数8,999,623,920。P2预测SHA=`93413ea0a1e4097be7376b497391cfe715122b5d481e52e762e90ae492d50f00`；实际`p2.zip` SHA=`1ef881a0d4d5993148c5263623297c618f0abdfa67ce7aaf6dfcfb9fbaac1141`。原运行与最终材料绑定，完整框晚间包未混入P2。

## 二、后处理探针、撤销分支与误判修正

- SP3E_EXPAND5报告42.49，但预测SHA与42.49完全相同；放大后又被合法边界截回，属于恒等探针。**不能据此说尺度已最优。**
- P3新A+旧B/cap24/QP.6在方向切换后撤销，只留下171视频、46363帧临时输出，没有完整提交包和回分。
- RAFT-A在这里指拒绝采样微调，不是光流。样本/混合清单完成，两次训练启动失败，目录空；后来撤销，没有新权重。
- GNMC已取得并审计为7000train/1000validation/2000test源；旧B420图均在train，val/test零重叠。小幅负坐标显式截边并留原值。

## 三、完整框与连续几何实验

| 实验 | 真正做到哪一步 | 公开结果 | 最终状态 |
|---|---|---|---|
| FULLCROP_A_V1 | 匹配center与full，同前4800 schedule样本、600步、有效batch8 | full=.625878、matched center=.628279，Δ−.002401，CI跨零 | raw门失败，未跑426 |
| token-CE机制分析 | train/dev尺度多样性、相关性、收缩对照 | train scale仅4值、Pearson=.0395；dev仅8值、.0517；固定.95收缩仅+.004537 | 连续数值塌缩，不只“训练不够久” |
| FULLCROP_A_IOU1 | 同600步，仅加期望数字坐标的可微窗口IoU loss | train scale Pearson=−.01753、仅6值 | teacher-forced改善未转greedy几何，学习门失败 |
| geometry linear/MLP32 | 从A现有prompt-last隐藏状态预测cy/scale，复用cx；后接600步联合LoRA | MLP32新增139395参数；raw比current+.006131，但真实40视频轨迹−.000620 | 15:47归档`official_candidate_ready=false`，无426/回分 |
| GNMC upright/native | 60upright、40native诊断 | upright新B−旧B+.014312、CI跨零；native三模型与全图相同.931291 | native无独立候选，且缺原生16:9验证 |
| FULLCROP_B_V2/full300 | 300步，212GNMC源raw+40LIVE真实路径 | raw=.872317，明显超过旧B；路径=.478158，低于新B=.512216，Δ−.034058、CI负 | 最终路径门失败，不跑426 |
| FULLCROP_B_MIX_V1/mix240 | 新B初始化，GNMC+LIVE均衡，240步、batch16 | 3840 schedule样本、3456唯一样本；**按当时用户要求未做公开质量评估** | 完成426、独立审计及单视频回放，无官方分 |

首轮full曾在step79遇1921/1920标签边界；审计前4800条只有8个1像素右越界，显式截边并保留失败现场后重跑。这是数据合同修复，不应把首次失败计作另一种算法负结果。

## 四、最值得重新评估的遗留候选身份

`FULLCROP_B_MIX_V1`是末期真正交付的独立完整包：2026-10-24 19:02:26补完，426视频、133921帧、2空、参数8.999771458B。它沿用安全母包A/native和时间集合，只改B。

相对42.73母包改变133视频33882框，其中32678框宽度实际改变，是真实尺度干预。ZIP SHA=`b1f3474bc8b95419acd7073d8e774d8236b8eba41b99b5048dcde3a88a5c24b7`，预测SHA=`a81190df26940c62d127b55e1a80932911623eb2484eab432a1a169f20e2420f`。本地保存ZIP、代码、A/B权重，不是“仅远程存在”。

它错过当轮18:00项目交付目标；现有材料没有平台最终实际截止日志，不能将19:02直接说成违反官方截止。该包未进入最终P2、没有回分、也没有公开质量评估，**质量未知**。其他完整框失败则已有raw/轨迹负证据，不能全部归因“时间不够”。

本次2026-11-07已重新读取P2与B_MIX本地ZIP，payload/ZIP的上述完整SHA、426唯一视频、133921帧、2空均吻合。这是产物身份核验，没有重新评分或GPU推理。

## 与 GPT 讨论的问题

1. 首先补B_MIX的公开静态+真实轨迹验证，是否比训练新模型信息价值更高？
2. 完整框失败究竟来自数值token化、尺度监督、跨域还是后处理压制？geometry head与B_V2已经分别提供了哪些反证？
3. 下一轮应按原生几何分域学习，还是训练单个连续框模块？怎样固定时间集合和非目标分支，使结果可归因？
4. 42.74附近只有单次微小差值，是否需要先量化可重复的部署差异，再讨论几百分之一分的配置胜负？

## 证据入口

- `reports/final_submission_20261024/NOTES.md`、`research_ab_20261023/PLAN.md`。
- `reports/videoora_3point_research_20261024/plan.md`、`reports/videoora_fullcrop_v1/gate_raw.json`。
- `reports/videoora_fullcrop_iou1/state.json`、`reports/geometry_head_20261024/gate_a.json`及`chain_state.json`。
- `reports/fullcrop_b_v2/deploy_gate.json`、`reports/fullcrop_b_mix_v1/selection.json`、`training_config.json`、`delivery_finish_state.json`。
- `reports/repository_audit_20261106/summary.md`、`submissions/20261024/FULLCROP_B_MIX_V1/manifest.json`。
