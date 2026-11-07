# 路线 05：时间预算、位置先验、决策目标与 Qwen3.5-4B 后训练

整理日期：2026-11-07。总入口：[index.md](index.md)。本页覆盖早中期决策研究及4B后训练，后期Video-ORA-9B的RL/SFT另见路线11。

## 主线与几个不同的问题

项目先试“预测逐帧重要性再top-k”，随后试概率校准、视频自适应预算、期望F目标、位置先验剥离、全轴训练及VLM分数生成。它们分别改变排序、预算、观察合同或训练目标，不能视为同一种调参。

## 子方向与实际状态

| 实验家族 | 实际尝试 | 结果／解释 |
|---|---|---|
| V10 adaptive keep | Platt/isotonic后expected-F；18维分布统计cross-fitted budget utility | 校准改善Brier，但固定keep仍更好；budget utility相对best fixed仅+0.0007、CI跨零。曾读真实正例数的+0.075是泄漏值，作废 |
| V11 VTREPLAY / EXACTDP | replay时间头与精确期望F-DP | 复赛34.95/34.74；公开DP确认+0.031没有变成官方提升 |
| R4 prior deconfound | 零特征、打乱、较小LR健康控制 | DP头99.97%片段top-k一致，real/zeros/shuffle F都约.6625；健康MSE与DP并列。原“目标函数增益”改判为位置先验鲁棒性 |
| R5 anchor/neutral | 由高光中心切段改为标签无关起点 | 相同模型从anchor约.653降到neutral约.545；位置先验约同幅下降，暴露切段捷径 |
| R6 T-CONTEXT | 前/中/后多位置训练、TCN/dense，2×2交叉评估 | multi在共同anchor任务比anchor训练约−.192；只在自己切段协议上贴近prior，不出包 |
| R7/R8 decision layer | monotone calibration、quantile/listwise/annotator protocol等 | 部分原天花板/单调变换方案数学上无信息或不改argmax；修正后不再把它们当独立有效增益 |
| R9 metric/encoder audit | 实际K、单位、内容和位置分离 | 8槽round预算使keep=.7/.8均选6槽；“不同keep同分”有些只是相同行为，不是实测最优区间 |
| R13–R15 Qwen3.5-4B SFT/RL | 点分/序列分SFT、teacher+GIF及27B打标监督、CE/分箱/rank、thinking、GRPO/OraRL、reward多版本、难例挖掘 | 实际训练与部署过；不是只有计划。历史thinking官方34.53/34.60；直接RL排序run7约35.04，比对应父链低.40。checkpoint格式/模板指标改善不等于内容改善 |
| VPB_GIFENS_A025 | 冻结4B视觉表示/小头、GIF ensemble与融合 | 历史复赛最佳35.52；不要把它与457M的R12_GATT_LFM=35.17混为一个4B包 |
| keep sweep | 同gif ensemble，keep=.50/.90/.92/.95等 | 32.57/35.22/35.11/35.06，低于当时.85的35.52。只能说该分数源已测范围内.85较好 |
| R15c/d | rankability、宏平均、天花板与密度反解 | 发现同分数无法识别密度d与平均IoU；后续又撤回39.54严格天花板，不能用作研究边界 |
| R16 | uniform/centered、点/短状态头，源互斥121/40/42 | 8臂落在位置/随机/全选控制附近，说明本协议信号弱；不是“任务没有内容信号” |
| R17–R19 | 全轴block、映射/exposure恢复、多标注员block oracle | R17小增量未过门；R18三源轴失败隔离；R19 block优于point是GT辅助oracle，不是可部署模型 |
| R20/R21 | 旧数字、脚本、缓存和源身份审计 | 撤销内容上限.3340、CLIP总体领先、均匀窗必败、shot优势等强解释；部分实验实际源重叠、行号错绑或目标单位不同 |
| R22/R22b/R22c | 公共源恢复、真实PTS、缓存重绑、VPB头身份恢复 | 测量链重建；A025历史输出不能直接从现存盘上复现的事实保留。是工程/溯源结果，不是新模型分数 |
| R23 random probes | 同B3空间、随机嵌套p=.25/.50 | 复赛18/26；后来明确随机rho=p只在期望成立，两个宏平均总分不能唯一反解测试密度 |
| R26/R27 | 精确85% AHA、中心连续窗、75%/95%预算控制 | 中心85+B3=36.04；AHA exact85=35.07；中心75/95=35.87/35.39。早/晚偏移包未回分 |
| R29 decision theory | 未知GT数量下精确期望F穷举及plugin反例 | 数学反例和穷举成立，但不证明真实后验可识别或训练能得到该后验 |
| R32–R36 | replay、exactDP、union90、consensus85 | 四个冻结缓存候选及k.90包已构建；未发现其独立官方新回分，不能补写为负分或成功 |
| R37–R41 | 转折、转折窗口、冗余、新颖性、内容门控 | R39/R40/R41复赛34.48/34.15/35.19，均低于中心36.04；R37/38只保留构包状态 |

## 必须废弃的强结论

旧文档曾说“平均IoU≈1，所以空间不是瓶颈”“GT头尾双峰”“没有内容信息”“39.54是严格上限”“本地门永远不可构造”。这些都是未识别参数、标签泄漏、宏平均简化或协议外推下的过强判断；本次只保留可复核的具体实测，不沿用永久禁令。

MrHiSum的`gif_hard=[0]*12`后来查明是生成器占位符，不能当真实全负；完整池、holdout、PHD2子池79%/87%/0.9%的全零比例分母不同。`gif_soft`仍完好，因此不能把全部弱监督一并判废。

4B生成的“分布丰富”曾被跨段pred-pred相关哨兵识别为模板化。工程上有非零梯度、权重变化和可解析输出，依然可能没有改变最终top-k。

## 与 GPT 讨论的问题

1. 下一步是否应学习有限连续区间动作，而非跨窗不可比的自由分数？需要什么数据证明动作优于位置控制？
2. 研究预算是否应先用于重建可识别的公共监督和部署合同，再改变loss或RL算法？
3. 自适应预算的正信号需要什么源外证据，才能避免从总分反推测试真值？

## 证据入口

- `reports/20261003_v10_exploration_report.md`、`reports/20261004_round4_report.md`、`reports/20261004_round5_report.md`。
- `reports/r8/decision_layer_correction.md`、`reports/r9/gpt6pro/AIC_20261006_R9_prior_metric_encoder_and_execution.md`。
- `old_docs/r15_reward_function_versions.md`、`old_docs/r15b_signal_diagnosis_record.md`、`old_docs/keep_scan_closure.md`：仅作历史原件，必须结合R21/R23更正阅读。
- `EXPERIMENTS.md`：R15d至R29；`reports/r26/official_analysis.md`、`reports/r37_content_candidates/official_score_analysis_20261017.md`。
