# 路线 08：空间效用驱动选帧、联合目标与自适应预算

整理日期：2026-11-07。总入口：[index.md](index.md)。本页所有“官方分”是项目保存的用户回分；不同公共代理不能合并成一个验证分数。

## 主假设与数学目标

官方每视频F为 `2·Σ(同帧命中的crop IoU)/(N_pred+N_gt)`，因此最值得输出的帧，既要是高光，也要能构好图。项目尝试把几何分歧、构图稳定性、预测crop utility和高光时间分相结合。

R29的理论穷举进一步指出：未知GT数量下，精确期望F不总等于边际高光概率排序或分子/分母期望之比。这是目标函数事实，不证明项目已有可校准后验。

## 子方向与结果

| 家族／实验 | 干预 | 结果／状态 |
|---|---|---|
| R59–R64 geometry joint | 用V9/R26框分歧或稳定性调整R54时间排序；正反向对照 | 原R63/R64 stable-rank把大量零分歧按帧号排序，注入位置先验，机制实验作废；编号还有重用，须按完整名称核对 |
| R67 tie-aware修复 | 只对geometry使用平均秩，数值噪声≤1e−6归零 | 相对R54换2427帧，官方36.16 vs36.22，−.06；分歧作为高光价值未获支持 |
| R68 variants | 几何tie/分歧派生与修正版 | 有构包及审计记录，不应把旧bug版本视为独立有效试验；无新确认回分 |
| R70 spatial utility pilot | 重训候选utility，native crop监督、topsoft目标 | 最终topsoft空间包官方36.42，R54时间集合固定，+.20；是较干净的空间读出正证据 |
| R71 adaptive budget | 按视频区间信号调整少量输出预算 | 官方36.43，较R54+.21，删除3823帧；公共十视窗soft微正、binary负，不能解释成已识别官方密度 |
| R72/R73 | 预测空间效用时间重排、signed incremental utility | 做过公共与部署审计；包括原B3资产恢复，避免拿近似权重冒充父链。没有独立已确认提分 |
| R74/R75 | tie-aware增量utility与有符号增量控制 | 修复平值排序；R74交换2418帧、245视频变化，完整包/checker已做，无独立已确认回分 |
| R76/R77 | 实际输出框对齐的additive/multiplicative utility | 修复“最优候选效用”与“实际部署框”错位；多个完整候选及审计，无独立已确认回分 |
| R78–R81 direct utility | 从候选argmax效用改为实际部署框效用，tie-aware/output-aligned；乘法时间重排 | 公共学习信号及完整候选存在；`R81`产物后来仍有未提交修改，不能只凭目录名确定历史上传身份 |
| R82 ensemble risk05 | 直接部署框utility集成、风险惩罚、乘法重排 | 外部156源Spearman +.02304，CI正；官方36.20，比R54−.02，未迁移 |
| R83 validity audit | 对R44/R54/R69/R70/R71/R82作目标与回分对照 | 证明局部crop排序、人工时间proxy和轨迹平滑都不能通用预测官方排名；不是“所有公共验证都无用” |

## 关键区分

R70“选择更好的框”与R82“用预测框质量选择更好的时间”是两种任务：前者固定集合和分母，后者改变命中帧与输出数量分布。公共crop utility可学，不保证`P(highlight)×utility`在另一域已校准。

模型分歧可能代表构图改善、主体切换、视角、数值误差或模型不稳定，不等于精彩程度。tie-aware修正后R67仍负，说明早期分歧增益不能由bug版本支持。

R71是一次同阶段正回分，但没有逐视频GT，不能从总分判定哪个视频应删多少帧；更不能将此反推信息用于训练或重加权测试分布。

## 当前判断与讨论题

空间监督重训有可保留正证据；直接把空间效用乘进时间分数尚未证明可靠。联合目标仍合理，当前瓶颈在监督与校准的可迁移性，而不是公式写成乘法或加法。

1. 应先做同一公共视频的高光+crop联合标注，还是让两个独立模型输出受限的联合动作？
2. 能否通过同时间、同框、同预算的逐项干预，把R70/R71收益与时间排序真正分开？
3. 若再次建联合utility，目标是否应直接对最终路径和输出集合训练，避免只拟合静态候选IoU？

## 证据入口

- `reports/r59_joint_spatial_utility_20261018/summary.md` 至 `reports/r64_joint_spatial_stability_20261018/summary.md`。
- `reports/r67_official_bottleneck_review_20261018/summary.md`、`reports/r68_spatial_bottleneck_audit_20261018.md`。
- `reports/r70_spatial_utility_pilot_20261018/README.md`、`reports/r70_topsoft_objective_20261018/README.md`、`reports/r70_utility_temporal_reassessment_20261018/summary.md`。
- `reports/r78_direct_output_utility_20261019/summary.md`、`reports/r82_direct_utility_ensemble_20261019/summary.md`、`reports/r83_official_local_validity_20261019/summary.md`。
