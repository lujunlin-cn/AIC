# 路线 11：Video-ORA 时间 RL、有限候选、精细 SFT 与 rank-1

整理日期：2026-11-07。总入口：[index.md](index.md)。冻结完整基线为复赛42.49。官方分只用于结果记录，不进入reward、监督或伪标签。

## 研究问题

在不破坏冻结基座、主体与空间分支的前提下，用公共时间标签提高事件边界与区间选择。项目不仅尝试自由输出RL，也尝试有限候选、精细等级、阶段辅助、原生segments SFT和参数预算内的独立rank-1 adapter。

## 实验谱系

| 实验 | 实际训练／干预 | 结果与最终状态 |
|---|---|---|
| RL qualification/smoke | BF16、K候选批rollout、reward/gradient/KL/同质化监控、save/reload | 证明工程可用；不单独算提分模型 |
| TEMPORAL_RL_PUBLIC_CONTEXT_V1 | 96/96更新，91训练行/8源组，K=8，KL/CE anchor | 13行3源dev：base与RL reward均.3736329826，raw输出0/13改变。非零梯度、更新与训练reward上涨没有变成留出行为增益 |
| T0候选headroom | 左/原/右等有限边界动作，审计oracle可选空间 | 已做候选合同和诊断，不能把GT辅助最优动作当可学模型 |
| T1 warm-start / T2 candidate RL | pairwise warm-start、candidate RLOO、deterministic rerank及置信门控代码/运行 | 最终决策记录：未形成留出增益，2026-10-22关闭，未作为新SFT初始化。不是“未来才准备尝试有限候选” |
| FINEGRAINED manual | rank-4 q/o LoRA，人工细等级，122更新 | Manual guarded F=.725，与base相同 |
| FINEGRAINED mix | public/manual混合，204更新、fresh reload | monitor F=.3705498866，full-dev .2960632186，均与base相同 |
| FINEGRAINED phase | 加101条高置信事件阶段，254更新 | raw JSON有效率约.8148，但guarded决策不变；改格式未提任务指标 |
| NATIVE_MANUAL_V1 | 改为部署原生`segments`，8个manual上下文，8更新 | F=.3333，与base相同，预算太小 |
| NATIVE_MANUAL_V2 | manual重复30次，240更新 | step96 F=.6305，step240回落.2875；存在过拟合，保留已回读中间点 |
| NATIVE_MIX_V3 | manual重复10+PHD2 observed-positive，loss权重.25，263更新 | step200 F=.7036、interval IoU=.4422；仅3个manual dev源，CI下界跨零，不能称独立成功 |
| merged T1-base native | 时间adapter先merge到compact，再沿用`disable_adapter()`分阶段逻辑 | 官方 **39.67**；165607帧，共同帧52432个crop改变。时间、主体、空间一起受污染，是结构失败，不是纯时间消融 |
| NATIVE_RANK1_TEMPORAL_V1 | 保持基座和空间adapter，时间阶段独立启用rank-1，step50 | 公开7源F .261905→.728112；426/155416帧/1空，参数8.999935298B，真实完整候选已提交 |
| rank-1官方闭环 | 比固定42.49增加21495帧，约+16.05% | 官方 **42.46**，−.03；不采用。早期“等待回分”被此结果覆盖 |
| TP1_SHRINK10 | 原时间区间对称收缩10%，不训练 | 120841帧，官方 **41.98**，较42.49−.51；只否定这次收缩策略 |
| shot/边界后处理候选 | 边界吸附、观察与输出预算分离的代码/方案 | 有实现和研究记录；没有每种变体独立完成回分，不能统一标成已提交成功 |

## 为什么结果需要这样读

自由RL没有改变greedy，但确实改变log-prob；这说明adapter活性成立，行为门未跨过，不是“没加载权重”。candidate/phase SFT也可能只改善语法。更换原生segments监督后公开输出发生变化，证明输出合同是有效变量；但小源集的公开大增益没有转成官方大增益。

merge时间adapter后再关闭adapter，关闭不了已经写入基座的变化。39.67属于全阶段污染；rank-1用分阶段独立adapter修复这个问题，并满足9B完整参数，因而42.46是更可解释的时间后训练结果。

旧复盘曾由42.46/41.98推断“覆盖轴饱和”“新增帧只有半价”“空间每+.01 IoU换官方+.4–.5”。后续已明确撤回：总分不足以识别新增帧信用、官方GT密度或命中框平均IoU，也不能关闭全部时间研究。−.03只有一次回分，不宣称已测出统计显著退化或平台噪声分布。

## 当前可复用资产与讨论题

保留公开源组manifest、原生PTS标注聚合、quality gate、候选合同、fresh reload、分阶段adapter隔离、rank-1参数预检及完整426审计。

1. 后续主要缺的是独立标注源、多事件边界监督、跨窗口比较还是训练动作空间？哪项有尚未覆盖的机制？
2. 如何在输出集合不大幅膨胀的条件下改善边界位置，并固定空间阶段做对照？
3. 若重开RL，哪一个held-out行为变化应先被SFT或候选oracle验证，而不是继续扫KL/entropy？

## 证据入口

- `reports/videoora_rl_training_20261021/summary.md`、`failure_analysis.md`、`next_routes_code.md`。
- `reports/videoora_finetuned_sft_20261022/summary.md`、`native_segment_sft_report.md`。
- `reports/videoora_route_deep_analysis_20261022.md`、`reports/videoora_rank1_temporal_20261023/summary.md`。
- `reports/rank1_42_46_postmortem_20261023.md`：正文含已撤回强推断，须连同顶部2026-10-24勘误读。
- `reports/repository_audit_20261106/summary.md`、`DECISIONS.md`：后续最终状态。
