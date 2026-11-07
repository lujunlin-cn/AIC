# 路线 02：视频编码器、时间头与监督源迁移

整理日期：2026-11-07。总入口：[index.md](index.md)。本文的公开数据指标与官方回分分列；所有官方分数均来自项目保存的用户回分记录，本次未登录平台。

## 主假设与路线

视频预训练表征、全局时间交互和更合适的监督，可以改善静态图像特征的时间排序。实际尝试涵盖：VideoMAE、InternVideo2、SigLIP/LFM 时间头、全局注意力 GATT、QVHighlights、PHD2/GIF、MrHiSum、TVSum、VideoXum、人工时间等级和事件阶段。

## 子方向与结果

| 阶段／实验 | 方法 | 已得到的结果 |
|---|---|---|
| QVH 弱标签 pilot | DeiT-S，37/9 可用 clip；最后一层解冻对照 | 弱标签 F1 可达 0.67846，但标签不是原生人工 GT；有界解冻未获晋级 |
| QVH human native | 96 train / 24 dev / 40 未读 holdout，冻结视频主干 | VideoMAEv2-Base 未超过匹配 DeiT；InternVideo2-1B 排序有潜力，CI 跨零，仅作为工程候选 |
| VideoMAE-L / IV2 Stage1、Stage2、K710 | 原生视频特征 + QVH 评分头 + YuNet | 174 视频完整初赛：K700=34.43、Stage2=34.42、K710=34.38。差别很小，输出时间集合近乎相同 |
| V6 E4 | MrHiSum 特征上的 TCN、U-Net+ranking，多种子 | TCN 三种子 confirm 有正向弱域增量；U-Net+rank 未稳定过门。YTH 32 视频 macro AP=0.6162，仅迁移线索 |
| V7/V8 时间头 | LFM pooled 特征 + TCN；TVSum 或 DHF1K 显著性监督 | TVSum nDCG 显著超过随机，但没有直接官方增益通道；显著性值头可减少观察成本，未证明提分 |
| V9 PHD2 temporal | 5,886 片段、软标签 TCN、keep 决策 | 局部公开部署模拟较全选改善；后续实际时间包约 34.88，不能由本地 AP/F1 预测比赛收益 |
| V10/V11 | 多尺度、mean/topk/attention/delta pooling、负片段补充、SigLIP2 LoRA | 多种配方处于同一 AP 带；内容破坏控制和采样审计暴露位置捷径，详见路线05 |
| R8/R9 双基座／slot 表征 | InternVideo2 Stage1/Stage2、Qwen 塔、SigLIP 等在同合同下探针 | 没有打破位置控制；旧 native 视频缓存存在错误 seek，相关绝对值后来作废 |
| R11 VLM event roles | 7B/32B 事件角色、密集观察前提调查 | 有工程与局部 pilot；部分缓存无法复用、角色判别退化；运行故障修复不等于任务增益 |
| R12 P0-b / GATT | MrHiSum MSE vs pairwise；8 臂 GATT 全局注意力 | 公共 MrHiSum pairwise 59.17±0.38 vs MSE 54.00±1.60；完整 `R12_GATT_LFM` 复赛 35.17，相对 34.88 +0.29 |
| R12/R13 外部监督 | QVH saliency、UV/user-vote、CLIP 文本语义、MrHiSum pairwise、相关辅助臂 | UV/P3 五臂未晋级；QV 两 seed 官方 35.08/34.94；MrHiSum pairwise 官方 32.74。不能把公开 benchmark 复现成功当目标域成功 |
| R17 | 全轴 block 训练 vs 短窗 block | 公共 Δ=+0.002011，低于当时 +0.005 开发门；没有证明全轴事件模型无效 |
| R24 | 公共 VideoXum，VPB/CLIP × point/block × 两 seed，共8臂 | VPB point F@.85=0.296544/0.298774，位置控制=0.297467；block−point 两 seed 均负，不晋级 |
| R30/R31 | 静态、高维动态差分、低维事件状态/PCA | 高维动态明显过拟合；低维 F=0.302285，但相对中心 CI 跨零，且 dev 已暴露 |
| R43 | InternVideo2 Stage2 6B 事件完成度 | 固定 B3 空间，复赛 34.95，低于中心85+B3 的36.04；“事件完成”不自动等于高光 |
| R44–R46、R48–R53 | 人工等级校准、1fps/4fps 状态、阶段、全局上下文、运动持续性、Farneback、画质门控 | R44 官方36.09，仅弱正；R46官方35.88；R49–R53多为公开诊断，未产生可靠新部署收益 |
| R83 interval policy | 从逐帧 rank 改为连续区间政策 | 已做 probe，不应把后续计划描述为已有官方新模型；边界路线另见路线06 |
| R96 | 人工12源、leave-one-source-out的前后事件上下文 | context AP=.6736、static=.6435，Δ+.03014但11有效源CI跨零；方向性正证据，非确认 |
| R97 | PHD/GIF区间前/中/后状态辅助，匹配RGB TCN | 448源F .416244→.415464，Δ−.000780，CI跨零；具体状态辅助配方负结果 |
| R99 | 低秩 local×global TCN | Δ=+0.000331，448源CI跨零，置换后增量几乎消失；没有足够新内容增量 |

## 特别需要保留的反证

K710 曾出现 17.46 的近半分读数，最终确认提交了 87 视频单 shard；完整174视频包为34.38。因此“主干性能减半”的解释作废。主干质量比较只能使用完整包。

QVHighlights 是 query-conditioned saliency/grounding；TVSum 是摘要重要性；MrHiSum 是 replay 行为；PHD2 是用户选 GIF 区间，未选时间常常只是未观察。它们都不等价于官方逐帧高光真值。曾经把未知标签当负例、把 weak 标签称 human GT、或者把不同数据分母混用，这些都已留下修正记录。

位置控制、内容置换、同容量控制比“换一个更强主干”的单次正结果更有解释力。但历史负结果只约束当时观测与监督；2026-10-17自主权修订不允许把这些配方失败升级成永久研究禁令。

## 当前判断与讨论题

这条路线已经系统尝试过“换编码器、换时间结构、换公开监督源”。最值得进一步讨论的是怎样改变监督语义与部署观察，而不是默认项目只欠一个更大的时间头。

1. 未来目标应是逐帧 rank、连续事件边界还是视频级有限动作？用哪种公共标注可区分它们？
2. 若引入新监督，如何保留 unobserved mask、用户粒度、真实 PTS，并设置源外位置控制？
3. 怎样证明长上下文确实提供内容增量，而非重复学习切段位置？

## 证据入口

- `EXPERIMENTS.md`：Native human QVH、K710完整回分更正、R12/R13、R17、R24、R30/R31、R43、R49–R53。
- `reports/20260927_official_score_root_cause_v2.md`、`reports/20261001_v9_p0p1_execution.md`。
- `reports/20261003_v10_exploration_report.md`、`reports/20261004_round4_report.md`。
- `reports/r12/20261009_r12_gatt_verdict_next.md`、`reports/r13/20261008_r13_retrospective_45gap.md`。
- `reports/r97_event_state_tcn_20261019/summary.md`、`reports/r99_lowrank_context_tcn_20261019/summary.md`。
