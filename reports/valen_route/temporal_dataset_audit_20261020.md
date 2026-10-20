# Valen 时序训练数据审计（2026-10-20）

## 研究问题

比赛的主要难点仍是时间轴上的帧级排序和输出预算。这里审计现有数据是否能够支持 Valen-2B 的时序训练，以及哪些数据更可能传递“官方偏好的高光”而不是一般摘要重要性。

本轮只做数据和标签合同研究，不启动新的训练。当前 Valen utility 空间训练完成后，再依据其结果决定是否进入时序实验。

## 结论

现有数据可以支持时序研究，但不能把所有数据合成一个逐帧 BCE 标签。最有价值的监督顺序是：

1. **PHD2/PHD-GIFs**：主监督。它记录用户制作 GIF 的区间和用户差异，语义最接近通用高光剪辑；未被选区间是未观察区域，不能默认是负例。
2. **YouTube Highlights**：第二监督源。MTurk soft votes 和同视频相对排序可以补充用户保留行为，但必须按源视频和动作域隔离。
3. **已有人工标签**：用于校准“为什么保留这帧”和事件阶段的辅助目标；数量小，适合诊断和小权重辅助，不足以单独训练一个稳健时序主模型。
4. **AIKU**：只适合研究时序 trunk 迁移。已有 R102 结果为 `+0.003413` fixed-85% F，CI `[+0.00080,+0.00610]`，但它只有预提取特征，没有原视频和特征提取器，不能直接部署。
5. **QVHighlights、TVSum、SumMe、MrHiSum、TRINITY**：作为辅助预训练、表示诊断或负对照；不能作为官方高光主标签。TVSum 直接迁移已有 R91 官方分数 `33.87`，低于当前强时间父链。

因此，Valen 时序训练是可行研究方向，最可靠的第一版是**源级隔离的 PHD2 soft-vote/pairwise 训练**，再用 YouTube Highlights 做跨源确认。它的监督目标应是窗口或候选帧的相对效用，而不是把未选帧写成 0。

## 数据源逐项判断

| 数据源 | 现有事实 | 可传递的能力 | 适合的训练角色 | 主要风险 |
|---|---|---|---|---|
| PHD2/PHD-GIFs | 已有约 5,886 个干净片段和原始 user/interval 字段；全库规模大，但媒体恢复不完整 | 用户选择的高光区间、同视频偏好、个性化分歧 | 主时间监督 | GIF 片段池有选择偏差；未出现区间不是负例；上传版本可能有时间轴变化 |
| YouTube Highlights | 已索引 6 个动作域，约 419 个媒体可用，含 MTurk 标注和 weak match | 编辑保留片段、同视频相对排序、多人投票 | 第二时间监督和跨域确认 | 动作域偏 sports；`match_label=-1` 不能当人工负例；来源规模小 |
| 已有 R42 人工标签 | 12 个公开源、540 份结构化标注，包含 generic highlight、event phase、information dependence | 阶段、完成度、信息依赖、人工高光判断 | 辅助校准、错误分析、少量多任务目标 | 不是独立大规模真值；存在重复源和标注一致性限制 |
| AIKU | 6,000 train groups，1024D 预提取特征，连续 `gtscore`、二值 `gtsummary`、change points | 摘要时间结构和时序 trunk 先验 | 条件式 trunk 迁移 | 无原视频、FPS/PTS、来源和特征提取器；不能形成官方部署输入 |
| QVHighlights | 本地有界子集，22 人 clip 评分 0–4，query-conditioned | 密集片段显著性 | 辅助密集监督、query 条件诊断 | 有查询；和无查询通用高光任务不等价；已有跨域迁移不稳定 |
| TVSum/SumMe | 摘要重要度，视频数量小；已有多轮跨域评估 | 摘要代表性、排序损失调试 | 负对照或冻结表示诊断 | 目标语义和比赛高光不同；TVSum+ResNet18 已有官方负结果 |
| MrHiSum/TRINITY | 预提取特征或多视角摘要标签 | 摘要/事件结构辅助 | 只做离线表示或结构实验 | 特征合同和任务语义无法直接映射到比赛输入 |
| RetargetVid/LIVE | 原生 crop 框 | 空间 IoU 与构图 | 空间路线 | 不含时间高光监督 |

## 为什么 PHD2 适合，但不能直接训练

PHD2 的行为目标和比赛“高光剪辑”更接近，因为用户主动把视频片段制作成 GIF。原始数据可以构造四种不同语义：

- `user_interval`：单个用户选择的真实区间；
- `vote_density`：同视频同时间窗被多个用户选中的比例；
- `pairwise`：同一视频内投票明确不同的两个候选片段；
- `unobserved`：没有被任何用户提交的时间区域，保留 mask。

当前历史 union-binary 做法会把“任一用户选过”写成正类，并把其余时间隐含写成负类。它同时混入了个性化分歧和缺失标签，无法作为官方偏好真值。

首轮 Valen 时序样本应使用固定预算窗口，例如 `left / keep / right`，由 PHD2 的区间覆盖率或 vote density 产生候选概率；未覆盖的时间只参与 mask。训练和验证按 YouTube source ID 隔离，不能按片段随机拆分。

## 对帧级排序的支持程度

现有数据对“帧级排序”的支持分为三层：

1. **直接支持**：PHD2/YTH 的区间或投票可以提供同视频候选片段的相对顺序，适合训练排序头或窗口 Choice。
2. **间接支持**：人工标签的阶段、完成度和信息依赖可解释某些高光帧为什么优于邻帧，适合作为小权重辅助任务。
3. **不支持**：没有任何现有公开源提供与 AIC 完全一致的、整视频每个原始 frame 的官方 GT 及匹配 crop IoU。因此不能声称训练后会直接优化官方逐帧 F。

真正的比赛决策仍需同时处理：

\[
F_v = 2S_v/(N_p+N_g)
\]

所以排序目标必须和固定或估计预算一起验证。单独提高片段 AP、Spearman 或摘要分数，不能作为官方收益的替代。

## 建议的 Valen 时序训练合同

等当前空间 utility 实验完成后，按以下顺序执行：

### T0：不训练的接口与候选测试

- 固定 R26/R54 的时间集合和 `K`；
- 构造同预算的 `left / keep / right` 候选窗口；
- Valen-2B 用视频 Choice 做批量判别，候选顺序循环置换；
- 对 PHD2 和 YTH 报 source-macro 窗口覆盖、相对排序和位置-only 对照。

T0 的作用是确认 Valen 是否真正利用内容。它不能证明精确帧定位能力。

### T1：PHD2 soft-vote/pairwise 微调

- 从原始 `training.csv` 重建 user interval、vote density 和 mask；
- source-disjoint train/dev/test；
- 同一视频生成固定 `left / keep / right` 或相邻片段 pairwise 样本；
- 先训练 Choice SFT/排序目标，保留视频窗口和原始帧映射 sidecar；
- 只在源级验证集选择 checkpoint。

### T2：加入 YouTube Highlights

- 仅加入 MTurk votes 和同视频 pairwise；
- PHD2、YTH 使用不同标签归一化和损失权重；
- 按动作域和源视频留出；
- 只有两个数据源方向一致，才进入联合训练。

### T3：人工标签辅助

将 `event_phase`、`generic_highlight`、`information_dependence` 作为低权重辅助任务，重点用于区分 setup/action/result/idle，而不是把人工标签扩展成大规模伪 GT。它更适合帮助模型学习“完成瞬间”和“信息依赖”这类官方可能偏好的事件结构。

## 当前应暂缓的方向

- 不把 AIKU 特征直接接到官方推理；
- 不把 TVSum/SumMe 的摘要分数和 PHD2 分数直接拼接；
- 不把 QVHighlights query-conditioned 评分改名为通用高光；
- 不用 PHD2 未选区间生成全视频硬负例；
- 不在当前 Valen 空间 utility 结果出来前启动时序大训练；
- 不把平台回分用于选择公开训练窗口或标签规则。

## 最小可行后续

当前空间 utility 训练结束后，先比较旧 utility checkpoint 与 `T006`，确认 Valen 路线是否值得继续。若空间路线保留，再准备 T0 的三窗口零样本时序 probe；若 T0 在 PHD2/YTH 的 source-macro 内容对照没有增量，就不进行 Valen 时序 LoRA。若 T0 有增量，再按 T1 的 PHD2 soft-vote/pairwise 合同训练，最后用 YTH 做跨源确认。

本报告没有使用官方测试标签、官方回分或由回分反推的信息。
