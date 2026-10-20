# DINOv3 时间监督源审计（2026-10-20）

## 结论

PHD2/PHD-GIFs 是现有候选中与 AIC 复赛语义最接近的时间监督源：都是公开视频中由人制作 GIF 的高光选择。但 **PHD2 不能按当前的 union-binary 方式直接使用**。它记录的是“某个用户把哪一段做成 GIF”，不是逐帧全视频真值；未被选中的时间默认未观察，不能统一写成负例。

首选监督组合应为：

1. PHD2 `training.csv` 的用户级软投票/区间监督，按 YouTube 源视频去重和留出；
2. YouTube Highlights 的 MTurk 软投票和同视频相对排序，作为小规模高质量补充；
3. 两者使用任务特定的归一化和 mask，不把它们拼成一种伪统一标签；
4. 先在现有 5,886 个干净 PHD2 片段上重建标签，再决定是否扩大媒体规模。

## PHD2 为什么合适

- 目标语义是用户制作 GIF 的精彩片段，和比赛的“通用视频高光剪辑”同属用户选择行为；
- 可提供区间、用户、视频时长和 `is_last` 等原始字段；
- 仓库已有 PHD2 解码、source split、1 FPS/8 点采样和特征缓存；
- 现有训练历史显示，PHD2 比 QVHighlights、TVSum、MrHiSum 更接近目标域。

## PHD2 为什么不能直接当真值

1. **个性化**：同一视频不同用户的 GIF 区间可能不同，union 会把“任一用户喜欢”误写成唯一正解。
2. **缺失机制**：CSV 只记录被选中的 GIF；未出现的区间没有被用户明确标成不精彩。
3. **标签聚合损失**：当前 `train.json` 丢掉了 user 级字段，用户数和分歧没有进入模型。
4. **采样偏差**：历史 65% 片段锚定在 GIF 区间，模型可能学习“片段被选入采样池”，不是帧本身的价值。
5. **媒体时间轴**：当前 YouTube 上传可能和用户制作 GIF 时的上传版本不同，必须保留 PTS/时长异常。
6. **域并非已证明相同**：P0-a 对官方 426 视频与在盘 PHD2 子集只得到 1/426 的直接重叠；PHD2 是谱系相近，不是逐视频官方监督。
7. **迁移反例**：PHD2 教师/KD 在本地 PHD2、LIVE 等池表现正向，但对应官方包低至 34.61，说明“PHD2 代理变好”不能自动推出官方变好。

## PHD2 的正确标签合同

建议从原始 `training.csv` 重新构造，不读取上游 `testing.csv` 或 `is_last` 测试目标：

- `user_interval`: 每个用户的真实 GIF 区间；
- `vote_density`: 同一视频同一时间窗被多少用户选择，作为软目标；
- `union_positive`: 仅作为弱正标签，不作为唯一真值；
- `unobserved`: 没有用户选择记录的时间，默认 mask 掉；
- `pairwise`: 只在同一视频中比较有明确投票差异的候选片段；
- `source_group`: YouTube ID，所有重编码、不同用户和不同片段必须同侧；
- `duration_mismatch`、当前上传版本变化和 VFR 异常单独记录并可屏蔽。

训练上优先使用 masked soft regression 或同视频 pairwise ranking；不要对所有未选帧做 BCE 负例。视频级空高光判断需要另建真实背景池，不能从 PHD2 缺失标签推断。

## 现有数据源的适用性

| 数据源 | 现有状态 | 能教什么 | 用途判断 |
|---|---|---|---|
| **PHD2/PHD-GIFs** | 原始标注约 119,938 视频、222,015 行；当前 release 只登记 40 个媒体样本，历史远端已有 5,886 个干净片段 | 用户 GIF 选择、个性化高光、同视频相对偏好 | **主时间监督**，必须重建 user/vote/mask |
| **YouTube Highlights** | registry 约 550 个标注视频、419 个媒体可用；6 个动作域；有 MTurk soft votes 和自动 match label | 编辑视频中被人保留的片段、同视频相对排序 | **主时间补充**；用 MTurk votes，`match_label=-1` 不当人工负例 |
| **QVHighlights** | 当前本地只有 20 个视频；原数据 query-conditioned，22 人对 clip 打 0–4 | 密集显著性、局部片段强度 | **辅助预训练/校准**；不能直接当 generic GIF 标签。历史跨域迁移已失败过 |
| **TVSum** | 50 个视频，摘要重要度 | 摘要代表性、全局相对重要性 | **弱辅助或表征诊断**；R91 已证明 TVSum+ResNet18 不能直接迁移官方 |
| **MrHiSum** | 约 27,883 条特征训练单元，但当前 release 无原始视频 | 新闻视频摘要的 pairwise/summary 排序 | **只能预训练时序头或做协议诊断**，不能做 DINOv3 原视频提取 |
| **AIKU** | 6,000 group 的预提取特征和 summary score | 视频摘要时序归纳偏置 | **条件式 trunk 初始化**；R102 迁移 `+0.003413`，但特征不可用于官方部署 |
| **TRINITY** | 公开版本主要为 CLIP 预提取特征，Event/Emotion/Nature 多视角 | 多视角高光定义、辅助头结构 | **借鉴结构/做离线头实验**；不能作为 DINOv3 原视频监督 |
| **RetargetVid/LIVE** | 已有 native crop 框 | 空间构图和候选 IoU | **只用于 S-0 空间头**，不能训练时间高光 |

## 推荐的最小训练数据方案

### 第一阶段：不新增大下载

- 用已有 5,886 个 PHD2 干净片段，重新生成 `user_interval`、`vote_density`、`unobserved mask`；
- 保持已有 448 source 留出，只改标签聚合和损失；
- 以 R92 RGB TCN 为控制，比较 `union-binary`、`soft-vote`、`pairwise` 三种标签合同；
- 这一阶段只判断标签合同，不能同时换 DINOv3、TCN 和 keep。

### 第二阶段：加入 YouTube Highlights

- 使用可解码的 419 个媒体单元，训练 300–350 个源，按 domain 留出；
- 只用 MTurk votes 和同视频 pairwise，自动 `match_label` 单独作为低权重辅助或完全排除；
- 用 dataset-specific head 或 source-normalized loss，不能把 PHD2 的分数直接当成 YTH 的同尺度标签；
- 只有 PHD2 和 YTH 两个留出方向一致，才把联合监督接入 DINOv3 T-1。

### 第三阶段：条件辅助

- QVHighlights：把 150 秒原视频切成 8–14 秒窗口后做辅助密集监督，保留 query 条件字段并单独报告；
- AIKU/MrHiSum：只初始化 temporal trunk，不将预提取特征伪装成 DINOv3 输入；
- TVSum/TRINITY：只用于诊断跨任务表示，不进入官方时间主损失。

## 对 DINOv3 计划的改动

DINOv3 T-0/T-1 不能直接用当前 PHD2 union-BCE 做唯一结论。新的顺序是：

1. 先跑 `PHD2-label-contract`：RGB 控制下比较 union、soft-vote、pairwise；
2. 只有 soft-vote 或 pairwise 在源级固定预算 F 上超过 union 控制，才把更好的合同用于 DINOv3 T-0/T-1；
3. 空间 S-0 不受这条时间标签合同影响，继续使用 RetargetVid/LIVE native crop GT；
4. 官方提交候选仍固定 R54 时间或固定父链空间，不能把多数据集代理分数直接合成 AIC Score。

## 最终判断

PHD2 **适合作为主监督源，但不适合作为未经重建的单一硬标签源**。YouTube Highlights 是最值得加入的第二时间源；QVHighlights、AIKU、TVSum、MrHiSum 和 TRINITY 各有辅助价值，但都不能替代 PHD2/YTH 的目标域行为监督。对当前 DINOv3 路线，先修标签合同比立刻扩大模型更有识别价值。
