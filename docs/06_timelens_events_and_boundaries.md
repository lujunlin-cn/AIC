# 路线 06：TimeLens2 事件区间、边界、采样率与事件拓扑

整理日期：2026-11-07。总入口：[index.md](index.md)。官方分来自项目回分记录；公共12源/18视图代理不等于官方标签。

## 主假设与参数约束

显式预测“事件从哪里开始、到哪里结束”，可能比泛化逐帧重要性更适合高光。实际链路为：`TimeLens2-8B区间 → 中心先验/区间mask/端点/生命周期解码 → 精确预算原始帧 → V9/B3空间框`。

TimeLens2本体8,767,123,696参数，加完整LFM空间系统会到9.217B而超限；改用已审计V9 vision-only空间87,407,729参数后为8,854,531,425，满足9B。早期构包复用了空间缓存，因此参数数值合规与可从原视频重跑是两个核验事项。

## 子方向与实际结果

| 实验 | 主要改变 | 结果／状态 |
|---|---|---|
| R47 public / direct | 固定generic prompt、2fps、高分辨率；区间直接变时间分 | 公开proxy中心=.704417，TimeLens=.721327；完成426完整推理与V9/中心空间两种包，没有把proxy当官方分 |
| R47 V9 center85 | 同R26时间集合，只换V9空间artifact | 已构建空间隔离候选；未发现独立回分，R54收益仍不能全部归时间 |
| R54 center residual | `rank(center)+.20*rank(interval)` | 官方36.22，相对R26=36.04 +.18；同时换了V9框，属于联合结果 |
| R55 phase / dense | 区间阶段、增加观测密度、2→4fps | 阶段/端点家族已形成多个诊断和候选；不是每个编号都有官方分 |
| R56 event edge | 端点性代替区间常数 | 公开比R54约+.001696，受单视图主导；完整候选已做，无已确认官方分 |
| R57 fixed late2s | 在相同V9空间下统一后移2秒 | 公开proxy最好之一，官方34.76，比R54−1.46；反证全局后移可替代内容定位 |
| 窄结果prompt | 要求只输出短结果区间 | 公开binary=.720168，低于generic=.721327；只做公共probe，没有扩量426 |
| R58 audio novelty | 20Hz无参数音频瞬态，交换固定5%帧 | 公共soft/binary均负；候选构建过，但不应描述为成功音频增益 |
| R58/R59 mask-edge ensemble | 2fps或4fps区间与端点集成 | 必须按完整名称区分。4fps R59官方35.04，比R54−1.18 |
| R60 direct4fps | 4fps区间直接解码 | 完整候选存在，未发现独立回分；不能与R59集成包混称 |
| R65/R66 | 4fps residual、mask/edge规则 | 构包及输出审计已做，未记录独立官方分 |
| R83 interval policy | 26,786参数、8槽、两个合法7/8连续窗，PHD2源级留出 | interval F=.025424，与中心相同，退化为中心策略；不是未执行的计划 |
| R93 dense event edge | 4fps generic intervals +固定端点residual | 官方36.32，比R54+.10、低于R71=36.43 |
| R106 shell | 2fps/4fps跨采样率事件壳层 | 已形成完整高风险候选，未发现独立官方分 |
| R107 lifecycle | 开始/完成/边界、2fps/4fps生命周期融合 | 官方36.43，与R71持平；不是新的突破 |
| R108 disagreement | 区间分歧/共识壳层作为价值 | 官方35.62，比R54−.60；分歧不自动等于高光 |

## 关键混杂与结论边界

R54使用2fps、`max_pixels=102400`；R59/R93的4fps链使用`max_pixels=50176`。因此观测密度、单帧分辨率、序列长度和token化同时改变，不能称为纯采样率消融。R59负结果只否定该具体组合，不能证明更密采样本身无益。

R54绝大多数输出仍为近连续窗口，选中位置从中心略后移；它与逐帧语义重排、固定晚移存在竞争解释。R57的负回分削弱固定晚移解释，但没有完成时间/空间全部因果分解。

R107/R108与R93共同说明，同一空间下事件政策有少量增量，激进分歧和公共代理最大化却可能损失。现有历史关闭决定约束当时配方，不是永久禁止新的事件模型。

## 与 GPT 讨论的问题

1. 应做保持分辨率、token预算和输出政策一致的密度对照，还是直接换能保留长事件上下文的观测方式？
2. 如何区分“generic事件区间”和“值得保留的高潮/结果”，而不重新落回12源代理过拟合？
3. 能否将事件区间变成受限动作的监督，迁移给更小时间组件，并保留高质量空间模型？

## 证据入口

- `reports/r47_timelens2_full_20261017.md`、`reports/r47_timelens2_v9_20261018/summary.md`。
- `reports/r54_timelens_center_residual_20261018/summary.md`、`reports/r54_r46_official_postmortem_20261018/summary.md`。
- `reports/r57_r59_official_postmortem_20261018/summary.md`、`reports/r93_dense_event_edge_20261019/summary.md`。
- `reports/r106_breakthrough_temporal_routes_20261020/summary.md`、`reports/r107_r91_r108_r93_official_postmortem_20261020/summary.md`。
