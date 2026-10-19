# 空间瓶颈审计与最小突破路线（2026-10-18）

## 判断

当前空间框是重要的可提升环节，但现有官方证据不能把全部35–36分差距归因于空间。最强可识别事实分三层：

1. **官方联合证据**：R25 原始时间掩码相同的空间对照中，B3 框为35.18，中心合法框为30.23，差值+4.95。它证明“空间框是否贴合主体”会强烈影响联合分，但比较的是中心空间对照，不是B3的上限。
2. **公开 native crop GT**：同一129个合法候选、同一帧上，B3当前head与候选oracle存在稳定差距。live_confirmation B3=.53495、oracle=.78188，gap=.24694，源级95% CI [.23131,.26295]，227/227源oracle更高；live_dev gap=.23426 CI [.21365,.25517]；rv_diag_exposed gap=.16187 CI [.14671,.17841]。这些结果支持优先修复候选内utility/readout，不能区分视觉特征、pooling、head优化和标注迁移。
3. **当前R54空间干预很小**：R53比较R26/V9全帧框，平均框IoU=.99804，仅9.678%帧达到几何变化阈值；landscape变化6.777%，portrait变化17.578%，R26中心85选中帧变化9.951%。因此R54=36.22相对R26的+.18不能解释为空间系统已发生大跃迁；V9是小幅替换，与时间共同干预。

结论是：空间仍可能是最大可提升来源之一，但当前提交分数没有识别“空间总误差占比”。R54/R67上的V9与R26几何分歧不能作为native IoU证据。

## 人工标注边界

Label Studio完整数据库有540条正式标注、349个框结果，但两个项目均为同一账号。120个同账号重复帧的高光等级完全一致率仅28.3%，构图字段在重叠样本上完全一致率虽为100%但样本只覆盖48个已有框；这些框可作弱监督和错误分层，不能当独立native GT。标注协议的自由框也没有绑定比赛目标比例，不能直接计算官方空间IoU。可用于学习“需要主体/多人互动/整体画面/字幕”的条件门控，给公开native GT实验提供弱先验，按composition_confidence和框理由分析失败类型。

## 最小可执行判别实验

### P0：空间合同与候选覆盖

用R46的公开CSV和现有R26/V9全帧框，重新按源级汇总B3、center、candidate oracle；native目标比例分层（1:3/3:1、portrait/landscape）；统计candidate gap分位数、连续帧gap、自身框几何变化。不把不同帧集的时间分数与空间IoU相乘。R46结果已经满足候选gap在两个源级留出池复现的门槛。

### P1：候选内utility readout（首选）

数据：公开live_train/rv_train native crop GT，冻结同129候选、同帧、同processor和目标比例。模型分三层：冻结B3 head；linear residual；2层小MLP。输入B3 129维utility logits/margin、候选位置、目标比例、主体/多人/字幕弱特征。目标直接回归每候选native IoU或候选相对B3的pairwise preference，按源级划分训练/验证，禁止帧级泄漏。每帧选最高预测utility，报告B3、center、oracle，以及1:3/3:1分层。主指标是源级宏平均IoU与top-1 candidate regret，保留候选数不变。

判别预测：linear提升且MLP不再提升，说明当前读出/校准不足，可部署小head；训练高、跨源低，说明监督政策/域迁移问题，不能换大模型；两者均无提升而oracle gap仍大，说明特征或pooling缺空间信息；oracle gap在域内消失，说明旧gap主要来源偏移或候选标签政策。

成功门：live_dev与live_confirmation两个源级池同时超过冻结B3，且bootstrap CI不跨0；x/y分层不能有一个方向明显退化。过门后才考虑426空间重提取。

### P2：Label Studio弱监督校准（无B3 logits时）

从SQLite导出349个人工框及对应图像SHA，只保留构图把握高、非重复争议帧；将自由框投影到比赛目标比例的合法窗口，形成软中心/偏移分布，不把投影框称native GT。只训练无参数后处理或极小线性偏移头，并在公开native GT上校准方向。若弱监督和native GT方向相反，立即停止该路线。

### P3：高风险结构路线

固定R54/R26时间集合和精确85%预算，只替换空间框：

1. candidate utility × context gate：用多主体/互动/字幕标签对candidate utility做门控；
2. expected IoU直接读出：估计候选的预期native IoU，而非预测高光后使用固定框；
3. uncertainty mixture：保留B3 top-1与top-2候选，按校准不确定性在多人/字幕帧启用第二候选。

不能将R63/R64的V9/R26几何分歧直接当作上述utility；原R63/R64还存在stable-tie位置偏差。部署前必须用公开native GT做源级验证。

## 资源与停止条件

P0纯CPU，少于10分钟。P1只训练线性/小MLP，CPU或单张V100，预计小于1小时；不需要官方426视频。P2读取本地SQLite和公开native GT，少于30分钟。只有P1过门，才在AutoDL上用V9视觉模型重算公开/426空间候选；正式训练预留可恢复checkpoint和外层timeout。

停止条件：候选oracle gap没有源级复现，停止readout瓶颈解释；新head在两个留出池均不超过B3，停止空间readout提交路线；只在单一比例提升，不包装联合候选；无B3原始logits/processor身份，不把V9或V7缓存冒称B3，转向V9公开权重独立诊断；没有同视频时间+空间GT，不把空间IoU增量换算成官方Score。

## 下一动作

优先恢复或核验B3候选logits、feature、head；随后执行P1，而不是继续提交box-jitter或平滑包。若B3资产无法恢复，使用V9 vision-only权重建立同129候选的公开native GT readout试验，先证明跨源空间增量，再决定是否进行426空间替换。
