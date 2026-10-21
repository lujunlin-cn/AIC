# Video-ORA 42.49 对 Jev 路线的迁移结论（2026-10-21）

## 判断

用户报告的 `VideoORA_42.48_semifinal_426_fixed.zip` 得分为 42.49。包审计和本地复现确认了代码合同，但没有独立核验平台上传字节或官方页面，所以 42.49 只能作为用户报告的结果记录。

这个包最有价值的证据不是“应该把 Video-ORA-9B 接进 Jev”，而是它把时间、空间和输出合同拆开了：时间路径使用冻结大模型的事件能力；空间路径按目标比例使用专用低秩 adapter；最后用真实 PTS、最大合法窗口、平滑和主体覆盖约束直接作用于评分输出。包内时间定位和主体检测都显式关闭 LoRA，两个几何 adapter 也不同时激活。

## 对当前 Jev 的具体迁移

### 1. 时间只做有限相位决策

Jev 不再直接生成任意原始帧号。固定当前连续时间父链和输出预算，构造同预算 `left / keep / right` 三个相位候选，Jev 只返回 Choice；程序把 Choice 映射到原始解码帧 ID。低置信度、候选顺序置换不稳定或概率不完整时回退 `keep`。

这保留了 AIC 已由 R26/R54/R71/R107 支持的中心连续先验，同时允许模型提供有限内容修正。它不会把一个不稳定的语言输出直接扩展成碎片化帧集合。

### 2. PTS 是唯一时间轴

迁移包的扫描、窗口和事件展开都基于解码 PTS，而不是平均 FPS 或 `frame_id / fps`。事件采用半开区间 `[start,end)`，输出帧保持原始解码 ID、升序和无重复。新增实现位于 `aic/jev_temporal_policy.py`，包括 PTS 校验、事件合并、半开映射、等预算相位候选和保守回退。

### 3. 观察合同先对齐，再谈时序 LoRA

参考包使用 30 秒窗口、5 秒重叠、64 帧、640 输入。Jev 的第一步是用这个合同完成 T0 零样本 probe，并与位置 only、候选顺序置换和中心回退对照；只有 PHD2 和 YouTube Highlights 两个 source-held-out 域都出现内容增量，才允许做 PHD2 soft-vote/pairwise 时序 LoRA。

### 4. 空间路径独立于时间路径

包内两个几何 adapter 分别对应横源到竖裁和竖源到横裁，且时间定位时关闭 adapter。Jev 不能把空间 crop SFT 或主体框输出混进时间分数。空间实验只使用目标比例专用分支和最大合法窗口；空间模型仍计入总参数，不能与现有约 8.855B TimeLens2 链叠加后超过 9B。

## 已实现代码与配置

- `aic/jev_temporal_policy.py`：可复用的 PTS/半开区间/等预算相位决策合同。
- `tests/test_jev_temporal_policy.py`：不规则 PTS、边界窗口、事件合并、顺序不稳定和低置信度回退测试。
- `configs/JEV_TEMPORAL_VIDEOORA_TRANSFER_V1.json`：固定的 T0/T1 观察与晋级合同。

该代码目前是部署合同和判别探针基础，尚未声称 Jev 官方收益，也没有读取官方测试标签或把 42.49 回分用于训练。

## 为什么不直接照搬 42.49 包

- 9B Video-ORA 主干已经接近 `9,000,000,000` 参数，不能再与现有 8.855B 时间链组合。
- 两个几何 adapter 的公开 B DEV 增益 95% CI 跨零；它们提示“按几何拆分”值得验证，但不证明 adapter 本身带来 42.49。
- 42.49 包使用的训练媒体和完整运行环境没有全部随 ZIP 提供；本地已完成授权公开视频的 fresh 合同复现和 426 输出审计，但不能把它写成独立分数复现。
- Video-ORA 的 RL smoke 已出现合法输出和非零梯度，但这只证明工程链路；不能把 RL adapter 融合到 Jev，除非 source-held-out 目标先通过。

## 下一唯一动作

运行 `JEV_TEMPORAL_VIDEOORA_TRANSFER_V1` 的 T0：在 PHD2 和 YouTube Highlights 上用同预算 `left/keep/right` 做 batch=16 的 Jev Choice，记录 source-macro observed coverage、位置 only、候选顺序置换和回退率。T0 只回答 Jev 是否有可利用的内容相位信号；不进行 prompt/温度扫描，不制作官方包。若两个域都通过，再进入 PHD2 soft-vote/pairwise LoRA；任一域失败则保留中心父链。
