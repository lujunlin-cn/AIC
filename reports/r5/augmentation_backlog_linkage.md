# 数据增强候选池 × round-5 裁定的衔接分析

日期：2026-10-04。输入：`AIC_数据增强候选池_模型锁定后执行.md`（BACKLOG 状态，
模型锁定后执行的约束**继续有效**）。本文不做增强实验，只把 round-5 已裁定的
机制事实映射到 backlog 的优先级上，供 L 矩阵读数后升级为正式计划。

## 1. round-5 裁定如何改变 backlog 的前提

| backlog 隐含假设 | round-5 裁定 | 影响 |
|---|---|---|
| "某类视频天然被模型打高分"是需要打破的捷径（§4.3） | **已证实**：champion 内容贡献是视频级（real−xperm 仅 +0.003~0.004） | §4.3 从"候选"升级为"已确诊病症的处方" |
| "视频固定位置被模型打高分"（§4.3 position debias） | **已证实**：champion 时间排序=位置先验，且锚定切片撑起全部测量 F（A6 +0.108） | position debias 与切片协议改造是同一病灶的两面 |
| 增强在"切片已定"的前提下做 | 锚定切片本身就是最大捷径源（A6） | **所有增强实验的评估必须双切片**（Anchor+Neutral），单切片读数不可信 |
| FPS jitter 是普通增强（§2 A 级） | intake 60fps vs 训练池 25fps 是**真实部署域差** | 该项升格：不是增强，是 train/deploy 一致性修复，优先级应排到 A 级之上 |
| 模型锁定后再执行 | temporal head **尚未锁定**（L 矩阵判定中），champion 已证=位置先验 | 文档的等待逻辑正确且更强：在位置先验头上做增强无意义，先要有读内容的学生 |

## 2. L 矩阵读数后的执行顺序（建议升级稿）

### 若 O > P（native 内容线存活）

1. **Feature-level augmentation（§4.1）**——第一优先：
   与 native tubelet 训练完全兼容（冻结 encoder，TCN 阶段做
   noise/dropout/masking/插值，零重编码成本）。R5 §3.4 的 128+128 开发源
   正适合 controlled ablation。扰动尺度按文档要求从真实 feature 统计定
   （不用固定高斯幅度）。
2. **Position debias augmentation（§4.3 = R5 §3.2 起点 re-windowing）**——
   两份文档独立指向同一操作（文档的 context replacement / R5 的合法上下文
   起点变化），合并为同一实验：从原媒体重新取窗（非循环移位）、标签时间
   坐标同步变换、Anchor+Neutral 双读数。
3. MomentMix / context replacement（§4.2）：依赖 foreground 区间重组合的
   标签重生成基础设施，成本高一档，等 1/2 有信号再投。

### 若 O ≤ P（native 内容线关闭）

增强实验全部暂缓——位置先验头上任何增强都在优化一个不读内容的函数。
主推力转向：更大/不同的时间表征（受 §4.4 大编码器决策规则约束）或
切片协议根本改造。

## 3. 需要写进未来正式计划的补充约束（文档 §7 之外）

1. **双切片评估**：任何增强的 dev 读数同时报 Anchor 与 Neutral（A6 纪律），
   只报 Anchor 的增益视为切片协议依赖，不晋级。
2. **FPS 一致性**：intake 60fps 与训练源 25fps 的差距进 feature/契约哈希；
   FPS jitter 实验的结果同时作为部署域适配证据。
3. **增强不改结论方向**：文档 §7 已写"不因增强实验改变 backbone 结论"——
   补充对称条款：不因增强实验重开已关闭的线（期望F、S1 特征细分、S4 平滑）。
4. 每种增强登记进 claim ledger，状态词表照用。

## 4. 与现有资产的兼容性速查

| backlog 项 | 依赖资产 | 现状 |
|---|---|---|
| Feature-level aug | native tubelet npz（契约哈希 ac3b657f） | ✅ 256 源就绪 |
| Position debias | 原媒体重取窗（native_frame_contract 管线） | ✅ 契约已冻结可复用 |
| MomentMix | GT 区间重组 + 标签重映射工具 | ⚠️ 需新建，唯一缺基建项 |
| 空间分支 flip/jitter | B3 grid 特征重池化（v8_build_samples） | ✅ 管线在，但空间线当前排序靠后 |
