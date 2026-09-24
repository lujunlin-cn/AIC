# 实验历史

2026-09-25 从空实现开始。真实训练资产和结果受许可证闸门单独标记。

## A0_001（已运行，TVSum 时间代理）

- Hypothesis：冻结 ImageNet ResNet18 特征上的 Temporal U-Net 可以从 TVSum 人工重要性学得时间排序信号。
- Current bottleneck：无端到端基线和可评测联合标注。
- Change：首个 A0，最大合法居中 crop；不启用 TSM/Feature Bank/teacher。
- Control：相同来源隔离验证视频上的常数选择/均匀预算参考。
- Expected impact：验证是否存在可重复的时间信号，不预报比赛收益。
- Model-size impact：完整 backbone + head 预计 FP32 约 51MB，最终以实际导出为准。
- Compute estimate：单卡物理 GPU 2；先测特征吞吐，训练正常目标远小于 11h。
- Failure condition：非有限 loss、帧映射不符、数据泄漏或不优于合理参考时不能锁定 SAFE_BASELINE。
- Metrics：TVSum temporal proxy 与官方公式合成测试分开；没有真实 crop GT 时比赛 F_video/score 留空。
- Actual run：远程物理 GPU 2；缓存 43 个 TVSum 视频；训练 10 epochs，11.65s；proxy F1 0.011019；导出 FP32 51,319,217 bytes（51.319 MB）/ FP16 25,685,041 bytes（25.685 MB），均 S 档数学假设。
- Result：用户授权可下载数据用于实验；A0 作为 TVSum 时间代理结果恢复。该 proxy 指标仍不是官方 F_video，也没有空间 GT。
- Artifact：从远程 `/data/aic/quarantine/TVSum_A0_001_blocked` 恢复到 `/data/aic/features/A0/cache` 与 `/data/aic/experiments/A0_001`。
- Status：Accepted for controlled research; not SAFE_BASELINE until official-like joint evaluation exists.

## A1_001（已运行）

- Hypothesis：在相同 TVSum cache、split、Temporal U-Net、seed、训练预算和模型大小下，参数为零的局部通道时移可改善短时边界代理指标。
- Change：仅启用 `temporal_shift=true`；A0 cache 与 backbone 固定不变。
- Control：A0_001。
- Expected effect：时间代理 F1 可能改善；不预设比赛联合收益。
- Model-size impact：新增 0 个参数；导出文件只因 metadata/序列化轻微变化，实际 bytes 重新审计。
- Actual run：远程物理 GPU 2；10 epochs，12.22s；proxy F1 0.016484。
- Result：相对 A0_001 的 proxy F1 提升约 49.6%（单 seed、TVSum summary proxy，不能外推官方 F_video）。新增参数 0；FP32 51,319,281 bytes，FP16 25,685,105 bytes，均 S 档数学假设。
- Failure condition：后续需多 seed 和联合空间评估确认；当前不锁定 SAFE_BASELINE。
- Status：Promising proxy result; continue controlled validation.

## A2_001（已运行，拒绝）

- Hypothesis：固定 32D motion/quality/audio/composition bank 融合到 A1 的 128D 时序投影，可以补充低成本边界与质量线索。
- Change：仅增加 Feature Bank MLP；A1 backbone/cache、split、训练预算和 TSM 固定。
- Actual run：物理 GPU 5；6 epochs，7.81s；proxy F1=0.000000。
- Size：12,817,921 parameters；FP32 51,362,293 bytes；FP16 25,707,253 bytes；仍为 S 档数学假设。
- Conclusion：当前 bank 定义/归一化造成明显代理退化；不继续无结构调参，保留代码供后续错误分析后重开。
- Status：Rejected for current proxy; A1 remains temporal control.

## A3a（实现完成，待空间 GT）

- Change：新增无权重合法 crop candidate、gradient saliency center、shot-aware EMA smoothing；最大合法居中 crop 仍是固定控制。
- Evaluation gate：必须在目标比例的人工/合法 crop GT 上固定 temporal frame 集合比较；TVSum 没有 spatial GT，因此暂不训练或宣称空间收益。
