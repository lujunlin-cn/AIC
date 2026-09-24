# 实验历史

无既有实验。2026-09-25 从空实现开始。

## A0_001（准备中，未运行）

- Hypothesis：冻结 ImageNet ResNet18 特征上的 Temporal U-Net 可以从 TVSum 人工重要性学得时间排序信号。
- Current bottleneck：无端到端基线和可评测联合标注。
- Change：首个 A0，最大合法居中 crop；不启用 TSM/Feature Bank/teacher。
- Control：相同来源隔离验证视频上的常数选择/均匀预算参考。
- Expected impact：验证是否存在可重复的时间信号，不预报比赛收益。
- Model-size impact：完整 backbone + head 预计 FP32 约 51MB，最终以实际导出为准。
- Compute estimate：单卡物理 GPU 2；先测特征吞吐，训练正常目标远小于 11h。
- Failure condition：非有限 loss、帧映射不符、数据泄漏或不优于合理参考时不能锁定 SAFE_BASELINE。
- Metrics：TVSum temporal proxy 与官方公式合成测试分开；没有真实 crop GT 时比赛 F_video/score 留空。
- Status：准备中。
