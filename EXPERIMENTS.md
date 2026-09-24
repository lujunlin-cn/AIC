# 实验历史

2026-09-25 从空实现开始。真实训练资产和结果受许可证闸门单独标记。

## A0_001（已运行，合规无效，隔离）

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
- Result：**无效，不得用于比赛或 SAFE_BASELINE**。TVSum archive 内 `WebscopeReadMe.txt` 要求签署 Yahoo DSA、获批非商业学术用途并禁止再分发/网络存储；AIC 竞赛训练许可未确认。
- Artifact：已移至远程 `/data/aic/quarantine/TVSum_A0_001_blocked`，不再被训练 manifest 引用。
- Status：Rejected/Quarantined（license gate）。此 proxy 指标不是官方 F_video。
