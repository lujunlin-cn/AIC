# 技术决定

## 2026-09-25：从 Phase 0 / 1 启动

Status：Accepted（工程选择，非模型效果结论）。

Evidence：本地只有三份文档，远程项目与数据目录为空，无历史 baseline 可继承。

Decision：先实现比赛契约和 A0；TVSum 作者原视频为首个数据源；用显式指标名称区分时间代理与联合比赛指标；无同版本官方脚本时称 official-like，不称官方一致性验证。

## 2026-09-25：TVSum 许可证闸门

Status：Blocked。

Evidence：包内 `WebscopeReadMe.txt` 要求签署 Yahoo Data Sharing Agreement、获批非商业学术研究并禁止再分发/网络存储；README 的 CC-BY 表述不能自动覆盖该冲突。文件 SHA-256=`407d340bcd06fdc6d17374ebe6760b4a96816bcace228559c8283d9fb2520dea`，50/50 视频字节与解码元数据已核验。

Decision：用户明确授权本项目将可下载的数据集和模型视为可用于实验。保留 `license_gate`、来源和条款哈希用于追踪，但不再因该字段暂停训练；恢复 A0_001 资产，后续实验按用户授权继续。A0 的 TVSum 标签仍是 summary proxy，不是官方联合 GT。

## 2026-09-25：资源与时间边界

Status：Accepted（用户硬约束）。

Decision：物理 GPU 0/3 永不默认使用；GPU1 已有任务，首轮优先2并启动前重查。所有训练使用内部预算、定期 best/last 与外层 timeout，断线不终止。大文件仅在 /data/aic；代码 rsync 不删除远端数据。

## 2026-09-25：固定 TVSum 评估协议并修复变长推理

Status：Accepted for temporal proxy research。

Evidence：50/50 个 MAT 记录的 `user_anno` 都是 `(20,nframes)`，旧 linspace 展开最大标签差低于 `5e-8`；但旧 GroupNorm/pooling 路径在 5→12 padding 时有效 logit 最大差 `0.3392`，8→12 为 `0.5733`。修复后通过 `lengths` 逐视频计算，padding 回归差为 0。

Decision：GT 使用固定 `tvsum_summary_mean_norm_ge_0.5_v1`，prediction threshold 独立并只在 train/dev 选择；模型选择使用 video-macro temporal proxy，micro 仅诊断。TVSum continuous score、固定二值 proxy、作者 15% summary protocol 分开命名。A1 历史结果改称 feature-level shift。

## 2026-09-25：阈值校准优先于扩展模型

Status：Accepted for current development protocol。

Evidence：修复前 A0/A1 在 0.5 threshold 的空预测率分别约 0.938/0.750；A0_006 在 dev threshold 0.40 达 macro F1 `0.15188`，A1_005 在 0.35 为 `0.14474`。all-positive baseline 为 `0.08715`，说明原先接近零分数主要受校准和阈值影响。

Decision：候选必须保存 threshold 来源；禁止用评估视频真实正例数量决定预算。拿到官方 lockbox 后重新锁 threshold，当前数值不得称 AIC F_video。

## 2026-09-25：Feature Bank 当前降级

Status：Rejected for current definition; route remains open。

Evidence：第 11、23–28、31 维恒零，0/8/9、5/22、7/16 重复；motion-only、quality-only、composition-only、audio-zero proxy F1 分别为 `0.02724/0.01892/0.01613/0`，均低于 repaired A1。

Decision：停止当前无结构融合；清理重复/恒零维度并使用 residual/gated fusion 后再重开。全零 audio 结果不解释为真实音频无效。

## 2026-09-25：B0 作为 M 档参照保留

Status：Promising probe, not default fallback。

Evidence：ViT-B/16 frozen features + 同一 Temporal U-Net 在 threshold 0.30 的 TVSum proxy macro F1 `0.15928`，真实 raw-video JSONL 通过 validator；FP16 bundle `174,986,447` bytes，约 M 档。

Decision：继续做多 split/seed 和联合 GT 前的工程复现；在没有证明 raw F_video 增益足以覆盖 size coefficient 前，不替换 S 档 ResNet fallback。

## 2026-09-25：SmoothL1 不替换 BCE

Status：Rejected for current proxy。

Evidence：A0_012 只替换 loss 为 SmoothL1，fixed-0.5 macro F1=`0.11679`，threshold 0.40=`0.14143`；A0_006 的 BCE 对照为 `0.15188`。

Decision：保留 BCE；ranking loss 和更简单 temporal head 仍是下一轮实验，不把本次结果外推为所有回归损失无效。
