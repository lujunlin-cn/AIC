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

## 2026-09-25：冻结 TVSum local validation protocol v1

Status：Accepted for local temporal-proxy model selection。

Evidence：`reports/tvsum_manifest_v3.jsonl` 已有 source-group 隔离的 27 train / 16 dev / 7 test 分区；远程五个历史 fold 覆盖 43 条非 test 视频。机器可读协议为 `splits/local_protocol_v1.json`，并由 `scripts/validate_local_protocol.py` 校验 manifest/assignment hash、分区覆盖、source-group 隔离和五折互斥。

Decision：把原 manifest test 的 7 条视频永久作为 LOCAL LOCKBOX；不得用其调 prediction threshold、smoothing、top-k、segment、checkpoint、模型或超参。候选冻结后才可一次性比较 lockbox。TVSum 无可靠 category metadata，v1 不宣称 category-stratified；协议变更必须新建 v2，保留 v1 历史。

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

## 2026-09-25：冻结 local lockbox 与 zero-parameter policy

Status：Accepted。

Evidence：`splits/local_protocol_v1.json` 固定 27/16/7 source-group partitions。A0 Gaussian window 5 在 DEV 只有 `+0.00287`，在 lockbox 从 raw `0.114304` 降到 `0.112405`；gap/min-duration 也下降。

Decision：保留 raw A0 作为工程 fallback；所有后处理参数必须从 train/dev 产生，lockbox 只做一次冻结比较。当前不升级 smoothing、rank normalization 或 hysteresis。

## 2026-09-25：canonical internal TSM 暂不升级

Status：Accepted for current route prioritization。

Evidence：真实 layer1 feature-map TSM cache 的 chunk/full max error `1.81e-5`，参数增量为 0。五折固定 threshold 0.30：A0 `0.14631±0.03251`，internal TSM `0.13800±0.06299`，paired mean difference `-0.00830`；锁箱 internal TSM `0.175143`，强 A0_006 同 threshold `0.182033`。

Decision：保留实现、cache 和 correctness regression；降低 canonical internal TSM 优先级，不把历史 final-embedding shift A1 与它混称。

## 2026-09-25：DeiT-S/16 进入 S-tier challenger

Status：Promising, not replacement。

Evidence：timm DeiT-S/16 frozen features + 同一 Temporal U-Net，完整 FP16 bundle `46,618,447` bytes、`23,280,257` 参数。DEV median-9/threshold-0.35 为 `0.140241`，同一冻结 policy lockbox 为 `0.291867`；五折为 `0.14389±0.05173`；raw video → JSONL validator 已通过。

Decision：保留为当前最强 S-tier temporal challenger。它仍只有 TVSum temporal proxy 证据，必须经过第二独立 split/OOD 和 AIC 联合 GT 才能替换 fallback。

## 2026-09-25：OOD 与 VLM 暂不阻塞主线

Status：Blocked subtask, mainline continues。

Evidence：SumMe ModelScope 仓库 raw 视频是无进展的 LFS object；本轮停止下载，没有生成 OOD 分数。服务器没有可快速运行的本地 VLM teacher 权重。

Decision：不把下载阻塞或缺 teacher 误写成模型结论；下一轮优先取得可验证的 SumMe/YouTube Highlights raw 子集，再做小规模 VLM pilot。
