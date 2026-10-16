# R31 低维事件状态探针（2026-10-16）

## 假设与协议

R30 的 768 维相邻差分在 TRAIN 高、DEV 低，可能是读出过拟合而非所有状态变化都无效。R31 将 VPB 观测压缩为范数、相邻余弦、窗口汇总等低维事件状态，再用 TRAIN-only 强正则读出；位置控制、精确 85% 预算、VideoXum 26 TRAIN/14 已暴露 DEV 和源级 bootstrap 固定。该 DEV 曾参与 R24/R26，因此只作机制诊断，不能作独立确认或官方分数预测。

## 结果

| 读出 | DEV F@.85 |
|---|---:|
| center | 0.288381 |
| position | 0.297399 |
| low-dimensional event state | **0.302285** |
| PCA(4) | 0.298328 |
| low + PCA | 0.301449 |

低维事件状态相对 center 为 +0.013904；源级 bootstrap 95% CI `[-0.000339, 0.031144]`，跨零。结果未达到预注册的“增量实用门”，不能宣称已找到可迁移内容信号。该读出也没有 AIC 426 对应特征缓存或可部署模型合同，不能直接包装提交。

## 判定

R31 支持“低维、强正则状态量比高维差分更不易过拟合”的弱证据，但位置/来源、DEV 暴露和标签语义仍可解释该收益。停止把 R31 probe 直接部署到官方集；不启动大时序训练。下一步使用已经在 AIC 426 上生成的冻结 temporal_k0.90 推理链，构建可复现的新候选 R32，并把 R31 作为诊断记录。

证据：`reports/r31/lowdim_event_probe/metrics.json`、`records.json`、`configs/R31_LOWDIM_EVENT_PROBE_20261016.json`、`scripts/r31_lowdim_event_probe.py`。
