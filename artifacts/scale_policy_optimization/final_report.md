# P2 Route-specific Scale Policy Optimization

更新时间：2026-11-08T05:00:59.128262+00:00。唯一父链 **P2**。状态：`PARTIAL_OFFICIAL_RESULTS_OWNER_REPORTED`。

## 结果与候选

|候选|A|B|Native|视频数|输出帧数|本轮官方 F1|相对新 C0|
|---|---:|---:|---:|---:|---:|---:|---:|
|C0|1.00|1.00|1.00|890|263970|null|null|
|C1|0.90|0.95|1.00|890|263970|42.23|null|
|C2|0.93|0.97|1.00|890|263970|null|null|
|C3|0.90|1.00|1.00|890|263970|42.87|null|

成绩更新（2026-11-08）：负责人通过聊天回传 **C1=42.23、C3=42.87**，来源见 `owner_feedback_C1_C3.json`，已绑定对应 ZIP SHA256；未由代理独立登录平台核实。C3 高于 C1 **0.64 分**。C0/C2 本轮成绩仍为 null，不能据此晋级。

原 P2 ZIP 与 C0 字节相同，SHA256 仍为 `1489ace9d0ab5108af257800fb7a0cf83d07cfccc2b8c9a0c2f53e16fd557644`。
本轮任务给定 P2 历史成绩为 44.63，旧记录为 44.65；二者保留来源冲突。它们没有被冒充为本轮新 C0 回分。
常数选择依据来自此前 P1 公开 LIVE 诊断；本轮全部坐标、时间、路由来自 P2。公开空间 IoU 不换算为官方 F1。

## 八项结论与证据

1. **已回传成绩最高候选**：C3=42.87，高于 C1=42.23 共 0.64 分（均为负责人聊天回传）。C0/C2 未回传，四候选最终胜者和相对新 C0 的增量仍为 null，未晋级新候选。见 `official_results.json`、`owner_feedback_C1_C3.json`。
2. **A/B 改动与贡献**：A=266 视频/71407 帧；B=228 视频/56537 帧。C1/C2 同时修改 A/B；C3 只修改 A。官方分数贡献为 null，不能由框变动量推算。见 `route_coverage.json`。
3. **Native**：396 视频/136026 帧，四候选逐框不变。见 `independent_audit.json`。
4. **QP**：C1 shot审计=544、失败=0、宽度变化=0、最大QP中心变化=0.0 px; C2 shot审计=544、失败=0、宽度变化=0、最大QP中心变化=0.0 px; C3 shot审计=297、失败=0、宽度变化=0、最大QP中心变化=0.0 px。额外固定位置等式是已登记的后处理修改，不能描述为完全未改 QP。见 `qp_scale_audit.json`。
5. **Position Drift**：C0 最大=0.0 px、非法窗口=0、回退输出帧=0; C1 最大=0.5 px、非法窗口=0、回退输出帧=0; C2 最大=0.5 px、非法窗口=0、回退输出帧=0; C3 最大=0.5 px、非法窗口=0、回退输出帧=0。来自整数取整的偏差满足 ≤1 px。未 clamp、未修改 P2 Adapter。见 `geometry_audit.json` 与 ZIP 独立复算。
6. **时间集合**：890/890 视频的 frame 顺序/成员/数量 100% 与 P2 一致；每包 263,970 帧，3 个空视频保持原样，新增/删除均为 0。见 `temporal_identity_check.json`。
7. **完整复现**：890/890 原 P2 CPU 部署重放逐框相同；协议、输入、源码、每源可恢复收据和 ZIP 均已 SHA256 锁定。C0 字节级复用，其他 ZIP 固定时间戳。媒体复核：890/890 原视频全解码，444581 原始帧，SHA256/尺寸/帧数/PTS 全相同。
8. **是否保留为最终提交**：四包保留为可评测候选；只有真实官方 F1 严格高于新 C0 才允许晋级。目前不覆盖原最佳提交。

## 部署几何与稳定性

从最终 P2 `[x,y,w]` 计算实际中心；新宽度为 `floor(w*Scale+1e-9)`，高度仅由目标画幅推导，两个轴分别就近取整。禁止 Scale 后再调用会恢复最大宽度的原整数导出器。
先重放原完整事件的 P2 平滑与 QP（max_speed=0.15、P2 qp_floor=0.6），再缩放并执行登记了 `lo=hi=scaled_baseline_position` 的原 QP；最后只恢复 P2 原 target_frames。任何非法窗口按原镜头回退，QP 失败按原源回退，均保存原因。
C1-frame 与 C1-segment 在原镜头/事件边界内分别计算；每路线固定一个动作，所以镜头内中位数仍为同一常数。两者 JSONL SHA256 均为 `df8113d8ecf62353aad59b39be9b8455e9a06cf423c0e9d1357622418efdf724`，与 C1 相同。没有独立收益，不额外提交 segment 变体。
**B 风险**：0.95/0.97 仅有转置 LIVE 的历史证据，没有原生竖屏转横屏公开域验证；本轮也未用隐藏标签验证。

## 预算、合规和复现入口

本轮 GPU 实际消耗 0.0 秒 / 36000 秒；无模型推理、无 SFT/DPO、无权重修改。独立共享预算台账 `gpu_budget_ledger.json` 不随重启重置。
完整推理参数仍为 8,999,771,458（含原保守 buffers）；本轮增加 0 参数。P2 原 base/A/B 身份沿用已核验记录，不把 ZIP 身份当作权重身份。

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/build_scale_policy_candidates.py --root artifacts/scale_policy_optimization --workers 8
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_scale_policy_candidates.py --root artifacts/scale_policy_optimization
# 仅远端 CPU 全媒体核验（所有包共用一次解码）
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/audit_scale_policy_candidates.py --root artifacts/scale_policy_optimization --video-root /root/autodl-tmp/半决赛/video --workers 8
PYTHONPATH=. python scripts/finalize_scale_policy_optimization.py --root artifacts/scale_policy_optimization
```

输入需保留 `inputs/`，官方媒体、原预测、缓存和 ZIP 只保留在本地/授权远端，不上传公开 GitHub。`protocol.json` 含输入摘要；`inputs/cache_manifest.json` 含逐缓存 SHA；`work/*/attempt_*` 含每帧四阶段中心/宽度/框与回退原因。
CPU 验证证据见 `validation_cpu.json` 和测试日志；原路径源码保存在 `inputs/runtime/` 并锁定 SHA。

## 剩余阻塞与下一步

公开网页可访问，但本地/远端未配置可复用的已登录官方提交适配器；历史公开排行榜查询不是官方评测接口，半决赛多次提交额度未确认。已请求负责人提供既有登录入口/脚本或手工回传结果。
已记录负责人回传的 C1=42.23、C3=42.87；C0/C2 本轮成绩仍为 null。等待剩余成绩和平台凭据；不将旧 P2 分数填作新 C0，不因局部回分晋级，不运行更多实验或训练。
可用 `--official-feedback` 导入带同一半决赛索引、对应 ZIP SHA256 和平台结果文件 SHA256 的反馈；不接受其他 ZIP 或代理 IoU。原结果会归档，不会被 null 重置。

## 提交包 SHA256

- C0: `1489ace9d0ab5108af257800fb7a0cf83d07cfccc2b8c9a0c2f53e16fd557644`；`candidates/C0/C0_P2_Scale.zip`
- C1: `1d2d9dd63ff6e22eba5124046c68656208e448a531f16d42e4d48ce3558478fa`；`candidates/C1/C1_P2_Scale.zip`
- C2: `ff0df4a23f3f4724d4ddde1934c373e622a35ffc3451305c03fcb8868bad653f`；`candidates/C2/C2_P2_Scale.zip`
- C3: `436abbef0316bfaa5a97347932fc6195d64d0a16001e53dbb30083667a18e518`；`candidates/C3/C3_P2_Scale.zip`
