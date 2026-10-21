# Video-ORA 42.48 复现验收（2026-10-21）

## 已确认

- 目标机使用 `/root/autodl-tmp/videoora42_clone/package/extracted/videoora_42_48` 中从 `VideoORA_42.48_复现包.zip` 解出的原始代码、配置、portrait/vertical LoRA 和 compact 基座。
- `verify_package.py`：`passed=true`，43 个包文件检查通过，未包含比赛测试数据。
- `self_test.py`：`passed=true`；parser、geometry routes、actor feasible QP、vertical axis invariants、public TRAIN schedules 均通过。
- 原始 8 个 safetensors 分片 SHA256 与 `method/MODEL_FREEZE.json` 的 `original` 完全一致；compact 第 1 分片 SHA256 为 `b0603904e611bf4c963ecbd581d72eecd95204e6a65a4c4eb85e46b0b76a621c`，与冻结清单一致，其余分片为原始分片链接并已由原始清单核验。
- 未改基座、未激活 temporal LoRA 的全包入口 `infer.py` 在授权公开 QVH 视频 `B6AEPicK9Bk_210.0_360.0.mp4` 上完成 fresh inference：1 视频、1350 帧预测、`validate_output.py` 通过，参数元素 `8,999,771,458`，官方分数保持 `null`。
- 该 native 目标比例回归产物为目标机本地新生成，未读取任何比赛测试预测或隐藏标签。

## 暴露的问题

同一授权视频强制使用 `[9,16]` 目标比例时，复现包的 portrait 路径在 `recovery_path_projection.apply_events` 的 y 轴不变断言处失败。该失败发生在包原始代码的运行时合同中，不能把它改成“复现通过”，也不能在修复前启动 RL。需要后续先定位 portrait adapter 输出、目标比例/路线假设与断言之间的关系，并在修复副本中补充回归。

包内 `LIVE-YT-VideoCropping` 原视频和 `GNMC` 原图没有随包分发；目标机当前只有公开 QVH/manual 资产。因此 A/B 两个 LoRA 的 240/210 步重训尚未宣称完成复现。当前完成的是基座/包审计和授权公开视频上的 fresh 推理合同复现。

## RL 状态

之前的 temporal RL 仅为资格 smoke 和吞吐实验，未使用来选择或融合 LoRA。正式 RL 在复现闭环完成前暂停；其 reward/gradient/output-homogeneity 监控代码保留在 `videoora_rl/temporal_rl_smoke.py`，不代表已获得官方增益。

## 完整 426 复赛推理与 RL 启动（2026-10-21）

- 在隔离副本 `portrait_fix_20261021` 中只放宽了投影器对非活动轴单像素舍入的前置断言；基座和两份包内 LoRA 未改。
- 三 shard 推理完成 `426/426`，输出 `133,921` 条预测，2 个空视频，参数元素 `8,999,771,458`。
- 12 进程并行独立审计通过：426/426 几何合同、426/426 源媒体 SHA256、预测顺序与索引完全一致。
- 本地 ZIP 根目录只有 `predictions.jsonl`；ZIP SHA256=`7da198e794f0e6029f92413f56b9e22bf01c97c83c840caaec30db4b909191ea`。本地副本位于 `repro_packages/videoora_42_48_semifinal_426_fixed/`。
- 该结果只有格式、几何和来源完整性证据；用户随后报告官方得分 `42.49`，本机没有平台上传字节或页面的独立核验。
- `TEMPORAL_RL_PUBLIC_CONTEXT_V1` 已在 PRO6000 启动，96 update qualification；当前由 `watch_videoora426_and_start_rl.sh` 每 30 分钟记录训练状态。尚未选择或融合 RL LoRA。
