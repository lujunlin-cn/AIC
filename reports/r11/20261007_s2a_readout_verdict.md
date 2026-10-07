# R11 S2a 读出判定（readouts a/b/c，live 池）

日期：2026-10-07。执行地：kmsp05 910A host CPU（提取 32 分片 ~35 min + 读出单进程）。
资产：`scripts/r11_s2a_live_extract.py`、`scripts/r11_s2a_readouts.py`、
`reports/r11/s2a_readouts.json`（910A 侧 `/data/aic/experiments_910a/LFM_V11/r11/s2a_readouts.json`）。

## 判定（预注册门对照，计划 v2 S2 步骤③）

| 读数 | 门 | 实测（eval=live_confirmation） | 裁定 |
| --- | --- | --- | --- |
| a. 新池 oracle 抬升 | ≥ +0.02 | **+0.0000045**（dev +0.0000065；top3 口径 +0.0000058） | **FAIL，低 3 个数量级** |
| b. B3 argmax 与检测框一致率 | （诊断项） | 0.506（dev 0.502） | 检测框与 B3 半数重合 |
| c. 检测先验条件重排 | ≥ +0.005 且 CI>0 | **+0.00454，CI95 [0.0021, 0.0071]**（dev β*=0.5 → +0.0044 CI>0） | **未过门（差 0.0005），弱正灰区** |

覆盖：RT-DETR r50vd person conf≥0.3 帧率 live_dev 56.5%、live_confirmation 58.4%
（低于 PHD2 池的 0.766；两池源内容不同，同池协议内自洽）。

## 机制判读（推断）

1. **a 门大败的机制**：9-16 窗在 640px 帧上自由轴 span≈437px，129 格点格距≈3.4px；
   检测窗（中心对齐检测框）与最近格点窗几乎重合。**格点网格已致密覆盖检测器给出的
   位置**——与 F3「33 格点 shortlist regret 0.0055 足够」同构。「dense 候选」支路按
   预注册关闭。
2. **c 的弱正是真实信号但不可用**：dev/eval 一致（0.0044/0.0045）、CI>0，方向对；
   但 0.0045 < 0.005 门，且其信息源（检测框位置）与 a 的证据矛盾——若检测框位置
   已被格点覆盖，c 的增益只能来自先验对 B3 分数的**重加权**而非新候选，量级受
   b（半数重合）封顶。不满足打包条件，按 S4 纪律不出包、不消耗名额。

## 处置

- S2 源 a（检测器）**收口关闭**：不再投入 dense 候选/检测先验方向。
- S2 三源剩 **S2c（Light-ASD active-speaker）**：前提审计已过门（speech_dominated
  41.7% ≥30%，day-1 判定），下一步是脸部轨迹×音频对位链路，等排期。
- day-1 的 PHD2 3917-frag 检测框与本轮 live 池检测框均为 S2c/S3 轨迹连续性的
  可复用资产（S2b 跟踪连续性读出仍开放，CPU 零依赖）。

## 过程教训

1. **池错位**：day-1 S2a 提取跑在 PHD2（keep-mask 协议池），而 readout 需要 B3 分数场
   所在的 live 池——执行前应先核对读数协议的数据合同，不是先跑提取。
2. PyAV seek(backward) 取不到首关键帧之前的帧（LIVE-YT-VC 需要 frame 0；PHD2 frag
   在中段掩盖了该缺陷）；LIVE-YT-VC 流首帧 pts 非零（+4 帧），帧号必须用**解码顺序
   计数**（与 v8_live_extract 一致），不得用 pts×tb×fps。
3. `pgrep -f` 链式任务自匹配死锁：链式 bash 命令行含目标串会永远匹配自己；
   用日志标记（如 grep pairs_total）判完成。
