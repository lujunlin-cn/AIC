# R46 空间候选效用审计（2026-10-17）

## 新判断

B3 的候选集合在公开源上明显够用，但当前候选排序器留下了很大的可达效用差距。`spatial_candidate_oracle.csv` 共 26,562 行，包含公开 `live_dev`、`live_confirmation`（seed0/seed1）和 `rv_diag_exposed`。候选 oracle 是同一 129 个最大合法窗口上的公开人工 crop GT 穷举，只作机制诊断。

- `live_confirmation`（227 个源）：B3 head IoU `0.53495`，center `0.46876`，129-candidate oracle `0.78188`。
  - B3−center `+0.06619`，源级 bootstrap 95% CI `[+0.04970,+0.08356]`，147/227 源胜出。
  - oracle−B3 `+0.24694`，CI `[+0.23131,+0.26295]`，227/227 源 oracle 更高。
- `live_dev`（124 个源）：B3 `0.54020`，center `0.46897`，oracle `0.77446`；B3−center `+0.07123`，CI `[+0.04690,+0.09576]`；oracle−B3 `+0.23426`，CI `[+0.21365,+0.25517]`。
- `rv_diag_exposed`（70 个源）：B3 `0.68203`，center `0.61983`，oracle `0.84390`；B3−center `+0.06220`，CI `[+0.03461,+0.09022]`；oracle−B3 `+0.16187`，CI `[+0.14671,+0.17841]`。
- `live_confirmation` 的两 seed 在源级相关 `0.963`，seed1−seed0 均值 `+0.00283`，CI `[-0.00181,+0.00760]`。因此 oracle gap 不是单纯 seed 噪声。

## 解释边界

该结果支持“优先修复 candidate utility readout / pooling / 域迁移”，不支持“129 候选需要加密”，也不证明编码器本身缺信息。候选 oracle 使用公开 GT，不能转换成官方涨分；`live_confirmation_s1` 与 `live_confirmation` 是相同源的两个 seed 视图，源级统计中没有把它们当独立确认集。

当前部署阻塞是：远程仍缺 V8 B3 原始 `head_s.pt` 与 LFM 基座；只有 V7 head 和 B3 point 输出。不能用 V7 冒称 B3，也没有足够证据直接替换官方 426 帧框。

## 下一最小实验

1. 恢复 B3 同一 129 候选 feature/cache/head 身份，或重建可核验的等价输入。
2. 在 `live_train/rv_train` 上训练受限新 readout（同候选、同最大窗口、同标签；先线性/小 MLP，再比较当前 B3 head），源级冻结 `live_dev` 与 `live_confirmation`，另报 `rv_diag_exposed`。
3. 仅当新 readout 在两个独立来源的源级 CI 同时超过冻结 B3 且不低于 center，才考虑对 426 重新提取视觉特征并做空间替换；否则停止空间 readout 路线。
4. 保持 B3 的时间 mask/85%预算不变，先测空间单变量；联合时间只能在 readout 通过后做 2×2。

复现：

```bash
python3 reports/r46_spatial_readout_20261017/analyze_oracle.py \
  --csv reports/spatial_candidate_oracle_remote.csv \
  --out reports/r46_spatial_readout_20261017/metrics.json
```

输入 CSV SHA256：`5cdfb5c8dc36255f2967bfd3f3c0d31756a954a9aed6cef8c56326cb0958d4dc`。

## B3 资产核验补记

2026-10-17 对新 AutoDL `22527` 做了有界库存检查：精确路径 `/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt`、`/root/autodl-tmp/evac/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt`、`/root/autodl-tmp/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt` 和 `/root/AIC/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt` 均不存在。对 `/root` 与 `/data` 中 4–12 MB 的 67 个 `.pt` 候选逐个 `torch.load` 检查 `D/nc/config/state_dict`，命中 D=2305、nc=129 的只有 V7、V9 A0/A1 及其 seed/KD 变体，没有 B3。

V8 B3 复赛 ZIP 的 `unzip -l` 只有一个 `predictions.jsonl`（11,015,344 bytes），不含模型、特征或 logits。V9 `A0_ctrl`（SHA256 `d8d5eebfaa3be87893030285105dbec5681654d8b8a0b74dd3ab00d6edb9917b`）是相同 GT 池的后续重训，不能宣称与 B3 权重等价。
