# R47 TimeLens2 + VISIONONLY B3（2026-10-17）

## 状态

已生成候选包，**未上传、无官方回分**：

`submissions/R47_TIMELENS2_V9B3/R47_TIMELENS2_V9B3.zip`

TimeLens2-8B 负责离线视频区间判断；本目录中的 `R47_TIMELENS2_V9B3` 使用 `/root/autodl-tmp/experiments/R44_RESIDUAL_001/b3_allframes.jsonl` 的完整 LFM B3 全帧框作为转换父表，因此它是用于时间集合审计的条件候选。真正以 VISIONONLY B3 框生成的包是同一工作区的 `submissions/R47_TIMELENS2_V9_B3`；上传时应优先选择后者。转换器只改变保留帧集合，按原始帧轴执行 `round(.85*N)`，没有重算或修改框。

经独立重建后的最终候选已整理到 `submissions/r47_final_candidates_20261018/R47_TIMELENS2_V9_B3`，其 VISIONONLY 框逐帧来自审计过的 V9 parent：`predictions_sha256=b177c6b389c4804479ba0923b63c13ab0d08a355479a420e68c6737f3974cb30`，ZIP SHA256 为 `067f9a47095bb8e7e3a2231ed2829358ef07e0246f94294d1941685e536be9e3`。此前使用完整 LFM B3 框的同名草包应废弃。

## 审计结果

- 426/426 视频，199,037 条预测，0 个区间 fallback。
- 远端逐视频检查：精确预算违例 0，保留帧的 B3 框不一致 0。
- 相对 R26 中心 85% 父集，平均每视频替换 11.48 帧，中位数 1 帧，平均帧集合交集 97.405%；143/426 视频完全不变。3 个视频出现多区间非连续集合，其余 423 个仍为连续集合。
- 本地 `scripts/independent_submission_check.py`：`valid=true`，426 视频，199,037 条预测。
- ZIP 只有 `predictions.jsonl`，解压往返通过。
- 参数声明 8,854,531,425（TimeLens2 8,767,123,696 + VISIONONLY B3 87,407,729）只适用于 `R47_TIMELENS2_V9_B3` 的实际 VISIONONLY 空间资产。当前目录的 full-LFM 框候选不能据此宣称实际部署已经替换为空间 vision-only runtime；完整 LFM B3 与 TimeLens2 的合计超过 9B，不能混用。

## 哈希

- predictions：`ef3088de38cc95b796618ac21c98b4c062246924fd979a23f2cf6991f2541261`
- ZIP：`fad3f20ad0cec607aa1b88a795b221211bed349eb3d80b1db2548d432eb8f55d`
- ZIP 大小：1,422,057 bytes

## 研究证据

12 个公开人工标注源、18 个视图上的固定 TimeLens2 区间转换代理由 `0.7044166` 提高到 `0.7213273`（`+0.0169107`，6/18 视图改善）。这是公共诊断证据，不等价于官方成绩；官方测试标签和回分未用于生成该包。
