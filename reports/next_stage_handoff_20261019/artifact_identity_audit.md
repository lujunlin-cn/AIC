# AIC 现有提交产物身份审计

审计日期：2026-10-19。仓库基线：`41f5e9113b92e22e00367ce266c76527872eecf4`。

## 结论

**R54、R70 topsoft、R71 和 R84 的现有预测文件均通过两套 current checker。** 4 个文件均覆盖当前索引中的 426 个视频。R70 和 R84 完全保留 R54 时间集合。R71 从 R54 移除 3823 帧，保留帧的框没有改变。本轮没有训练、GPU 推理、生成 ZIP 或官方上传。

参数计数只来自已有 manifest。checker 没有加载权重，不能证明实际推理组件清单完整。本轮也没有重新解码原视频。

## 预测、时间集合与空间框

| 产物 | 预测数 | 相对 R54 改变时间集合的视频 | 新增帧 | 移除帧 | 共同帧中改变的框 | 参数声明（B） | 两套 checker |
|---|---:|---:|---:|---:|---:|---:|---|
| R54 | 199037 | 0 | 0 | 0 | 0 | 8.854531425 | PASS / PASS |
| R70_TOPSOFT | 199037 | 0 | 0 | 0 | 101797 | 8.854531425 | PASS / PASS |
| R71 | 195214 | 209 | 0 | 3823 | 0 | 8.854531425 | PASS / PASS |
| R84 | 199037 | 0 | 0 | 0 | 112141 | 8.853590257 | PASS / PASS |

各产物的逐字段结果、两套 checker 的真实命令与完整 JSON 输出、6 对候选差异均保存在 `artifact_identity_audit.json`。

### 预测文件 SHA256

| 产物 | 当前文件 SHA256 | 与原 manifest 一致 |
|---|---|---|
| R54 | `62371bc54d62b72eafd7168898ab5b23fad79a6c252ebde458b8fc78d033983e` | 是 |
| R70_TOPSOFT | `58b401a2c35f56760354ac880b64cbd661e891ea1441d49fed007c06e882d506` | 是 |
| R71 | `937857b27672982d676e40262c6e3f391bb6a1481688101d2b75d6083ee546dc` | 是 |
| R84 | `22e668f1dbd9d36694d60d5ee191a3437508cf9cb2b4cb0ebc3d9945be7c1616` | 是 |

当前索引：`reports/r47_timelens2_index_enriched_426.jsonl`。SHA256：`02d52321e82a70cafd33d566319b2ebf5e405af43946d6da32181eccdbc46a93`。索引包含 426 个视频；原始媒体路径没有在当前环境重新验证。

## R70 topsoft 身份与原 manifest 缺陷

R70 当前 JSONL 的 SHA256 与原 manifest 内的 `prediction_sha256` 一致。原 manifest 还将该 JSONL 绑定到报告中的 ZIP SHA：

`da5fcc5dccbfcd960b112852295dbbba35be89aaad61c20fcbb81d1ba8fed4d2`

ZIP SHA 同时出现在原 `.zip.sha256` 侧车文件和 topsoft 报告。因此，现有记录支持该 JSONL 属于 `da5fcc...` 版本。当前 ZIP 字节不存在，不能独立验证 ZIP 内容，也不能证明平台实际上传的是该文件。用户给出的官方分数单独记录为用户报告。

原 manifest 末尾含字面量 `\n`。严格 JSON 解析报 `Extra data: line 29 column 2`。本审计仅在内存中移除这 2 个字符以读取字段。**原 manifest 没有修改。** 原文件 SHA256：

`70a1cef9647cfdaeb82a6071ab653c74de1c5cc3ab1800b01bf6338026219930`

## R84 现有产物路径

预测文件：`/workspace/scratch/0ef05647638d/AIC-current/submissions/R84_COORD_CONSISTENCY_SPATIAL_R54TIME/predictions.jsonl`。文件存在，SHA 已核对。

历史记录中的 ZIP 路径：`/workspace/scratch/0ef05647638d/AIC-current/submissions/R84_COORD_CONSISTENCY_SPATIAL_R54TIME/R84_COORD_CONSISTENCY_SPATIAL_R54TIME.zip`。该 ZIP 当前不存在。

历史记录中的 ZIP SHA256：`526d3e0b41c329806c7746e0f7c1404b0ec8d3d2470495e8470c0f25983934f4`。本轮没有重建 ZIP，也没有把 R84 命名为新实验。

R84 声明参数组成为 TimeLens2 8,767,123,696、V9 vision tower 85,843,200 和 head 623,361，总计 8,853,590,257。只核对了数值求和和预测字段。没有权重文件可重新计数。脚本不加载 YuNet，但最终导出仍读取继承的 raw/reset 缓存；完整上游缓存依赖尚未复现。

## 当前计算与资产可达性

本环境 Python 为 `3.12.14`。检查时没有 torch、GPU 设备、nvidia-smi 或 SSH agent。SSH 目录中普通文件数为 0；本轮没有读取任何凭据内容。

| 已记录入口 | 只读检查结果 | 可以得出的结论 |
|---|---|---|
| 最新 R70 配置中的 AutoDL westc:22527 | 普通和提升网络权限后均 DNS temporary failure | 尚未建立连接；不能据此断言实例停机 |
| 历史学校服务器 119.62.14.24:22022 | 普通沙箱阻止连接；提升网络权限后 Connection refused | 检查时该端口拒绝 TCP 连接 |

两次检查只尝试 DNS、TCP 和读取 SSH banner，没有发送凭据，没有读取或修改远端文件，没有运行训练或杀进程。没有取得远端 GPU 或路径状态。

当前工作树内，视频、pt/pth/bin/safetensors 权重、npy/npz 特征、pkl 源表、ZIP 的扫描计数均为 0。已有 JSONL 预测和 JSON/Markdown 报告不等于原始训练或推理资产。

## R70 / R84 的真实输入路径与标签版本

R70 代码读取 `T5_CROPHEAD_V3/tables/{rv_train,live_train,rv_dev,live_dev}.pkl`。R84 读取 R70 导出的 `[129,2305]` 特征 shards，沿用其中的 `source` 和 `name`。这条路径不同于后续 V8 B3 的训练 manifest。

| 项目 | 当前代码给出的版本信息 | 本会话验证边界 |
|---|---|---|
| RV 划分 | T5 生成器：001–080 训练，081–100 开发；每源两种原生比例 | 实际 pkl 不在本地，不能证明缓存与当前生成器完全一致 |
| LIVE 划分 | T5 的 live_train_index.jsonl，sha1(video_id)%10==0 为开发；R70 历史指标记录开发 148 源 | 原 index 缺失，无法核对实际训练源数与源互斥 |
| RV 标签 | 6 名标注者原框负坐标裁到 0；R70 重算 half-open IoU 后对标注者取均值 | 未读取实际训练标签，不声称重新验收 |
| LIVE 标签 | T5 将 raw_ltrb 的 x 裁至 [0,W−1]、y 裁至 [0,H−1]；R70 对这些框重算 half-open IoU | 属于经 T5 处理的原生监督，不是未经处理的原始框 |
| R84 划分 | name 以 _train 结尾为训练，其余输入行为留出；归一化只用训练行 | 训练器没有独立 source 互斥断言；实际 shards 不存在，尚未核验 |

V8 B3 的 RV 001–030+621–700 训练、601–620 开发划分属于另一条路径。不得将该 110 源规则套用到 T5 支持的 R70/R84。早期 frozen spatial_crop_v1 的 RV train=0 也不能替代这两套后续实验配置。历史公开验证暴露仍保留，不把这些集合称为新的盲测。

远端待核对的权重、processor、T5 表和 shards 路径已记录在 JSON 中。当前本地均不存在。

## 可复现检查命令

以下命令只核对现有 R84 预测。参数值来自历史声明，执行命令不等于重新计量权重。

```bash
python -m aic.contract validate \
  --index reports/r47_timelens2_index_enriched_426.jsonl \
  --submission submissions/R84_COORD_CONSISTENCY_SPATIAL_R54TIME/predictions.jsonl \
  --stage current --require-sorted \
  --actual-model-params-b 8.853590257

python scripts/independent_submission_check.py \
  --index reports/r47_timelens2_index_enriched_426.jsonl \
  --submission submissions/R84_COORD_CONSISTENCY_SPATIAL_R54TIME/predictions.jsonl \
  --contract current --parameter-count 8853590257
```

## 停止边界

当前可以确认现有预测身份与结构合同。获得可用 GPU 会话、源表、模型权重和输入媒体之前，不能声称完成新模型训练或原视频端到端复现。现有 checker PASS 也不能推导官方涨分。
