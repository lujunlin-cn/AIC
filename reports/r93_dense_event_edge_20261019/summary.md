# R93：4fps TimeLens2 dense event edge（2026-10-19）

## 目的

R54 的官方正向变化主要表现为连续窗口的事件相位修正；R92 已关闭低层多视图路线。R93 改变可观测信息：用同一 generic-highlight prompt 将 TimeLens2 从 2fps 提高到 4fps，再用区间端点证据做固定 `eta=0.20` residual。空间框、每视频输出数量和 V9/B3 父链保持固定。

## 推理合同

- 服务器：`autodl-container-d71c439741-85cf907a`，RTX 4080 32GB。
- 环境：`/root/miniconda3/envs/train/bin/python3`，PyTorch `2.14.1+cu130`，Transformers `5.19.0`；补装 `qwen-vl-utils` 和 `decord`。
- 模型：`/data/aic/pretrained/TimeLens2-8B`，总推理参数和 V9/B3 组件共 `8,854,531,425`，`model_params_b=8.854531425`。
- 输入：复赛 426 个原视频，4fps、`max_pixels=50176`、`max_new_tokens=256`；原始响应无错误，覆盖 426/426。
- 原始响应 SHA256：`dffce676ee1cf8ef8942c1ec5c6dc937884c847f8bc02217a6e661134808822c`。

## 解码

每帧使用完整视频时间轴上的端点证据：对预测区间 `[a,b]`，区间内相位为 `(t-a)/(b-a)`，端点效用为 `2*abs(phase-0.5)`；多区间取最大值。最终分数为中心85排序加端点秩 residual，严格选择 `round(0.85*N)` 帧。

## 包级结果

- 包：`submissions/R93_DENSE_EVENT_EDGE_V9_B3/R93_DENSE_EVENT_EDGE_V9_B3.zip`
- 426/426 视频、199,037 条预测、0 fallback。
- 独立 checker：通过；ZIP round-trip：通过。
- predictions SHA256：`00c4e2bd875e7584b16c1f947e4bcf55f733b69aec0d8ffa9a6242f1945e4d26`。
- ZIP SHA256：`bc05c7cbd4fa60ea2c01d9a928c55b5704bc7b83748279cf39530c33fd76a28`。
- 官方分数：待用户上传，当前不能称官方收益。

与已有候选的帧集合差异：

| 对照 | 变化视频 | 平均交换帧数 | 最大交换帧数 |
|---|---:|---:|---:|
| R56 2fps edge | 186/426 | 4.90 | 174 |
| R60 4fps direct | 337/426 | 10.59 | 897 |
| R55 2fps phase | 332/426 | 11.67 | 174 |
| R54 | 330/426 | 11.77 | 221 |

R93 不是 R60 的重复包；同样使用4fps观测，但把区间端点结构作为独立时间决策。它的机制证据来自“观测路径改变”，不是公共标签或官方回分拟合。

## 决策与限制

R93 是当前事件观测路线唯一已生成的新提交候选，建议优先保留一个提交额度验证。不要在官方回分前继续扫 `eta`、prompt 或端点函数。若 R93 高于 R54/R70/R71，下一步只研究 dense interval 的 phase/edge 联合；若下降，关闭简单 4fps 加密和端点 residual，转向新的事件状态监督。

R93 仍使用 V9/B3 静态空间 artifact，参数合规依赖该已审计 artifact 的部署口径；官方测试标签、官方回分和伪标签均未用于训练或调参。原始响应保留在远端 `/data/aic/experiments/R55_DENSE_4FPS_20261019/`，不提交 Git。
