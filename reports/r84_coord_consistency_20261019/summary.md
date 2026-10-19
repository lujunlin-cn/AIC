# R84 coordinate consistency spatial readout

日期：2026-10-19

## 机制

R84 保持 R54 的时间集合和 85% 预算不变，只替换空间框读出。它使用与 R70 相同的 129 个合法窗口特征，但采用 623,361 参数的逐候选小型 MLP，并在 native crop IoU 回归之外加入 soft-argmax 坐标一致性损失。训练只使用 RetargetVid/LIVE 的公开 native crop 标签，按 source split 留出。

## 公共留出结果

| 留出域 | R84 逐行 IoU | center 逐行 IoU | 增量 |
|---|---:|---:|---:|
| LIVE dev | 0.51277 | 0.48142 | +0.03135 |
| RV dev | 0.63009 | 0.62645 | +0.00365 |
| 合并 source-macro | 0.52306 | 0.49759 | +0.02547 |

R70 topsoft 在相同公开协议的两个域平均 source-macro 增量为 `+0.07006`，因此 R84 没有通过“超过 R70”晋级门。但它在 LIVE 的行为和 RV 的弱变化与 R70 不完全相同，仍可作为一个结构不同的高风险官方候选。

## 官方推理与包

- 426/426 视频，199,037 条预测，0 fallback；
- 112,141 个保留帧的空间框相对 R54 改变；时间集合完全继承 R54；
- 实际参数：TimeLens2 `8,767,123,696` + V9 vision tower `85,843,200` + R84 head `623,361` = `8,853,590,257`（8.853590257B）；
- predictions SHA256：`22e668f1dbd9d36694d60d5ee191a3437508cf9cb2b4cb0ebc3d9945be7c1616`；
- ZIP SHA256：`526d3e0b41c329806c7746e0f7c1404b0ec8d3d2470495e8470c0f25983934f4`；
- independent checker、final contract、ZIP roundtrip 均通过；
- 尚未上传，官方分数未知。

## 决策

R84 是真正的新算法和新读出，但公共增量小于 R70，不能把它当主候选。保留一个官方名额时，R84 的价值在于检验“R70 的收益是否来自特定大 readout，还是空间域存在可迁移的小模型信号”。如果 R84 官方明显低于 R70，停止坐标一致性读出；如果 R84 接近或超过 R70，再研究固定的 R70/R84 域门控，不扫描融合权重。
