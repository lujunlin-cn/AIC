# R78/R81 直接部署框 utility 路线（2026-10-19）

## 目标

把空间效用目标从“129候选中的最佳窗口”改成最终提交框的 native IoU。公开数据中的目标定义为：冻结 B3 head 选出候选 `j`，训练目标为 `y[j]`。部署时输入实际可获得的候选 logits 与最终 V9 B3 输出位置，输出该部署框的预期 IoU。

官方标签没有读取，官方测试没有用于训练或校准。

## 公共实验

R70 full feature cache 共12822行、1663源，按 `*_train`/`*_dev` 源级隔离。

可用的直接读出变体包括：

| 输入 | 参数 | dev结果 |
|---|---:|---|
| selected candidate feature | 1,247,489 | MSE .06209，row Spearman .3001，source Spearman .0631 |
| selected feature + confidence | 1,248,513 | MSE .05864，row Spearman .3617，source Spearman .1119 |
| selected feature + 全129 B3 logits | 1,313,537 | MSE .05659，row Spearman .3868，source Spearman .1561 |
| topsoft 129 logits + B3输出位置 | **50,561** | MSE .04678，row Spearman **.4747**，source Spearman .1595 |

另一个独立回归头在 RV dev 的 top85 source macro 从 V9 score 的 `.68564` 到 `.69180`，LIVE dev 从 `.52972` 到 `.53424`；148源 bootstrap 区间为 `[-.0050,+.0142]`，因此这不是稳定官方增益证据。源内 pairwise objective 的 LIVE 为 `.53114`，没有超过回归头，停止 pairwise 分支。严格 nested source 检查把训练源拆为1196 inner-train、299 inner-tune，168 external dev 完全不参与选模；50,561参数头 external row Spearman=`.47334`、source-macro Spearman=`.15850`，与普通拆分一致，信号不是早停泄漏。

## 部署包

R81 使用公共训练出的 50,561 参数 head。官方推理只读取已有 R70 topsoft 关键帧 logits，并从审计过的 V9 B3 全帧父包取得每个关键帧的实际输出位置；不重跑或替换 V9 spatial artifact。时间决策采用：

```text
rank(R54 temporal value) * (0.5 + 0.5 * midrank(direct_output_utility))
```

包：[R81_DIRECT_OUTPUT_UTILITY_MULT_TIME_R54_V9_B3](../../submissions/R81_DIRECT_OUTPUT_UTILITY_MULT_TIME_R54_V9_B3)。使用nested checkpoint `topsoft_stats_nested.pt`，SHA256=`bfd026d5710ef7b85bfb0e7567b14b7e1a14f5ebd2f7e0163db5e272e02c0248`。

- 426/426 视频，199037 条预测；
- 8.856094947B 参数；
- 相对 R54 换帧 2393 次；
- ZIP SHA256：`80087b6e75fe70887db1f408e6b69baeae20812e5bb3e57ab3723de66dd1dfdb`；
- checker、ZIP roundtrip 通过；未上传，官方分数未知。

## 识别限制

A0 surrogate smoke 曾与历史 V9/B3 point 文件产生明显差异，因此不能用 A0 重跑结果冒称当前 V9 B3。R81 避开了这个问题，直接使用审计过的 V9 全帧输出位置；但公共目标的 B3 logits 生成资产仍必须保留来源和 SHA 绑定，之后若要正式上传应再次核对该 checkpoint 与训练标签合同。若这项身份核验失败，R81降级为 surrogate 诊断包，不上传。

R81没有联合时间 GT，仍不能把公开 utility 指标换算成官方 F。官方分数是下一识别步骤；R81不替换用户明天指定的2、3、5包。

## 决策

- 保留 R81 作为新的高价值联合候选，暂不自动上传。
- 不再继续扫回归损失、pairwise、MLP宽度或融合系数。
- 若 R70/R69/R71 官方回分支持空间路线，下一次优先验证 R81；若空间候选官方下降，关闭 direct utility temporal 分支。
