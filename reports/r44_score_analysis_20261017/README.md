# R44/R26 微扰随机交换 null（2026-10-17）

`random_swap_null.py` 使用公开人工标签做诊断性 null，不读取官方 test 标签。它固定中心 85% 选择，在每个标注视图随机交换约 `6352/199037=3.19%` 的已选帧，交换数量为 `Poisson(0.032*K)`，并用 R44 已有的 soft-grade `proxy_f` 计算 18 个视图的宏平均差异。seed=1，20,000 trials。

结果见 `random_swap_null.json`。这个 null 只回答“同等规模近邻交换产生多大偶然波动”，不能给官方分数做显著性检验：标签来自同一账号，且不是比赛 GT。官方评估若固定测试集应是确定性的；“随机”指未知 GT 下这次交换的偶然净收益。

复现：

```bash
python reports/r44_score_analysis_20261017/random_swap_null.py
```
