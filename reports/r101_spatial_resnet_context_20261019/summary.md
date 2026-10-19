# R101：当前帧高分辨率候选 × ResNet18 邻帧视觉上下文

这是 VistaHL 思路在当前 AIC 资产上的最小真实实验：

- 当前帧高分辨率分支：R70 `[129,2305]` 候选特征；
- 低分辨率时序分支：冻结 ImageNet ResNet18（11,176,512 参数）对前后各最多两个 keyframe 提取 512D RGB embedding；
- 交互：rank-32 Hadamard product；
- 对照：同容量零交互 control；负控制：跨 source 置换邻帧上下文；
- 训练目标：RetargetVid/LIVE native crop IoU，RV/LIVE source-disjoint split；不使用音频和官方测试标签。

结果：

- control dev macro keyframe IoU：`0.588335`；
- interaction：`0.588532`；增量 `+0.000197`，188 个 source unit 的 95% CI `[-0.001942,+0.002463]`；
- 置换上下文：相对 control `+0.001546`，CI `[-0.000439,+0.003707]`，没有显示真实 source-specific context 增量。

结论：在当前候选池、邻帧采样和 native crop 监督下，低分辨率 ResNet18 上下文没有可识别独立收益；关闭这条具体配方，不制作提交包，也不继续扫交互秩或融合权重。该结果不排除完整视频上下文、不同采样密度或保留空间 token 的更大结构，但下一次必须改变观察合同或监督，而不是继续微调此头。
