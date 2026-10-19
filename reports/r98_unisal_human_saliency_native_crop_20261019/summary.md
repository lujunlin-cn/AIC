# R98：AIM 2024 / UNISAL 视觉注意力迁移

## 研究问题

AIM 2024 的视频显著性挑战表明，多帧视觉上下文可以产生人类注意力图。这里测试一个最小可部署迁移：把已有 UNISAL 时序视觉模型的 saliency mass 中心用于 AIC 合法 crop，是否能超过当前 Qwen 构图点。声音通路按项目已有负迁移证据排除。

这是空间构图诊断，不把显著性图当作高光标签，也不改变 R54 时间集合。

## 实现与输入

- UNISAL `weights_ft_mit1003.pth`，参数 `15,368,928`，冻结推理；权重身份已登记在 `研究1/open_models_registry.md`。
- LIVE-YT-VC 的既有 T5 表格与每个源 1 秒 keyframe，148 个 source-level dev 源。
- 每个 keyframe 使用真实时序 UNISAL 视觉路径产生 saliency map，取质量质心、前 50% mass 质心和合法 9:16 crop。
- 对照为几何中心和同一 T5 表格中的 Qwen point-centered crop；只计算有 native crop GT 的 keyframe，不把缺失标注当负例。

## 结果

完整 148 源结果见 [`r98_unisal_human_saliency_native_crop_20261019.json`](../r98_unisal_human_saliency_native_crop_20261019.json)。源级 keyframe IoU 均值：

| 方法 | mean IoU |
|---|---:|
| Qwen point | 0.52594 |
| center | 0.47903 |
| UNISAL saliency centroid | 0.48871 |
| UNISAL 50% mass centroid | 0.49230 |

相对 Qwen point 的配对差：

- UNISAL centroid：`-0.03723`，95% bootstrap CI `[-0.06317,-0.01117]`，60/148 源更好、88/148 更差。
- 50% mass centroid：`-0.03364`，95% CI `[-0.05943,-0.00794]`。

20 源 smoke 也为负；全开发集结果排除了运行偶然性解释。

## 结论与边界

关闭“UNISAL/通用人类显著性中心直接替代 Qwen 构图点”这一具体配方，不出官方包。该结果不证明 AViMoS 或 SalFoM 的空间 token 没有信息，仍有三点未被测试：

1. 显著性图作为候选 utility 的二级特征，而非直接中心；
2. 目标域 crop 监督下的高分辨率局部读出；
3. AViMoS 视觉注意力预训练后再用 AIC native crop 监督适配。

首轮不再扫 UNISAL 的中心公式、平滑或音频分支；这些改变不会解决“注意力与赛事构图目标错位”的已测问题。
