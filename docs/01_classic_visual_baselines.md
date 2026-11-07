# 路线 01：轻量视觉基线、手工特征与传统多视图

整理日期：2026-11-07。本文可单独提供给 GPT 网页版；全项目入口为 [index.md](index.md)。这是实验回顾，末尾问题是讨论题，不是已批准的新实验。

## 主假设与路线

用冻结图像主干提取逐帧特征，再训练小型时间头，能以很低成本获得高光排序；运动、画质、构图等传统信号可能补充语义特征。

实际链路：`视频 → ResNet18 / DeiT-S / ViT-B 特征 → Temporal U-Net / 线性头 / TCN → 时间选择 → 中心或轻量空间框`。历史 A/B/C 指 CNN/ViT/teacher 路线，与后期 Video-ORA 的空间 A/B 无关。

## 子方向与实际结果

| 子方向／实验 | 实际尝试 | 结果与证据边界 |
|---|---|---|
| A0_001–012 | ResNet18 + Temporal U-Net；种子、batch、标签修复、BCE/SmoothL1 | A0_006 的 TVSum dev 阈值校准后 binary proxy F1=0.15188；固定 0.5 仅 0.05661。不是 AIC 联合分 |
| A1_001–010 | 最终 embedding shift；随后真正 ResNet 层内 TSM | 内部 TSM 五折相对 A0 平均 −0.00830；参数增量为零，但没有稳定收益。两种 shift 不可混称 |
| A2_001–005 | 32D motion/quality/composition/audio bank 及分组消融 | 发现恒零、重复维；motion/quality/composition 的 proxy F1 仅 0.02724/0.01892/0.01613；audio-zero=0。此处没有测试真实音轨 |
| B0、Bs0 | ViT-B/16、DeiT-S/16 冻结特征，同一 U-Net | DeiT-S 约 23.28M 参数，七视频历史 lockbox F1=0.291867，但五折均值约 0.14389；小 holdout 的高值没有支持普遍领先 |
| RG_NCV_* | 3 表征 × 2 repeat × 5 outer fold × 3 seed，共 90 次训练 | TVSum summary F1：A0/DeiT/ViT-B=0.21521/0.23073/0.23064；DeiT−A0 CI 跨零 |
| RG_LINEAR_* | 60 次嵌套训练，只改线性头 | A0/DeiT summary=0.21324/0.21781，未超过 U-Net；SumMe OOD 也未支持替换 |
| RG_RANK_001 | BCE + 0.1 pairwise logistic，A0/DeiT 各 30 个外层评估 | summary=0.217504/0.233615；有局部排序信号，未获得 AIC 提升证据 |
| SUMME_OOD / CV_OOD | 恢复 raw 视频、作者 evaluator 对照、冻结头跨域 | 合并 14 源，CV summary A0/DeiT/ViT-B=0.21255/0.13408/0.14361；说明 TVSum 单池排序不可直接迁移 |
| SUB_A/B/C | 174 视频原视频部署，分别 A0+中心、DeiT+中心、DeiT+YuNet | 初赛官方回分 1.01/6.08/6.64。B/C 时间集合相同，YuNet 的受控增量 +0.56 |
| R90 | TVSum 静态 ResNet 与 32D 低层时序 bank | 静态 F=0.256248；bank=0.246228；静态+时序 bank=0.256212。简单差分、novelty 无可靠增益 |
| R91 | ResNet18+TVSum 时间读出 + V9/B3 空间，完整约 98.6M 参数 | 复赛官方 33.87，低于 R54=36.22。否定的是这套监督和部署合同，不能推广为“小模型不行” |
| R92 | RGB、低频、高频、全局运动补偿残差四视图，gated fusion | 5,886 PHD2/GIF 片段、448 dev 源：RGB F=0.415595，多视图=0.415335；Δ=−0.000260，CI 跨零。运动置换没有破坏所谓收益 |
| R94 | U²-Netp 主体观察与同容量零输入控制 | LIVE dev Δ=−0.000463，confirm3 +0.000834，CI 均跨零；不出包 |
| R95 | 显式片段 shot 结构与匹配 RGB 控制 | 448 源 Δ=−0.000364，CI 跨零；只是 embedding 代理镜头结构配方，不代表真正 shot 边界全部无效 |

## 哪些结论经过了修正

最初的 padding 处理改变了 GroupNorm、pooling 和 interpolation；同一短序列补齐到更长长度会出现最大 0.57 的输出差。加入真实 lengths 后差为零，旧读数不能与修复版直接混用。

阈值和评价目标会改变排序：固定 0.5 时 A1 看似领先，dev 校准阈值后 A0 更好。TVSum binary proxy、作者式 summary、Spearman 和 AIC 同帧连续 IoU 加权 F 是不同指标。

早期文档的“文件大小档位”“M/S tier”和规模系数属于历史解释；当前总推理参数 ≤9B、复赛无 k_size。不能再为压缩文件字节数预支分数收益。

## 当前可保留的资产与判断

轻量特征缓存、长度正确的 U-Net、嵌套源级划分、作者 evaluator 对照和原视频 JSONL 链是可复用基建。传统多视图已实际执行，但当前低层观察、冻结主干和监督组合没有显示可靠增益。R91 提供了真正的小参数完整系统负结果，不能用“尚未试过小模型”描述项目。

## 与 GPT 讨论的问题

1. 若重新做轻量模型，新增价值应来自目标域监督、连续事件政策还是更好的空间监督？怎样避免仅重跑已有 TVSum/PHD2 配方？
2. 传统处理是否应只用于边界、候选生成或质量诊断，而不承担逐帧高光排序？需要什么控制才能区分这几种用途？

## 证据入口

- `EXPERIMENTS.md`：Repaired A0/A1、Representation generalization、RG_RANK、R89/R90。
- `reports/20260925_representation_generalization.md`、`reports/internal_tsm_20260925.md`。
- `reports/r91_lowparam_resnet_20261019/summary.md`、`reports/r92_multiview_traditional_route_20261019/summary.md`。
- `reports/r94_u2_h1_20261019/summary.md`、`reports/r95_shot_structure_20261019/summary.md`。
- 后续真实回分覆盖早期“R91 未上传”：`reports/r107_r91_r108_r93_official_postmortem_20261020/summary.md`。
