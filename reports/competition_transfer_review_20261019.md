# 外部视频竞赛经验的 AIC 迁移裁决（2026-10-19）

本报告对应 `video_competitions_research_and_agent_handoff.md`。我重新核对了 AIM 2024 原文与组织者仓库、SalFoM 官方代码、YouTube-8M 三篇 workshop 论文，并用当前 AIC 的 R92/R94/R96/R97 结果约束迁移结论。

## 已核实的外部事实

### AIM 2024 / AViMoS

论文 [arXiv:2409.14827](https://arxiv.org/abs/2409.14827) 的任务是二维人类注意力图预测，不是高光选择。冠军 420.5M UMT 多级解码，亚军 187.7M HiSal 将低分辨率多帧上下文和当前高清帧分开，季军 SalFoM 以动态、静态和局部—全局分支解码。组织者仓库 [ECCVW24_Saliency_Prediction](https://github.com/msu-video-group/ECCVW24_Saliency_Prediction) 确认 AViMoS 包含 1,500 个约19秒、FullHD、含音频视频，fixation 坐标 JSON、连续 saliency map 和原始划分；Hugging Face 文件总量约43.4GB，`Videos.zip` 23.48GB、`SaliencyTrain.zip` 11.98GB、`FixationsTrain.zip` 1.18GB。

项目已有音频负迁移证据，因此只保留视觉路径。R98 使用已有 15.37M UNISAL 时序视觉权重做直接空间中心诊断，结果相对 Qwen 构图点显著下降，关闭“通用注意中心替代构图点”。这并不关闭“注意力作为二级候选 utility”或“AViMoS 预训练后用 native crop 监督适配”。

### YouTube-8M 2019

Layer6 的 CCRL 真正利用的是条件相关性和高召回候选；候选过滤发生在跨视频的类别候选层，不能翻译成 AIC 先删掉每个视频的大多数帧。BigVid 的视频—片段一致性依赖“视频没有该类”弱负例，不能把 PHD/GIF 未选帧当硬负。RLin 的 MOD 价值是训练期多模型共享知识；AIC 单一高光 logit 沿类别轴做 softmax 会恒为1，必须改成 Bernoulli 或同视频有效时间步分布蒸馏。

这些判断由原论文交叉核对：

- [Layer6 PDF](https://research.google.com/youtube8m/workshop2019/c_02.pdf)
- [BigVid PDF](https://research.google.com/youtube8m/workshop2019/c_13.pdf)
- [RLin MOD PDF](https://research.google.com/youtube8m/workshop2019/c_05.pdf)
- [RLin code](https://github.com/linrongc/solution_youtube8m_v3)

### AIKU × brAIn 2025

Kaggle API 在不暴露凭据的前提下已列出真实文件：`train.csv` 14.3MB、`train.h5` 5.00GB、`test.h5` 1.25GB、`test.csv` 24KB、`baseline.ipynb` 19.9KB。下载接口返回 403，当前账号还没有完成该比赛数据访问确认；因此标签字段、特征提取器、预算和许可仍未知。AIKU 不能进入训练，直到取得真实样例并完成来源/时间/标签审计。

## 下一步实践路线

### P0：AIKU 数据审计（最高信息价值，尚不训练）

完成 Kaggle 数据访问确认后只下载 `train.csv`、`baseline.ipynb` 和一个受控 `train.h5` 分片，建立字段、特征维度、时间步、标签覆盖、预算和原视频/片段关系登记。检查与 TVSum、MrHiSum、QVHighlights、PHD/GIF 的 source overlap。通过后做三个严格对照：

1. PHD/GIF-only；
2. AIKU 预训练后 PHD/GIF 适配；
3. PHD/GIF-only 但总优化步数相同。

主指标仍是 source-level fixed-budget observed-selection proxy；AIKU 自身摘要分数只作外部任务指标。任何缺少原视频或无法重算特征的包只能作 head benchmark，不能成为可部署模型。

### P0：高分辨率当前帧 × 时序上下文构图读出

这是 AIM VistaHL 思想在 AIC 中最直接的合法改写，且不依赖声音。固定 R70 的 129 合法候选、R54 时间集合和精确输出预算，用 native crop GT 训练一个小交互头：当前候选的 V9 spatial token 作为高分辨率分支；同视频邻近 keyframe 的低维 RGB/语义摘要作为低分辨率上下文；二者用 FiLM 或低秩 Hadamard 交互。必须有同容量零交互控制、随机上下文置换控制和中心控制。

这条路线与 R94 不同：R94 只测 U²-Netp 前景统计；与 R99 不同：R99 的 8 点时间上下文只用于 observed-selection 时间标签，且未接入 native crop。首轮只测空间 IoU，不把空间 utility 重排到时间轴，避免重犯 R82 的错配。

晋级条件：RV 与 LIVE 两个 source-disjoint 留出池都相对 R70 强控制达到至少 `+0.010` native IoU，且源级 bootstrap 下界为正；否则关闭当前头，不出包。

### P1：完整事件上下文时间头

R99 的低秩局部—全局交互相对同容量控制只有 `+0.000331`，95% CI 跨零；跨 source 置换后增量归零。它说明“上下文可能有微弱信号”，不能支持当前配方。若重开，改变的是观测粒度：完整视频或更密事件边界，再用 Bernoulli/时间分布目标；不继续扫 rank、loss 或全局均值。

### P1：AViMoS 视觉预训练

只有在 P0 的直接构图读出没有足够增量、且 AIKU 审计未提供新时间监督时投入。先下载训练视频与视觉 saliency/fixation，抽取 100 train/30 dev 做对齐 smoke：方向、帧数、PTS、坐标和常数图行为。训练轻量视觉注意力学生，随后在 LIVE/RetargetVid native crop 上用固定时间/候选做适配。音频永不进入首轮。完整 43GB 数据下载前需再次检查 `/data/aic` 空间和许可证记录。

### 暂停：MOD、音频、直接显著性中心、类别候选剪枝

MOD 需要先有已被目标域验证的教师；当前 AHA/R43/R97 尚不满足。音频已被 AIC 实验反复证实负迁移。R98 已关闭直接显著性中心。YouTube-8M 的跨视频类别候选剪枝不适用于 AIC 的视频内精确帧分母；若研究昂贵观察预算，必须让所有原帧保留最终候选资格。

## 建议的明日执行顺序

1. 完成 AIKU 数据访问确认，生成 `aiku_asset_audit.md`；未通过则停止下载训练。
2. 在 R70 native crop shards 上实现高分辨率当前候选 + 低分辨率邻帧交互的 1 个控制/1 个主模型/1 个置换对照，先跑 200 个 source profile，再决定全量。
3. 保留 R54/R70 作为时间与空间回退；不因公共代理小幅提升生成官方包。
4. 只有第2步超过预设 native IoU 门，才制作一个空间单变量候选；官方上传仍由用户执行。
