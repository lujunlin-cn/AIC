# 项目状态

更新：2026-09-25（Asia/Shanghai）。本轮完成评估链修复、TVSum 标签审计、修复版 A0/A1、5-fold split 稳定性、Feature Bank 分组、ViT B0 probe 和空间/raw 一致性诊断。

## 当前证据最强的候选

- **开发集时间候选：A0_006 + prediction threshold 0.40**。TVSum temporal proxy video-macro F1=0.151875（threshold 在开发 val 上选择），micro F1=0.161220，平均选中率=0.07397，空预测率=0。它是时间代理候选，不是 AIC `F_video`。
- A1_005（feature-level shift，准确命名，不是 backbone-internal TSM）在 threshold 0.35 时 macro F1=0.144745；固定 0.5 时为 0.121079。A0_006 在同一 proxy-v2 split 上更好。
- 5 个 source-group fold 以固定 threshold 0.30 的结果：A0 mean=0.11457，std=0.07900；A1 mean=0.10787，std=0.07764。按每 fold 单独调阈值的探索值分别为 A0 0.14983±0.03692、A1 0.15422±0.03665，只能作为乐观诊断，不能作为锁箱分数。
- ViT-B/16 frozen-feature B0 probe：threshold 0.30 时 macro F1=0.15928，FP16 bundle 174,986,447 bytes（M 档假设）；已完成真实视频 JSONL + validator，但仍需更多 split 和正式提交集验证。
- Loss ablation：A0_012 只换 SmoothL1，fixed-0.5 macro F1=0.11679、threshold 0.40=0.14143，低于 BCE A0_006；当前保留 BCE。

## 评估和标签协议

- 固定 GT 定义：`tvsum_summary_mean_norm_ge_0.5_v1`；prediction threshold 独立配置，只能在 train/dev 选择。
- 已冻结 `splits/local_protocol_v1.json`：TRAIN=27、DEV=16、LOCAL LOCKBOX=7（原 manifest 的 test 划分）；manifest SHA-256=`b8db2bb3f1e36e2ecc32a1c307531e43d69bae3f74605c9112b8eda8b679e565`。Lockbox 禁止 threshold、checkpoint、模型、后处理和超参选择，只有候选冻结后比较使用。
- v1 另记录 5 个 source-group development folds（验证集规模 8/9/10/7/9），并校验每折只覆盖 43 条非 lockbox 视频；TVSum 当前没有可靠 category metadata，因此不宣称 category-stratified。
- 报告 per-video F1、video-macro、micro 诊断、选中率、GT 比例、空预测率、precision/recall、分数分位数、连续 MAE 和 Spearman。
- TVSum v1.1 MAT 的 50 条 `user_anno` 全部为 `(20,nframes)`，2 秒是收集评分的片段语义，发布文件已经逐原始帧展开。`published_mat_per_frame_identity` 审计显示旧 `linspace` 对这些视频是恒等映射，标签错位不是接近零 F1 的主要原因。
- 作者 15% summary/knapsack evaluator 与本项目固定二值 temporal proxy 分开记录；TVSum 没有合法 composition crop GT。
- 审计产物：`reports/tvsum_annotation_audit_v3.jsonl`、`reports/tvsum_manifest_v3.jsonl`。

## 一致性和空间状态

- padding 确有影响：旧实现 5→12 timestep 最大有效 logit 差异 0.3392，8→12 差异 0.5733；`lengths` 路径逐视频计算后回归误差为 0。
- 远程同环境 PyAV 15.1 的 raw→backbone→temporal 与 cache 路径：frame/timestamp 一致，feature 最大差约 2.5e-5，aux 完全一致，probability 最大差约 2.1e-7。当地 PyAV 18.1 与远程解码会产生明显漂移，正式推理固定远程环境/版本。
- inference 现在显式支持 `spatial_mode=center|saliency|subject`，A2 raw inference 会构造同定义 32D aux；三种模式真实视频均通过 JSONL validator。空间诊断只报告合法率、速度、加速度、jerk 和可视化，**没有伪造 IoU**。
- 代表性空间接触表和诊断：`/data/aic/experiments/A1_004/spatial_diagnostics_J0nA4VgnoCo.jpg`、`reports/spatial_diagnostics_20260925.json`。

## Feature Bank / TSM / B0

- Feature Bank 维审计：第 11、23–28、31 维恒零；0/8/9 与 5/22、7/16 存在重复或近重复。motion-only F1=0.02724、quality-only=0.01892、composition-only=0.01613、audio-zero=0；当前融合定义没有收益，不能解释成“真实音频无效”。
- `temporal_shift_feature_map` 已实现并在 ResNet 中间 feature map 上做 canonical TSM probe：mean shift difference=0.12085，GPU batch overhead≈17.8%，参数为零。尚未全量 recache/end-to-end 训练，因此不把它与 A1 embedding shift 混称。
- B0 使用 torchvision ViT-B/16 ImageNet-1K frozen features + 同一 Temporal U-Net；真实 raw-video → JSONL → validator 已通过。bundle 实际 174.99 MB，超过 S 档，暂作为高分参照而非默认 fallback。

## Engineering fallback

当前可运行 fallback 是 ResNet18 + repaired Temporal U-Net + center crop，模型 FP16 25,685,169 bytes、FP32 51,319,345 bytes；阈值暂用开发集锁定值，拿到官方输入/GT 后必须重新选择。没有官方 evaluator、比赛视频或联合 spatial GT，`official_f_video` 和 competition score 保持 null。

## 远程资源

- 远程代码：`/home/supie/AIC`；大数据/权重/实验在 `/data/aic`。
- 当前使用的 V100 物理卡仅为 2、4、5、6、7；0、3 未使用，1 保留既有进程。
- 远程 PyTorch 2.9.1+cu128、torchvision 0.24.1、PyAV 15.1；所有训练均远低于 12 小时并使用外层 timeout。

## 未验证项

- 没有 AIC 联合 temporal+crop GT，不能宣布任何 TVSum 数值为官方 `F_video`，也不能比较空间 IoU 或最终 size-weighted score。
- 尚未取得官方 evaluator/比赛测试索引；提交流程只完成本地契约和 validator。
