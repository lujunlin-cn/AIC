# 项目状态

更新：2026-09-25，representation generalization 阶段。最新完整证据见 `reports/20260925_representation_generalization.md`、`reports/spatial_benchmark_status.md`；旧报告保留为历史快照。

## 当前候选

- **Engineering Fallback / 当前保守 Best S：A0_006**，ResNet18 + repaired Temporal U-Net，raw threshold 0.40（历史 DEV 选择）+ center，完整 FP16 文件 **25,685,169 bytes**。原视频到 JSONL 可运行。
- **S-tier Promising Challenger：DeiT-S/16**，完整 FP16 **46,618,447 bytes**。nested CV 的摘要均值较好，但配对区间跨 0，SumMe 小样本 OOD 没有复现优势；不升级 Primary。
- **Best M / semantic reference：ViT-B/16**，完整 FP16 **174,986,447 bytes**。TVSum 摘要与 DeiT-S 近乎持平，目前无证据证明额外体积值得。
- **Current Temporal Best：没有同时在全部指标/数据集可靠胜出的单一模型。** TVSum summary 均值 DeiT-S 略高，Spearman ViT-B 略高，首批 SumMe OOD A0 较好。
- **Current Spatial Best：true_face_smooth 是本地均值最高的探索配置，非已确认胜者。** Center 保持默认；YuNet 是人脸观察器，不是完整主体理解；新增权重 232,589 bytes。
- **Best Overall / official_f_video / competition_score：null**，尚无 AIC 联合 GT/官方反馈，不能用 temporal 与 spatial 两个不同数据集的分数拼接出比赛成绩。

## 已完成的最新证据

- TVSum `TVSUM_RANKING_V2`：Spearman、Kendall tau-b、线性增益 tie-aware NDCG/NDCG@15%、固定 GT AP、明确命名的 top15 relevance。
- `TVSUM_SUMMARY_V1_FIXED`：作者 60 原帧分段（尾段合并）、15% 预算、0/1 knapsack、20 annotator F1 宏平均；预测分段确定性替代作者随机示例。10 个合成/真实案例与未修改 MATLAB 函数经 Octave 核对，mask 零差异。
- **90 次真实配对 nested-CV 训练已完成**：3 representations × 2 repeats × 5 outer folds × 3 seeds；每折 30 train / 10 inner-dev / 10 outer。每次 20 epochs，checkpoint/threshold 只使用 inner-dev。总登记训练 1032.39 秒，单次最大 14.73 秒。
- A0 / DeiT-S / ViT-B 的 TVSum summary F1：**0.21521 / 0.23073 / 0.23064**；Spearman：**0.43216 / 0.41439 / 0.44127**；binary proxy F1：**0.16084 / 0.16869 / 0.17301**。
- DeiT−A0 summary 配对 delta **+0.01552，95% CI [-0.00653, 0.04070]**；Spearman delta **-0.01777，CI [-0.07939, 0.04530]**。50 source-video 为单位，先平均 repeat/seed，不把重复观测当独立样本。
- 600 组配对外层记录的 frame indices、timestamps、labels、mask 完全一致；逐视频/类别表、FP/FN/空输出/过选清单和图像已保存。
- **SumMe raw 已取得**。首批 8 个预先按压缩大小选取的视频，6 条帧数/FPS 严格匹配，Cooking / playing_ball 排除。镜像 PTS 破损，使用显式 annotation-ordinal CFR 协议；比赛 PTS 校验保持严格。Python native mean/max summary F1 与 SumMe 未修改 MATLAB evaluator 的 18 个案例误差 ≤1.12e-16。
- SumMe 已完成两批14条原视频、每模型全部30个matched CV heads：mean human summary F1 **A0 .21255 / DeiT-S .13408 / ViT-B .14361**；第二批8条单列为 **.17883/.11955/.15411**。DeiT−A0合并paired delta **−.07848，95%CI[−.13175,−.02734]**。镜像对齐与size-biased抽样限制保留，不代表完整SumMe。
- 固定随机预算32draw对照mean F1 **.13755**；A0 matched-head相对delta **+.07501 [+.02407,+.13841]**，DeiT/ViT-B相对随机CI跨0。50TVSum×14SumMe=700对SHA/pHash筛查无flag；不能排除所有近重复或预训练重叠。
- 历史完整FP16 bundle合并14条OOD mean F1 **.20727/.16074/.15916**；三路全部JSONL验证通过。Native摘要mask和实际threshold JSONL是两种不同输出，不能混用指标。
- **RetargetVid 已进入真实 GT 阶段**：DHF1K 001–020，12,365 原帧 ×2比例 ×6标注者；6方法，完整原函数 890,280 次 IoU 核对误差为0。Center IoU 1:3 **0.48190**、3:1 **0.73951**；face+EMA **0.48634 / 0.75214**。平均增益 CI 跨0。
- `spatial_protocol=dense_v1` 接入原始帧 observation→association→shot reset→EMA→最大合法 crop，6种显式模式保留旧接口。8 个 raw-video E2E 检查通过，3600条预测，crop 与 GT benchmark 保存结果完全相同；detector bytes 计入。
- 固定发行清单/入口 `releases/20260925_v1`、`python -m aic.release`：三种候选完整权重在本地/远程双份保存，加载前校验SHA和bytes；历史DEV两原视频真实阈值输出1263/2297/1263帧，均valid。DHF1K三样例A0全空也如实保留，无temporal GT不解释成FN。
- 本地与远程均72 tests passed；远程运行代码与本地核心代码哈希逐一核验。FP16指存储文件，实际加载为FP32标准kernel。

- **追加60次线性head对照**已完成：A0/DeiT summary .21324/.21781、Spearman .32930/.33129；SumMe14 .14771/.12230。当前线性替换无收益，保留U-Net；只淘汰这个固定预算假设。

## 协议与历史边界

- TVSum MAT 的 `user_anno` 是 20×nframes；已完成的标签/padding/raw-cache审计不重做。MAT **有真实 category metadata**：10类，每类5视频；旧文档“不可取得 category”已过时。
- 原7条只称 **comparison_holdout_v1**；原协议文件不改写，状态另记 `splits/local_protocol_v1_status.json`。TVSum 50条均为开发暴露数据，没有新 pristine test。后续使用 `splits/tvsum_nested_cv_v1.json`。
- feature-level shift、canonical internal TSM、SmoothL1、Feature Bank v1、A0首轮 smoothing 均保留历史并暂停 sweep。
- VLM pilot 尚未运行，没有 teacher ranking/蒸馏收益证据。

## Current Bottleneck / Running Experiments / Latest Failure

- **主要证据缺口**：OOD 样本少且镜像时轴有局限；没有赛事联合标签。技术上同时存在 ranking/domain shift、过选校准和多人主体选择错误，不能跨 benchmark 排名谁是比赛最大瓶颈。
- **Running**：本阶段全部150次训练、原视频OOD、固定候选和线性头外部复核已完成，无后台训练。独立Git代码副本3候选与既有JSONL逐项相同；本阶段记录、配置和代码已整理，后续按TODO的新数据/主体错误队列推进。
- **Latest Failure**：ENGINEERING_RELEASE_001在CUDA初始化前请求显存统计失败；修复后新ID002/003全部完成。SUMME_OOD_001非递增PTS失败仍保留；比赛reader不放宽。RetargetVid负坐标clamp已补齐，重计分v2保留v1，自有分数不变。

## 资源与下一任务

远程 `/home/supie/AIC`；数据/模型/结果 `/data/aic`；Python `/opt/miniconda3/envs/cv/bin/python`，PyTorch2.9.1+cu128、PyAV15.1。仅使用空闲授权物理 GPU2/4/5；GPU1既有任务保留，0/3未使用。

下一步：不同来源highlight训练/验证小子集与时轴核验 → 独立空间协议下的多人主体选择 → 有界loss/head验证。已有TVSum/14条SumMe均已开发暴露；不能重新称未见lockbox。TSM、旧bank、无结构threshold sweep不重开。
