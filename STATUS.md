# 项目状态

更新：2026-09-29 14:15。当前是初赛阶段，只计 raw F_video（规则 01 §3.4），不乘规模系数。

## 最新（2026-09-29 下午）：V5 P0 官方包已打包待评（QWEN32B_INTERP_XRERANK_V5_FINAL）；P1/P2/数据侧进行中

- **P0 通过预注册晋级规则并已打包**：INTERP 母本上 x 行替换为 E4 式视觉复核点，dev2 x +1.87 CI[+0.73,+3.23]、confirm2 x +1.88 CI[+0.79,+3.26]、全视频 +0.93/+0.94、y 行严格零变化；E4 sanity 复现 ±0.0001 内。`QWEN32B_INTERP_XRERANK_V5_FINAL.zip` sha `027fd12a…`，174 视频/78,992 帧，keys==TEMP 逐视频 0 差异，vs INTERP 实际差异 21 视频/10,273 帧。成绩**待评**，正式最佳保持 INTERP 48.67。详见 `reports/20260929_v5_p0_xrerank_package.md`、`registry.jsonl`（V5_P0_XRERANK_OFFICIAL）。
- P1（H3 视觉评分头 G/V/V+Q）：RV+LIVE 六表构建完成（rv_train 6,795 kf / rv_dev 1,470 / rv_confirm2 8,004 / live_val 1,032 全 axis0），DINOv2 缓存并行中；决策规则见 `configs/V5_H3_VISUAL_PREREG.json`（含实现附录：候选恒 33 网格、mother 吸附最近格点）。
- P2（Qwen 命名对象 + GroundingDINO 定位）：25 个 confirm2 分层视频（x/y × fast/slow 四层），32B 命名阶段重跑中。
- 数据侧：PM-400/AVE-PM 直链实测全灭（12 链接 × 3 头全 502）；改用社区 GDrive 缓存（68.07 GB）后台下载中（~4.6 MB/s）；`portrait_reframe_pilot_v1` 补标任务已建（320 源/86 类全覆盖，215/51/54 划分，240 片段上下文引用；状态 pending_annotation；AVE-PM 事件/BGM/类别仅作上下文，不冒充裁剪 GT）。见 `ext_data/configs/portrait_reframe_pilot_v1/ANNOTATION_SCHEMA.md`。
- 分支 `teacher-v5-20260929`（主题提交进行中）；V5 基线冻结 `configs/V5_BASELINE_MANIFEST.json`。

## 2026-09-29 上午：V4 正式成绩已回（INTERP 48.67 = 新正式最佳）；V5 周期启动

- **正式最佳更新为 INTERP 48.67**（`501cebce…`，+0.79 vs DT_V3）；OBS025 48.32、MEDIAN3 48.21、H1_XONLY 46.73（−0.11 vs TEMP）。证据：用户 2026-09-29 08:28 平台截图，四项 DONE；逐视频分数未取。详见 `reports/20260929_official_v4_feedback.md`、`registry.jsonl`、`configs/V5_BASELINE_MANIFEST.json`。
- H1_XONLY 拆分：x 轴 −0.11，y 轴 −1.73（≈94% 伤害在上下方向）。
- V5 主攻（母本一律 INTERP 48.67）：P0 左右方向视觉复核（E4 缓存信号）、P1 冻结视觉编码器+候选评分头（G/V/V+Q 消融）、P2 "Qwen 选对象、目标帧检测器定位" pilot；数据侧接 PM-400/AVE-PM 小样本与竖屏补标任务。时间 mask / face 路由 / 后处理冻结不变。
- V4 周期记录（母本 DT_V3 47.88，历史快照）：
  1. `QWEN32B_DT_INTERP_V4_FINAL`（`501cebce…`）：同镜头关键帧间主体点线性插值。RV confirm2 +.0119 [.0084,.0160]；无新增查询。
  2. `QWEN32B_DT_OBS025_V4_FINAL`（`f575c9ba…`）：空间关键帧 0.25 s。confirm2 +.0063 [.0041,.0087]；官方新增 5,765 次查询。
  3. 可选：`QWEN32B_DT_MEDIAN3_V4_FINAL`（`d62ed910…`）：三点中值。confirm2 +.0044；与插值重叠。
- 诊断包：`QWEN32B_TEMP_H1_XONLY_DIAG_V4_FINAL`（`6ecb766e…`），只用来拆分 45.00 的方向贡献。
- 否决（不出包）：
  - E2 上下文主体点：dev2 −.0277。
  - E3 事件角色：YTH val 人工 −.0009。
  - E3B 只删：YTH train 人工 −.0037。
  - E4 视觉裁剪复核：dev2 +.0093；confirm2 +.0072，CI 跨 0，y −.0034，否决。x 轴收益在两个划分上都复现了（+.018），瓶颈在选择器。
- H1 审计：公式与方向无误。y 轴只在 RV 3:1 上拟合，官方 y 轴为竖屏 → 16:9，是迁移失败的主要嫌疑，但未证实。
- 报告：
  - `reports/20260929_teacher_4788_cycle_v4.md`
  - `reports/20260929_candidate_results.csv`
  - `reports/20260929_h1_transfer_audit.csv`
- 基线冻结：`configs/V4_BASELINE_MANIFEST.json`。

## 2026-09-28 夜：46.84 之后（指南 v3）

- **最高已知平台成绩：DT_V3 47.88**（QWEN32B_NOFACE_DENSE_TEMP_V3_FINAL，zip `c702de926450717ff3c18c4a11aabeee9f3ad3a49ab28f0f512768b14722cf67`，78,992 帧，用户 22:33 告知）。晋级为新的已评分母本。
- 其余：TEMP 46.84（`631d0104…`）；dense 43.76（`972b3ed1…`）；NOFACE 42.83；B0 34.42。SHA 均从远端文件重算。32B 教师原始成绩，33.36B 参数，非合规学生。
- 2×2：DENSE 在全选/TEMP 下 +0.93/+1.04，TEMP 在 N0/DENSE 空间下 +4.01/+4.12，交互 +0.11（单次提交，不作因果解释）。距 60 参照 12.12。
- **T4 边界精修否定**：YTH val 代理 −0.0014 [−.0034,+.0003]，不出包。
- **oracle 复核**：0.70 静态上限不约束当前逐帧移动的 N0；49/42 是池化阈值份额，1:3 以主体认错为主、3:1 以位置为主。
- **T5 窗口评分器晋级并打包**：H1 几何校准（4 标量）在未用过的 LIVE-YT-VC val 上 +0.0062 [+.0034,+.0093]。包 `QWEN32B_NOFACE_TEMP_CROPHEAD_V3_FINAL` zip `519a718a2f16d499cf0a2b99bd8c5525bed34440d5e8b0a3dad40a27f63eaade`，174 行 / 78,992 帧，keys==TEMP，未上传，官方分 null。母本 TEMP、N0 观察（与 DT 不叠加）。见 `reports/20260928_teacher_crop_decision_probe.md`。
- 报告：`reports/20260928_score_46_84_followup.md`、`reports/20260928_dense_temporal_interaction.md`、`reports/20260928_v3_candidates.csv`。

## 最新（2026-09-28 晚）：教师方向调整（指南 v2）

- 平台分数（用户截图）：**NOFACE 42.83**、POINT 40.09、B0 34.42。冻结 N0/P0/B0，远端提交目录只读。
- 新公共集划分（pre-registered）：used=001–030，dev2=031–100，confirm2=601–700（200 个 RetargetVid 视频本就在外部数据中）。
- T0 门控审计通过：N0/P0 逐帧复现，3,106 关键帧齐全；N0 有脸帧 74% 与 B0 不同（EMA 状态继承，符合定义）。
- **T2（关键帧加密 0.5 s）成立并已打包**：confirm2 +0.0135 [+.0096,+.0176]。官方集 5,988 个密集关键帧（新增 2,882 次查询，3,106 个共享回复逐字节复用）。包 `QWEN32B_NOFACE_DENSE_V2_FINAL` zip `972b3ed1…`，174 v / 87,781 f，mask/width diff 0，独立校验与解包复检通过，未上传。
- **T1（比例感知区域）不成立**：confirm2 −0.0239 [−.034,−.015]，三种映射全负；官方推理中止（23/174），不打包。
- **T3（时间只删不增）已完成并以假设包形式打包**：YTH 弱标签代理 vs keep-all +0.0031 [−.005,+.011]（不确定），vs 随机同删量 +0.0208 [+.011,+.031]（有选择性）。按预注册规则（均值≥0 且官方帧差异 ≥10%；实际删除 10.01%）出**假设性诊断包** `QWEN32B_NOFACE_TEMPORAL_V2_FINAL` zip `631d0104…`，保留帧 bbox 与 N0 逐字节一致（0 diff），未上传。**不是**已确认改进。
- 教师瓶颈定位：位置决策 ~49% 缺口 > 主体认错 ~42% > 传播 ~5%；每镜头单窗上限 ~0.70。
- 详见 `reports/20260928_score_42_83_followup.md`、`reports/20260928_teacher_60_gap_analysis.md`。

## 2026-09-28 下午：最大窗口位置实验与新候选

- 平台分数（12:46，用户截图）：B0 34.42，S2 30.37，S1 30.13，Q2 27.40。缩小尺度的两条路线分别下降 4.05 / 3.80，已暂停；YuNet 跟随比居中高 4.29。
- P1a（只改路径）：镜头内 L1 DP 与 B0 的 EMA 持平（confirm +.001，CI 跨 0），更平滑但没有更准。已拒绝，不再调路径。
- P1b（COCO 覆盖候选）：候选集 oracle 有潜力（confirm +.090），但实际选择器 CI 跨 0，不交付。
- P2（Qwen 主体点，只改位置观测）：RetargetVid 30 视频合并 +.042，CI [+.007, +.081]；确认集 021–030 +.067，CI [+.015, +.125]。用 PyAV 15 复核，结果一致。
- 新候选（READY_TO_UPLOAD，探索性，测试时加载 32B 教师）：
  - `MAX_WINDOW_QWEN_POINT_V1`：`684e566f…`，先传。
  - `MAX_WINDOW_QWEN_NOFACE_V1`：`f73c399f…`，后传。
  - 两个包都是 174 个视频 / 87,781 帧，时间掩码与宽度相对 B0 的差都是 0。见 `reports/20260928_max_window_path.md`。
- 时间监督契约 v1 已实现，消除了 +0.99 s 系统偏移；native QVH 没有可靠负例，时间路线继续全选。见 `reports/20260928_temporal_supervision_contract.md`。

## 最新（2026-09-28）：空间裁剪候选与全选诊断

- 新上传候选：S2（可变尺度，`4d39831a…`）→ S1（居中最大窗口，`7c4ef54a…`）。都是探索性候选，都是 174 个视频 / 87,781 帧，全部校验通过，未上传。详见 `reports/20260928_spatial_candidates_release.md`。
- B0 = 预测与 V0 完全相同、实际部署不同的低参数实现（53,104 参数，`ab6e7c0f…`）。初赛与 V0 同分，不优先上传；复赛计系数后再作为候选。
- Q2（Qwen 帧 + S2 裁剪）：真实规模 33.38B 参数 / 66.8 GB，超限，是内部教师探针，不上传。350.5 占位值已撤回。
- 时间路线保留全选。V0 全选的原因是 loss 只在 query 相关片段上计算（覆盖 .31，72% 目标 ≥ .75）。dev 代理上没有时间选择可靠胜过全选；E1/E2（未标注=0）为失败探索。见 `reports/20260927_all_select_diagnosis.md`。
- 规模档纠正：按参数量口径，V0 = 305.6M → 0.95（此前按字节归入 0.90 是错的）；InternVideo2 系列 ≈ 1.02B → 0.90。
- Qwen dev 审计阻塞：GPU 4–7 被非本项目服务占用。见 `reports/20260927_qwen_temporal_audit.md`。

## 历史状态（2026-09-26 快照）

更新：2026-09-26，正式评测集冻结推理已完成，且已收到首轮官方 raw 分数。研究证据仍见 `reports/20260925_representation_generalization.md`、`reports/spatial_benchmark_status.md`；旧报告保留为历史快照。

## 最新官方反馈（2026-09-26）

- `SUB_A` A0/ResNet18 + center：raw **1.01**。
- `SUB_B` DeiT-S/16 + center：raw **6.08**。
- `SUB_C` DeiT-S/16 + YuNet `true_face_smooth`：raw **6.64**。
- 三者均低于当前 S 档边界，`k_size=1.00`；相对 C，M 档需 raw `>6.98947`，L 档需 raw `>7.37778`。
- A→B 是近控制 frozen-representation 对照；backbone、threshold 与 AMP/trainer implementation 仍有差异，不能把分数差全归因参数量。
- B→C 的 temporal frame list 174/174 完全相同，8,130/8,133 bbox 改变；`+0.56` 是 YuNet 空间替换的独立官方差分证据。
- 诊断详情：`reports/20260926_official_score_diagnosis.md`。

## 当前候选

- **Engineering Fallback：A0_006**，ResNet18 + repaired Temporal U-Net，raw threshold 0.40（历史 DEV 选择）+ center，完整 FP16 文件 **25,685,169 bytes**。原视频到 JSONL 可运行。
- **TVSum/OOD challenger：DeiT-S/16**，完整 FP16 **46,618,447 bytes**。nested CV 的摘要均值较好，但配对区间跨 0，SumMe 小样本 OOD 没有复现优势；不升级 Primary。
- **Best M / semantic reference：ViT-B/16**，完整 FP16 **174,986,447 bytes**。TVSum 摘要与 DeiT-S 近乎持平，目前无证据证明额外体积值得。
- **Current Temporal Best：没有同时在全部指标/数据集可靠胜出的单一模型。** TVSum summary 均值 DeiT-S 略高，Spearman ViT-B 略高，首批 SumMe OOD A0 较好。
- **Current Spatial Best：true_face_smooth 是本地均值最高的探索配置，非已确认胜者。** Center 保持默认；YuNet 是人脸观察器，不是完整主体理解；新增权重 232,589 bytes。
- **最新多人主体对照：`SPATIAL_GROUP_003` 未通过升级门槛。** 固定 top-3 人脸面积/置信度群体中心相对 `true_face_smooth` 双比例 IoU `-0.00264`，20/20 视频没有正增益；相对 center `+0.00590` 但 CI `[-0.01690, +0.02739]` 跨 0。保留接口与负结果，不改默认。
- **Best measured official candidate：SUB_C，raw 6.64**。官方反馈来自用户，未获得本地联合 GT；local proxy 不能冒充官方成绩。A0 仅保留工程 fallback。

官方平台 raw 反馈已记录为外部证据；本地 `official_f_video` 字段仍只表示有无可复现联合 evaluator，不能用代理指标冒充官方 F_video。

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
- 本地与远程核心代码哈希已逐一核验；当前本地回归为 86 tests passed。FP16 指存储文件，实际加载为 FP32 标准 kernel。

- **追加60次线性head对照**已完成：A0/DeiT summary .21324/.21781、Spearman .32930/.33129；SumMe14 .14771/.12230。当前线性替换无收益，保留U-Net；只淘汰这个固定预算假设。

## 协议与历史边界

- TVSum MAT 的 `user_anno` 是 20×nframes；已完成的标签/padding/raw-cache审计不重做。MAT **有真实 category metadata**：10类，每类5视频；旧文档“不可取得 category”已过时。
- 原7条只称 **comparison_holdout_v1**；原协议文件不改写，状态另记 `splits/local_protocol_v1_status.json`。TVSum 50条均为开发暴露数据，没有新 pristine test。后续使用 `splits/tvsum_nested_cv_v1.json`。
- feature-level shift、canonical internal TSM、SmoothL1、Feature Bank v1、A0首轮 smoothing 均保留历史并暂停 sweep。
- VLM pilot 尚未运行，没有 teacher ranking/蒸馏收益证据。

## Current Bottleneck / Running Experiments / Latest Failure

- **主要证据缺口**：OOD 样本少且镜像时轴有局限；没有赛事联合标签。技术上同时存在 ranking/domain shift、过选校准和多人主体选择错误，不能跨 benchmark 排名谁是比赛最大瓶颈。
- **Running**：无 AIC 训练；`YTH_ACQUIRE_001` 在本地对 YouTube Highlights 9.9GB tar 做可断点 Range 索引，已取得人工/弱标签 metadata；`SPATIAL_CONFIRM_001` 尝试恢复 DHF1K 021–030 作为固定参数空间确认集。GPU 2/4/5/6/7 空闲，GPU1既有任务保留。
- **Latest blocker**：本地 YouTube Highlights tar Range 连接在 metadata 之后出现 TLS EOF/timeout；远程 DHF1K 021–030 Drive Range 也在启动阶段 TLS timeout，未生成新视频。两条路径均保留失败日志，不影响已完成证据。
- **Latest Failure**：ENGINEERING_RELEASE_001在CUDA初始化前请求显存统计失败；修复后新ID002/003全部完成。SUMME_OOD_001非递增PTS失败仍保留；比赛reader不放宽。RetargetVid负坐标clamp已补齐，重计分v2保留v1，自有分数不变。

## 资源与下一任务

远程 `/home/supie/AIC`；数据/模型/结果 `/data/aic`；Python `/opt/miniconda3/envs/cv/bin/python`，PyTorch2.9.1+cu128、PyAV15.1。仅使用空闲授权物理 GPU2/4/5；GPU1既有任务保留，0/3未使用。

下一步：不同来源highlight训练/验证小子集与时轴核验 → 独立空间协议下的多人主体选择 → 有界loss/head验证。已有TVSum/14条SumMe均已开发暴露；不能重新称未见lockbox。TSM、旧bank、无结构threshold sweep不重开。

## Pairwise ranking continuation（2026-09-25）

- `RG_RANK_001` 已完成 60 个 nested outer evaluations（A0/DeiT-S，各 30；2 repeats×5 folds×3 seeds），使用 `BCE + 0.1*pairwise_logistic`，不增加推理权重。
- Pairwise A0：proxy F1 `.155797±.042412`，Spearman `.426467±.108241`，NDCG@15 `.651274±.064576`，summary `.217504±.028750`。
- Pairwise DeiT-S：proxy F1 `.172148±.031270`，Spearman `.424183±.065153`，NDCG@15 `.669796±.045305`，summary `.233615±.013983`。
- Per-video paired DeiT−A0：F1 `+.01635`、Spearman `-.00228`、NDCG@15 `+.01852`、summary `+.01611`；正增益视频比例约 `52/44/52/50%`；200k paired bootstrap CI 分别为 F1 `[-.00684,.04461]`、Spearman `[-.06504,.06303]`、NDCG15 `[-.01741,.05856]`、summary `[-.00573,.04064]`，均跨 0。保留为 exploratory，A0 fallback 不变。
- 远程 GPU 2/4 已释放；YouTube Highlights 代理索引到358成员，DHF1K 021–030 RAR恢复但7z不支持 AVI 压缩方法，暂无新增 OOD/spatial 分数。
- 新报告：`reports/20260925_pairwise_ranking_and_proxy_ood.md`。

## Evaluation-set readiness（2026-09-26）

- 新增 `AIC_EVAL_INTAKE_V1`：`scripts/prepare_eval_set.py` 对 compact index 中每个视频做真实 probe、完整 PTS/帧数核对、SHA-256 和 coded-pixel 坐标记录，生成不可覆盖的 enriched index 与 intake manifest。
- `scripts/run_eval_candidates.py` 可在 intake SHA 与 release manifest 核对后，按新 run 目录依次运行 A0/DeiT-S/face candidate，并保留 command、report 和 JSONL；不读取训练 cache，不调参。
- `scripts/release_batch.py` 支持直接给原始视频建立临时 index，已用真实短视频完成 A0 raw inference；loaded weight bytes `25,685,169`、JSONL validator 均通过。
- 本地演练结果：compact index → 5 帧真实解码/PTS/SHA → final dummy JSONL → final validator valid；A0 真实权重 release smoke 也 exit 0。SMOKE 不代表比赛成绩。当前回归测试 `86 passed`。
- 评测集接入手册：`EVAL_HANDOFF.md`。官方 `F_video`/competition score 仍保持 `null`。

## Remote asset audit（2026-09-26）

- 远程 `/data/aic/asset_inventory/asset_manifest_20260926.jsonl` 已完成：11,038 条记录，7,293 个 regular files 已 SHA-256，3,702 个 dataset 文件 metadata-only，43 个 features symlink；无 missing/error/changed 条目。
- 已核对 A0 `25,685,169` bytes、DeiT-S `46,618,447` bytes、YuNet `232,589` bytes 以及 code snapshot SHA；资产清单不进入 Git，原始数据仍留在 `/data/aic`。

## Official test inference closure（2026-09-26）

- 正式包已保留原 ZIP（`757,234,263` bytes，SHA256 `130bdceb2574ef3ca853f6937fdf5f24c3fb1f054c92477f0c2d075e53efe0b6`），174/174 视频完整解码；16:9=119、9:16=55；标签不存在；官方联合评测仍不可用。
- 三个候选均只做冻结推理：SUB_A=A0 ResNet18+Temporal U-Net+center，SUB_B=DeiT-S+center，SUB_C=DeiT-S+YuNet `true_face_smooth`。没有测试集训练、标注、逐视频调参或第三方 API。
- SUB_A：25,685,169 bytes，1,809 predictions，空输出率 `.862069`，742.213 s；SUB_B：46,618,447 bytes，8,133，`.557471`，742.764 s；SUB_C：46,851,036 bytes，8,133，`.557471`，1,959.029 s。
- 根目录上传包均 `READY_TO_UPLOAD=yes`，并通过项目 validator、独立 checker、解压回归；官方 `F_video` 和 `competition_score` 保持 `null`。上传顺序：SUB_A → SUB_B → SUB_C。完整记录见 `reports/20260926_official_test_inference.md`。

## Official train package audit（2026-09-26）

- 本地挑战训练包包含 33 个 QVHighlights-derived shard，压缩 129.308 GB、解压 129.988 GB；11,245 个视频文件中只有 987 条标注记录，889 条已匹配、98 条缺失。
- 标注全部携带 `seed_weak_training_label_v1`、Doubao seed provenance；624 条有 seed crop observations，363 条为 `dropped_center_default`。当前没有 native human temporal/crop GT，不能将其写成官方 AIC 标签。
- `scripts/prepare_qvh_training_manifest.py` 已生成外部可复现 manifest v3：800 verified train / 89 verified val，33.208 h / 3.703 h，标签协议 `qvh_seed_timeline_linear_v1`；canonical dataset manifest 与 cache manifests 均已通过生成和 schema 检查。
- 当前 A0/DeiT 冻结模型使用的是 27 条 TVSum train 视频，未使用该 130 GB 包。首批 QVHighlights frozen/finetune 仅为有界弱标签诊断，结果和协议见 `reports/20260926_scaling_experiments.md`；不得与 TVSum proxy 或官方 F_video 混称。

## Continuation correction (2026-09-26)

- 用户确认赛方不提供130GB训练集；本地QVH资产来源供应方未独立核实，不能因目录名称其官方提供。它包含自动seed弱标签，不是人工赛事GT。
- 模型档位按用户最新参数量口径探索：≤100M / 100–500M / 500M–9B；同时保留附件中的文件bytes口径作双重审计，不能把86M参数VideoMAEv2称为100–500M参数候选。
- 最新远程核验：VideoMAEv2源权重344,924,592B；InternVideo2 Stage1-1B K700文件2,042,600,861B，已下载并可读取state dict。下载完成不等于已验证泛化。
- VideoMAE旧probe发现归一化与模型配置不一致、缓存缺timestamps，结果无效并保留；修正run由实验进程继续。

## 2026-09-26 model-base continuation

- Corrected `VIDEOMAE_QVH_PROBE_V2_20260926` completed on a source-disjoint 40/10 local QVH weak-label split. VideoMAEv2-Base (86,227,200 params; source 344,924,592 B) achieved weak validation clip F1 `.54444`, Spearman `.41459`, empty rate `.30`; matched clip-mean controls A0 `.33571/.42129/.00`, DeiT-S `.40000/.38757/.90`, ViT-B `.46667/.42070/.70`. VideoMAE−DeiT paired F1 CI `[-.15569,+.38889]`, so no promotion. The train-mean position baseline Spearman `.55405` demonstrates strong weak-label position bias.
- The first VideoMAE run had wrong normalization/cache schema and is quarantined; it is not evidence.
- InternVideo2 Stage1-1B K700 compatibility passed on V100 FP16: 1,020,710,144 loaded encoder parameters, 2,042,600,861 B source file, 8-frame `.2095s`/2267 MiB and 16-frame `.5385s`/3084 MiB. Compatibility only; no temporal head or quality score.
- Public native QVHighlights annotations were acquired locally (7,218 train / 1,550 val query rows). 7,738 `vid` stems intersect the local extracted archive, but no alignment/training has been done; duration/source semantics still need validation. This is a future OOD/native temporal route, not AIC joint GT.

## Native human QVH continuation (2026-09-26)

Native public annotations now match 6,384 train / 1,354 val raw videos; first 5/5 per split pass full decode, duration, monotonic PTS and saliency schema audits. Official native train/val original source intersection is zero. A new bounded protocol `splits/qvh_native_bounded_v1.json` freezes 96 train / 24 dev / 40 holdout, one clip per original source, excluding prior weak-data/audited sources from holdout. This is QVH query-conditioned human supervision, not AIC joint GT.

Frozen weak-trained clip heads evaluated on human-rated clips in the previously exposed ten-video weak validation set: Spearman VideoMAE .23530 / DeiT .17771 / A0 .12251 / ViT-B .02553. VideoMAE−DeiT delta +.05759, paired 95%CI [-.21716,+.33909]; no upgrade. Nine queries belong to native TRAIN and one native VAL: this is a label sanity check, not native validation or OOD. No human labels were used to change checkpoints or thresholds. Reports: `reports/qvh_human_sanity_20260926/`.

Cache provenance bug fixed for future extraction: explicit encoder identity no longer overwritten by ResNet18. Existing caches preserved with external provenance audit.

## Candidate D/E controlled build in progress

GitHub main synchronized through prior evidence; native training code frozen at c1156e7. No official upload. `NATIVE_CANDIDATE_V1` runs 96 train / 24 dev only with a matched retrained DeiT control. Original holdout is not materialized. Masked query-averaged human saliency /4; target binary threshold .75; prediction threshold fixed at SUB_C .35. Same U-Net family, original PTS interpolation and YuNet dense_v1. Explicit additional variables versus SUB_C: native supervision and 2-second feature anchors; VideoMAE16/InternVideo8 temporal context. These are not pure backbone-only comparisons against historical SUB_C. Candidate export waits for DEV evidence. Current regression 97 passed.

## Native foundation candidate gate (2026-09-26)

- Matched native-QVH DEV control: DeiT-S F1 `.76543`, Spearman `.11364`, NDCG `.96287`.
- VideoMAEv2-Base: F1 `.76543`, Spearman `.06824`, NDCG `.96109`; paired ranking deltas are non-positive/uncertain. **SUB_D NOT READY; no official inference.**
- InternVideo2-Stage1-1B: F1 `.76609`, Spearman `.23474`, NDCG `.96763`; paired deltas Spearman `+.12111`, NDCG `+.00476`, but bootstrap CIs cross zero. **SUB_E promising engineering candidate, not proven replacement.**
- SUB_E frozen manifest is `/data/aic/experiments/NATIVE_CANDIDATE_V1/SUB_E_frozen_manifest.json`; its 174-video raw release is running with unchanged threshold `.35`, 2 FPS, and YuNet `true_face_smooth`. Official score remains null and no upload is authorized.

## SUB_E engineering release completed (2026-09-26)

- InternVideo2-1B native-QVH candidate completed frozen raw inference for 174/174 official videos using fixed parity shards on physical GPUs 2 and 4; no test-specific tuning or content inspection.
- Merged output: 87,637 predictions, zero empty videos. Project validator, independent checker, and unzip regression all valid with zero errors.
- Weight bytes `2,049,501,387`; parameters `1,022,373,889`; L tier, expected `k_size=0.90`. ZIP SHA256 `342d36e229f2502d863e8944a4906030050e16282cc94f5cb81138bc9a0d419a`.
- ZIP path: `/data/aic/official_test_20260926/submissions/SUB_E_INTERNVIDEO2_NATIVE_V1_FINAL/upload.zip`. Official score remains null; do not auto-upload.

## Official SUB_E feedback (2026-09-27)

- The InternVideo2 Stage1-1B K700 submission received official platform score **34.43**. This is recorded as the platform-reported score; no raw/size decomposition is inferred without an official breakdown.
- This materially validates video-native foundation scaling over the previous SUB_C score 6.64. Next priority is higher-ceiling foundation candidates, followed by controlled post-training.
- VideoMAE-Large native DEV: Spearman `.17574`, NDCG `.96752`; InternVideo2 Stage2-1B native DEV: Spearman `.21433`, NDCG `.96962`. Stage2 full package subsequently received official platform score **34.42**; no raw/size decomposition is inferred.

## Foundation scaling continuation（2026-09-27）

- SUB_F InternVideo2 Stage2-1B native-QVH head + unchanged YuNet completed frozen 174-video inference. Output is 174/174, 87,781 predictions, zero empty videos; project merge validator, independent checker, and post-unzip checker all valid.
- SUB_F loaded weight bytes `2,827,511,457`, parameters `1,020,930,561`, L tier, expected `k_size=0.90`. ZIP SHA256 `a7622bae7b78712549a344f5b5d61fd1762c395a734d82a2c85173bc344a6537`; it is READY_TO_UPLOAD and has not been uploaded.
- Full K710 native-QVH extraction and fit-only completed on 96/24 without holdout access. Best epoch 12: F1 `.76543`, Spearman `.19818`, NDCG `.96576`; this is below Stage2 ranking and remains a diagnostic result, not a submission candidate.

## K710 official submission package（2026-09-27）

- K710 frozen candidate `SUB_G_INTERNVIDEO2_K710_NATIVE_V1` completed full official-test inference: 174/174 videos, 87,585 predictions, zero empty videos.
- Loaded weights `2,049,516,187` bytes, `1,022,373,889` parameters, L tier (`k_size=0.90`). Merge validator, independent checker, and fresh unzip regression all passed.
- ZIP: `/data/aic/official_test_20260926/submissions/SUB_G_INTERNVIDEO2_K710_NATIVE_V1_FINAL/upload.zip`; SHA256 `a5be193b72e0f18d4a91aabc94f3fd237aa229803db74abc6086100527f61366`. Official score was subsequently reported as **34.42**; raw/size decomposition remains unavailable.

## Official score root-cause analysis（2026-09-27）

- New platform feedback: G `17.46`, E01 `17.46`, Stage2 `34.42`.
- Audit found the parallel-release trap: `SUB_G_shard0` and `SUB_E_shard0` each contain only 87 even-ID videos, while the corresponding `*_FINAL` packages contain all 174 videos. The 17.46 value is approximately half of the 34.42–34.43 full-coverage score range.
- K710 full JSONL differs from K700 on only 3 videos / 340 net frame-set changes, with identical YuNet boxes on common frames; model quality cannot explain a 2x score drop. Treat G=17.46 as a package-coverage/upload-file issue until the platform-side uploaded SHA is confirmed.
- Full report: `reports/20260927_official_score_root_cause.md`.

## Official score correction（2026-09-27）

- 用户确认 K710 完整 174-video FINAL 包的官方平台分数为 **34.38**。因此此前 G=`17.46` 已确认是单 shard 覆盖问题，而不是 K710 模型质量。
- 完整包官方结果：K700 `34.43`、Stage2 `34.42`、K710 `34.38`。三者最大差异 `0.05`，均使用相同 L-tier 规则、threshold、2 FPS、Temporal U-Net 和 YuNet spatial；当前没有证据证明 Stage2/K710 超过 K700。
- 最新报告：`reports/20260927_official_score_root_cause_v2.md`。
