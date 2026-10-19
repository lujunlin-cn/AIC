
## P0 — 研究1 PDF 执行后的下一路线（2026-10-19）

- [x] R94 U²-Netp H1：真实 forward、同容量零输入消融、LIVE 源级 dev/confirm，具体配方关闭。
- [x] R95 shot structure H2：同容量零结构控制、448 源配对 bootstrap，embedding 代理配方关闭。
- [x] R96 事件上下文可观测性：12源源级留一，context AP表面+0.0301但CI跨零，冻结上下文优先级。
- [x] R97 GIF 区间前/中/后状态辅助 TCN：448源差 -0.000780，CI跨零，关闭具体状态损失。
- [ ] 目标域事件状态：从 R42 V1.3 人工标签和公开 PHD/GIF 区间中构造显式进行中/结果/依赖状态，保持 `label_status` mask，不把未选帧当硬负。
- [ ] 在源级留出上比较状态读出、R54 时间和中心控制的联合集合效用；未超过固定控制前不生成官方包。
- [ ] 若状态监督通过，才接入轻量完整上下文时间头并重新核算全推理组件参数；R94/R95 权重与数据只留远端，不进 Git。

## 竞赛迁移研究后续（2026-10-19）

- [ ] 用已授权 Kaggle 凭证完成 AIKU `train.csv`/`train.h5` 访问确认；当前可列文件但下载返回403，先不训练、不假定字段。
- [x] R98 UNISAL/AIM 2024 视觉注意力中心诊断；通用显著性中心负增量，关闭直接替代构图点。
- [x] R99 低秩 local×global 交互与跨源置换；方向性信号低于实用门，关闭当前8点配方。
- [ ] 目标 crop 监督下的高分辨率当前帧 + 低分辨率时序上下文读出：固定 R70 候选、固定预算和时间集合，先做同容量零交互控制，禁止音频。
- [ ] 若 AIKU 访问成立，先做 PHD/GIF-only vs AIKU→PHD/GIF vs 等步数 control 的 source-disjoint 时间迁移审计。
- [ ] AViMoS 只保留视觉注意力预训练候选；下载前先做存储/许可/帧对齐审计，不把 SaliencyTrain 或 UNISAL 输出当 AIC GT。

- [x] AIKU 压缩包审计：6,000 train / 1,500 test、1024D预提取视觉特征、连续 gtscore、二值 gtsummary、change_points；原视频/特征提取器/来源缺失，保持 AUDIT_ONLY。
- [x] R100 R70 pooled-context 空间代理；增量低且置换不支持，关闭。
- [x] R101 R70 当前候选 + ResNet18 邻帧视觉 context；增量 `+0.000197`、CI跨零，关闭当前配方，不出包。
- [ ] 等用户决断：是否投入资源取得 AIKU 特征来源/原视频协议，或转向新的观测合同/监督。

# 近期任务

## P0 — R85 target-domain transfer (2026-10-19)

- [ ] 重建PHD/GIF源级目标清单：`gif_soft`与GIF区间union分开保存；已知`gif_hard`占位行不得当真负。
- [ ] AHA目标域适配pilot：冻结基座+小时间头，固定R54时间预算和R70/B3空间，比较源级留出集合效用。
- [ ] HL-CLIP式saliency pooling候选：先核对代码、权重和许可证，再与AHA适配在同一PHD/GIF分割上比较。
- [ ] 建立真正的负视频池后再测试empty-video gate；在没有负池前不根据官方榜单假定空GT比例。
- [ ] 只在目标域集合效用超过PHD2 frame-rank和中心控制后生成一个官方候选；不把QV/TVSum单项指标作为出包门。

## P0 — 当前复赛（2026-10-16）

- [x] R29精确期望F推导、可复算反例和全动作穷举；区分数学结论与项目可学收益。
- [x] R29公共26TRAIN/14DEV标量校准、插件预算与同K位置；独立复算520项通过，当前配方不晋级。
- [x] R29原生1fps覆盖检查：所有400条rater序列正run均被采样命中，不能沿用12点漏采解释。
- [x] R29补完R28公共24项、空间GT候选诊断与失败图；独立复算、资源、hash及退出状态留档。
- [x] 当前官网独立核实参数≤9B、无k_size、`model_params_b`及赛期，更新01并保留旧ZIP。
- [ ] R42已开放12源双人通用高光机制诊断；待完成后按`configs/R29_NEXT_SUPERVISION_STAGE.json`判断是否需要native目标比例联合监督。当前自由比例框不能替代native crop GT。
- [ ] 先用相同输入的简单读出和位置控制检验源外内容增量；静态不可辨而连续可辨时再投入时序观测，未过门不扩模型。
- [x] R30冻结12点VPB静态/相邻差分读出：动态高维配方源外失败，未启动大时序训练。
- [ ] R31低维事件状态特征（norm/cosine/窗口汇总+强正则）与位置控制；失败则停止视觉公共读出路线。
- [ ] 新候选生成器/validator采用`model_params_b` B单位；取得当前平台schema/样例后验证；不改历史回分ZIP。
- [x] 原生BF16 AHA全426视频推理、固定B3空间、独立校验与提交ZIP。
- [x] 更正9B参数上限，撤回旧文件大小否决；更新规则、实验与决策记录。
- [x] 记录用户报告R25_AHA_BF16_K085复赛35.18及本地交付ZIP SHA；上传字节对应未独立验证。
- [x] 同赛段比较：R25较35.52低0.34、较全帧34.75高0.43，不晋级；初赛分不可参考。
- [x] 复验426视频包、量化预算超额和位置倾向、核对AHA上游协议，见reports/r25_score_analysis/summary.md。
- [x] AutoDL恢复；取回426份scores，完成双头饱和/相关性/时间漂移与原mask重建。
- [x] R26-A：26TRAIN/14DEV冻结AHA与位置/随机控制，精确预算+配对CI；预留14未打开，未证明胜主要控制。
- [x] R26-B/C：精确85%包、等预算中心时间控制、原R25固定mask中心空间包已双校验，未自动上传。
- [x] R26用户回分36.04/35.07/30.23已记录；匹配对照分析并晋级中心85+B3。
- [x] R27固定B3中心75/95、85前/后移四包已生成，426视频双校验，SHA与环境记录。
- [x] R27中心75=35.87、95=35.39已记录；停止预算/位置扫描，前/后移包暂停。
- [x] R28现役B3全426视频空间缓存精确回放，无坐标差或实际关键帧回退。
- [x] R28公共pilot原13条取回核验，R29补11条，完整24条/metrics/日志/环境已取回，独立重算通过。
- [ ] 恢复B3_s0头、LFM基座与RV原视频来源绑定；同帧B3控制及原生x/y数据后再决定候选。
- [ ] 核对新最佳B3完整权重、环境和原视频推理入口，完成最终作品部署材料。
- [x] R29在TRAIN源拟合双头/位置校准与预算对照，未过源外门，停止当前配方；完整AHA特征/query/未来信息仍未排除。

## P0 — 2026-09-30 初赛截止保护

- [x] V6-NEXT10H 四线（E1-AUTO/E3/E4/E5）收口，0 新候选包，正式最佳保持 INTERP 48.67。
- [x] 复赛包 SUB_SF_K700_SEMIFINAL_V1 交付（待上传）。
- [ ] 09:30 冻结候选确认、10:00 材料就绪（初赛 11:30 截止）。
- [ ] T1 骨架在学生原生特征域重训 ×3 种子 → confirm 门 → 过门则做单一因素官方提交（NEXT_ACTIONS #1）。
- [ ] E1-AUTO 扩展 108 条（教师 ~3.2h @0.19qps）+ 重跑 A/B 对照（报 CI）。
- [ ] AVE-PM 下载完成（68,067,427,328B 校验 → 停临时 mihomo → 恢复 AIC_EXT_PROXY 默认 18890）。

## P0 — 本轮收口

- [x] 重新读取协议和实际本地/远程状态；保留旧失败和checkpoint。
- [x] 原7条降级为 comparison_holdout_v1，不伪造 TVSum 新 lockbox。
- [x] ranking/author-style 15% summary实现和 MATLAB/Octave 数值交叉验证。
- [x] 真实 MAT category + repeated nested source-group CV：90次训练、3seed、2×5fold。
- [x] 视频级配对 delta CI、逐视频/类别/FP/FN/过选分析、timeline/contact sheets。
- [x] SumMe替代 raw 路径恢复、严格帧数筛选、原生 evaluator 交叉验证。
- [x] 三个历史完整bundle OOD + 三条表示各30个匹配CV heads的OOD。
- [x] RetargetVid annotations、20条DHF1K原视频、6方法真实IoU、native前置clamp核对。
- [x] dense spatial raw inference/EMA/真实face observer/权重计量/JSONL一致性验证。
- [x] 完成有界 SumMe 独立第二批8条固定配置评测；两批14条分开报告并合并，全部CV heads与完整bundle均完成。
- [x] 加入固定预算常量/均匀/32次随机非学习对照，视频级paired CI；700对SHA/稀疏pHash筛查。
- [x] 冻结3个候选的完整权重SHA/bytes、离线运行入口；本地备份、真实阈值raw JSONL验证。
- [x] 完成A0/DeiT线性head容量控制60次nested训练及已有OOD复核，结果拒绝当前线性替换。
- [x] 最终报告、registry、provenance、状态与干净代码副本复现；本地回归通过，Git/远程同步留审计。

## P1 — 按当前证据排序

- [ ] 扩大可靠 raw OOD 覆盖并核对源视频版本/近重复；模型参数冻结，不能用新增 OOD 反复调参。
- [ ] 取得 AIC 样例/index/同版本 evaluator；先复现 A0 center；有联合GT才做oracle和官方大小加权。
- [ ] 独立空间DEV/评估划分后，针对多人选错主体做一个受控 observation/association 实验；保持center控制和全权重计量。
- [x] `SPATIAL_GROUP_003` 完成固定 top-3 人脸群体中心对照；相对单脸/center 均未形成稳定增益，降级该具体假设，保留后续真正主体关联研究。
- [ ] 完成 DHF1K/RetargetVid 021–030 固定参数确认，若原视频与标注恢复成功则只做一次性 transfer check，不再用其调参。
- [ ] 继续 `YTH_ACQUIRE_001` Range tar 索引；先取得所有小 metadata，再决定是否仅恢复有界人工标注视频 subset。
- [x] 记录 YTH/DHF1K 新来源连接 blocker；不把未取得原视频写成 OOD 或空间泛化证据，后续改用稳定镜像/预提取 benchmark 或继续已有固定候选。
- [ ] 基于 native summary/ranking 错误提出单一 loss/head 假设，通过nested inner选择；不进行DeiT超参扫。
- [ ] 仅在teacher能快速本地部署时，50–200训练/开发clip pilot，用human correlation测质量；暂无收益证据，不蒸馏扩量。
- [x] 保存fallback发行配置、环境、权重SHA和完整提交入口；严格官方环境仍待公布，不能声称已官方验收。

## P2 — 当前明确低优先级

- [ ] Feature Bank v2仅修复恒零/重复后做独立残差门控组；真实音频需实际波形。
- [ ] 只有新OOD机制证据时才重开canonical TSM；保留代码/cache/tests。
- [ ] 官方联合数据到位后研究size-score Pareto、压缩；不把proxy加权冒充competition score。

- [x] `RG_RANK_001` 完成预注册 pairwise loss nested 对照并登记结果；不升级候选。
- [ ] 对 `RG_RANK_001` 的 50-video paired vectors 生成正式 bootstrap CI，并与 BCE 控制做同协议 paired comparison。
- [ ] 用 Clash 代理恢复 YouTube Highlights 有界人工标注视频；完成 native frame/alignment 后才纳入 OOD。
- [ ] 处理 DHF1K 021–030 RAR Unsupported Method（寻找兼容 unrar/恢复工具），成功解码后只做冻结空间 transfer check。

## 2026-09-26 评测集准备

- [x] 建立 `AIC_EVAL_INTAKE_V1`：真实解码、PTS、尺寸/帧数、SHA 和 intake manifest。
- [x] 建立多候选 frozen release batch 入口，防止覆盖输出并核对 intake/release hash。
- [x] 用短视频完成 compact index 到 A0 raw JSONL 的端到端演练，当前本地回归 86 个测试通过。
- [x] 远程 `/data/aic` 完成资产 inventory：权重/代码哈希，数据 metadata，symlink 和异常状态审计。
- [x] 评测集到手后保留原始压缩包，生成 intake manifest，并在独立目录完成 174/174 解码核验；禁止直接覆盖原始目录。
- [x] A0 center、DeiT-S center、DeiT-S + YuNet spatial 均完成全量冻结推理，记录逐候选运行时间、空输出率、权重清单和 validator 审计。
- [x] 三个正式 JSONL 通过项目 validator、独立 checker、ZIP 解压回归；根目录上传 ZIP 已生成并记录 SHA256，见 `reports/20260926_official_test_inference.md`。
- [ ] 等待用户按 SUB_A → SUB_B → SUB_C 上传；官方平台字段/联合 evaluator 仍待确认，禁止 Agent 自动提交。
# P0 — Official score driven next experiments

- [x] Record R86 official score 34.88 and audit the R54 parent contract; identify A1raw/B3 point-source confound.
- [x] Generate R88 with the same T5 H1 coefficients on `points_B3_vo/points`; local semifinal contract passes. Do not upload automatically.
- [x] Repair R87 AHA FP16 overflow with BF16 and verify finite trace/full pilot; keep AHA out of submissions until target-domain adaptation clears a source-level gate.
- [ ] If a submission slot is available, use R88 as the single-variable corrective test; bind the uploaded SHA to the recorded package.
- [ ] Add point-source identity and parent observation-contract checks to future spatial package manifests/builders.

- [ ] Finish and archive `QVH_FINETUNE_PROBE_20260926` frozen vs DeiT-S last-block controls on the matched 37/9 weak-label subset.
- [ ] Build a fair ViT-B + YuNet official inference candidate; audit both parameter count and loaded bytes before any submission.
- [ ] Re-run A0/DeiT temporal heads with one identical trainer/AMP/checkpoint rule to reduce the A→B causal confound.

# P1 — Public data and model scaling

- [ ] Expand QVHighlights-derived extraction only after the 5% pilot passes label/time-axis checks; keep `official_aic_gt=false`.
- [ ] Add a bounded VideoMAE-V2/InternVideo2/UniVTG candidate inventory with V100 compatibility, license, parameter count and weight bytes. Start with one downloadable M-tier reference.
- [ ] Test 4 FPS against the frozen DeiT/SUB_C protocol on a source-group validation subset before increasing backbone scale.
- [ ] Use the official raw break-even values: M `>6.98947`, L `>7.37778` relative to C=6.64.

# P2 — Spatial and multi-task

- [ ] Train/evaluate a generic subject/crop head on the 624 weak crop-labelled QVHighlights rows only after temporal controls are stable.
- [ ] Add low-cost audio features only with decoded waveform evidence; do not revive the zero audio proxy.

## Continuation P0

- [x] Audit original source IDs: 23 shared sources in 800/89, emit train-purged 775/89 protocol.
- [x] InternVideo2 Stage1-1B V100 FP16 compatibility (8f/16f); no highlight quality claim.
- [ ] Complete corrected VideoMAEv2 V2 and matched clip-target controls before comparing representations.
- [ ] Use native human QVH labels for confirmation; current seed-label agreement cannot establish human highlight quality.
- [ ] Evaluate a genuine 100–500M-parameter encoder; VideoMAEv2-Base and ViT-B are below that parameter range.

## P0 continuation after corrected probe

- [x] Correct VideoMAEv2 preprocessing/cache and run matched clip-mean controls.
- [x] Acquire public native QVHighlights annotation JSONL without sending videos to a third party.
- [ ] Validate raw-video/annotation segment alignment for a bounded native QVH subset before using it.
- [ ] Only then run a source-disjoint native temporal benchmark; keep `official_f_video=null`.
- [ ] If pursuing user-requested 100–500M parameter sweet spot, audit a genuinely 100–500M-parameter encoder separately from byte-based VideoMAEv2/ViT-B references.

- [x] Native QVH bounded full-decode audit: 10/10 pass; 7,738 raw ID matches.
- [x] Frozen-checkpoint human-saliency sanity on exposed videos; CI crosses zero.
- [x] Freeze source-disjoint native 96/24/40 protocol, exclude historical weak sources from holdout.
- [ ] Build query-aware or explicitly masked native-human training targets; compare frozen/partial FT on train/dev without opening holdout metrics.

## Current narrowed task

- [x] Push historical official scores/probe/native protocol evidence to GitHub main.
- [x] Freeze native 96/24 masked-label D/E training controls; no holdout.
- [ ] Finish three matched representations and paired DEV analysis.
- [ ] Only evidence-supported D/E: freeze -> raw smoke -> 174-video JSONL -> validators -> ZIP.
- [ ] Publish final READY/NOT READY, weight accounting and size break-even; do not upload.

## Native foundation candidates

- [x] Complete matched DeiT / VideoMAEv2 / InternVideo2 native-QVH DEV training and paired analysis.
- [x] Reject VideoMAEv2 candidate on ranking gate; preserve probe report.
- [x] Freeze InternVideo2 manifest with checkpoint/code/index hashes and YuNet weight accounting.
- [ ] Finish SUB_E raw 174-video release, independent validation, ZIP regression, and final candidate report.
- [ ] Do not upload SUB_E unless the user later requests it and accepts its exploratory evidence status.

- [x] Complete SUB_E frozen 174-video raw inference, merge, validator, independent checker, and unzip regression.
- [ ] Decide whether to upload SUB_E after weighing its positive but CI-crossing native ranking signal against the L-tier raw break-even `7.3778`.

## 2026-09-27 foundation scaling

- [x] Record official SUB_E platform score `34.43`.
- [x] Download and adapt public VideoMAE-Large (M-tier) and InternVideo2 Stage2-1B (L-tier) with V100-safe inference.
- [x] Complete native-QVH DEV probes for VideoMAE-Large and Stage2-1B.
- [x] Finish Stage2-1B official frozen inference and validator/ZIP release.
- [ ] With Stage2 official score `34.42`, begin controlled post-training/adaptation on the stronger base using train/dev only.

## 2026-09-27 foundation release

- [x] Complete SUB_F Stage2-1B frozen 174-video inference, merge, validator, independent checker, unzip regression, and manifest.
- [x] Complete full K710 96/24 extraction and fit-only gate without holdout access.
- [ ] Obtain official feedback for SUB_F if user uploads it; do not auto-upload.
- [ ] On positive Stage2 feedback, run controlled post-training/adaptation on Stage1/Stage2 using train/dev only; preserve SUB_E 34.43 control.
- [ ] Keep K710 out of official submission queue unless a new adaptation produces independent positive ranking evidence.

- [x] Produce K710 frozen official JSONL and ZIP package on request.
- [ ] Wait for user/platform feedback on K710; compare against SUB_E K700 and SUB_F Stage2 without inferring raw score decomposition.

## Official score discrepancy follow-up

- [x] Audit E/G shard and FINAL ZIP coverage against the 174-entry index.
- [x] Write root-cause report for G/E01 17.46 versus Stage2 34.42.
- [ ] Confirm platform-side uploaded-file SHA for G and E01; if shard was uploaded, resubmit only the corresponding FINAL ZIP when allowed.
- [ ] Do not conclude K710 underperforms until a 174-video FINAL upload is scored.

## Official score correction

- [x] Confirm K710 full FINAL official score `34.38`.
- [x] Reclassify G/E01 `17.46` as shard coverage failure.
- [ ] Build a submission preflight that rejects any ZIP whose JSONL coverage is not exactly 174 IDs before upload.
- [ ] Move next official candidate design from frozen foundation swap to temporal/spatial adaptation with train/dev evidence.

## P0 — R23 探针（2026-10-15 建成，待提交）

- [x] 探针包建成 + 三层验证（合成干跑 / 闭环 / 真包 validator）。
- [x] 更正读数形式为逐视频精确期望；更正主读数点 p 0.5→0.25；声明尺寸改整字段省略。
- [x] 用户已上传 R23 P250/P500，回分18/26。
- [x] 撤回双点唯一反解 d：逐视频异质密度不能由两个宏平均总分识别。
- [x] 官方回分及推导限制写入 STATUS.md。

## P0 — 用户新授权（2026-10-15）
- [x] 记录 R23 官方复赛 P250=18 / P500=26；禁止跨初复赛比较。
- [x] R24：现算 VPB point/block ×2 seed；门失败，不晋级。
- [x] R24：同协议 CLIP 对照；未见替换优势。
- [x] R24：已按20min上限尝试AHA；缺wandb失败，未完成视觉前向。

## R24 后续
- [ ] 固定源划分，做12/48采样GT oracle分辨率诊断，后续训练以其机制证据决定。
- [x] 补齐AHA依赖，完整校验SigLIP，BF16真实视觉smoke通过；保留FP16失败。
- [ ] AHA若继续蒸馏，先固定公共train/dev小批协议，测人工标签相关性及无teacher控制，不能凭smoke扩量。

## R32 — 2026-10-16

- [x] 记录R31低维事件状态probe及暴露DEV限制。
- [x] 取回并核验AIC426 `temporal_k0.90`缓存，生成R32_VT_K090_B3。
- [x] 426/426覆盖、207292帧、独立checker和semifinal contract通过；参数0.450529778B。
- [x] 生成确定性ZIP、manifest和SHA256。
- [ ] 用户上传R32并回报官方分；上传前保留R26_POS_CENTER_EXACT085_B3回滚包。
- [ ] 若R32低于36.04，停止无结构keep扫描，转向可追溯时序观测/空间部署合同；若高于36.04，冻结其时间链路并做独立复核。

## 明日提交（2026-10-16）

- [x] 生成R33 replay、R34 exact-DP、R35 union90、R36 consensus85四个新包。
- [x] 四包426/426覆盖、独立checker、semifinal contract、ZIP解压回归通过。
- [ ] 按R35、R36、R34、R33、R32顺序上传并记录每次官方回分与上传ZIP SHA。
- [ ] 不用官方回分反推GT；根据五包成对差异决定是否冻结双头并集/共识路线。

## 2026-10-16 — R37–R41之后

- [x] 记录用户2026-10-17回分R41=35.19、R40=34.15、R39=34.48；绑定本地ZIP SHA，保留R26=36.04安全回退。
- [x] 完成三包失败范围分析；停止当前变化/冗余/新颖性/门控配方扫参，不由总分反推测试GT。
- [ ] R37/R38如有回分再记录；当前无已知分数，不登记为失败。
- [ ] 若内容候选全部失败，转音频/事件结构或恢复可复现空间部署合同。

## R42 — 人工先验诊断

- [x] 从 AutoDL `QVH_NATIVE_CANDIDATES` 固定抽取12个30秒源视频、360张1fps图和12个低码率MP4（约22MB）。
- [x] 生成盲标 Label Studio interface、local-files/file-uri任务和操作协议；任务不带类别、抽样原因或QVH query。
- [x] 完成两源60帧Smoke正式标注和只读导出；视频ID、source_time_sec、框原尺寸及坐标合同检查通过，原始结果保留。
- [x] 复查用户修订14题后的Smoke，原18题旗标已缩至6题；近零框、非高光留框及构图缺项消除，保留新旧快照和2份草稿，不自动改人工答案。
- [ ] 甲补修34、35、38、57、59、60六题多框为一个完整保留窗口；见`reports/r42_smoke_revised_20261017/repair_queue.csv`，不机械取并集，不因框问题改等级。
- [ ] 完成其余10个视频；6个重叠视频由第二位标注者在独立项目中完成。
- [x] 编写并运行Smoke只读导出审计，报告源级分布、信息依赖、框几何、草稿冲突及逐题复核队列；60帧不当60个独立来源，不报告单人一致性/准确率。
- [ ] 完成12源后分析查询错位/构图冲突；有第二位标注者时单独分析一致性。没有同批合法GT前不把自由比例框当native crop GT，不把本批标签用于提交训练。
- [x] 冻结人工先验标注标准 v1.1：完整视频含声音优先、0/1/2/3/不确定裁决树、阶段与信息依赖字段、仅高光2/3画框、自由比例框不作为 native GT。
- [x] 用Smoke导出验证v1.2四核心字段60/60完整、2/3信息依赖43/43完整、框坐标通过；空间语义校准仍待完成，未宣称整体验收通过。
- [x] 标注界面v1.2隐藏逐帧“未看查询”确认，信息依赖改可选，允许0/1跳过并补齐填法；保留旧结果和任务ID。
- [x] 配置 Label Studio 用户级常驻服务与 FRP 公网中转；公网入口为 `http://47.108.220.174:3002`，Smoke 项目已导入 60 个任务。
- [x] 在Smoke现有60任务中原位增加带声音30秒播放器和单独打开链接，保留任务ID和标注/草稿；同步360任务生成器；两个Smoke视频公网播放、音频解码与跳转验收通过。
- [x] V1.3通俗标准、裁决树和8个虚构例子部署到任务顶部与Instructions；取消去冗余/高光数量配额，强调一帧一个窗口。
- [x] 按冻结双人分工去重扩展：甲新增240题、乙240题，540份任务覆盖360图/12源；旧60正式结果、2草稿及旧任务data哈希保留。
- [x] 公网浏览器验证视频播放/跳转、0/1跳过依赖及2/3无合理框备注提交；全部测试标注写请求拦截。360图可读取，12源公网媒体200/206，幂等重跑新增0题，自有验证session已撤销。
- [x] 两个项目各完成新增240题；按项目身份完成导出。审计发现两个项目实际使用同一账号，不能把它们称作独立标注员。
- [x] 完成双方V1.3新增4重叠源及旧Smoke重叠源的重复测试分析；保留分歧/未知和每人框，按原视频源统计，不平均框。真正第二标注者复核仍是下一证据门。
- [ ] 对四个V1.3重叠源或最小高分歧集做第二账号盲复核，获得可报告的人际一致性；不自动重写现有540条标签。
- [x] 540条正式标注完成质量审计；结构合同零错误，有限语义/构图修补队列见`reports/r42_annotation_v13_full_20261017/`。

## R43 — InternVideo2 6B 事件完成度（正交人工标签）

- [x] 冻结事件/准备/结果提示、四帧窗口、0.75/0.25秩融合、精确85%预算和B3继承规则；未用人工标签。
- [x] AutoDL公开smoke：加载、参数、显存、prompt顺序、anchor到原始frame映射和有限值审计通过。
- [x] smoke通过后完成可恢复的426视频全量推理（batch=88中途OOM，batch=4续跑；单次任务受外层timeout保护），未扫权重/keep，未使用测试标签。
- [x] 零合同问题、0回退、资源可行，已生成R43上传包并记录ZIP SHA。
- [x] 记录用户回报R43=34.95（相对R26 -1.09），停止R43提示/融合扫参。
- [ ] 完成冻结特征的“进行中/结果”双头、音频/语言门控和源级留一诊断；通过预注册门后再考虑新候选。

## R44 — 可观测性与联合效用诊断

- [x] 修正项目2审计修补链接，按项目ID重生成队列；不改写正式标签。
- [x] 记录同账号重复性：等级二次加权κ=0.185，阶段κ=0.164，依赖κ=-0.015；标签只作诊断/不确定性掩码。
- [ ] 按源留一建立通用等级、进行中/结果、信息依赖三个目标的低维读出。
- [ ] 做中心→静态→局部前后状态→音频可靠性门控的嵌套对照；保留每源配对效应和失败例。
- [ ] 仅当源外效用增量稳定时，冻结“中心先验+受限内容残差+B3”的候选函数；否则保留R26并停止当前内容替换路线。

R44_DIAG_001已完成CPU留一：局部状态描述子使通用等级MAE 1.0668→0.9695，9/12源改善；进行中/结果与依赖目标均未改善，低级音频特征暂停。下一步只做低容量局部状态残差读出，不直接生成提交包。
### R44 package

- [x] 以 R26 36.04 中心85+B3为父包完成人工标签校准残差候选；仅改变时间集合。
- [x] 完成 426/426、199037 帧、精确 `round(.85*N)`、B3 框逐帧一致性、独立 checker 和 ZIP 解压回归。
- [ ] 由用户决定上传 `R44_HUMAN_CALIBRATED_RESIDUAL_ETA020_B3`；不自动上传，官方回分待确认。
- [ ] 回分后只做机制分析：超过 R26 才保留；不超过则停止 eta/alpha 扫描并转新观测或原生空间监督。


## R44 回分后（2026-10-17）

- [x] 记录 R44 官方回分 36.09、R26 36.04、差值 +0.05；更新 R44 manifest 为用户回分状态。
- [x] 完成 R44 随机交换 null 分析并保存可复现报告；结论为弱信号，不能宣称稳定收益。
- [x] 修改 AGENTS.md，解除旧路线、旧资源、停止门和扫参限制对研究思考的束缚，保留官方规则、隐私安全、禁止测试标签训练和可复现记录等硬约束。
- [x] 构建并审计 `R45_BOUNDARY_GUARDED_RESIDUAL_B3`：R26 中心窗口中间冻结，两端各 5% 边界带允许 R44 局部交换；不扫 eta/alpha。
- [ ] R45 回分后只做机制判断；若仍低于实用收益门，停止人工残差微扰，转同域空间或真实音频/事件结构。

## P0（2026-10-18）

- [ ] 完成AutoDL D3/D4高频候选，取回predictions与运行日志，验证源SHA、全帧覆盖、预算和B3框一致。
- [ ] 用固定5%边界规则从D3/D4各派生一个包；禁止eta、alpha、边界比例扫描。
- [ ] 更新四包manifest、SHA、checker和提交登记；保留R26/R44/R45作为控制/回退，不自动上传。

## 2026-10-18 后续候选与明日额度

- [x] 完成TimeLens2固定prompt的426视频分片推理、合并和原始响应SHA审计。
- [x] 构建`R47_TIMELENS2_V9_B3`（8.8545B）及`R47_V9_B3_CENTER85`空间对照；不使用超9B的TimeLens2+完整B3组合。
- [x] 构建`R47_TIMELENS2_CENTER_EXACT085`严格参数合规的中心构图对照。
- [ ] 等待D3 4fps local-state-only分片完成；合并后构建含`model_params_b`字段的完整包及固定5%边界包。
- [ ] 统一明日5个提交候选：优先R26、R44、D3或D4中证据较强的一包、R47 V9 B3、V9 B3 center空间对照；R45作为边界机制备用。只提交全量覆盖、ZIP往返和字段checker通过的包。
- [x] 将明日五包冻结为R26、R44 current-field、R46 D3 local、R47 TimeLens2+V9 B3、R47 V9 B3 center；独立checker与SHA审计见`reports/next_day_five_candidate_audit_20261018.json`。
- [ ] 记录每次官方回分及上传SHA；不依据一次回分做eta/alpha扫描。

## P1

- [ ] 若官方回分允许，恢复精确V8 B3 head/base或同SHA的原始logits，再做129候选utility residual；V9/V7近似head只做诊断。
- [ ] 高频路线若官方无稳定收益，转向真正第二账号复核或合法音频/事件结构监督，不继续扩大同一账号标签。
## R48（2026-10-18）

- [x] 完成共识门控公共判别：1fps/4fps同侧门控代理`+0.02360`，停止起始变化门控和阈值扫描。
- [x] 完成R48全量部署、合并、逐帧框继承和ZIP往返审计；明日最多只占一个机制名额。完整包与固定5%边界包在`submissions/r48_candidates_20261018/`，尚未上传。

## 2026-10-18 时间残差扩展与下一主线

- [x] 完成R49事件阶段残差判别；公共增量+0.00291，停止。
- [x] 完成R50全片轨迹上下文及去位置消融；均低于统一D3，停止。
- [x] 完成R51持续性门控反例；增量降至+0.00654，停止。
- [x] 完成R52 Farneback光流观测对照；增量+0.01093，停止。
- [ ] 等待并审计R48全量输出：426/426、199037条、精确预算、R26 B3框逐项一致、ZIP往返；不上传。
- [ ] 明日五包中将D3/R46作为时间残差主候选，R48最多作为其可靠性门控控制；不提交R49–R52。
- [ ] 完成TimeLens2 V9 B3、TimeLens2中心、V9中心三个候选的最终checker/SHA/参数清单复核；优先把提交额度用于结构不同路线。
- [ ] 空间侧继续寻找精确B3 utility资产或建立不依赖人工native GT的候选集合诊断；不能把V9空间替换代理写成官方收益。

## 2026-10-18 R53/R54

- [x] 完成R53低维音频、画质条件化和可部署阶段预测判别；均未形成新包，原因与结果写入`EXPERIMENTS.md`。
- [x] 完成R54 TimeLens2中心锚定残差公共代理：相对中心`+0.024997`，10/18视图正；固定`eta=.20`，不扫参。
- [x] 生成并审计`submissions/R54_TIMELENS_CENTER_RESIDUAL_V9_B3/`：426/426、199037、8.854531425B、contract/checker通过。
- [ ] 明日若试水，优先上传R46与R54；记录官方分数、平台参数字段和上传SHA。
- [ ] 若R54低于R26，停止TimeLens2硬/残差解码微调；若高于R47，保留中心锚定规则并分析V9空间分层。

## 2026-10-18：R54/R46回分后的激进执行

- [x] 记录R46=35.88、R54=36.22，并完成包SHA、换帧、位置、V9空间混杂复盘。
- [x] 明确停止D3/R46低级RGB残差族；报告见`reports/r54_r46_official_postmortem_20261018/summary.md`。
- [x] 生成R56 `TIMELENS_EVENT_EDGE_V9_B3`：端点性结构候选，426/426、199037、8.854531425B，ZIP SHA `43845ef4085e65985c768a7c9391e895d28ed0c89f0d04ef69bc3801da96f455`。
- [ ] 完成AutoDL R55_DENSE 4fps TimeLens2全量426推理，保存原始响应SHA、环境和日志。
- [ ] 用同一V9空间父集从4fps响应生成一个独立候选，运行checker、contract、ZIP往返并绑定SHA。
- [ ] 下一提交额度优先：R47_V9_B3_CENTER85空间隔离、R55_DENSE观测密度、R56端点性；R54为最高回分锚，R26为安全回退。
- [ ] 不再提交R46/R48或扫描eta、alpha、keep；除非新观测给出结构证据。

- [x] 构建R57_FIXED_LATE2S_V9_B3固定+2秒后移竞争控制；426/426、199037、checker通过。
- [ ] 将R57与R47_V9_B3_CENTER85、R56、dense 4fps候选放入下一批提交的结构对照；不再加入未区分机制的重复包。
- [x] 完成R58 dense audio novelty 5%交换激进试验；公共proxy为负（soft -0.00592），不占正式提交名额，包仅保留为最后风险备选。


## 2026-10-18：明日五个额度前

- [x] 完成TimeLens2 4fps全量426响应并记录SHA。
- [x] 生成4fps residual/endpoint候选R65/R66及更激进R59，完成预算、checker、ZIP往返审计。
- [x] 生成真实FPS且中心表达式严格复现的联合空间分歧/稳定性候选R63/R64；后续审计确认两包作废，R59–R62旧稿同样作废。
- [x] 审计并作废R63/R64的stable-tie实现：平值几何差异按帧号注入隐藏位置偏置。
- [x] 生成R67 tie-aware修正版：center/interval保持R54 stable rank，只有geometry disagreement平均秩，gamma=0精确复现R54；作为有效负结果归档，不再作为提交候选。
- [x] 记录R67官方回分`36.16`（R54=`36.22`）；禁止上传旧R63/R64、旧R67及无epsilon R68包。
- [x] 在人工标签上严格复现音频5%交换为负，停止音频主线。
- [x] 已按R67官方回分`36.16`完成机制判别：停止无监督空间分歧路线，不根据一次回分做系数扫描。
- [x] 完成TimeLens同权重直接空间 grounding smoke 与 426 全量；`R69_TIMELENS_DIRECT_CROP_V1`通过解析率、完整覆盖、checker、contract 和 ZIP 回归，待官方上传回分。

## 2026-10-18 R70 空间候选效用读出

- [x] 完成R69直接TimeLens构图包；确认原始V8 B3 head缺失，不冒称复现；最终候选统一为`R69_TIMELENS_DIRECT_CROP_V1`，旧`CENTER_MAXWIN`变体不占名额。
- [x] 完成R70 smoke：V9 vision tensor/processor合同、129候选`2305`维特征和新head训练链通过。
- [ ] 完成RV/LIVE source-disjoint train+dev特征抽取与readout训练：远端`/root/autodl-tmp/experiments/R70_SPATIAL_UTILITY/full/`。
- [ ] 报告RV、LIVE分别的新head/center/oracle源级IoU；任一留出池不超过中心即关闭R70。
- [ ] 仅在公共门通过后固定R26 center85时间集合，重提取426 V9空间候选并制作一个空间单变量包。

## 2026-10-18 R70 completion

- [x] Repair R70 CPU/GPU starvation: GPU-batched candidate pooling/head, bounded prefetch, FP16 host transfer; controlled helper equivalence verified.
- [x] Run official R70 spatial points on 426/426 videos with 8,848/8,848 keyframes and zero fallback.
- [x] Build and validate `R70_SPATIAL_UTILITY_R54TIME_BATCH32`; 426/426, 199,037 predictions, exact R54 time set, independent checker and contract valid.
- [ ] Obtain official R70 score and bind it to the uploaded ZIP SHA; no score is inferred locally.

## 2026-10-18 R69/R71/R72

- [x] 完成 R69 分段构图 426 全量、1,281 段响应、111,306 框替换、0 fallback；生成并审计 `R69_SEGMENTED_CROP_R54TIME`。
- [x] canonical R69 ZIP 仅含正确 `predictions.jsonl`；本地副本、ZIP round-trip、independent checker、final contract 和 canonical SHA 已核对。
- [ ] 用户上传后记录 R69 平台显示参数、上传 SHA 和官方分数；在回分前不把分段响应变化写成收益。
- [x] 完成 R71 自适应预算包：195,214 条、删除 3,823 帧、8.854531425B、checker/contract 通过；记录公共 soft/binary 方向冲突。
- [ ] R71 仅保留最后高风险槽；禁止无新机制依据继续扫预算公式。
- [x] 完成 R72 原始 B3 库存审计并确认精确资产不可恢复；缓存 B3 预测保留。
- [ ] 关闭“恢复原 B3”路线；如重训新 head，另建实验 ID 并重新做 source-disjoint native crop 读出/参数合规审计，不能冒称原始 B3。
- [ ] 当前候选排序：R69 分段构图 > R70 空间 utility（待官方） > R69 全局构图 > R71 自适应预算；R54=36.22/R26=36.04 保持回退，所有新包官方分数均为空。
- [ ] 明日五额度队列按 `R54 -> R69_SEGMENTED -> R70_UTILITY -> R69_GLOBAL -> R71_ADAPTIVE` 复核上传；仅使用 canonical ZIP，不上传重复命名目录。

## 2026-10-18 R69 gate / R70 topsoft objective / joint utility

- [x] 完成 R69 run-level stability gate：固定 `max_global_delta=.12`、`max_run_spread=.12`，生成 `R69_SEGMENTED_GATE_R54TIME`，checks/ZIP 通过。
- [x] 完成 R70 MSE/pairwise/listwise/topsoft public source objective 对照；topsoft 作为空间新目标。
- [x] 生成并审计 `R70_TOPSOFT_SPATIAL_R54TIME`：426/426、199037、101797 框变化、8.855B、0 fallback。
- [x] 生成并审计 `R70_TOPSOFT_UTILITY_TEMPORAL`：固定乘法联合时间—空间排序，426/426、199037、平均 5.122 swaps/video。
- [ ] 等待以上新包官方回分；不得用公共 utility IoU 或稳定性指标替代官方 Score。
- [ ] 明日原“R69 分段”槽位优先替换为 `R69_SEGMENTED_GATE_R54TIME`；R70/R71 继续作为用户指定的 3、5 候选。
- [x] 记录 R72 additive utility residual 对照；与 topsoft 乘法联合候选分开，不混称为同一实验。

## P1 — 空间效用时间重排后续

- [x] 完成R70 max/center/marginal/output utility公共诊断与报告。
- [x] 修复R73增量utility的stable-tie位置偏置，生成R74 tie-aware包。
- [x] 生成R76 output-aligned additive与R77 output-aligned multiplicative诊断包；均未上传。
- [ ] 等待R69/R70/R71官方回分；不把公共utility代理替代官方联合分。
- [ ] 若R70官方正向，训练source-disjoint的V9实际部署框IoU head，再验证R77形式。
- [ ] 若R70官方负向且R69正向，关闭公开候选utility官方迁移路线，保留output-aligned诊断为负结果。

## P1 — R78/R81 直接部署框 utility

- [x] 完成部署框 IoU source-disjoint 目标构造和回归/排序对照。
- [x] 完成 topsoft logits + B3 output position 的 50,561 参数读出。
- [x] 生成 R81 output-aligned multiplicative package，checker/ZIP通过，未上传。
- [ ] 绑定并复核公共 B3 logits 资产来源/SHA；失败则停止R81上传。
- [ ] 等R69/R70/R71官方回分后，最多用一个额度验证R81；不再扫描损失、宽度或融合权重。
- [x] 修正R81参数清单：计入生成R70 topsoft logits的1,511,425参数，总计8.856094947B；改用nested source head。


## 2026-10-19：direct utility 后续

- [x] 完成 R78 直接监督最终部署框 IoU 的 nested source 诊断。
- [x] 核对 V9/B3 full-frame artifact SHA、参数清单和 R54 共享框身份。
- [x] 完成五头 direct utility ensemble + uncertainty penalty，生成并审计 R82。
- [ ] 不自动上传 R82；若用户保留一个高风险名额，优先用 R82 做一次官方验证。
- [ ] 若 R82 上升，做一次新的 source split 预注册 beta 固定；若下降，关闭 direct utility temporal branch。

## P0 — R83 低容量视频级政策拟合

- [x] 记录 R70=36.42、R71=36.43、R82=36.20、R69_GATE=24.95，并完成官方—本地代理有效性审计：`reports/r83_official_local_validity_20261019/`。
- [x] 关闭把公共 direct utility 排序直接迁移到官方时间选择的解释；关闭把平滑性当官方收益的解释。
- [ ] 盘点并核验公开 PHD2/GIF 训练标签、源级 split、视觉/运动特征缓存和历史 frame-rank checkpoint；禁止把官方 test 426 用作训练或调参。
- [ ] 实现 `interval policy` 小模型：全片统计 → 中心/宽度/边界置信度 → 固定 85% 合法集合；先做源级留出，不做公式扫描。
- [ ] 只有 interval policy 同时超过中心和现有 PHD2 frame-rank 的集合效用，才制作一个 R83 官方单变量包。

- [x] 完成 R83 interval policy probe：5 折源级留出、26,786 参数、CUDA；结果未超过 center，关闭受限连续区间形式，报告见 `reports/r83_interval_policy_20261019/`。
- [ ] 设计 R84 低容量空间连续坐标/候选集合读出：RetargetVid/LIVE native crop source split，固定 V9 vision tower 与 R54 时间集合；不读取官方 test。
- [ ] R84 promotion gate：RV 与 LIVE 留出都超过 R70 topsoft，且连续坐标误差改善不能只来自容易的横向样本；否则关闭该空间读出。

- [x] 完成 R84 公开 source-disjoint 坐标一致性空间 probe：LIVE +0.03135、RV +0.00365，总 source-macro +0.02547。
- [x] 完成 R84 官方 426 推理和包审计：`submissions/R84_COORD_CONSISTENCY_SPATIAL_R54TIME/`，ZIP SHA `526d3e0b41c329806c7746e0f7c1404b0ec8d3d2470495e8470c0f25983934f4`。
- [ ] 不自动上传 R84；若需要检验小模型空间路线，R70 与 R84 中只选一个高风险名额，R70 仍有更强公开依据。
# 2026-10-19 新增

- [ ] 用户决定是否上传 `R86_T5_CROPHEAD_V3_R54TIME`；上传前只复核本地 ZIP、SHA 和平台字段，不改包内预测。
- [ ] R87 AHA 路线先做单片段连续帧有限值 trace：复现原 `LiveInferForBenchmark` 的 SinkCache、dtype、processor 和 attention 配置，确认每层 head 输出有限后再扩展源级适配。
- [ ] 不把 R87 NaN pilot 登记为迁移失败，不制作 AHA 官方候选。
# 2026-10-19 更新

- P0：若有提交额度，上传并记录 `R89_LOWLEVEL_COMPOSITION_R54TIME_B3` 的真实官方回分；保留 R54/R88 回退。
- P0：有额度时先验证 R88 的 B3 点源纠错，避免把 R86 混杂实验当结论。
- P1：用轻量语义表征的差分、局部峰值和镜头边界做下一项源级留出时序诊断。
- P1：不要重复训练独立 32D 低层 bank 替代时间模型；R90 已显示它及其简单差分结构不足。
- P0：若要用一次额度检验“100M 轻量语义路线”，优先上传 R91；其余额度保留 R88/R89/R54 对照。
# P0 — R92 传统多视图时间路线（2026-10-19）

- [ ] 核对 PHD2/GIF 真实区间 union、缺失状态和源级 train/dev 清单；严禁使用 `gif_hard` 占位字段当负例。
- [ ] 实现批量多视图缓存：RGB、低频、高频、全局配准残差，保存原始 frame ID；记录 decode/transform/forward 时间、batch、GPU利用率和峰值显存。
- [ ] 在相同源级划分和预算上完成 R92-A RGB、R92-E gated fusion、R92-F 运动视图时间置换；固定 R54/R70 空间框，不接入 ROI utility 时间重排。
- [ ] 仅在 E 相对 A 的留出源集合效用达到 `+0.005` 且 bootstrap CI 下界为正、F 的增益消失时生成官方候选；否则关闭该多视图时间路线并转向目标域监督/观测路径。

- [x] R92-A/E/F 已执行；多视图未超过 RGB，运动置换未破坏分数，关闭该具体配方，不制作提交包。
- [ ] 保留 R89 空间低层候选；若重开传统路线，必须改变监督或观测粒度，不能重复四视图 gated fusion。

- [x] R93 4fps dense TimeLens2 + event-edge residual 包已生成并通过 checker。
- [ ] 用户上传 R93，记录官方回分并与 R54/R60/R71 绑定；回分前冻结 eta/prompt/端点规则。


## 2026-10-20 截止前 P0

- [x] 完成 R102 AIKU 600-group transfer pilot，记录完整资产与子集 SHA。
- [x] 完成 R103 genuine V9 token context control/actual/permuted。
- [x] 完成 R104 AViMoS visual-only saliency smoke 与 R105 native crop utility/permutation。
- [x] 固定三份可直接提交候选：R70、R71、R89；保留 R93 作为观测密度风险备选。
- [ ] 对三份候选执行最终 `independent_submission_check.py`、ZIP 往返、426覆盖、参数字段和 SHA 清单；不自动上传。
- [ ] 截止前若无新的官方证据，不继续扫 AViMoS blend、V9 attention 或 AIKU pilot 超参数。
# P0 — 2026-10-20 截止前

- [ ] 用户上传并记录 R107、R108、R106 官方 SHA 与回分；不能把公开 proxy 当官方成绩。
- [ ] 若 R107/R108 显著高于 R54/R70/R71，保留跨采样率事件拓扑；若三者均低于安全锚，关闭壳层/分歧解码，转向新的观测合同。
