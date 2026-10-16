# 近期任务

## P0 — 当前复赛（2026-10-16）

- [x] R29精确期望F推导、可复算反例和全动作穷举；区分数学结论与项目可学收益。
- [x] R29公共26TRAIN/14DEV标量校准、插件预算与同K位置；独立复算520项通过，当前配方不晋级。
- [x] R29原生1fps覆盖检查：所有400条rater序列正run均被采样命中，不能沿用12点漏采解释。
- [x] R29补完R28公共24项、空间GT候选诊断与失败图；独立复算、资源、hash及退出状态留档。
- [x] 当前官网独立核实参数≤9B、无k_size、`model_params_b`及赛期，更新01并保留旧ZIP。
- [ ] 按`configs/R29_NEXT_SUPERVISION_STAGE.json`建立可追溯native公共联合监督pilot：先试标一源实测两人耗时，冻结来源/目标/validity/分歧；没有人工通道时不生成伪人工GT。
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

- [ ] 用户上传并记录R41、R40优先；保留R26作为安全回退。
- [ ] 回分后按视频集合差异解释内容增量；不得根据五包排序继续扫参。
- [ ] 若内容候选全部失败，转音频/事件结构或恢复可复现空间部署合同。

## R42 — 人工先验诊断

- [x] 从 AutoDL `QVH_NATIVE_CANDIDATES` 固定抽取12个30秒源视频、360张1fps图和12个低码率MP4（约22MB）。
- [x] 生成盲标 Label Studio interface、local-files/file-uri任务和操作协议；任务不带类别、抽样原因或QVH query。
- [ ] 在 Label Studio 中先验收 `B6AEPicK9Bk_210.0_360.0`、`rAHQY2FHcOo_210.0_360.0`，导出并检查视频ID、source_time_sec和矩形框。
- [ ] 完成其余10个视频；6个重叠视频由第二位标注者在独立项目中完成。
- [ ] 编写导出解析与源级一致性/上下文依赖/查询错位/构图冲突分析；人工结果出来前不扩大数据集、不把标签用于提交训练。
