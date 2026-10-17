# 近期任务

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
