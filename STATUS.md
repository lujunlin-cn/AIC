# 2026-10-19 研究1 PDF 执行更新

- R94 H1 已执行：U²-Netp 真实 forward + 同容量零输入消融。LIVE dev U²−control = -0.000463，95% CI [-0.003289,+0.002383]；confirm3 +0.000834，CI 跨零。关闭具体 U² 主体增量配方，不出包。
- R95 H2 已执行：448 源 PHD2/GIF dev，显式片段 shot 结构−同容量 RGB 控制 = -0.000364，95% CI [-0.002038,+0.001271]。关闭 embedding 代理 shot 配方，不出包。
- 当前判断：研究1 PDF 的实验顺序合理，但两条低成本通路尚未解释 36 分瓶颈；下一主攻是目标域事件状态/完整上下文监督，固定 R54/R70 空间，避免继续同特征扫参。
- 复现报告：[R94](reports/r94_u2_h1_20261019/summary.md)、[R95](reports/r95_shot_structure_20261019/summary.md)。

## 2026-10-19：竞赛迁移路线审查与 R98/R99

- 已阅读 `video_competitions_research_and_agent_handoff.md` 并核对 AIM 2024、YouTube-8M 原始论文/代码入口。迁移价值收敛为目标域新监督、局部—全局交互、视觉注意力空间读出和观察预算分离；声音按 AIC 历史负迁移证据关闭。
- **R98 UNISAL**：冻结时序视觉显著性在 LIVE native crop dev 148 源上，质心/50% mass crop 相对 Qwen point 分别 `-0.03723/-0.03364`，95% CI 均低于0。关闭通用显著性中心直接替代构图点，不出包；报告见 `reports/r98_unisal_human_saliency_native_crop_20261019/summary.md`。
- **R99 local×global**：R92 RGB/PHD2 observed-selection 诊断上，真正低秩同视频上下文交互相对同容量零交互 `+0.000331`，448源 CI `[-0.000916,+0.001633]`；跨源置换后 `+0.000006`。机制方向很弱但未过 `+0.005` 门，不出包、不继续扫秩。
- 当前下一主线：先取得 AIKU 文件访问并审计真实标签；若重开上下文，必须提升完整事件观测粒度；空间上优先目标 crop 监督下的高分辨率局部读出，而不是显著性中心。

# 项目状态

## 2026-10-19：传统处理多视图路线复核（R92 已执行，未晋级）

已阅读并归档用户提供的 `传统处理接入方案.pdf`（SHA256=`04482190c7ba9d90b527175648abfdddaccc9687b26d9309d98f45055635a7ef`）。它提出的有效新机制是保留 RGB、低频结构、高频结构和全局运动补偿残差的空间形状，在轻量时序 head 中做 gated fusion；这与 R89 的标量空间候选读出、R90 的 32D 低层 bank 不同，现有负结果不能关闭该假设。

PDF 复核后的下一实验为 R92-A/E/F：冻结轻量 RGB 主干，使用 PHD2/GIF 源级目标域时间监督，固定 R54/R70 的空间框和精确预算；比较 RGB 基线、完整多视图和运动视图时间置换对照。只有留出源集合效用相对 RGB 至少 `+0.005` 且源级 bootstrap 下界为正，才进入官方候选。ROI utility 暂不接入时间排序，R82 的官方负回分仍是跨域校准警告。报告见 [R92 传统多视图路线复核](reports/r92_multiview_traditional_route_20261019/summary.md)。

**R92 实际执行已完成**：5,886 个 PHD2 片段、RGB/低频/高频/phase-correlation 运动补偿四视图、448 个源级开发集；RGB 固定85%集合F=`0.415595`，多视图=`0.415335`，配对差=`-0.000260`，95% CI=`[-0.003184,+0.002589]`。运动视图时间置换=`0.415474`，没有破坏所谓增益；低频置零反而=`0.417219`。因此未通过预注册门，不制作 R92 包，关闭该具体冻结主干+gated fusion 配方；R89 空间低层路线仍独立保留。执行记录在报告 `remote_metrics/`。

**R93 已生成**：4fps TimeLens2 dense generic intervals + 固定端点 residual，426/426、199037预测、8.854531425B，独立 checker 和 ZIP round-trip 通过。ZIP SHA=`bc05c7cbd4fa60ea2c01d9a928c55b5704bc7b83748279cf39530c33fd76a28d`；与旧 R60 4fps direct 仍有337视频不同，平均交换10.59帧。待用户上传，尚无官方分数。报告见 [R93 dense event edge](reports/r93_dense_event_edge_20261019/summary.md)。

## 2026-10-19：R89/R90 低层构图与时序结构

R89 已完成一个高风险空间候选 `R89_LOWLEVEL_COMPOSITION_R54TIME_B3`，未上传。它固定 R54 的时间集合、精确预算和 TimeLens 排序，在 B3 空间链上加入公开 RetargetVid/LIVE 源级均衡训练的参数为零低层构图先验（梯度、局部对比、相邻帧差分、清晰度和合法窗口几何）。Confirm 272 源的原生 crop IoU 源宏平均从 0.512363 提到 0.516466（+0.004104）；去掉差分后为 +0.004108，当前差分未显示增量。官方候选 426/426、199037 帧、8.854531425B，ZIP SHA=`e2830a5fb539969e9dbde5e5133728f37d9207bb23449e746990a134b0973d2e`，待上传。

R90 在 TVSum 27/16 源级划分上验证固定 top85 连续摘要重要性：静态 ResNet 512D 的源宏平均 0.256248；32D bank 0.246228；bank 加差分/短均值 0.246542；静态+时序 bank 0.256212；静态加语义 novelty/速度/加速度 0.256299（差值 +0.000051，CI跨零）。低层 bank 不能替代轻量语义表示，简单时序拼接也没有可识别收益。下一步暂停独立 bank 替代时间模型，转向更明确的事件状态与镜头结构。

R91 已生成一个真正约 98.6M 参数的独立候选 `R91_LOWPARAM_RESNET18_B3_V3`：冻结 ResNet18+TVSum 连续读出负责时间，V9/B3 全帧框负责空间；精确沿用 R54 每视频数量。426/426、199037 帧，相对 R54 替换 27768 帧，共同帧框差异0，ZIP SHA=`c8ba77bfbcd3ac46ea0bcb8578e107f2e1007a8ad31bfa8c23cc35a17385fe27`，独立 checker 通过，未上传。它是验证“100M 轻量语义是否能替代 TimeLens”的高风险包。

## 2026-10-19：R86/R87 两条路线接管结果（R86 官方回分复盘）

**R86 官方回分已报告为 34.88**，相对 R54=36.22 下降 1.34。复盘发现 R86 实际使用 `points_V9_A1raw/points`，而 R54 的空间父链是 `points_B3_vo/points`；因此 R86 同时改变了点源和 H1 映射，不能作为“B3 上 T5 迁移失败”的证据。R86 归档为混杂实验。原包仍见 `submissions/R86_T5_CROPHEAD_V3_R54TIME/`，ZIP SHA=`308275cd2a3ac3caef0aaf3ae237e3d2552bc048500cb23f12b79c3cdd8f714a`。

**R88 修正版已生成但未上传**：同一 R54 时间集合、预算、H1 系数，只把点源改为 `points_B3_vo/points`。426/426、199,037 条预测、8.854531425B，local semifinal contract 通过；ZIP SHA=`db6408cdff86649ab25913b225f61991d5e221c0af711483624131014fd71207`。这是当前唯一能识别 H1 在 B3 观察链上是否有效的最小候选。

**R87 运行时阻塞已修复，但适配证据不足**：BF16 + 原始空 token causal transition 使 trace 与 96/48 pilot 的 `finite_fraction=1.0`；32 源留出上 Ridge row AP=0.3486，source-macro AP=0.0952，不能制作 AHA 提交包。详见 `reports/r86_r87_observation_and_transfer_20261019/summary.md`。

## 2026-10-19：官方榜单揭示新的任务对齐缺口

公开复赛榜单快照返回182条记录：最高47.13，中位数36.44，≥40分42条，≥45分4条，34–38分52条。R71=36.43约在第92名，距离第40名40.14约3.71分。该分布支持“存在高价值任务策略差异”，但不能从榜单反推出官方测试集中无高光视频的比例；官方未公开空GT比例。快照与统计见[ R85榜单/迁移复核](reports/r85_leaderboard_transfer_review_20261019/summary.md)。

当前主攻调整为**PHD/GIF目标域迁移**：利用已下载的AHA作为可适配初始化，比较HL-CLIP式saliency pooling、AHA高层头和PHD/GIF源级监督；第一阶段固定R54/R70空间输出，只测时间策略。旧AHA零样本R25=35.18、InternVideo2事件完成度R43=34.95、QV跨域迁移和MrHiSum pairwise负结果，不足以否定“目标域适配”；它们只关闭了直接运行或错配监督。

更新：**2026-10-19**。当前复赛。官方回分不乘 k_size（结案），现役最佳 **R71=36.43**（用户报告；上传SHA待独立核验），其次R70=36.42、R54=36.22。R23随机探针已回分：P250 **18**、P500 **26**；两者明确低于现役包，且复赛分数与初赛分数不可混比。
**历史本地读数失效**（R21）；R24已完成新特征、源互斥、原帧标签对照，block门失败，未晋级。
完整计划：`docs/r21_direction_and_plan_20261015.md`；决定：`DECISIONS.md` 2026-10-15 两节；
实验史：`EXPERIMENTS.md` R21 节。

**R42 标注V1.3已完成（2026-10-17）**：甲项目1共300/300题，乙项目2共240/240题；合计540份正式标注、360张独立图片、12个公开源，6源/180图片重叠。结构合同零错误，但两个项目实际都由同一Label Studio账号完成，重叠只能作重复测试，不能报告人际一致性。四个V1.3重叠源高光等级重复一致率28.3%，相差不超过1级62.5%；高光2/3中25.9%明确需要前后过程或声音/语言。质量报告见[完整审计](reports/r42_annotation_v13_full_20261017/summary.md)，标准及判断流程见[标注标准V1.3](reports/annotation_protocol_v1_3.md)。

**R43 官方回分已记录（2026-10-17）**：InternVideo2 Stage2 6B 事件完成度路线得分 **34.95**，相对R26中心85+B3=`36.04`下降`1.09`。426/426、199,037条预测、精确`round(0.85*N)`预算和B3空间框合同均通过；因此下降来自时间集合排序。与R26每视频平均选帧交集88.7%，平均替换53帧，说明0.25中心秩没有保住安全基线。停止R43提示/权重扫参；下一步使用人工诊断做进行中/结果阶段、音频/上下文门控的源级判别。包：[R43 ZIP](submissions/R43_IV2_EVENT_COMPLETION_20261017/R43_IV2_EVENT_COMPLETION_20261017.zip)，SHA256=`f20b91eadd89c2e6a73fd249a0dcba64078203008f80edf5b9a8902947074cb1`。

**R44 已完成包级候选（2026-10-17）**：确认继续以R26的36.04包为父包。R44只改时间排序，保留B3空间框和精确`round(0.85*N)`预算；用540条人工标签训练的源级留一低容量局部状态+冻结IV2残差，`eta=.20`、Ridge `alpha=10`。426/426视频、199,037帧，419个视频相对R26改变时间集合，保留帧B3框0项改变；独立checker和ZIP解压回归通过。公共标签代理提升不是官方分数，候选尚未上传、官方分数null。包：[R44 ZIP](submissions/R44_HUMAN_CALIBRATED_RESIDUAL_ETA020_B3/R44_HUMAN_CALIBRATED_RESIDUAL_ETA020_B3.zip)，SHA256=`b814699f1bdc71f2fec80e8e0b9465fe600327d44eab731f41f4d75ff6deac9e`。R26仍为安全回退包。

R43与标注质量综合判断见[回分与R44路线报告](reports/r43_r42_postmortem_and_r44_plan_20261017.md)。标注540/540结构合同通过，但两项目均为同一账号；四个V1.3重叠源等级二次加权κ=0.185，不能当双人GT。R44先做源级留一的静态/局部时序/音频可观测性诊断，配置见`configs/R44_OBSERVABILITY_UTILITY_20261017.json`，尚未启动训练或新提交。

R44_DIAG_001已完成CPU留一：局部状态描述子使通用等级MAE从1.0668降至0.9695（9/12源改善），但进行中/结果和依赖目标没有改善，低级音频特征暂停。下一步仅验证低容量局部状态残差，R26仍是唯一安全上传基线。

用户修订14道Smoke后，四核心字段和媒体映射仍零错误；近零误框、非高光留框和构图说明缺项已消除，只剩34、35、38、57、59、60六题多框需合为一个完整窗口，不再阻塞扩展。[修订复查](reports/r42_smoke_revised_20261017/review.md)保留旧结果与2份草稿，不自动修改人工答案。旧60仍记录为V1.2用户修订；双人一致性主分析使用双方V1.3新标的4个重叠源，另2个旧Smoke源单列版本敏感性。

两项目公网视频播放/跳转及两种提交路径验收通过，360张JPEG本地验证通过，12源公网媒体读取通过；幂等重跑新增0题。正式标注与草稿内容哈希保持一致，未保存测试答案。Label Studio及FRP均active，公网health=200。当前未配置账号访问隔离，甲乙只进各自项目、独立完成。自由比例框不是native crop GT；本次为协议与工程验收，无新训练、独立确认或新提交包。完整证据见[扩展报告](reports/r42_annotation_v13_20261017/summary.md)。

## R39–R41 官方回分（2026-10-17）

用户报告 **R41=35.19、R40=34.15、R39=34.48**，相对 R26 中心85+B3 的36.04分别 **−0.85、−1.89、−1.56**，均不晋级。三份本地 ZIP 的 SHA、426视频完整性及包内 predictions SHA 已重新核验；平台与实际上传字节未独立核验。回分事件见 `reports/r37_content_candidates/official_score_events.json`，分析见同目录 `official_score_analysis_20261017.md`。

停止当前视觉变化/冗余/新颖性/门控配方的继续扫参；结果不能推广为所有内容或动态信息无效。R37/R38尚无已记录回分。三包198842帧，R26为199037帧，不能称与R26严格等帧数。下一步先完成12源盲标诊断，识别事件过程、声音/语言和标注语义的缺口；保留R26安全基线。

## R29 当前研究结论（2026-10-16）

报告：[reports/r29/summary.md](reports/r29/summary.md)。**冻结AHA两标量校准配方，下一主攻为有效原生监督下的源外内容增量；空间候选读出为备选。** 没有新可上传候选、独立新源确认或官方涨分。

- 公共26TRAIN/14已暴露DEV、相同85%预算：位置＋AHA头F=.294206，位置=.297467，Δ=−.003262，源级95%CI[−.008499,+.001084]，未过预定门。插件F=.302208，相对自身固定预算和同K位置的CI均跨零。独立复算520项最大F误差0；不是“完整表征无内容”的证明。
- AHA实际1fps在DEV平均133.214独帧，40源×10人所有标注正run均有采样命中；12帧漏采不能解释该系统。是否缺关键动态/未来/音频仍未知。
- R28原13条保留并补11条，24/24完成、exit0：Qwen crop IoU=.599645，中心=.579117；y轴完全相同，20/24中心输出；同候选GT辅助oracle=.868315。提示0.5示例也是竞争解释；无B3同帧控制，不能归因LFM或折算官方涨分。
- 精确期望F推导及穷举通过：固定K按E[Y·IoU/(K+M)|X]排序；期望之比可改变排序/预算，校准不保证解码最优。数学反例不是项目增益估计。

## R30 当前研究结论（2026-10-16）

R30 已执行并记录于 [reports/r30/summary.md](reports/r30/summary.md)。在同一 12 点 VPB 观测、同一 26 TRAIN/14 DEV 公共来源和精确 85% 预算下，冻结 VPB 静态读出 DEV F=.292491，位置读出=.297399，中心=.288381；相邻差分读出=.260556，静态+差分=.278525。静态相对中心 +.004109（源级95%CI跨零），动态相对中心 −.027826（CI全负），未过预设 +.005 且下界>0 门。动态 TRAIN=.304126 而 DEV=.260556，提示 768维差分读出过拟合，不能据此断言视觉动态无信息。停止当前高维动态配方；先做低维事件状态探针，未过则转音频/事件结构或恢复空间部署合同，不启动大时序训练。
- 官网10月16日独立核实：全部推理参数≤9B、Score=100F、当前字段`model_params_b`；结果10月23日20:00，材料10月25日23:59，每日5次。新包使用当前字段，旧回分ZIP保留；同版本执行evaluator仍未取得。
- 下一协议：[configs/R29_NEXT_SUPERVISION_STAGE.json](configs/R29_NEXT_SUPERVISION_STAGE.json)，状态为设计完成未执行。缺真实native双人联合标注通道及B3原头/基座，不用教师框替代真值。普通公共诊断继续可做，不把该依赖写成全项目停工。

> 本文件此前停留在 2026-09-30 初赛阶段（INTERP 48.67 / 174 视频集 / 32B 教师），
> 那些数字**已不是当前口径**：32B 系统超过当前9B参数上限，48.67 与复赛 35.52 不可比。

## R25 AHA提交包（2026-10-15）

**用户回报复赛官方分35.18，低于现役35.52，不晋级。** 分数来源为用户报告，平台和上传SHA未独立核验。
包：`submissions/R25_AHA_BF16_K085/R25_AHA_BF16_K085.zip`。
原生未量化BF16合并AHA 8,030,363,168参数＋固定B3空间450,283,377参数＝**8,480,646,545参数**，满足用户确认的9B参数上限。
1fps、完整视频因果上下文、relevance＋informativeness排序、keep=0.85整格选择；空间框复用B3。
R23是随机抽帧探针，不是AHA；18/26不代表AHA质量。现役35.52保留；R25比其低0.34，比B3全帧34.75高0.43。
旧“文件超过9GB所以AHA只能teacher”的限制撤回；文件bytes不决定资格，量化不减少参数数。
完整证据与局限：`reports/r25_aha_submit/summary.md`。
原因分析：`reports/r25_score_analysis/summary.md`。实际帧保留87.51%、视频均值88.75%，多于精确85%预算5870帧；首采样帧仅28/426保留，中1/3平均保留98.77%。优先排查位置倾向、预算与query适配；空间瓶颈待独立对照，不能据此归因容量或推算天花板。
AutoDL40206已于2026-10-16恢复；raw头分数及R26公共对照已执行，见下节。

## R26 排查完成（2026-10-16）

报告：[reports/r26/summary.md](reports/r26/summary.md)。426份原评分逐视频重建R25 mask完全一致；双头时间趋势相反（info/time rho中位+.635，rel/time −.827），未发现rel饱和/大量同分故障。
公共26TRAIN/14DEV冻结AHA：F@精确.85=**.290883**；中心位置.288381、前缀/训练位置回归.297467、随机.285110，主要差值CI均跨零，尚未证明超越这些控制。静态重复训练帧也随上下文变化；不据单源归因全部测试删头。
三包已双校验、零contract issues、均≤9B：`R26_AHA_EXACT085_B3`（199037帧）、`R26_POS_CENTER_EXACT085_B3`（同预算）、`R26_R25MASK_CENTER`（固定原204907帧，116620框改变）。位于`submissions/R26_DIAG/`；用户回报依次35.07、36.04、30.23，本地ZIP SHA重核通过，平台上传字节未独立核验。
中心85+B3以36.04晋级最佳；中心时间比等预算AHA高0.97，预算修正比原R25低0.11，固定mask的B3空间比中心框高4.95。保留B3，优先检验中心预算/位置；预留14源未打开。详细分析：`reports/r26/official_analysis.md`。

## Current Best

| 口径 | 包 | 分 | 说明 |
|---|---|---|---|
| 现役最佳记录 | `R71_ADAPTIVE_BUDGET_R54_V9` | **36.43** | 用户报告，8.855B参数；上传SHA待核 |
| 次优记录 | `R70_TOPSOFT_SPATIAL_R54TIME` | **36.42** | 用户报告，8.855B参数；上传SHA待核 |
| 历史最佳保留 | `VPB_GIFENS_A025 / pkgB` | 35.52 | 上传SHA未独立核实 |
| R26精确AHA | `R26_AHA_EXACT085_B3` | 35.07 | 同预算同B3，低于中心0.97 |
| R26中心空间 | `R26_R25MASK_CENTER` | 30.23 | 同R25 mask，低于B3空间4.95 |
| keep 扫描最低 | KF050 | 32.57 | p=0.5451；单压keep未改善该包，历史lift反解不作依据 |
| 全帧锚点 | v8b3 | **34.75** | p=1.0；联合指标控制，不能精确反解统一密度 |
| R25 AHA | R25_AHA_BF16_K085 | **35.18** | 用户回报；实际p=0.875076，不晋级 |
| R23 随机探针 | P250 / P500 | **18 / 26** | 官方复赛回分；用于校准探针链，不作为模型候选 |
| 材料合规 | — | **未闭环** | 新最佳B3完整部署材料待核；历史VPB包说明/参数材料亦未闭环 |

历史初赛 48.67 为 32B 教师系统、另一 174 视频集，**超过复赛9B参数上限，不可当复赛合规最佳**。

## 官方轴（2026-10-15 更正）

R23 复赛 P250=18 / P500=26 为用户报告，保留原始精度，未独立登录平台核验。
不将初赛读数用作复赛模型比较或收益估计。

历史单一密度 d 公式不能作为异质视频宏平均下的精确反解。
随机 rho=p 是期望关系，有限样本不是恒等式；两个官方总分不能唯一识别逐视频密度。
因此撤回“天花板39.54”“现役已经达到89.8%”“随机探针钉死d”的研究约束。
历史VPB包 p=0.8534 与本机复算0.839126的差异仍是溯源问题，不能靠这两个分数消除。

## 熔断/阻断（做任何实验前）

1. ~~1979/2508 MrHiSum `gif_hard` 全零，构造器未恢复~~ → **已裁定**（阶段 0.3）：
   `r14_assemble_sft_v2.py:107` 是字面 `gif_hard = [0] * 12`，**占位符，不是标签**。
   标 `label_status=placeholder`，不当真负、不删行；`gif_soft` 完好不受影响。
   全池全零 **79.07%**（0.9%=仅PHD2 / 87%=holdout125 各自分母不同，不矛盾）。
2. 源身份未绑定：2508 行 = **1329** 规范源；legacy hold125 有 98 行同源。
3. **缓存合同破裂 —— 机制已定位（R22c）**：`train_vp_head.py:56` 有
   `assert len(vids)==len(rows)`，但 `e04_compare.py:118-124` / `resid_e03.py:85-93`
   只有 `concat → argsort(order)`，**无断言无截断**，把 2700 行缓存喂给索引域 <2508 的
   `hold_idx` ⇒ **静默行号错配**。禁令：禁止跨池复用行号；评估路径必须自带长度断言。
4. **VPB 头身份不一致 —— 已恢复（R22c）**：真身是 `/255.0` + `(x−mu)/sd` + **GELU**
   （`train_vp_head.py:86/106`、`extract_features.py:65`）；四处副本改成 ReLU 且漏归一化。
   按真身复算已**精确重现** ckpt 记录的 `0.5696324319689352`（Δ=+0.000e+00）。
5. **抽特征不可复现（R22c 新）**：`vpb_artifacts/feat` 盘上缓存**重算不出来**
   （未被改写过的 `shard_0` 缺口一样大，单行最高 0.1298 ≈1.45σ；fp16/fp32 权重实测相同）。
   且 `shard_3of4` 被追加过 192 行重复 vid。
   **处置：阶段 2 起冻结 VPB 特征必须现算，禁止复用该缓存**，并记录当次权重 sha256。
6. 确认池耗尽：LIVE 1800 源全 exposure；227 reserve ≡ 旧 R11 confirmation。

## 已作废的本地读数

`0.3340`（内容上限）· `0.7404`（CLIP>VPB）· `−0.0994`（均匀窗失败）· `0.7085`（shot 优势）
· 残差 B（"纯内容"）· LFM 排除。理由见 `EXPERIMENTS.md` R21 节。

## 仍然有效的本地读数

R16_EXEC 的 T1/T1S 8 条臂（源级互斥：121/40/42，三向交集全 0）。
但 **8 条臂全部落在控制带 [0.2404, 0.2470] 的 ±0.008 内**，无一对预定控制稳定占优
⇒ E1.1 结论 = **"本协议下可学信号弱" ≠ "任务无内容信号"**。

## Current Bottleneck

**当前阻塞是目标语义可追溯的源外联合效用证据与B3部署闭环。**
R26回分确认中心时间优于当前AHA排序，修正预算未提升，B3优于中心空间但尚非空间上限。R27中心75=35.87、95=35.39，均低于中心85=36.04；停止预算/位置扫描，前后移包暂停。R29简单校准未过源外门，原生1fps并未完全漏标注正run；优先有效监督和可观测状态差异。R28缓存全部426视频已精确回放；R29公共坐标pilot补完但没有B3同帧控制，原头/基座缺失仍影响完整部署复现。
过去 12 轮"换监督/换 backbone/换损失都无迁移"的读数，
无法与泄漏、错配、指标混比区分开。

## Running / Next

| 阶段 | 内容 | 成本 | 阻塞 |
|---|---|---|---|
| 0 | 全部 **闭**：额度=每日5次 · manifest sha（虚惊）· gif_hard（占位符）· 35.52 包审计完成 | 0 GPU，~1h | — |
| 1 | 全部 **闭（4/4）**：64 源（1.395 GB）· **54 过轴门 / 10 隔离** · 源级绑定 + 双 sha · **缓存重绑 `aligned=true` 冻结 2508 行** · **头身份恢复并精确重现 0.5696324319689352** | CPU 有界，280 s | — |
| 2 | R24完成：VPB block−point均值−.002058，两seed均负，**门失败** | 26train/14dev，新特征 | 停止当前12帧配方 |
| 3 | R24完成：VPB block均值.295601 / CLIP block .282120 | 同头/同初始化规则 | 不支持换CLIP，CI跨零 |
| 4 | R26中心85=36.04晋级；AHA35.07、中心空间30.23 | 官方复赛回分 | 上传SHA未独立核验 |
| 4b | R27中心75=35.87、95=35.39；前后移暂停 | 0训练 | 低于36.04，停止扫描 |
| 4c | R28空间缓存426视频精确回放；R29补完24条公共pilot、独立IoU复算 | GPU27.29秒＋CPU | B3头/基座记录路径缺失，无同帧比较 |
| 4d | R29标量校准/预算对照、原生覆盖与精确F理论验证已完成 | CPU小实验；无训练 | 当前配方未过门；下一步有效监督/源外增量 |
| 5 | 空间：LFM vs DINO | — | **需新 native x/y 人工 GT** |

**随机排序探针 —— 已回分 P250=18 / P500=26（用户报告，2026-10-15）**
远端 `/root/autodl-tmp/experiments/R23_RANDPROBE/`，两个已验证 arm：

| arm | p 实测 | n_pred | zip sha256（前16） | 声明尺寸 |
|---|---|---|---|---|
| `R23_RANDPROBE_P250` | 0.24996384 | 58536 | `51910acc96f14f70` | 整字段省略 |
| `R23_RANDPROBE_P500` | 0.4999734696 | 117077 | `98862a8f78526209` | 整字段省略 |

R23 包已由用户上传并回分。完整性验证记录：426/426 视频，P250/P500 分别
58536/117077 个预测框，框沿用 B3。声明尺寸字段省略的构建事实不构成复赛合规证明；
最终作品必须遵守 `01` 的复赛必填要求。
此前双点反解唯一密度与锚点校准的解释已撤回，保留在历史记录中供追溯。

阶段 0/1 结论与证据：`reports/r22/r22_instrument_report.json`。
阶段 2/3 当前执行 R24，重新提取特征，不复用旧缓存。

## Failed Ideas（勿重开）

历史配方未见稳定提升（不等于监督源维度已穷尽；污染本地读数不能排除新数据）·
空间缩放/居中/RV-rescore 均负迁移 · 本地门不可构造（5 包 4 次独立证伪，绝对禁令）·
回退型自适应（熵 0.874–0.912，|mono_rho| 8/377，n_eff 0/377）· 均匀切段池独立可用性。

## 资源

当前AutoDL `connect.westd.seetacloud.com:40188`（用户新端点），单卡 **RTX 4080 32760MiB**，数据盘`/root/autodl-tmp`。使用旧主机同一pinned key严格验证；旧40206仅历史。R29结束复查：GPU1MiB、0%、无计算进程。单次训练严格<12h；本轮无正式训练/新付费资源。

## R24 结果与下一步

证据：[reports/r24/summary.md](reports/r24/summary.md)。8臂×3keep独立重算通过，预留源未打开。
VPB point均值F@.85=.297659，位置控制=.297467；尚无稳定高光排序优势。
该轮后续已由R29完成12/24/48和实际1fps诊断，见上；不再把12点漏采泛化为当前AHA1fps病因。不按开发结果换seed或阈值。
AHA依赖已补齐，BF16真实视觉smoke通过；FP16非有限输出失败已保留。尚无评分质量或蒸馏收益依据。上述均为公开摘要时间指标，不是复赛分数。

AHA修复证据：[reports/r24_aha_repair/summary.md](reports/r24_aha_repair/summary.md)。三个评分头精确匹配适配器；不同帧响应差2.05078125，重复帧差0。上述是旧smoke记录；原按字节否决的判断已撤回。R25未量化合并AHA＋B3空间共8.481B参数，已完成直接提交包，详见本文件R25节。

## R27 基于新最佳的后续实验

中心75/95已回分35.87/35.39，分别低于85%的36.04达0.17/0.65。停止该扫描，前后移包暂停。转固定中心85%下的空间主体评分、表示和原生几何数据覆盖。R28核验426视频234159帧缓存回放误差0，但B3原头/基座记录路径缺失；见`reports/r28/summary.md`。

## R28 空间方向（本轮）

停止R27预算/位置扫描。完整426视频B3缓存路径误差0；记录路径的B3头和LFM基座缺失，旧RV原视频符号链接也已失效，缓存存在不等于数据/权重完整。空间方向优先主体评分、表示与原生几何数据，避免重复旧插值/加密/检测先验。
Qwen3.5-4B公共4源24预测pilot已加载并产出部分结果，随后SSH断开，完整metrics未取回，**不宣称收益或完成**。恢复后先取回结果，不直接推理官方集。见`reports/r28/summary.md`、`pilot_status.json`；预留14源未打开。

## R31/R32 交接（2026-10-16）

R31低维事件状态probe：DEV `.302285` 高于中心 `.288381`，但暴露DEV、源级CI跨零，且没有AIC对应特征，不作为部署收益。R32使用AIC 426冻结 `temporal_k0.90` 时间选择与B3空间框：426/426视频，207292帧，实际保留率 `.885262`，总推理参数 `450,529,778 = 0.450529778B`。`submissions/R32_VT_K090_B3/` 已含 `predictions.jsonl`、确定性ZIP、manifest和SHA256；独立checker与semifinal contract均通过。R32尚未上传、官方分数为null；用户上传时保留 `R26_POS_CENTER_EXACT085_B3` 回滚。ZIP SHA：`9d6f148b4e42c48ff8553edd052f226503d4d304a1c046a3a496dc16aab3a434`。

## 明日五提交候选（2026-10-16）

已准备五个可上传包：`R35_VT_UNION90_B3`（两头并集，210827帧，90.036%）、`R36_VT_CONSENSUS85_B3`（共识优先+中心B3补齐，199037帧，85.001%）、`R34_VT_EXACTDP_B3`（183584帧，78.401%）、`R33_VT_REPLAY_B3`（183505帧，78.368%）、`R32_VT_K090_B3`（207292帧，88.526%）。全部426/426覆盖，独立checker和semifinal contract通过；新包尚无官方分。建议上传顺序：R35 → R36 → R34 → R33 → R32。每个包总参数在0.450530B或0.450776B，均满足9B规则。

## R42 人工先验诊断批次（2026-10-16）

已从 AutoDL 现有 `QVH_NATIVE_CANDIDATES` 固定抽取 12 个公开源视频的 30 秒窗口，导出 360 张 1fps JPEG 和 12 个低码率 MP4，约 25MB，已放入 `artifacts/annotation_batch_v1/`。6 个视频预留双人重叠，其余 6 个单人；不下载整套 QVH、TVSum、SumMe、VideoXum 或 LIVE-YT 到电脑。

Label Studio 导入任务已改为盲标版本：任务只含图像、上下文视频、视频 ID 和时间，不含类别、抽样原因或 QVH query；query 与原标签仍只在独立 manifest 中，供标注结束后比较。先标 `B6AEPicK9Bk_210.0_360.0` 和 `rAHQY2FHcOo_210.0_360.0` 验收导出，再完成其余 10 个视频。此批次只用于区分目标语义、过程/音频依赖与时间—空间冲突，不能当官方 GT 或直接训练 AIC 提交模型。

当前状态：素材和任务已生成、SHA 已核对；尚无人工结果，因此尚无新的模型或官方分数结论。结果出来前不扩大数据集，也不开展无机制的提交包扫描。

**标注标准 v1.1 已冻结**：先完整播放带声音的 30 秒片段建立事件时间线，再逐帧判断；`0/1/2/3/不确定` 按同一视频内部的保留价值裁决，过程/声音依赖单独记录。由于本批次没有 `targetRatioWH`，`crop_box` 明确是自由比例构图可行性诊断，不是 native crop GT；只有高光 2/3 才画框，没有稳定窗口可留空。Label Studio 核心字段已设为必填，首轮 `query_relevance` 只允许“未看查询”。协议、XML和manifest的版本字段已同步为 v1.1。

**Label Studio 部署已验收（2026-10-16）**：用户级 `label-studio.service` 已 enabled/active，监听 `127.0.0.1:18081`；系统级 `frpc.service` 已 enabled/active，将其转发到 `http://47.108.220.174:3002`。公网 `/health`、登录、项目 API 和 Smoke 图片访问均已通过。项目 `AIC Manual Highlight Smoke v1`（ID 1）部署时导入60个未标注任务，2026-10-17已完成60份正式标注；质量检查见本文件顶部。先修订并验收Smoke，再按video_id+time_sec去重导入其余300任务。

**视频入口补齐（2026-10-17）**：在现有60任务中原位补充`context_video`，图片上方显示完整30秒有声标注窗口，任务ID和已有标注/草稿保留。完整360任务生成器同步支持视频路径。数据库备份置于本机Label Studio私有目录；验收结果见`reports/annotation_video_player_20261017.json`。操作：打开项目任务，先在顶部播放器看完并打开声音，再按图片时间顺序标注；换video_id后再完整看新窗口。原始长视频全片未加入本批次。


## 2026-10-17：R44 官方回分与当前判断

用户报告 R44_HUMAN_CALIBRATED_RESIDUAL_ETA020_B3 官方复赛分为 **36.09**，父包 R26_POS_CENTER_EXACT085_B3 为 **36.04**，差值 **+0.05**。R44 的 ZIP SHA256 为 `b814699f1bdc71f2fec80e8e0b9465fe600327d44eab731f41f4d75ff6deac9e`；平台上传 SHA 尚未独立核验。R44 426/426 视频、199037 帧、精确 round(.85*N)，B3 框逐帧继承。

这只能记为**弱实打实信号，尚未确认稳定收益**：R44 相对 R26 仅替换 6352/199037（3.19%）帧，随机交换 null 在公共人工视图上得到 `P(ΔScore≥+0.05)=0.2369`，且该 null 不是官方 GT 检验。历史同协议官方分数的 0.05 级波动也与该差值同量级。R26 仍是安全回退，R44 不是已证明突破。

下一条机制候选固定为 `R45_BOUNDARY_GUARDED_RESIDUAL_B3`：只保留 R44 在 R26 中心窗口两端各 5% 边界带内的局部换帧，冻结中间区域；不再扫描 eta/alpha。若仍未形成至少有解释力的增益，停止人工残差微扰，转向同域空间读出或真实音频/事件结构。

## 2026-10-18：转入高频边界残差主攻

R46空间审计确认：同一129候选池上 B3 IoU 约0.54，而候选oracle约0.77；候选集合不是空间瓶颈。但精确V8 B3 head/base在远端库存缺失，67个`.pt`候选中只有V7/V9近似head，不能冒充B3或生成空间替换包，空间utility readout暂缓。

时间侧完成4fps源级留一诊断。D1静态MAE=1.07136，D2 R44 1fps局部状态=0.98537，D3加入子秒状态=0.99133，D4再加冻结IV2事件锚=0.96739；D3单独总体回归略退，但固定85%集合代理增量为+0.02283（11/18视图），D4为+0.01306（10/18）。这说明子秒变化更像边界决策信号，IV2更像校准信号，不能把两者简单拼接后宣称收益。5%边界规则公共代理为0.706636对中心0.703889，只有3/12源为正，作为风险控制而非收益证据。

语义×状态交互未改变集合且回归略差，已停止。D3、D4完整426视频候选正在AutoDL上生成；完成后各派生一个固定R26边界包，保持B3框和精确85%预算。官方回分仍为空，R26=36.04安全回退，R44=36.09为最高已报告包。

**R47 TimeLens2-8B 研究支线（2026-10-18）**：12个公开源的固定prompt区间输出已完成，18视图代理 `0.7044166→0.7213273`（`+0.0169107`，6/18正），说明直接事件区间建模值得保留。服务器真实视觉smoke通过；原官方失败是视频路径错误。随后核实四个safetensor分片精确参数 **8,767,123,696**，与B3空间450,283,377合计 **9,217,407,073>9B**，不能和B3组成合规提交；全426推理已停止，未生成R47包。TimeLens2若继续只能走合规学生蒸馏或无参数空间基线，不能把离线教师与B3偷偷合并。

### R47 完整候选更正（2026-10-18）

上段的“停止全426推理”针对完整450M B3组合，现已被合规V9空间审计取代。远端已有完整的V9 vision-only B3父集：SigLIP2 vision tower 85,843,200 + B3 head 1,511,425 + YuNet 53,104 = 87,407,729参数；与TimeLens2合计 **8,854,531,425**，低于9B。TimeLens2分片0–8已覆盖426/426，精确85%转换候选已完成：

- `R47_TIMELENS2_V9_B3`：V9 vision-only B3空间 + TimeLens2时间，8.8545B，426/426、199037帧；这是明天最高风险、最高信息价值候选。资格条件和完整组件清单见包内manifest及`reports/r47_timelens2_visiononly_compliance_20261017.md`。
- `R47_V9_B3_CENTER85`：V9 vision-only B3空间 + R26中心时间，87.4M；作为空间替换对照。
- `R47_TIMELENS2_CENTER_EXACT085`：TimeLens2时间 + 确定性中心框，8.7671B；作为严格参数合规的空间降级对照。

上述包均未上传，官方分数为空。TimeLens2+完整LFM B3仍明确禁止使用，不能用缓存结果规避9B计算。
### R48 时间残差共识门控（2026-10-18）

D3的4fps残差回归MAE略差于D2，因此新增固定可靠性门控：仅当1fps和4fps读出位于同侧中位数时才允许改动R26中心排序。公共18视图代理为`+0.02360`（9/18正），略高于D3的`+0.02283`；这仍是同一12源人工诊断，不能当官方收益。426视频包正在生成，保持R26 B3框、精确85%预算和0.450283377B参数口径。

### 2026-10-18 时间残差扩展判别结果

今天没有提交额度，转做结构判别而非包名扫描。三个固定机制均未超过D3：

- R49事件阶段读出（铺垫/进行中/结果/反应/无事件）代理`+0.00291`，2/12源为正；阶段标签不能从当前低维时间观测稳定恢复。
- R50全片轨迹上下文（起止/中位状态、累计变化、局部运动密度）完整臂`+0.02001`，去掉原始归一化位置后`+0.02211`，仍低于统一聚合的D3 `+0.02302`。它没有形成新提交候选，说明收益主要来自D3局部状态而非更复杂的全局位置形状。
- R51固定“前后双侧持续变化”门控将D3降至`+0.00654`，反证了高光不能被定义为双侧持续动作；单侧启动/结果瞬间可能正是有效帧。
- R52 Farneback光流统计臂仅`+0.01093`，二值代理`+0.00119`，低于灰度局部状态；当前瓶颈不是简单光流观测。

结论：时间残差暂时保留D3/R46作为唯一有可重复诊断信号的时间候选；停止阶段、全局轨迹、持续性门控和光流微扰。下一优先级转向已完成的TimeLens2事件区间＋合规V9 vision-only空间候选，以及空间组件的独立对照。R48只作为一个“可靠性门控”验证名额，不把公共代理差异当官方涨分。

R48全量输出已完成并通过审计：426/426、199037条、每视频`round(.85*N)`，相对R26有415个视频发生集合变化，平均每视频交换12.65帧；B3框逐项继承，独立checker与ZIP往返通过。候选：`submissions/r48_candidates_20261018/R48_CONSENSUS_GATED_RESIDUAL_B3/`，ZIP SHA `d3efa721eb7c2c5ad4d4fbb2000e7e79b78d667aecfe3355b16dd236ad0aa1d9`；固定边界派生包 ZIP SHA `90d6769e771489fc632fef3dbdc41ed306869f4d76d40a361b80924f00f15bb8`。两包均未上传、官方分数为空。

### 明日五包结构审计（2026-10-18）

已冻结五个互补候选，全部通过独立checker：R26安全锚、R44已知最高回分、R46 D3高频局部残差、R47 TimeLens2+V9时间空间联合、R47 V9空间替换的R26中心时间对照。每包均426/426、199037条、精确预算、ZIP往返通过，未上传。精确路径、参数、预测SHA/ZIP SHA及排除R48的理由见`reports/next_day_five_candidate_audit_20261018.json`。

### R53/R54 输入扩展与 TimeLens2 中心残差（2026-10-18）

R53低维音频统计没有超过D3：audio-only公共代理`+0.00749`，RGB+audio`+0.01162`；固定画质交互`+0.01498`低于D2`+0.02235`；从RGB预测事件阶段后仅比D2增加`+0.00238`且CI跨零。停止这些微扩展，不把失败外推为“音频语义无效”。

R54把TimeLens2区间从硬替换改为中心锚定`eta=.20`残差。公共12源18视图中心`.704417`→R47硬区间`.721327`→R54`.729414`，R54相对中心`+.024997`、10/18视图为正。已生成并审计`submissions/R54_TIMELENS_CENTER_RESIDUAL_V9_B3/`：426/426、199037帧、参数`8.854531425B`，independent checker和semifinal contract均通过；ZIP SHA=`1e0f92f2292da0c057aa9b45a27134783af4def37a60c95ffde6d8fa9a8ae210`。V9 artifact资格条件与R47相同，官方分数为空。

如果只试一至两包，当前建议优先 **R46 D3局部残差** 与 **R54 TimeLens2中心残差**；R26保持安全锚，R44保留为已知36.09对照。R54不是官方涨分承诺，上传后必须绑定平台显示参数和SHA。

## 2026-10-18：R54/R46 官方回分复盘与激进主线

用户回报 **R54=36.22、R46=35.88**。相对 R26=36.04，R54 为 `+0.18`，R46 为 `-0.16`；R44=36.09 仍作为弱正对照。R46 的预测 SHA 为 `ebdf0f18...39004dc`、ZIP SHA 为 `be52cfd4...4e1b3a`；R54 的预测 SHA 为 `62371bc5...d033983e`、ZIP SHA 为 `1e0f92f2...8ae210`。平台上传 SHA 尚未独立核验。

R46 只替换时间集合，B3 框逐项继承 R26，换帧 5,636、424/426 视频，故其官方负分直接削弱 4fps 灰度局部状态残差的跨域迁移假设。停止 D3/R46 同类微扰、eta/边界扫描和把公共人工标签代理当官方收益。

R54 与 R26 换帧 6,152、417/426 视频，且 V9 空间框在共享选中帧中改变 28,730/199,037（153 个视频），所以 `+0.18` 是 TimeLens2 时间、V9 空间及交互的联合证据。R54 的选择在 425/426 个视频近似连续 85% 窗口，中心位置均值 `.49957→.52344`，区间覆盖仅 `.75695→.76531`；删除主要在早端、增加主要在晚端。当前最强解释是 TimeLens2 估计了可迁移的视频级后段相位，而不是散乱逐帧高光。

完整复盘见 [`reports/r54_r46_official_postmortem_20261018/summary.md`](reports/r54_r46_official_postmortem_20261018/summary.md)，逐帧差异见 [`reports/r54_timelens_center_residual_20261018/package_diff_20261018.md`](reports/r54_timelens_center_residual_20261018/package_diff_20261018.md)。

### 激进路线正在执行

AutoDL 新启动 **R55 dense TimeLens2**：同一 8.767B TimeLens2、同一 generic prompt、同一 V9 vision-only 空间组件，输入由 2fps 提高到 4fps，固定 `320×320` 上限和确定性解码。10 视频 smoke 已显示区间发生结构性变化（例如视频 0 从 `[[3,10]]` 变为 `[[0,5],[7,9]]`），全量 426 视频推理在 `/root/autodl-tmp/r55_dense_full/` 后台运行。该路线测试事件边界观测缺失，完成后生成独立高风险上传包。

已完成两个不扫系数的解码控制：`R55_TIMELENS_EVENT_PHASE_V9_B3`（显式完成相位，主要用于机制对照）和 `R56_TIMELENS_EVENT_EDGE_V9_B3`（区间端点性）。R56 公共 18 视图 proxy 比 R54 高约 `.00170`，但仍无官方确认；包为 426/426、199037、8.854531425B，ZIP SHA=`43845ef4085e65985c768a7c9391e895d28ed0c89f0d04ef69bc3801da96f455`。

下一批额度的结构优先级：

1. R47_V9_B3_CENTER85：隔离 V9 空间贡献；
2. dense 4fps TimeLens2 + V9：测试观测密度；
3. R56 endpoint residual：测试事件端点先验；
4. R54 保留为当前最高回分锚；
5. R26 保留安全回退。

不再把 R46、R48 或人工低级残差路线列为突破候选。

R57 已完成固定后移控制：每视频精确中心85%窗口统一后移2.0秒，使用V9 vision-only空间，426/426、199037、参数0.087407729B，独立checker通过。预测 SHA=`c1fc69324a7dafa3c2faca2de80af24438c20df466b3dfb5cf9815eaacae84c9`，ZIP SHA=`eac6c8610a43f6bb20b536018984febc683ac97ebf0ea3b986acb50f681f7fce`。公共18视图 proxy 高于R54，但这是竞争解释控制，不是官方涨分承诺。报告见`reports/r57_fixed_late_window_20261018/summary.md`。

### R58 音频瞬态正交路线（2026-10-18）

R58在R54 TimeLens集合上按固定5%交换引入无参数dense audio novelty；同一12源18视图人工标签的严格公共proxy为soft `-0.00592`、binary `-0.00585`（2/18视图正），因此该路线没有进入明日高优先级提交。完整426包仍保留在`submissions/R58_TIMELENS_AUDIO_NOVELTY_V9_B3/`，独立checker通过，386/426视频有音频、共9515帧变化；新增帧复用R26 B3全帧框，存在小空间artifact混杂。复现脚本为`scripts/r58_extract_test_audio.py`、`scripts/r58_public_audio_probe.py`和`scripts/r58_audio_novelty_package.py`，报告为`reports/r58_public_audio_probe_20261018.json`。


## 2026-10-18：超级激进候选完成

R54仍是已核实最高官方分（36.22）；R26为36.04安全锚。R46=35.88，低级RGB残差停止。

已完成并通过独立checker/ZIP往返的高风险候选：

- `R59_TIMELENS4FPS_MASK_EDGE_ENSEMBLE_V9_B3`：TimeLens2 4fps密集观测，响应SHA `bf712c77d94cdfd41d07e1d9aa0ec2719bd33328787df43662d96d6a64dc1906`，ZIP SHA `ef46457ab64b9a6d2a160cf45ba60ec75f2139fe2f74209470cc4da162cdad76`，8.855B。
- `R63_JOINT_V9_DISAGREEMENT_G1`：历史包，**作废、不可上传**（stable ordinal rank 对平值几何差异注入帧号位置偏置）。
- `R64_JOINT_V9_STABILITY_GM1`：历史包，**作废、不可上传**（与 R63 相同的 stable-tie 实现缺陷）。
- `R65_TIMELENS4FPS_RESIDUAL_V9_B3`：4fps二值区间残差，ZIP SHA `70b41e0493a67e614ce0233e62e2c1040e2279f4fabd540f5be7b788535129a5`。
- `R66_TIMELENS4FPS_MASK_EDGE_V9_B3`：4fps端点集成，ZIP SHA `369fe63a56bea9dc8bdce0a12c72e8a090577a2c18c19272a8e09a8bc125dc28`。

R59/R60/R61/R62旧稿因FPS索引/浮点边界问题全部标记作废，不得上传。R58音频候选公共严格复现为负（soft -0.00592），只保留为最后风险备份。

**当前瓶颈判断**：R54 的正效应仍只能归因于 TimeLens2 时间、V9 空间和交互的联合变化；R57/R59 已反驳全局后移与未校准的 4fps 端点替换，R67=36.16 又削弱了无监督空间分歧作为时间信号的解释。R63/R64 因 stable-tie 实现缺陷作废。当前唯一结构不同的高风险候选是 R69_DIRECT_CROP_V1；其官方分数尚未回传，不能把结构差异写成收益。

### 2026-10-18：R57/R59复盘与最后高风险槽

R57 官方回分 `34.76`（相对 R54 `-1.46`）反驳统一后移窗口；R59 官方回分 `35.04`（相对 R54 `-1.18`）说明 4fps 实验同时改变了 TimeLens 输入分辨率和上下文，不能解释为单纯增加观测。两者都不支持继续做全局时间平移或未校准端点替换。

审计发现原 R63/R64 的几何分歧项使用 stable ordinal rank，约 90% 的几何差异为平值，却按帧号分配不同秩，混入隐藏位置先验；两包因此作废。修正版 `R67_JOINT_V9_DISAGREEMENT_TIEAVG_G1` 保留 R54 的 center/interval stable rank，只对几何分歧使用平均秩，并将 `<=1e-6` 的数值噪声归零。`gamma=0` 逐视频逐帧复现 R54；`gamma=1` 相对 R54 仅交换 2427 帧、81 个视频，位置均值 `.519774`。官方回分为 **36.16**（R54 的 -0.06），因此该无监督空间分歧信号停止作为候选；R67 只保留为有效负结果，R63/R64 不上传。

### 2026-10-18 R57/R59 与最后名额

R57=`34.76` 和 R59=`35.04` 均低于 R54=`36.22`；R67 tie-aware 修正版=`36.16`，进一步说明无标签 V9/R26 空间分歧不是可靠的高光价值信号。R63/R64 由于 geometry stable-rank 平值位置偏置作废，旧 R68 变体也禁止上传。

最后一个名额切换到结构不同的 R69：TimeLens2 完整视频直接输出目标比例构图框，固定继承 R54 时间集合和精确 85% 预算。最终候选为 `submissions/R69_TIMELENS_DIRECT_CROP_V1/`，不是旧的中心投影变体：426/426 视频、199037 条预测、8,767,123,696 参数（8.767B）、0 fallback、0 帧集合差异，替换 129443 个选中帧框；独立 checker、contract 和 ZIP 往返均通过。预测 SHA=`aad8d3455490c2d3d43d42bf98c93117b405993452f985fb840126109452dfcf`，ZIP SHA=`f5d6dcadbd511f286c98360a72cfb568146056eb77d8cee6d62194b9e7839871`。R69 官方分数待用户上传后回传；在此之前不能预报收益。R54/R26 保持回退，R63/R64/R68 及旧 R69_CENTER_MAXWIN 变体不占提交名额。

### 2026-10-18：R70 空间候选效用读出启动

R69“TimeLens2直接构图框”与R70严格分开。R70针对另一条假设：在固定V9 vision-only塔和129合法窗口下，重新训练候选效用读出，检验空间信息是否已存在于视觉token而只在读出层丢失。原始V8 B3 `head_s.pt`仍缺失，因此R70 head不称B3复现。

R70 smoke已通过视觉197 tensor覆盖、processor合同和`[129,2305]`特征/训练链；完整RV/LIVE source-disjoint public native crop抽取已迁移到远端`/root/autodl-tmp/experiments/R70_SPATIAL_UTILITY/full/`运行。只有RV与LIVE两个留出池都超过中心，才考虑R26 center85时间集合上的426空间单变量包。

### R70 official spatial utility package (2026-10-18)

R70 public native-crop utility readout was run on the official 426 videos. The retained public source-level metrics are RV dev +0.06602 and LIVE dev +0.07225 over center (20 and 148 held-out sources); the official run itself has 426/426 videos, 8,848/8,848 keyframes and 0 fallback. These are public readout diagnostics, not official score evidence. The initial CPU-bound implementation was repaired: candidate pooling and head inference are batched on GPU, one bounded CPU prefetch batch overlaps preprocessing, and FP16 transfer is used. The 10-video profile took 2.7s; full official inference completed in 174.3s with one process, batch 32, prefetch worker 1. GPU samples reached 97–100% during active batches.

Output had 426/426 videos, 8,848/8,848 keyframes, 0 fallbacks, and all 8,848 statuses `ok_r70`. The package `submissions/R70_SPATIAL_UTILITY_R54TIME_BATCH32/` keeps the R54 time set and replaces only spatial boxes on 102,981 retained frames. It has 199,037 predictions, 8.854531425B declared parameters, prediction SHA `5d34388eb39a2d8f49ba9cd29420cbb5c2e450f2c88ec2aee16be11cbeeff3da`, ZIP SHA `a5d3031cb6ab896d73e83e032be0e385bdb17efa3c0024af06036dcc63179b7a`, and passes local contract, independent checker, and ZIP round-trip. It is not uploaded and has no official score yet. See `reports/r70_official_package_20261018/README.md`.

### 2026-10-18：R69 分段构图、R71 逐视频预算与 R72 B3 资产审计

**R69_SEGMENTED_CROP_R54TIME**：在固定 R54 时间集合、精确 85% 预算下，把每个连续选中 run 按原始 frame 轴切成 3 段，每段用 5% 选中长度做上下文，TimeLens2-8B 逐段请求 16:9 构图。426/426 视频、1,281 段响应全部有效、0 fallback；199,037 条预测中 111,306 个空间框替换。相对全视频单 crop R69，102,617 个选中帧框不同，277/426 视频变化，188 个视频拥有多个不同 segment crop。总参数 8,767,123,696（8.767B）。包 `submissions/R69_SEGMENTED_CROP_R54TIME/`，ZIP SHA `0b1de985c709b2b4d31d2c1a57b3d0420a3f66afb9dbad82605ff534e10a8010`；本地 checker、final contract 和 ZIP 回归通过，尚未上传、官方分数为空。它直接检验构图是否随高光阶段变化，风险高；不能从局部响应变化推断 IoU 收益。

**R71_ADAPTIVE_BUDGET_R54_V9**：保留 R54 排序和 V9 框，只按 TimeLens 区间占用率自适应保留率 `clip(0.75 + 0.10*occupancy/median_occupancy, 0.70, 0.85)`。相对 R54 删除 3,823 帧，195,214 条预测，426/426 覆盖；参数 8.854531425B。ZIP SHA `511fd5a946134c25a3cf537854f0c6a049b4bbc5a55e43c1e8742a44092880b0`，checker/contract 通过，未上传。10 个公共手工视窗诊断 soft `+0.00120`、binary `-0.00287`，方向冲突且无独立确认；暂列高风险备选，不能称收益或据此扫规则。

**R72 B3 原始资产恢复审计**：冻结配置期望 `/data/aic/experiments_910a/LFM_V8/arms/B3_s0/head_s.pt`；当前克隆相关路径均不存在。边界库存检查 67 个 4--12MB `.pt`，只发现 V7/V9 A0/A1/KD 变体，没有 B3 身份；历史 V8 B3 ZIP 仅有 predictions.jsonl，Git 无二进制权重；旧服务器端点审计时不可达。结论：原始 B3 head/base 当前无法恢复；缓存的 B3 预测仍可复用，但 V7 不能改名 B3。重新训练只能命名新 head 并单独验证。TimeLens2（8.767B）+原始 B3 约 0.450B = 9.217B，超 9B，不能作为最终组合。报告：`reports/r72_b3_asset_recovery_20261018/README.md`。当前最高官方锚仍 R54=36.22，以上新包均无官方回分。

### 明日五个提交额度的候选队列

按结构差异和证据价值排序：

1. `R54_TIMELENS_CENTER_RESIDUAL_V9_B3`：已知官方 `36.22`，作为最高分锚点。
2. `R69_SEGMENTED_CROP_R54TIME`：分段 TimeLens 构图，8.767B，ZIP SHA `0b1de985...10a8010`。
3. `R70_SPATIAL_UTILITY_R54TIME_BATCH32`：source-disjoint native crop 新 utility head，8.855B，ZIP SHA `a5d3031c...1cb7b7a`。
4. `R69_TIMELENS_DIRECT_CROP_V1`：整视频单 crop，对照分段构图，8.767B，ZIP SHA `f5d6dcad...e7839871`。
5. `R71_ADAPTIVE_BUDGET_R54_V9`：逐视频预算，8.855B，ZIP SHA `511fd5a9...092880b0`。

R26 `36.04` 和 R44 `36.09` 保留为已知回退/对照；R67 `36.16` 已知低于 R54。`R69_SEGMENTED_CROP_R54_TIME`、`R69_TIMELENS_DIRECT_CROP_R54_TIME` 等重复命名包不占名额。R63/R64/R68 及旧 R59/R60 无效或作废，禁止上传。

### 2026-10-18：空间稳定性门控与 R70 topsoft / 联合时间候选

**R69_SEGMENTED_GATE_R54TIME**：在 R69 分段构图上加入固定 run 级稳定性门控：合法自由轴中心相对整视频 crop 最大偏差和段内 spread 都必须不超过 `0.12`，否则整个 run 回退整视频 crop。427 runs 中 375 个采用分段框、52 个回退；相邻中心总变差下降 46.5%，p95 跳变由 0.2000 降至 0.1367。包保持 426/426、199037 帧、8.767B，ZIP SHA `f7d17806697bd677580fcddda2fe02c441c1dacb3c7f979cd000823a57c5cf58`，本地 checks 通过，官方分数未知。若明日仍选分段路线，优先用 gated canonical 包替换无门控 R69。

**R70 topsoft spatial objective**：在相同 public native crop source split 和 V9 特征上比较 MSE、pairwise、listwise、topsoft 四种新读出目标。topsoft 的 RV/LIVE source-macro 增量为 `+0.07121/+0.06892`，固定 split 平均 `+0.07006`，略高于 MSE 的 `+0.06901`。基于 topsoft head 完成 `R70_TOPSOFT_SPATIAL_R54TIME`：426/426、199037、101797 个框变化、8.855B、0 fallback，ZIP SHA `da5fcc5dccbfcd960b112852295dbbba35be89aaad61c20fcbb81d1ba8fed4d2`。这是公共源证据，不是官方涨分证据。

**R70_TOPSOFT_UTILITY_TEMPORAL**：把 topsoft 每关键帧 top-candidate utility 插值到原始 frame 轴，用固定联合规则 `rank(R54时间价值) * (0.5 + 0.5*rank(R70 utility))` 重排精确 85% 集合。426/426、199037、平均每视频相对 R54 交换 5.122 帧、377 视频变化、8.855B，ZIP SHA `aadb42cadc97a9880b63cc7367ea5285d53a365ea4802040a934131e1440883f`。独立 checker/final contract/ZIP 回归通过，官方分数未知。报告见 `reports/r70_topsoft_objective_20261018/README.md`。

**R72_SPATIAL_UTILITY_TIME_R54_V9_B3**：另有一个更保守的联合时间对照，固定复用 R54 的 `eta=.20` 结构，只加入 R70 max-score 的同权 residual；共 4120 次换帧、393/426 视频变化、8.855B，ZIP SHA `666a83f8dddd691022c66193eff30b54af6845fafadba3bb20ed3a6c99f64d88`。它与 topsoft 乘法联合包保持为两个不同机制，不自动占明日名额。

### 2026-10-18：空间效用驱动时间重排深审计

第一轮R72/R73不等于严格的联合效用决策：R70预测的是129合法候选的最佳分数，但R72/R73输出V9 B3框。将R70分数插值到V9实际框后，8848官方关键帧的`output-max`均值=-.03886、std=.08233，仅56.5%在0.001内。

公开168源空间诊断显示，`max_score-center_score`与实际边际IoU增益相关系数Spearman=.377，而绝对`max_score`与实际选中IoU=.293；这支持区分“相对中心改善”和“绝对可构图性”，不能把代理直接相乘称为官方F。R73的增量效用有44.7%零平值，stable rank注入帧号偏置；R74 tie-aware修复后换帧5004→2418。

已生成未上传、仅供后续结构验证的输出对齐候选：R76 additive（4251 swaps，ZIP SHA `18d4986d84b705cd4076c4f640c77155eca5d1ac024583318fac0e7e3e7b2265`）和R77 multiplicative（2308 swaps，ZIP SHA `aae5cfa6ffc56e8276aa5aef741781143a23e71ca8f428a608228700aa7d4106`）。后续更高价值的训练是直接监督V9部署框IoU的source-disjoint utility head；明日2/3/5不变。

### 2026-10-19：R81 direct deployed-box utility

R78 nested source split produced a 50,561-parameter head from R70 topsoft 129 logits plus rounded V9 B3 output position; untouched external dev row/source Spearman was `.47334/.15850`. R81 uses that nested head and existing V9 full-frame boxes, includes the 1.511M R70 head in its parameter count, and has 8.856094947B total parameters. It contains 426/426, 199037 predictions, 2332 temporal swaps, checker/ZIP pass, and remains unuploaded. ZIP SHA=`80087b6e75fe70887db1f408e6b69baeae20812e5bb3e57ab3723de66dd1dfdb`.


### 2026-10-19：R82 直接部署框 utility 集成

R81 的 B3 输出框逐帧与 R54 完全一致（共享帧 196,705/196,705）；远端 V9/B3 full-frame artifact SHA=`95b9219676a5eece74fbce7bc108d08ba2b84759d0f754bf5660cad9a44c8e8a`，参数清单 `87,407,729` 已核实。

R82 在相同 R78 nested source contract 上训练五个 52,097 参数直接 IoU head，部署 utility 为 `mean - .5*ensemble_std`。公共外部 156 源的 source-macro Spearman 为 `.19120`，高于单头 `.16869`；源级 paired bootstrap delta `+.02304`，95% CI `[+.00856,+.03812]`。这是公开迁移诊断，不是官方分数。

R82 包已完成：`submissions/R82_DIRECT_OUTPUT_UTILITY_ENSEMBLE_RISK05_MULT_TIME_R54_V9_B3/`，426/426、199037 条、8.856303335B、2294 次相对 R54 换帧；checker 和 ZIP 往返通过，ZIP SHA=`de99b3c5fcaee15966d893425e1b39a896645546092a4d00dc00133bc31529fd`，未上传。它是高风险候选；R54=36.22 仍为官方安全锚。

### 2026-10-19：官方回分—本地验证审计

用户回报：`R70_TOPSOFT_SPATIAL_R54TIME=36.42`、`R71_ADAPTIVE_BUDGET_R54_V9=36.43`、`R82_DIRECT_OUTPUT_UTILITY_ENSEMBLE=36.20`、`R69_SEGMENTED_GATE_R54TIME=24.95`。相对 R54=36.22，R70/R71 分别为 +0.20/+0.21，R82 为 -0.02，R69 为 -11.27。

本地诊断不能作为官方涨分度量。R70 的 RV/LIVE native crop 留出平均 +0.07006 与官方方向一致；R71 十个公共视窗 soft +0.00120、binary -0.00287，与官方正回分方向冲突；R82 外部源级 Spearman bootstrap +0.02304（95% CI 不跨零）而官方下降，是直接反例；R69 平滑指标下降 46.5%而官方骤降，说明平滑不是评分目标。完整对照见 `reports/r83_official_local_validity_20261019/summary.md` 与 `metrics.json`。

当前判断：本地验证可用于合同、实现和同目标机制诊断，不能用于候选官方排序。R70 保留为正向空间证据，R71 保留为一次预算机制成功证据；关闭“公共 utility 排序提升自然转成官方时间收益”的 R82 解释。

下一主攻改为“低容量视频级政策拟合”：使用公开 PHD2/GIF 训练标签，训练约 50–150M 的 interval policy，预测整段的中心、宽度和边界置信度，再用固定 85% 集合解码。它针对“官方偏好是稳定的事件阶段/边界政策，而不是逐帧排名”的新假设；PHD2 AP 变好但集合效用不超过中心时立即关闭。历史 PHD2 指标与平台曾脱钩，因此该路线只在源级策略门通过后进入一个官方单变量包。

### 2026-10-19：R83 interval policy probe 结果

公开 PHD2/GIF 5,886 片段、2,262 源级留出（留出426源）上，26,786 参数的连续 7/8 interval policy 退化为中心策略：中心集合 F=`0.025424`，小型逐帧 BCE=`0.025281`，policy=`0.025424`，interval oracle=`0.028871`。这削弱了“只要用小模型输出连续区间就能拟合官方偏好”的简单版本；由于只有 8 槽和两个合法窗口，它不关闭更丰富的低容量路线。报告：`reports/r83_interval_policy_20261019/summary.md`，远端训练使用 CUDA，未读取官方 test。

下一实验转向低容量空间策略：R70 官方 +0.20 且 RV/LIVE native crop 留出同向，优先在相同公开 native crop 上训练连续坐标/候选集合读出，固定 R54 时间集合；两个源留出都超过 R70 topsoft 才做官方单变量包。

### 2026-10-19：R84 低容量空间坐标一致性候选

R84 使用 RetargetVid/LIVE native crop 标签训练 623,361 参数的逐候选 MLP，并加入 soft-argmax 坐标一致性损失；固定 R54 时间集合和预算。公开 source-macro 相对中心 `+0.02547`，LIVE 逐行 `+0.03135`，RV 逐行 `+0.00365`，低于 R70 相同协议平均 `+0.07006`，但提供了结构不同的空间信号。

R84 已完成官方推理和打包：426/426、199037 条、0 fallback、112141 个保留帧框变化；实际参数 `8.853590257B`（TimeLens2 + V9 vision tower + R84 head，不加载YuNet）；prediction SHA=`22e668f1dbd9d36694d60d5ee191a3437508cf9cb2b4cb0ebc3d9945be7c1616`，ZIP SHA=`526d3e0b41c329806c7746e0f7c1404b0ec8d3d2470495e8470c0f25983934f4`。independent checker、final contract、ZIP roundtrip通过，尚未上传。详见 `reports/r84_coord_consistency_20261019/summary.md`。
