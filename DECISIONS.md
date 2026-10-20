# 技术决定

## 2026-10-19：关闭通用显著性中心，保留有条件上下文信号

**Decision**：不将 UNISAL/AIM 2024 saliency centroid 直接接入 AIC 构图，也不把 R99 低秩上下文头制作官方候选。

**Evidence**：R98 在148 LIVE native-crop源上，UNISAL centroid−Qwen point `-0.03723`，95% CI `[-0.06317,-0.01117]`；50% mass `-0.03364`，CI `[-0.05943,-0.00794]`。R99 在448 PHD2/GIF源上 interaction−matched control `+0.000331`，CI `[-0.000916,+0.001633]`，global permutation后 `+0.000006`。

**Interpretation**：通用人类注意中心与赛事 native crop 构图目标错位。R99 的置换对照提示完整上下文可能有微弱内容条件信号，但当前8点片段和 observed-selection 标签不足以支持部署。

**Next**：先做 AIKU 真实文件/许可/标签审计；若重开上下文，改用更密或完整事件观测并保持 source-level fixed budget；空间路线改为目标 crop supervision 下的高分辨率局部读出。

## 2026-10-19：AIKU 只登记为外部摘要监督，关闭当前 VistaHL 空间代理

**Evidence**：AIKU 训练包真实包含6,000个预提取1024D特征 group和1,214,730个特征步，但没有原视频、音频、来源或特征提取器。`gtscore`与`gtsummary`相关性有限（每视频 Spearman 均值0.398），不能把摘要标签当AIC高光真值。R101 当前候选×ResNet18邻帧 context 相对控制 `+0.000197`，188 unit CI `[-0.001942,+0.002463]`；置换后不归零。

**Decision**：AIKU 在特征提取器和来源协议确认前保持 `AUDIT_ONLY`；不进行迁移训练。关闭 R100/R101 当前空间 context 配方，不制作提交包。

**Reversal**：若取得 AIKU 原视频/特征生成合同，或新的联合 native crop 监督显示上下文达到预设增量门，才重开；重开时需改变观测合同而非继续扫参。

> 当前规则更正（2026-10-15）：总推理参数≤9B，文件bytes仅作工程记录；下文旧字节否决及AHA只能teacher的判断已撤回，最终决策见末尾R25节。

## 2026-10-19：R86 回分不是 T5 H1 的有效否证；锁定点源合同

Status：Accepted。

Evidence：R86 官方回分 34.88，相对 R54=36.22 为 -1.34。审计确认 R86
使用 `points_V9_A1raw/points`，而 R54 的空间链路使用
`points_B3_vo/points`；因此改变了点来源与 H1 映射两个变量。R86 相对 R54
的改变框均值绝对位移为 `[17.95,31.60,0]` px，最大 748.07 px。
修正版 R88 只把点源改回 B3，local contract valid，改变框均值绝对位移为
`[9.69,25.80,0]` px，最大 244.87 px。

Decision：R86 归档为混杂 transfer；以后空间迁移实验必须在 manifest 中记录
point source，并与父包的 observation contract 完全一致。R88 是唯一可识别
“T5 H1 在 B3 点上是否有效”的最小候选；未取得官方回分前不宣称收益。

Reversal：R88 在官方回分仍明显低于 R54 时，才关闭 H1→B3 迁移；若 R88
接近或高于 R54，停止使用 A1raw 作为空间替代链。

## 2026-10-19：R87 BF16 修复运行时，暂不包装 AHA

Status：Accepted。

Evidence：FP16 AHA causal path 在第二帧起产生 NaN；BF16 + 原始空 token
transition 后，trace 与 96/48 pilot 的 finite_fraction 均为 1.0。32 源留出
上 Ridge 适配 row AP=0.3486，但 source-macro AP=0.0952；监督是弱 GIF
interval label，样本量不足以支持官方迁移。

Decision：关闭“运行时 NaN”阻塞，保留 AHA 作为目标域监督适配初始化；当前不做
AHA 官方包。下一次适配必须扩大源级留出、报告空/正标签分层，并与中心和现役
R54 时间集合在同一合法公共目标上比较。

## 2026-10-16：R29——先有效监督与源外增量，停止当前标量挽救配方

Status：Accepted for next research allocation；无新部署候选。

Evidence：同85%公共时间预算下位置＋AHA两头比位置−.003262，源级95%CI[−.008499,+.001084]，未过冻结门；plugin的增量CI跨零。40公共源所有标注正run均被实际1fps采样命中，不能用12点完全漏采解释当前AHA。R28续跑完整24项显示Qwen20项输出中心，候选GT辅助oracle显著更高，但无B3同帧，提示0.5锚定仍是替代解释。详见`reports/r29/summary.md`，独立算术review不等于新源确认。

Decision：冻结现有两标量校准/keep配方，保留中心85＋B3安全包。主攻可追溯原生监督＋任务特定读出，检验内容相对位置的源外增量；空间候选读出为备选，不凭oracle差距直接换LFM。先判断信息是否可观测、监督是否奖励目标，再选新表示/训练位置。新人工native联合GT尚缺，不以教师框补真值；普通独立诊断继续。

Reversal：新源同K稳定增量可重开现有分数决策；相同静态输入不可判而短连续输入可判则改观测；目标可靠可见而读出无法拟合才升级表示；训练成功跨源失败优先审查标注政策和域。计划协议`configs/R29_NEXT_SUPERVISION_STAGE.json`尚未执行。

## 2026-10-16：R30——高维 VPB 动态读出未过源外门

Decision：不把 12 点 VPB 全维相邻差分读出升级为主路线，也不据此启动更大时序模型。静态 VPB 只有小幅且不确定的开发增量；动态差分 DEV 下降，且 TRAIN/DEV 落差显示过拟合。下一判别为低维、强正则事件状态特征；失败后转音频/事件结构或空间合同恢复。

Evidence：`reports/r30/summary.md`；配置 `configs/R30_DYNAMIC_CONTENT_PROBE_20261016.json`；脚本 `scripts/r30_dynamic_content_probe.py`。所有输入来自已绑定 R29 公共证据，未访问 protected/test。

## 2026-10-16：官网规则重新核实，新候选使用model_params_b

Status：Accepted（当前一手官网证据）。

Evidence：当前赛题页/复赛通知直接HTTPS获取，公式图目视确认，hash/摘录在`reports/r29/official_rules/evidence.json`。全部实际推理模型≤9B、共享计一次、量化不减参数；当前Score=100F，无k_size；工程要求`model_params_b`。截止结果10月23日20:00、材料10月25日23:59，每日5次。

Decision：新包按总参数/10^9申报，更新01；旧回分ZIP与SHA保持原件，旧字段只记历史构建。尚未取得登录后schema及同版本可运行evaluator，不声称平台兼容已核验。

## 2026-09-30：V6-NEXT10H——T1 不出包、E5 判负、E1-AUTO 达扩展标准

Status：Accepted（预注册规则执行）。

Evidence：T1 confirm 配对门 3/3 过 + YTH 迁移 macro AP 0.616(n=32, TRANSFER_INDICATIVE)；
E5 零样本 0/12 可解析；E1-AUTO pilot parse 100%/保留 90.6%，A/B delta +0.0069（噪声内）。

Decision：(1) T1 仅作开发主候选，不出包——把 MrHiSum 权重换入现有 bundle 不是单一
因素干净包，正确路径是学生原生域重训同骨架；(2) 450M 零样本空间定位路线终止，
空间蒸馏监督改用教师点→合法窗；(3) E1-AUTO 扩展至 108 条（用户预设扩展标准已达成），
扩展后对照必须报种子置信区间而非点估计。

## 2026-09-25：从 Phase 0 / 1 启动

Status：Accepted（工程选择，非模型效果结论）。

Evidence：本地只有三份文档，远程项目与数据目录为空，无历史 baseline 可继承。

Decision：先实现比赛契约和 A0；TVSum 作者原视频为首个数据源；用显式指标名称区分时间代理与联合比赛指标；无同版本官方脚本时称 official-like，不称官方一致性验证。

## 2026-09-25：TVSum 许可证闸门

Status：Blocked。

Evidence：包内 `WebscopeReadMe.txt` 要求签署 Yahoo Data Sharing Agreement、获批非商业学术研究并禁止再分发/网络存储；README 的 CC-BY 表述不能自动覆盖该冲突。文件 SHA-256=`407d340bcd06fdc6d17374ebe6760b4a96816bcace228559c8283d9fb2520dea`，50/50 视频字节与解码元数据已核验。

Decision：用户明确授权本项目将可下载的数据集和模型视为可用于实验。保留 `license_gate`、来源和条款哈希用于追踪，但不再因该字段暂停训练；恢复 A0_001 资产，后续实验按用户授权继续。A0 的 TVSum 标签仍是 summary proxy，不是官方联合 GT。

## 2026-09-25：资源与时间边界

Status：Accepted（用户硬约束）。

Decision：物理 GPU 0/3 永不默认使用；GPU1 已有任务，首轮优先2并启动前重查。所有训练使用内部预算、定期 best/last 与外层 timeout，断线不终止。大文件仅在 /data/aic；代码 rsync 不删除远端数据。

## 2026-09-25：固定 TVSum 评估协议并修复变长推理

Status：Accepted for temporal proxy research。

Evidence：50/50 个 MAT 记录的 `user_anno` 都是 `(20,nframes)`，旧 linspace 展开最大标签差低于 `5e-8`；但旧 GroupNorm/pooling 路径在 5→12 padding 时有效 logit 最大差 `0.3392`，8→12 为 `0.5733`。修复后通过 `lengths` 逐视频计算，padding 回归差为 0。

Decision：GT 使用固定 `tvsum_summary_mean_norm_ge_0.5_v1`，prediction threshold 独立并只在 train/dev 选择；模型选择使用 video-macro temporal proxy，micro 仅诊断。TVSum continuous score、固定二值 proxy、作者 15% summary protocol 分开命名。A1 历史结果改称 feature-level shift。

## 2026-09-25：阈值校准优先于扩展模型

Status：Accepted for current development protocol。

Evidence：修复前 A0/A1 在 0.5 threshold 的空预测率分别约 0.938/0.750；A0_006 在 dev threshold 0.40 达 macro F1 `0.15188`，A1_005 在 0.35 为 `0.14474`。all-positive baseline 为 `0.08715`，说明原先接近零分数主要受校准和阈值影响。

Decision：候选必须保存 threshold 来源；禁止用评估视频真实正例数量决定预算。拿到官方 lockbox 后重新锁 threshold，当前数值不得称 AIC F_video。

## 2026-09-25：冻结 TVSum local validation protocol v1

Status：Accepted for local temporal-proxy model selection。

Evidence：`reports/tvsum_manifest_v3.jsonl` 已有 source-group 隔离的 27 train / 16 dev / 7 test 分区；远程五个历史 fold 覆盖 43 条非 test 视频。机器可读协议为 `splits/local_protocol_v1.json`，并由 `scripts/validate_local_protocol.py` 校验 manifest/assignment hash、分区覆盖、source-group 隔离和五折互斥。

Decision：把原 manifest test 的 7 条视频永久作为 LOCAL LOCKBOX；不得用其调 prediction threshold、smoothing、top-k、segment、checkpoint、模型或超参。候选冻结后才可一次性比较 lockbox。TVSum 无可靠 category metadata，v1 不宣称 category-stratified；协议变更必须新建 v2，保留 v1 历史。

## 2026-09-25：Feature Bank 当前降级

Status：Rejected for current definition; route remains open。

Evidence：第 11、23–28、31 维恒零，0/8/9、5/22、7/16 重复；motion-only、quality-only、composition-only、audio-zero proxy F1 分别为 `0.02724/0.01892/0.01613/0`，均低于 repaired A1。

Decision：停止当前无结构融合；清理重复/恒零维度并使用 residual/gated fusion 后再重开。全零 audio 结果不解释为真实音频无效。

## 2026-09-25：B0 作为 M 档参照保留

Status：Promising probe, not default fallback。

Evidence：ViT-B/16 frozen features + 同一 Temporal U-Net 在 threshold 0.30 的 TVSum proxy macro F1 `0.15928`，真实 raw-video JSONL 通过 validator；FP16 bundle `174,986,447` bytes，约 M 档。

Decision：继续做多 split/seed 和联合 GT 前的工程复现；在没有证明 raw F_video 增益足以覆盖 size coefficient 前，不替换 S 档 ResNet fallback。

## 2026-09-25：SmoothL1 不替换 BCE

Status：Rejected for current proxy。

Evidence：A0_012 只替换 loss 为 SmoothL1，fixed-0.5 macro F1=`0.11679`，threshold 0.40=`0.14143`；A0_006 的 BCE 对照为 `0.15188`。

Decision：保留 BCE；ranking loss 和更简单 temporal head 仍是下一轮实验，不把本次结果外推为所有回归损失无效。

## 2026-09-25：冻结 local lockbox 与 zero-parameter policy

Status：Accepted。

Evidence：`splits/local_protocol_v1.json` 固定 27/16/7 source-group partitions。A0 Gaussian window 5 在 DEV 只有 `+0.00287`，在 lockbox 从 raw `0.114304` 降到 `0.112405`；gap/min-duration 也下降。

Decision：保留 raw A0 作为工程 fallback；所有后处理参数必须从 train/dev 产生，lockbox 只做一次冻结比较。当前不升级 smoothing、rank normalization 或 hysteresis。

## 2026-09-25：canonical internal TSM 暂不升级

Status：Accepted for current route prioritization。

Evidence：真实 layer1 feature-map TSM cache 的 chunk/full max error `1.81e-5`，参数增量为 0。五折固定 threshold 0.30：A0 `0.14631±0.03251`，internal TSM `0.13800±0.06299`，paired mean difference `-0.00830`；锁箱 internal TSM `0.175143`，强 A0_006 同 threshold `0.182033`。

Decision：保留实现、cache 和 correctness regression；降低 canonical internal TSM 优先级，不把历史 final-embedding shift A1 与它混称。

## 2026-09-25：DeiT-S/16 进入 S-tier challenger

Status：Promising, not replacement。

Evidence：timm DeiT-S/16 frozen features + 同一 Temporal U-Net，完整 FP16 bundle `46,618,447` bytes、`23,280,257` 参数。DEV median-9/threshold-0.35 为 `0.140241`，同一冻结 policy lockbox 为 `0.291867`；五折为 `0.14389±0.05173`；三个 seed 的 lockbox 为 `0.29187/0.22378/0.28128`，mean `0.26565±0.03664`；raw video → JSONL validator 已通过。

Decision：保留为当前最强 S-tier temporal challenger。它仍只有 TVSum temporal proxy 证据，必须经过第二独立 split/OOD 和 AIC 联合 GT 才能替换 fallback。

## 2026-09-25：TVSum comparison holdout v1 不再作为 pristine lockbox

Status：Accepted。

Evidence：原 7 条视频已经被 A0、DeiT-S、ViT-B、internal TSM 和多 seed temporal-head 实验反复比较，结果已影响路线选择。

Decision：将 `splits/local_protocol_v1.json` 的 7 条记录称为 `comparison_holdout_v1`。保留历史结果，但新模型、loss、postprocess、threshold 和 feature 不得用它调参。TVSum 后续使用 nested/repeated source-group CV，独立泛化证据必须来自新数据集。

## 2026-09-25：RetargetVid annotation-only 接入

Status：Accepted for spatial benchmark preparation。

Evidence：远程 `/data/aic/datasets/RetargetVid` 已包含 200 个视频、6 个 annotator、1:3/3:1 的逐帧 crop 标注和官方 evaluator。审计脚本输出 200 个 video pairs，帧数一致。

Decision：先完成标注格式和 evaluator 接入；在取得 DHF1K 原视频前，`spatial_iou` 保持 null，不把合法率、轨迹平滑或 saliency 分数称为 IoU。

## 2026-09-25：OOD 与 VLM 暂不阻塞主线

Status：Blocked subtask, mainline continues。

Evidence：SumMe ModelScope 仓库 raw 视频是无进展的 LFS object；本轮停止下载，没有生成 OOD 分数。服务器没有可快速运行的本地 VLM teacher 权重。

Decision：不把下载阻塞或缺 teacher 误写成模型结论；下一轮优先取得可验证的 SumMe/YouTube Highlights raw 子集，再做小规模 VLM pilot。

## 2026-09-25：真实category与nested CV替代已消费holdout

Status：Accepted；修正早期metadata不可得判断，不覆盖旧实验。

Evidence：实际MAT包含10类各5视频。90次配对训练完成；2×5外fold、3seed，每fold30train/10inner-dev/10outer；600组跨backbone外层frame/label/timestamp/mask精确相同。

Decision：原7条只作comparison_holdout_v1历史解释，不再调参；保留原JSON，新增status sidecar。TVSum全体已开发暴露，nestedCV不是全新独立test。只用inner-dev挑checkpoint与threshold。

## 2026-09-25：DeiT-S保留challenger，不提升Primary

Status：Accepted for current evidence；不是否定ViT路线。

Evidence：DeiT−A0 nested summary delta+.01552，95% CI[−.00653,.04070]；Spearman delta−.01777，CI跨0。ViT-B summary近乎等于DeiT。首批6条strict-alignment SumMe上30 matched heads平均summary为A0 .25752、DeiT .15345、ViT-B .12961。

Decision：A0继续工程fallback/保守S方案；DeiT是TVSum summary challenger；ViT-B保留M参照，不扩大模型。不能把7条高分或一次seed当泛化胜出。

## 2026-09-25：SumMe原视频恢复，时间轴协议显式隔离

Status：Accepted as bounded OOD pilot with limitations。

Evidence：Zenodo嵌套ZIP经Range+CRC恢复8视频；6条decoded count和GT/FPS相符，另2条排除。PTS确实非单调，PyAV/OpenCV前30帧像素完全一致。原生作者MATLAB evaluator经Octave与18项Python mean/max F1误差≤1.12e-16。

Decision：使用显式annotation-ordinal CFR benchmark，保留镜像时轴局限；不改变比赛严格PTS reader，不拉伸GT。第二批按预注册大小顺序扩展，冻结模型/摘要规则，独立报告；旧LFS失败保留。

## 2026-09-25：空间首次真实GT证据，不把微小均值收益升级

Status：Accepted for benchmark and deployment; candidate gain uncertain。

Evidence：20条DHF1K×2ratio×6human。Center IoU .48190/.73951，face+EMA .48634/.75214，pair delta+.00854 CI[−.01054,.02855]；890280次原IoU函数比较误差0，前置负坐标clamp已复刻。8条raw E2E/3600预测有效，crop与benchmark完全一致。

Decision：center保留默认；face/proxy+EMA仅候选，不能称成熟A3。YuNet232589bytes计入总模型。优先解决多人主体错误，再研究复杂路径/大小。无AIC联合GT，不制造oracle/官方得分。

## 2026-09-25：第二批OOD确认与预算参照

Status：Accepted for bounded frozen-model evidence，不外推整个架构。

Evidence：新增8条在冻结配置下CV-head summary A0/DeiT/ViT=.17883/.11955/.15411；合并14条=.21255/.13408/.14361，DeiT−A0 CI[−.13175,−.02734]。A0−32draw随机预算+.07501 CI[.02407,.13841]，其余表示对随机CI跨0。历史完整bundle方向一致。

Decision：A0继续fallback，DeiT只保留TVSum challenger，ViT-B只保留M reference；不继续扩大backbone或重新扫旧阈值。14条SumMe已用于模型比较，后续不得称pristine lockbox。近重复700对筛查无flag只是有限覆盖证据。

## 2026-09-25：冻结三种可运行工程候选

Status：Accepted for engineering release, no official acceptance claim。

Evidence：完整权重SHA在本地/远程匹配；真实DEV两视频A0center1263/DeiTcenter2297/A0faceEMA1263条预测全valid；A0center33.01s、faceEMA157.78s。DHF1K三视频A0全空保留，无temporal GT不判错。

Decision：统一hash-audited离线入口，默认A0center。FaceEMA空间探索虽然权重仅增加232589bytes，dense CPU代价明显，不能只因小权重就替换默认。DeiT raw .35与历史median9/.35分开命名。保留首次CUDA统计失败及修复后新run_id；不覆盖历史。

## 2026-09-25：不以线性头替换当前U-Net

Status：Rejected for this matched-budget linear-head hypothesis。

Evidence：60次2×5fold×3seed对照，仅替换head。A0 Spearman delta−.10286 CI[−.14801,−.05926]；DeiT summary delta−.01292 CI[−.02128,−.00475]。既有SumMe14外部comparison，A0 summary delta−.06485 CI[−.10564,−.02562]，DeiT无改善。

Decision：保留U-Net，停止此线性头扩搜；本轮不导出/升级表现退化的完整候选。不能把结果泛化为所有简单head无效，后续loss/head须有新的错误机制依据。

## 2026-09-25：固定多人群体脸中心假设降级

Status：Rejected for this fixed observation rule; spatial route remains open。

Evidence：`SPATIAL_GROUP_003` 在同一20条DHF1K/RetargetVid source、2 ratios、6 raters、无GT调参下运行；top-3 area×confidence group center 的 IoU 为 1:3 `.48004`、3:1 `.75115`，group+EMA 为 `.48106/.75216`。相对 single-face+EMA 的 paired delta `-.00264`、95% bootstrap CI `[-.00726,0]`，20/20 source 没有正 delta；多人样例020由 `.48228` 降至 `.44026`。

Decision：不启用 `true_face_group*` 为默认空间候选；接口、raw-video path、JSONL validator 和 E2E 回归保留。后续只研究有明确主体关联/互动覆盖机制的假设，不继续该 top-3 规则 sweep。

## 2026-09-25：pairwise ranking objective remains exploratory

Status：Promising branch, not accepted default。

Evidence：`RG_RANK_001` completed 60 nested outer evaluations with identical folds/seeds and zero inference-weight increase. DeiT-S pairwise summary F1 `.233615` and NDCG@15 `.669796` exceeded A0 pairwise `.217504`/`.651274`, but paired per-video positive fractions were only 50–52%, Spearman delta was `-.00228`, and no independent new raw-video OOD benchmark was available.

Decision：保留 pairwise loss、诊断和 checkpoints；不替换 A0 fallback，不继续 DeiT 超参 sweep。下一步只在独立 OOD/native summary 数据取得后验证，或对预注册 loss 做 paired bootstrap CI。

## 2026-09-25：代理下载可用但媒体证据未完成

Status：Acquisition partial / benchmark blocked。

Evidence：算力服务器 Clash/Mihomo `127.0.0.1:7890` 可完成 Hugging Face Range 请求；YouTube Highlights 索引到358个成员但9.9GB tar未完成；DHF1K 021–030 RAR已恢复，7z对 `video/021.AVI` 报 `Unsupported Method`。

Decision：代理包装脚本作为后续下载入口；未完成媒体解码、帧对齐和native evaluator前，不把两者记为OOD或空间transfer证据。

## 2026-09-26：正式评测集只做冻结推理

Status：Accepted for upload preparation; no official score claim。

Evidence：正式包 174/174 视频完整解码，16:9=119、9:16=55，未提供标签；SUB_A、SUB_B、SUB_C 均使用已冻结 checkpoint、DEV-derived threshold 和固定 spatial policy，在物理 GPU 2/4/5 完成全量推理。三个根目录 ZIP 只有 `predictions.jsonl`，项目 validator、独立 checker 和解压回归均为 zero errors。SUB_C 是 DeiT-S + YuNet `true_face_smooth` spatial differential candidate，不称为 A0_face_ema 或官方优胜。

Decision：保留 SUB_A 作为第一上传的 engineering fallback，随后上传 SUB_B（DeiT-S temporal 对照）和 SUB_C（空间差异对照）。正式测试集没有训练、人工修正、逐视频调参或第三方 API 使用；`official_f_video` 和 `competition_score` 保持 `null`。等待用户在平台上传，不由 Agent 自动提交。
# 2026-09-26：官方 raw 反馈改变模型规模优先级

Status：Accepted for next search; not a claim about local proxy generalization。

Evidence：平台反馈 `SUB_A=1.01`、`SUB_B=6.08`、`SUB_C=6.64`。B/C temporal outputs are identical and YuNet changes 8,130/8,133 boxes, giving an isolated `+0.56` raw spatial differential. Relative to C, M and L raw break-even values are `6.98947` and `7.37778`。

Decision：保留 A0 作为工程 fallback，优先测试更强 S/M 组件；不因 0.95 系数放弃 100–500M 甜点区。每个候选同时记录参数量、实际加载权重 bytes 及两种 size interpretation。ViT-B 是第一个 bounded capacity reference；在没有官方新分数前不把它升级为 winner。

# 2026-09-26：130GB package 的正确语义

Status：Accepted。

Evidence：本地发现 `/home/hajimi2025/datasets/data-challenge-2026/video_highlight`，129.308GB 压缩 shards、11,245 MP4、987 条 QVHighlights-derived seed weak labels；889 个标注 clip 可匹配，98 个缺失。记录的 seed model/prompt 是自动弱标签，包内没有独立 license/readme/manifest。

Decision：将其版本化为 `aic_qvh_seed_weak_training_v1`，与 TVSum summary proxy 分开；所有结果标明 `official_aic_gt=false`。先做有界 5% ingestion/head/finetune pilot，再决定是否迁移更多数据；不把该包称为原生 QVHighlights human GT。

# 2026-09-26：首轮数据规模与微调诊断

Status：Exploratory。

Evidence：TVSum DeiT frozen-head D25/D50/D100 macro F1@0.5 为 `.0528/.0422/.1214`，单 split 非单调；QVHighlights weak-label 37/9 frozen DeiT pilot validation macro F1 `.67846`、Spearman约`.393`，仅说明新域 pipeline 可运行。

Decision：不根据单次弱标签分数升级候选。完成 frozen vs last-block raw-video control 后，若收益在第二个 source-group split 保持，再考虑扩大 QVHighlights 抽取；否则停止无边界全量特征化。

## 2026-09-26：来源与参数口径纠正

用户明确赛方不提供训练数据。本地130GB资产只能称local QVHighlights-derived weak-label asset，供应方未核实；保留既有审计数值但撤回官方提供的描述。100–500M参数为本轮重点，不等同100–500MB权重；所有候选同时记录参数和实际bytes。VideoMAEv2-Base约86M参数不满足该参数甜点区，InternVideo2-1B属于L参考。无效VideoMAE归一化run不得进入模型排名；修复使用新run并匹配clip target对照。

## Native QVH next-stage gate

Weak-label position prior outranks all tested encoders, so do not scale weak-supervision claims alone. Native human saliency is query-conditioned and only partially rated: no silent conversion of missing scores to negative generic highlights. QVH_NATIVE_BOUNDED_V1 freezes 96/24/40 distinct sources; holdout only for frozen candidates. The ten exposed-video human sanity result is exploratory and cannot promote VideoMAE.

## D/E candidate freeze gate

Do not consume official test until native DEV supports representation ranking, not merely calibration F1. Train a matched DeiT control because old SUB_C uses TVSum. Do not call weak/native saliency a local AIC score. Keep .35 threshold and original dense_v1 YuNet. All candidates require immutable weight+code hashes, complete 174-video coverage, independent schema check, project validator and ZIP regression before READY. GitHub synchronization authorized; no platform upload.

## 2026-09-26：Video foundation candidate gate

Decision：VideoMAEv2-Base 不进入官方候选；其 matched native DEV ranking 低于 DeiT-S。InternVideo2-1B 只保留为 promising exploratory engineering candidate，因为 paired Spearman/NDCG CIs 均跨零。SUB_E 可以完成冻结的 raw-video/JSONL/validator/ZIP 闭环，但不自动上传、不宣称已优于 SUB_C。

## 2026-09-26：SUB_E release artifact

Decision：InternVideo2-1B native candidate is READY_TO_UPLOAD as a frozen engineering artifact. It has 174/174 coverage and zero validation errors, but remains exploratory because paired DEV CIs cross zero and its L-tier penalty requires raw `>7.3778` to exceed SUB_C weighted score. No automatic platform upload.

## 2026-09-27：InternVideo2 K700 official feedback

Evidence：SUB_E（InternVideo2 Stage1-1B K700 + native-QVH head + YuNet）获得官方平台分数 `34.43`，显著高于 SUB_C `6.64`。平台没有提供 raw/size 分解，因此只记录平台分数，不反推 F_video。

Decision：确认视频原生 foundation 是当前主线。下一步优先测试 Stage2-1B 和 M-tier VideoMAE-Large，再进行后训练；不继续在 DeiT 上做小修小补。

## 2026-09-27：Stage2 release与K710 gate

Decision：将 InternVideo2 Stage2-1B 冻结为 `SUB_F_INTERNVIDEO2_STAGE2_NATIVE_V1` READY_TO_UPLOAD 工程候选，等待用户决定上传，不自动提交。其完整 JSONL/ZIP 闭环已通过，正式测试集只用于冻结推理和格式验证。

Evidence：174/174 coverage，87,781 predictions，zero empty videos；weight bytes `2,827,511,457`，L-tier `k_size=0.90`；ZIP SHA256 `a7622bae7b78712549a344f5b5d61fd1762c395a734d82a2c85173bc344a6537`。解压后 independent checker valid。

Decision：K710 完整 native 96/24 结果不升级。其 best Spearman `.19818` / NDCG `.96576` 低于 Stage2 `.21433/.96962`，且只是 native-QVH proxy，停止其官方推理，保留作为 action-pretraining 对照。

## 2026-09-27：按请求制作 K710 提交包

Decision：虽然 K710 native DEV ranking 低于 Stage2，但按用户要求制作独立冻结官方提交包，命名 `SUB_G_INTERNVIDEO2_K710_NATIVE_V1`。它与 Stage1 K700 形成 action-pretraining 差异对照；不将其 DEV proxy 结果写成官方成绩。

Evidence：174/174、87,585 predictions、zero empty videos；2,049,516,187 loaded bytes；ZIP 与解压回归 validator 全部 valid。正式测试只用于推理和格式检查。

## 2026-09-27：17.46 分数先按单 shard 上传错误处理

Evidence：`SUB_G_shard0` 和 `SUB_E_shard0` 都是 87/174 个视频、偶数 video_id；对应 FINAL 包是 174/174。17.46 约为 Stage2/K700 完整包 34.42/34.43 的一半。K710 与 K700 完整 JSONL 共同帧 bbox 完全一致，只有 3 个视频少量帧集合变化。

Decision：在平台侧确认上传文件 SHA 前，不把 G 的 17.46 解释为 K710 模型能力。后续只上传 `*_FINAL/upload.zip`，上传前检查 ZIP 内 JSONL 行数必须为174，并记录 SHA256。

## 2026-09-27：K710 完整包分数修正

Evidence：用户确认 K710 完整 174-video FINAL 包得分 `34.38`；K700 `34.43`、Stage2 `34.42`。此前 G/E01 的 `17.46` 是 87-video shard 上传造成的覆盖问题，不能归因于模型。

Decision：将 K700、Stage2、K710 视为当前冻结 protocol 下同一官方分数窄区间；停止继续只换 foundation 的同构提交。下一轮优先改变 temporal adaptation、采样或 spatial head，并保留 K700 作为 tie-break fallback。

## 2026-09-28：空间尺度候选与时间路线

Evidence：
- 初赛只计 raw F；平台上 V0（305.6M）与 SUB_F（1.02B）预测相同、同为 34.42，与"未应用系数"一致。
- 时间：dev 代理上所有可部署的时间选择，相对全选的 CI 下界都 ≤ 0；E1/E2 显著变差。
- 空间：RetargetVid 真值都是最大窗口，无法验证尺度。S2 在那里平均 −.049，这是平均估计，不是上界。

Decision：
- 时间保留全选。未标注片段不当负例，E1 不再扩训。1 s 映射偏移登记为重启时间路线前必须统一的缺陷。
- 上传顺序 S2 → S1，都是探索性候选，用平台反馈检验"始终最大窗口""人脸跟踪"两个假设。
- B0 身份为"预测与 V0 完全相同、实际部署不同的低参数实现"，初赛不重复提交，不预报分数。
- 规模按参数量计档，冻结参数计入：V0 为 0.95 档，不是 0.90。
- Q2 按真实部署规模记录（33.38B 参数），撤回 350.5 占位值；它只是内部教师探针。

## 2026-09-28 下午：最大窗口位置

Evidence：
- 平台上，缩放两次都降了约 4 分，居中比 YuNet 低 4.29。
- RetargetVid（含未暴露的 021–030）上，只改路径无效；COCO 覆盖选择器无效；Qwen 语义主体点在确认集和合并统计上 CI > 0。

Decision：
- 暂停尺度缩小；拒绝路径平滑方向。
- 交付 Qwen 主体点的两个探索性候选，顺序是 POINT → NOFACE。它们与 B0 只差位置观测。
- 32B 在测试时加载，参数量和字节都超过上限，所以 JSONL 不填 model_size_mb，不写占位值。它们的初赛可用性依据是：SUB_Q/Q2 已被平台正常计分。复赛前必须蒸馏到合规的小模型，或向组委会确认。
- 时间路线继续全选，直到出现可靠负例监督。

## 2026-09-28 晚：教师优先（指南 v2）

Status：Accepted。

Evidence：
- 平台：NOFACE 42.83 > POINT 40.09 > B0 34.42；空间语义定位可独立涨分。
- 公共集（170 个未见视频）：T2 加密 +0.0135（CI 不跨 0）；T1 区域三种映射全负（−0.024 起）；位置决策而非主体识别或传播是最大误差来源。

Decision：
- 32B 继续作教师探索，4B 替换和大规模学生训练暂缓。
- T1（区域）按预注册规则拒绝，不打包；已中止官方推理。
- T2（加密）晋级，官方包 QWEN32B_NOFACE_DENSE_V2_FINAL 准备中。
- T3 只删不增按"假设性诊断包"处理；保留帧 bbox 与 N0 逐字节一致。
- 公共 170 个新视频上，教师缺口主要在"主体对但窗口位置差"与 1:3 窄窗的主体认错，不是提示词数量问题。

## 2026-09-28 夜：46.84 之后（指南 v3）

Status：Accepted（T5 待定）。

Evidence：
- 平台：TEMP 46.84 > dense 43.76 > N0 42.83；TEMP 保留帧 bbox 与 N0 0 差，+4.01 归于时间掩码。
- T4 边界精修在 YTH 弱代理上 −0.0014，恢复帧的正例率低于基线。
- oracle 复核：N0 路径已逐帧移动（静态镜头 6.3%），"每镜头 0.70 上限"不适用；位置/主体份额随池化方式与比例变化很大。

Decision：
- 交付 DT_V3 组合包；若官方 S_DT > 46.84 则晋级为新母本，否则保留 TEMP。
- 不交付 TEMP_BOUNDARY_V3；不在平台上扫删帧率/阈值。
- 不重复 T1 区域；4B 替换与大规模蒸馏继续暂缓。

## 2026-09-28 22:33：DT_V3 = 47.88

Status：Accepted。

Evidence：DT_V3（DENSE bbox × TEMP 掩码，`c702de92…`）平台 47.88，> TEMP 46.84（+1.04）；无交互参照 47.77，交互 +0.11。

Decision：
- DT_V3 晋级为新的已评分母本，远端目录设为只读。
- T5 预注册母本保持 TEMP / N0 观察协议不变；如通过确认，先出 TEMP 单变量包，与 DT 的叠加作为单独记录的组合。

## 2026-09-28 22:50：T5 窗口头晋级

Status：Accepted（待官方）。

Evidence：dev 上 H1 ≥ H2；从未使用的 LIVE-YT-VC val +0.0062，CI 下限 > 0；x/y 轴各自确认 > 0。

Decision：交付 `QWEN32B_NOFACE_TEMP_CROPHEAD_V3_FINAL`（母本 TEMP）。不在同一个包内与 DENSE 合并；若官方 > 46.84，下一步在 DT 母本上叠 H1，单独记录。H2 MLP 不采用。

## 2026-09-29：V4 周期交付选择

Status：Accepted（待官方）。

Evidence：
- 插值、0.25 s 观察、三点中值在 RV dev2 与 confirm2 上均为正，且 CI 下限 > 0，两个方向都不低于 −.002。
- 三者公开收益高度重叠：插值之上加 d4 只多 +.0057；加中值只多 +.0005。
- 上下文点、事件角色、只删变体未过预注册门槛。

Decision：
- 每个方法单独以 DT_V3 为母本出包，互不叠加。
- 建议提交顺序：插值 → 0.25 s → R0 诊断 → 中值（可选）。
- 不把 H1 叠到 DT。
- 时间方向保持 TEMP 掩码：YTH 代理本身不认可 TEMP（人工 −.003），不能用它判断 AIC 时间收益。下一轮如继续时间方向，需要与比赛目标更一致的标签。

## 2026-09-29：V5 周期决策

Status：Accepted（P0 待官方评分；P2 否决；H3 待训练结果过门）。

Evidence：
- P0（E4 复核点接回 INTERP）：dev2 x +.0187 [.0073,.0323]、confirm2 x +.0188 [.0079,.0326]，CI 下限均 >0，全视频均值 ≥0，y 行构造性零变化；E4 sanity 复现 ±.0001。这是 E4 在 V4 被否决的正确重启方式：E4 当时败在选择器拖累 y 轴与非空间行，P0 用管线级拼接把收益限制在它真正有效的 x 行。
- P2（Qwen 点名 + 检测器落位）：两个变体、四个分层一致为负（−.066 / −.039）；单帧开放词表框中心引入的方差大于它替换掉的 DENSE 蒸馏点。负结果成立。
- H3 视觉评分头：若 gate1（视觉头 dev 均值 − 几何头 >0）与 gate2（rv_dev+confirm2 管线配对均值 >0 且 CI_low >−.002、axis1 ≥−.002）全过才打包，否则只留消融证据。

结果（同日 18:00）：
- gate1 通过（+.0110）；gate2 confirm2 干净通过（VQ +.0190 [+.0071,+.0313]，y +.0172）但 rv_dev（40 视频）CI 下界 −.0112 未过 → **按预注册不出包**。
- 纯几何 G 四集全正：候选重评分框架本身有效，留作下轮基础；重启前置条件 = 扩大 rv_dev 划分并重新预注册。
- 提交顺序维持单一包 `QWEN32B_INTERP_XRERANK_V5_FINAL`；正式最佳保持 INTERP 48.67。

Decision：
- 提交顺序：先 `QWEN32B_INTERP_XRERANK_V5_FINAL`（唯一新候选包）；H3 过门才追加 `QWEN32B_INTERP_VISUALHEAD_V5_FINAL`。
- 正式最佳保持 INTERP 48.67，新包成绩填写待评，不预报分数。
- P2 路线关闭；重启前置命题（跟踪平滑后的检测中心 > DENSE 点）需先离线证实。
- 时间 mask、face 分支、B0 EMA、后处理全部冻结不动。

## 2026-09-29 晚：XRERANK 官方回分 48.49

Status：Accepted（记录性）。

- 48.49 = 母本 −0.18。RV x +1.9 IoU 的空间复核信号在官方集不迁移；空间类改动三连败（缩放、居中、XRERANK）。
- 正式最佳保持 INTERP 48.67。后续不再投纯空间 rerank 方向；H3 VQ 的 y 轴正信号（+0.0172 confirm2）是当前最有希望的未开发线索，重启前提是扩 rv_dev + 重新预注册。

## 2026-09-29 深夜：V6 E0 重判与 E2 复现结论

Status：Accepted（记录性）。

- H3 V5 按 V5 预注册完整版重判：gate1 FAIL（rv_dev best(V,VQ)−G=−0.0031 点估计，不是 CI 宽度问题，"扩样本翻正"假设不成立）+ gate2 FAIL（rv_dev 源组级 CI_low=−0.0114）。不出包维持；重启前置命题改为"视觉头在 rv 域点估计不胜 G"，且必须以 E1 原生 y 域人工标注为主指标重做修正头。
- E2 部署可复现性成立（特征提取逐位、前向逐位）；训练不可逐位是 `upsample_linear1d_backward` CUDA 核所致，声明 ±0.013 极差运行间方差，不隐瞒。后续任何引用 TemporalUNet 训练结论须报告该方差或多 seed 均值。
- 本轮 0 个新上传候选；正式最佳保持 INTERP 48.67。

## 2026-10-15：R15d 撤回"空间轴不是瓶颈" + 确认现役已在 89.8% 天花板

Status：Accepted（推翻 `docs/keep_scan_closure.md` §3 C3）。

Decision：
- **撤回** `keep_scan_closure.md` §3 的 C3 结论（"空间轴不是瓶颈，不要再做 bbox 校准类试验"）。
- 该结论取 `d = 0.2103` 解得 `R̄ = 1.0000`，而 `d = 0.2103` 恰是 `R̄ ≤ 1` 给出的**下界**
  （`d ≥ F_ANCHOR/(200−F_ANCHOR) = 0.21029`）。把下界当估计 = 取 R̄ 最大值，再从最有利情形外推，方向反了。
- 保留 C2（"时间轴机制净贡献 +0.77"）不动 —— 那部分正确。

Evidence（`scripts/r15d_ceiling_frontier.py`，纯解析）：
- `d ∈ [0.21029, +∞)` 全部与实测 6 点相容；(d=0.2847, R̄=0.784) 与 (d=0.2103, R̄=1.000) 不可区分。
- 两端点后果相反：修好空间到 R̄=1 分别值 **+0.00** vs **+9.42** 分。
- 天花板 `F_ceil(p) = F_ANCHOR·(1+d)/(p+d)`，只依赖 d：在 p=0.8534 处，
  d=0.2103→39.54，d=0.2847→39.23，d=1.0→37.50。
- **现役包已在 89.8% 天花板**（35.52 / 39.54）；即使排序器完美也只能到 39.5。
- **在现役 keep 率上拿不到 40 分**（需 keep ≤ 0.84，d=0.2103）。
- 反解 lift=ρ/p 仅 **1.05–1.16**；随机选择器 lift ≡ 1.000。

Consequence：
- "提高排序质量"与"压低 keep 率"两轴耦合且都已接近用尽：要超越 35.52，
  需 lift ≥1.170 **且** keep ≤0.55，是带门槛的跳跃，不是调参。
- `d` 是测试集属性，对全场所有选手相同，是我们唯一可反推的全局常数，**从未测过**（U2）。
- 新增 P0：**随机排序探针** —— 空间 bbox 逐字节复制 B3（R̄ 恒定），时间换固定伪随机排列，
  此时 `ρ ≡ p` 精确成立，得 `S_rand(p)=F_ANCHOR·(1+d)·p/(p+d)`，**只剩 d 一个未知量**。
  p=0.5 单点即可分辨 d∈[0.21,1.0]（极差 6.4 分，大于全部 6 次实测总跨度 2.95 分）。详见 `docs/r15d_ceiling_and_identifiability.md` §4。

## 2026-10-15：分数口径结案 —— 6 个实测点同一单位，混用合法，禁止 /k_size 换算

Status：Accepted（结案，关闭 U3）。

**问题**：锚点 `F_ANCHOR = 34.75`（v8b3 全帧）测于 ~10-02，keep 扫描 6 点测于 10-14，
中间隔 10-05 的 k_size 取消。若锚点带 k、扫描点不带（或反之），6 点不可混，`d` 反解全部作废。

**裁定：同一单位，混用合法。**

Evidence：
1. `reports/r5/score_ledger.json`（10-04）逐条记 `score_platform` 与 `score_raw_percent`
   **同值**，`score_unit: "raw_no_size_penalty"`；并把 `raw F = 38.83 = 34.95/0.90`
   明列于 **`withdrawn_conversions`**（"conversions from score/k_size are BANNED"）。
   ⇒ 平台**从来不在回分里乘 k_size**；34.75 自始至终是 F_video×100 本身。
2. **配对实验判定机制**（`reports/20261003_v10_exploration_report.md`）：同一预测集，
   声明尺寸 862 MB（k=0.90 档）→ 178 MB（k=0.95 档），两分 **34.75 vs 34.73（Δ0.02）**。
   若 k 作用于回分，Δ 应为 ~1.9 分。实测 Δ0.02 ⇒ **回分与尺寸无关**，k 在内部核算、
   不进返回值。10-05 取消它只是删掉一个从未生效的乘数 ⇒ **不产生单位断裂**。
3. `submissions/LFM_V11_VTREPLAY_SEMIFINAL/manifest.json`：`scored_at 2026-10-04`，
   `score_unit: raw_no_size_penalty`，operator-confirmed。

Consequence：
- `scripts/r15d_ceiling_frontier.py` / `scripts/r15d_random_probe_package.py` 的
  `F_ANCHOR = 34.75` **正确，不改**。`d ≥ 0.21029`、p=0.8534 天花板 39.54、
  "现役 89.8% 天花板"、"现役 keep 率下 40 分不可达" 全部保持。
- **反而是 `34.75/0.90 = 38.61` 那个换算是错的** —— 它正是 ledger 已撤回的那类操作。
  若曾按 38.61 重算，会把下界推到 ≈0.2393、天花板推到 ≈43.8，是**虚假**地推翻现结论。
- 唯一残留不确定：六点是否同**阶段**（同一测试集/同一评分窗口）。但 10-02 与 10-14
  同属复赛，无阶段变更记录，且 ledger 把 45.0 的"对手情报"与 34.95 并列在同一
  `raw_no_size_penalty` 口径下 —— 无证据支持阶段断裂，维持同单位。

Method lesson（并入铁律）：本类"单位疑虑"应当**先查 ledger/manifest 的显式字段**，
再考虑改动已提交的分析；本次若径直 rescale，会在"修正"的名义下注入一个更大的错误。

## 2026-10-15：R21 审计——撤回"0.3340 = 内容上限"，本地尺子全部失效，方向回到"先造尺子"

Status：Accepted（推翻 `r15b_experiment_plan_table.md` §1 的 E1.1 激进判据）。

**触发**：Claude 截图方向经实物/独立 CPU 审计（远端 `R21_CLAUDE_AUDIT_20261015`，
commit `d45cc09`，干净源码 `a7e4a51`），逐条核对原脚本、日志、缓存与既存 checkpoint。
审计未训练、未加载视觉 backbone、未出官方包。

**撤回 R21-C6**：`0.3340` 曾被当作"内容信息上限"。
实为 **1624 段 / 203 源，324 段 dev holdout 的源全部在 train**，且该 holdout 用于选 best epoch
⇒ **有泄漏的开发读数，不是上限**。同类作废的还有：
- `0.7404`（CLIP）：125 段中仅 **19** 个非恒定标签有效，19 段里 **14** 段与 train 同源；
  同一头喂**全视频一致的训练均值特征**仍得 `0.699287`，纯位置曲线 `0.731791`，
  与原始值只差 **0.008639**。另：VPB 头被改成 ReLU 且漏掉 checkpoint 的 `norm=(mu,sd)`。
- `−0.0994`（"均匀窗失败"）：`e01_eval.py` 用 812 行 uniform 标签的**行号**读 2508 行
  centered RGB 缓存，hold125 **124/125** 源错配。
- `0.7085`（"shot 优于 frame"）：该脚本**未做镜头边界检测**，测的是片段标签密度回归的
  跨片段 Pearson，与帧内 Spearman 目标/单位/指标三者皆不同。

**R21-C6 的数学反例**（`counterexamples.json`）：残差 B 用无截距单基函数投影，
不是"纯内容"——常数预测配全一标签得 B=**1.0**（未用图像）；纯位置 ramp 得 **0.882349**；
整体加常数 100 后降到 **0.406277**；完美逐样本预测时"预测均值曲线 vs GT 均值曲线"相关同样 =1。

**保留（不受影响）**：分数口径结案（§上一节）、`F_ANCHOR=34.75`、`d ≥ 0.21029`、
p=0.8534 天花板 **39.54**、"现役 keep 率下 40 分不可达"。这些是**官方轴代数**，
不依赖任何本地标签。

**我独立复核的修正**（对本决定的补充）：R21 的泄漏结论针对**旧池**成立，
**不适用于 R16_EXEC 的 T1\*/T1S\* 臂**。`public_uniform_group_v1.json` 实测
train 121 / dev 40 / eval 42 源、并集 203，**三向交集全 0**（裸 11 字符 YouTube ID 直接比对），
`T1S_uniform_s13_v1` 的 42 个 eval 源 ∩ train = 0。
⇒ 8 条臂读数有效，E1.1 判决成立，**但理由要换**：
`r15b_content_signal_verdict.py` 自己的判决文本写的是
「若模型下降，仅说明本协议可学信号减弱；源级独立评估前不得判内容不存在或 35.52 终局」——
**计划表 §1 写得比脚本激进，且该激进版本与 R21 冲突**。
采用脚本版本：**E1.1 结论 = "本协议下可学信号弱"，≠ "任务无内容信号"。**
8 条臂全部落在三控制带 [0.2404, 0.2470] 的 ±0.008 内，无一对预定控制形成稳定优势。

**残留待核**：`T1S_uniform_s13_v1/config.json` 声明 `manifest_sha256=a5de5742f51a…`，
而同路径 `public_uniform_group_v1.json` 实测 `016a3f034086…`，**不一致**——
在称其为"已认证源级划分"前必须闭掉。

**新增五项阻断**：①1979/2508 个 MrHiSum `gif_hard` 全零（全池 79.07%，非 87%），
构造器未恢复，既不当真负也不删；②源身份未绑定（2508 行 = 1329 规范源，legacy hold125 有 98 行同源）；
③禁止跨池复用行号，每行绑 `(source_id, media_hash, actual_pts, preprocess)`；
④VPB 头身份不一致；⑤确认池耗尽（LIVE 1800 源全 exposure，227 reserve ≡ 旧 R11 confirmation）。

**Consequence**：过去 12 轮"换监督/换 backbone/换损失都无迁移"的读数，
**无法与泄漏、错配、混比区分开** ⇒ 方向从"第 13 轮排序头"改为
**先重建测量仪器，再做一次可判定实验**。计划见 `docs/r21_direction_and_plan_20261015.md`。
其中**随机排序探针**（p=0.5/@0.8，`ρ≡p` 精确成立，只含 d 一个未知量）
**不依赖本地标签有效性**，是本阶段唯一不受阻断影响的最高信息量实验。

---

## 2026-10-15（补）：阶段 0.1–0.3 收口 —— 一项撤销、一项推翻、一项裁定

状态：**已接受**。三项都在 R21 审计的阻断清单上，现全部闭掉。

### 0.1 官方提交额度 = 每日 5 次，不是总量

**Decision**：按"每天 5 个名额"规划提交，诊断包与真候选**并行、互不挤占**。
此前的 hour-36 一次性决策、D/S-T 诊断包与真候选互斥的 slot_strategy 作废。

**Evidence**：用户 2026-10-06 明确更正。`reports/r6/claim_ledger.md` 记"memory 已存"，
但本地 memory 目录复查只有一个文件——台账记了一次没发生的写盘。
本次补写 `memory/aic-submission-quota-daily.md`。

**Consequence**：**随机排序探针可负担**（p=0.5 与 p=0.8 两点）。这是本阶段唯一
不受其余四项阻断影响的实验，额度确认后它从"奢侈"变成"必做"。

### 0.2 `manifest_sha256` 不一致 —— **虚惊，阻断撤销**

**Decision**：R16_EXEC 的源级划分**认证通过**，`docs/r21_direction_and_plan_20261015.md` §2
的"待核实"项关闭。

**Evidence**：`T1S_uniform_s13_v1/config.json` 声明的 `a5de5742f51a…` 指向
`R16_EXEC/temporal_rgb_v1/manifest.jsonl`（**特征池清单**），不是划分配置。
该字段名 `manifest_sha256` 被我误读为"划分文件哈希"。
该路径下 `public_uniform_group_v1.json` 实测 `016a3f034086…` —— 与声明值本就不同名。

另：两个划分副本（`016a3f03` / `48cb0c2c`）**字节内容不同但分区逐字段完全相同**
（train 121 / dev 40 / eval 42，`indices` 968/320/336，`positive_rows` 344/107/142）。
差异来自格式化，不是重新生成。

**Consequence**：§2 的修正全部成立且现在无保留项 —— T1/T1S 8 条臂的读数
不因 R21 的行号错配作废；E1.1 结论仍按脚本版本表述（"本协议下可学信号弱"）。

### 0.3 MrHiSum `gif_hard` 溯源 —— 裁定 `PLACEHOLDER_OR_FAILURE`

**Decision**：`gif_hard` 对这 1979 行**不是标签**，是一个字面硬编码的占位符。
标 `label_status = placeholder`；**不当作已验证真负，也不删除行**。
`gif_soft` 不受影响（0% 全零），继续可用。

**Evidence**（决定性，逐字节）：

```
/root/AIC/scripts/r14_assemble_sft_v2.py:107
    gif_hard = [0] * 12          # 字面常量，不是计算出来的
```

同文件的 PHD2 分支（84–100 行）**确实计算** `gif_hard`。上游
`r13_sft_data_build.py:129` 对 `mrhs` 写的是 `"gif_hard": None`，
而 `r13_grpo_pool.py:27-30` **依赖这个 `None` 把 MrHiSum 排除出 GRPO 池**。

> v2 把 `None` 换成 `[0]*12`，语义静默地从一个显式的"没有标签"
> 变成了一个貌似合法的"这 12 帧都不是高光"。

MrHiSum 原生标注只有 `gtscore` / `change_points` / `gt_summary`，**没有帧级硬标签**。

三个流传的全零比例各自在不同分母上可复现，不矛盾：

| 数字 | 分母 |
|---|---|
| 0.9% | 仅 PHD2 子组 |
| 87% | holdout-125 |
| **79.07%** | 完整池（1983/2508） |

**Consequence**：①阻断项 1 从"构造器未恢复 ⇒ 悬置"改为**已裁定**，
STATUS.md 与本文档 522 行的悬置措辞同步收口。②任何依赖 `gif_hard` 的历史数字
（含 GRPO 池构成）都建立在占位符上，需重算。③`gif_soft` 完好，弱监督线不受影响。

### 阶段 0 余项

- **0.4**（35.52 上传 SHA / 材料合规）仍在后台核实中。

---

## 2026-10-15（补 2）：阶段 0.4 —— 35.52 包审计，以及一次被拦下的错误判断

状态：**已接受**。调查只读，未改任何包、未接触平台。

### 结论：35.52 ↔ 磁盘文件 = 部分核实

**已核实**：
- 包在**远端** `/root/autodl-tmp/vpb/pkg_vpb_gifens_a025/`（本地 `submissions/` 里没有）。
  zip `33f55914cf4e…`（1,451,241 B），内含 `predictions.jsonl` + `manifest.json` 两个文件。
- JSONL sha256 `076016f6d9f1…` 与现役记录**逐字符一致**；zip 内外解出后 sha 相同（自洽）。
- 结构：426 行 / 426 唯一 `video_id` / 196,489 帧；帧号升序、无重复、无空行、无 NaN；
  `bboxes` 全为 3 元 `[x,y,w]`；`targetRatioWH` 16:9 ×303、9:16 ×123。
- 空间分支与 B3 逐字节相同：**196,489 / 196,489** 帧 bbox 逐值相等。

**未核实（两项）**：
1. **这份 JSONL 是否就是平台打 35.52 的输入** —— 无任何直接证据。manifest 自称
   `uploaded: false`、`official_platform_score: null`；全仓库全远端找不到 platform asset id
   或上传回执；`reports/r5/score_ledger.json` 的 `meta.sources` 亦自述
   *"not independently verified on the platform"*。
2. **A025 是哪一次 build** —— `jobs.log`/`jobs3.log` 显示该 zip 被重建**至少三次**，
   sha 依次 `be861ba9…` → `16341a09…` → `33f55914…`，`pred_frames` 分别
   195080 / 195551 / **196489**。而现役文档（`keep_scan_closure.md`、`e2_1b_keep_sweep_note.md`）
   **从未给 A025 记过任何 SHA**（只有 KF 族有）。⇒ 在日志层面不可判"35.52 对应哪次 build"。

**命名空间警告**：`e2_1b_keep_sweep_note.md:37-40` 那列 SHA 是 **zip 级**；
本次的 `076016f6…` 是 **JSONL 级**。两个命名空间，混用即错。

### manifest 与实物不符（逐项）

| 字段 | 声称 | 实际 | 判定 |
|---|---|---|---|
| 模型说明 | "Qwen3.5-4B **zero-shot** … **no training of any kind**" | 实际加载 ckpt_030 LoRA（84,972,248 B） | ❌ **严重失实** |
| 权重清单 | 4 项（LFM450 / V8 B3 头 / YuNet / Qwen 零样本） | 前三项确实列了；**漏 VideoPrism-B、漏 LoRA、漏 VPB gif_hard 评分头** | ❌ 漏列 |
| `status` | `…_ZEROSHOT_VLM_FRAME_SCORER` | 实为 SFT ckpt_030 + 头融合 | ❌ 同源 |
| `declared_model_size_mb` | 9503.763 | 实测合计 **9,813,117,568 B** | 见下 |
| `model_params_b` | **缺** | 4.8B（保守上界 5.25B） | ❌ 关键字段缺失 |
| `parent_baseline` | R12_GATT_LFM_SEMIFINAL (35.17) | 与本包自身分数无关，未更新 | ⚠️ 陈旧 |
| `video_count`/`prediction_count` | 426 / 196489 | 一致 | ✓ |

> 修正我先前的假设：任务简报里担心的"漏列空间分支（SigLIP2 NaFlex / B3 头 / YuNet）"
> —— **manifest 实际是列了的**。真正漏的是 VideoPrism-B、LoRA、VPB 评分头。

### ⚠️ 一次被拦下的错误判断（必须留档）

子代理按**规则文件 §4.1 的字节口径**（`M > 9216 MB` ⇒ 可判该次提交无效）算出
实测 9813.12 MB，得出「**这个包按附件1口径会被判无效，是致命风险**」，并建议
「降精度（fp16/量化 Qwen）+ 重算全量字节」。

**这个判断是错的，用户当场纠正：**

> 「注意事模型参数事9B限制，不是文件大小限制9GB」

正确口径：**上限是参数量 9B，不是文件字节 9GB**；k_size 惩罚系数已取消。
⇒ 现役包 ≈**4.8B 参数（保守上界 5.25B）⇒ 合规**。

**这个错误的两个来源，都必须记下**（否则会第三次犯）：
1. `01_赛事规则与评分标准.md` §4.1 的大小表**逐字归档的是附件1口径**，
   而该文件 §0 自己就写着这是**官方内部量纲冲突**
   （通知正文「9B／100M／500M」vs 附件1「9GB／100MB／500MB」）。
2. `aic/contract.py:283` 的 `0 < M <= 9216` 和 426 条 `model_size` error
   **检查的也是字节口径** —— 本地 validator 在这一项上不反映现行规则。

**教训**：把规则文件当 source of truth 是对的，但当文件**自己声明存在未解决的量纲冲突**时，
不能默认选那一支归档得最详细的口径去下"致命"结论。
`AGENTS.md` 的优先级里"官方最新规则 > 规则文件"正是为这种情况写的。

### validator 首次运行

`python -m aic.contract validate --index <intake_426.jsonl> --submission <pred.jsonl> --stage final`
→ `valid: false`：426 条 `model_size` error（**字节口径，按上面已过时**）
+ 426 条 `model_params_missing` warning（**这条是真的，需补 `model_params_b`**）。
**除大小字段外，几何/覆盖/排序/重复帧全部通过。**
历史运行记录：本地与远端**均无**任何 validator 报告 —— 这是第一次跑。

### 待办（合规侧）

| # | 缺口 | 优先级 |
|---|---|---|
| 1 | 补 `model_params_b ≈ 5.25`（真实缺口） | P0 |
| 2 | 改述 "zero-shot / no training of any kind" 为 "Qwen3.5-4B + ckpt_030 LoRA 微调" | P0 |
| 3 | 补齐权重清单为 7 项，逐件给 bytes/sha | P0 |
| 4 | 回填资产记录（asset id / 分数 / 时间），消掉 `uploaded:false` + `score:null` | P1 |
| 5 | 定版后把 zip + jsonl 双 SHA 写入 `keep_scan_closure.md`（**区分两个命名空间**） | P1 |

**不在待办里**：降精度/量化 Qwen 以规避 9216 —— 那是基于错误口径的动作，禁止。

---

## 2026-10-15（补 3）：阶段 1 完成，及一条必须降级的推论

状态：**已接受**。

### 阶段 1（仪器重建）完成

64 公共源全部到手（1.395 GB / 280.17 s），**54 通过 PTS 轴门、10 永久隔离**。
10 个失败全部是 `raw_interval_outside_media`（标注末段越过媒体末尾，最大 +0.168 s），
Δ 全在容差内 ⇒ 真实边界不一致，非解码错误。
54 个可用源 `label_status` 明确、**全零 0/54**，每源绑 `(source_id, media_hash, actual_pts,
preprocess)` 并留双 sha256。**阶段 2 的输入条件已满足。**

修掉三个会污染读数的执行层缺陷（探测吞失败 / **镜像 429 被记成源不可用** / `--resume` 字节双计），
其中 429 那条最危险 —— 它会把传输层限流伪装成数据可得性结论。
修后退避版 **64/64 成功、0 失败**。

### 降级：现役 A025 的 `p = 0.8534` 在本机不可复现

对 8 个包逐一用 `intake_426.jsonl` 复算 pooled `p = N_pred/frame_count`：

**KF 族 7/7 与文档差 0.0000（精确复现）**，而**现役 A025 差 −0.0143**
（文档 0.8534，实测 **0.839126**）。=> 复算口径正确，A025 是唯一例外。

盘上 A025 缺 3,342 帧；`jobs.log` 显示该 zip 重建过至少三次
（pred_frames 195080 / 195551 / 196489），**而 0.8534 所需的 199,831 哪一版都不是**。
`keep_frac=0.85 × 234159 = 199,035` 也 ≠ 196,489（隐含 keep≈0.840）。
旁证 `a0.15/a0.35/a025` 三包 p 只差 0.0004 ⇒ 后缀不是 keep（疑为 α），
但 `audit2.py:2` 读的 `scores_fuse_gifens_a025` 里后缀又是 α —— **两处记账互相矛盾，磁盘不可判**。

**为什么这条必须降级而不是绕过**：它是**可算出方向**的，不是模糊的不确定性 ——

| p | d=0.2103 天花板 | 天花板=40 所需 d\* | vs 下界 0.21029 |
|---|---|---|---|
| 0.8534（文档） | 39.54 | 0.1170 | **低于下界 ⇒ 40 恒不可达** |
| 0.8391（磁盘） | **40.08** | 0.2257 | **高于下界 ⇒ 40 可达（窄窗）** |

⇒ 「现役 keep 率下拿不到 40 分」**取决于一个从未在磁盘上复算过的 p**。
STATUS.md 与本文档 469/504 行那几句据此写下的断言，**降级为"依赖未复算的 p"**，
在阶段 2 之前不据此做任何设计决策。

**最省的解法**：随机排序探针（`ρ≡p` 精确成立，只含 d 一个未知量）
—— 它同时钉死 d **并**给出真实的 p，一次提交解决两件事。
额度已确认为每日 5 次 ⇒ 可负担。

**不采用的解法**：去 `score/k_size` 反推、或从 keep_frac 推断 p
（已证 keep_frac→N_pred 连 floor/ceil/round 都复现不出，**p 必须实测**）。

---

## 2026-10-15（补 4）：阶段 1 四项收口 —— 头身份可恢复、缓存可重绑、评估路径缺断言

**Decision:**
阶段 1 判定为**完成（4/4）**。此前它在 `STATUS.md` 里被写早了：R22/R22b 只做了第 1、2 项
（64 源 + 源级绑定），第 3 项（缓存重绑）与第 4 项（头身份恢复）直到本轮才做。
交付两个可复算执行器：`scripts/r22_rebind_cache.py`、`scripts/r22_head_identity.py`。

**Evidence:**

1. **头身份可恢复，且旧读数精确重现。**
   `vp_score_head.pt` 自带 `report.best_holdout_spearman = 0.5696324319689352`。
   在重建的干净缓存视图上按真身身份（输入 `/255.0`、特征 `(x−mu)/sd`、激活 GELU）复算得
   **0.5696324320**，`Δ = +0.000e+00`（字面相等）。
   横向对照：有归一化+ReLU `+0.004048`；无归一化+ReLU `−0.008014`；无归一化+GELU `−0.013384`。
   ⇒ 被审计副本的偏差会**改变读数**，但只在小数第三位；真正致命的是它同时丢了归一化。

2. **缓存可重绑。** 训练时缓存为 `627 × 4 = 2508` 行（与 `sft_data_v2.jsonl` 相等），
   分片是 **strided** `rows[i::4]`，四个 shard 前 627 条已逐位核实。
   盘上 `shard_3of4.npz` 在 13:35（头存盘后 7 小时）被**追加 192 行**，
   全部是重复 vid 且**逐值不同（0/192 相同）**，把缓存从 2508 污染成 2700。
   冻结训练时视图后 `aligned=true`、`frozen_rows=2508`。

3. **评估路径的行号错配从"怀疑"升级为已定位机制。**
   `train_vp_head.py:56` 有 `assert len(vids) == len(rows)`；而
   `e04_compare.py:118-124` 与 `resid_e03.py:85-93` 只有 `concat → argsort(order)`，
   **无长度断言、无截断**，把 2700 行喂给索引域 <2508 的 `hold_idx` ⇒ 静默错配。
   **断言只存在于训练路径，评估路径被丢掉了。**
   受影响读数（E0.3 VPB head、E0.4 VideoPrism-B 列）**均已在 R21 判废**，
   本条补充机制证据，不构成新判决，**不重开任何已废结论**。

4. **新增独立测量缺陷：抽特征不可复现。** 用当前 jpg 重算，`shard_0of4`（**从未被改写过**）
   的缺口与 `shard_3of4` 一样大（0.011–0.033 vs 0.011–0.12），单行最高 **0.1298 ≈ 1.45σ**。
   fp16/fp32 权重实测完全相同（最大差 0.000000），帧目录 mtime 全在 10-12（早于两次抽特征）
   ⇒ 不是追加、不是精度、不是输入变化，是**抽取路径本身不可复现**。
   原始抽取用的 `.npz` 权重已不在盘上（全盘唯一 >50MB 的 npz 就是这份 shard 自身）。

**为什么这条不足以推翻任何已废结论：**
在已验证的干净视图上做同尺度扰动，`amp=0.014`（≈0.0236σ）对应的 spearman 变化
**恰好是 0.000**；要到 `amp=0.033` 才 `−0.0032`、`amp=0.12` 才 `−0.0088`。
⇒ `0.3340` / `0.7404` / `−0.0994` / `0.7085` 的作废**仍然成立**，
理由仍是 R21 的泄漏 / 行号错配 / 指标混比，**不因本条而削弱或加强**。

**Status:** Accepted。

**代价与残余风险：** 特征抽取不可复现这一条**未闭环**。已定位到它存在、量化了幅度、
并证明它对本项目的裁决无实质影响；但**没有**找到原因（怀疑权重版本或模型构造差异，
但两者都不可及：npz 已丢、`.fp16.pt` 与缓存本身也不一致）。
**处置**：阶段 2 起，任何"冻结 VPB 特征"的实验**必须现算**，不得复用
`vpb_artifacts/feat` 的盘上缓存；冻结时必须连同**当次实际用的权重 sha256** 一起记录。

**Lesson（与 [[aic-size-limit-9b-params]] 同类，这是本会话第三次）：**
"这个缓存是权威的、我只需要正确地索引它"是一个**未经检验的假设**。
本轮之前，`vpb_artifacts/feat` 被当作冻结资产引用了至少四处；实际它是
**被追加污染过的、且重算不出来的**。凡是被称为"冻结"的东西，都要有一条
**能当场复算并精确命中已知读数**的验收路径，否则它只是"看起来稳定"。

---

## 2026-10-15（补 5）：随机排序探针的读数形式、最优点、与声明尺寸

Status：Accepted。包已建成并通过独立 validator（`R23_RANDPROBE_P250` / `_P500`）。

### Decision 1：读数用**逐视频精确期望**，不用计划里那个聚合闭式

计划 §主线 2 预注册的是 `S_rand(p) = F_ANCHOR·(1+d)·p/(p+d)`。
**这个式子是逐视频形式在 `F_anchor,i ≡ const` 时的特例**，一般情形下不成立。

推导：空间 bbox 逐字节复制锚点包 ⇒ `Rbar` 与现役相同、空间轴零变化；
时间选择换成与内容无关的固定伪随机抽样 ⇒ **命中集合与 IoU 取值独立**。
独立性是关键一步，它给出 `E[Rbar | 命中] = Rbar_i`，于是逐视频

    F_anchor,i = 2·Rbar_i·d/(1+d)                     （p=1 时的官方分）
    E[F_i(p)]  = F_anchor,i · (1+d) · p_i/(p_i+d)
    E[S(p)]    = mean_i { F_anchor,i · (1+d) · p_i/(p_i+d) }

官方 `F_video` 正是**视频级宏平均**（规则 §3.4），所以逐视频形式才是对的那支。

**两条直接后果：**

1. **`Rbar` 的异质性精确抵消** —— 不需要知道任何 `Rbar_i`。计划里被当作残余风险的
   "`Rbar` 抽样噪声"**不存在**。这同时退役了此前挂着的最后一条探针残余风险。
2. **`p_i` 是逐视频量、不是全局常数**：精确计数抽样给 `p_i = round(p·N_i)/N_i`，
   与 `p` 差至多 `0.5/N_i`。bench 表按逐视频形式逐条求和，并同时给出聚合形式以
   **暴露两者之差**（Jensen + 与 `F_anchor,i` 的协方差，二阶项）。

残余系统项（已界、**不可由重跑估计**，因为 `p_i` 只由 `(p, N_i)` 决定、与 salt 无关）：
`|cov(F_anchor,i, h(p_i))| ≤ sd(F_anchor)·max(0.5/N_i)·h'(p)`，`h=(1+d)p/(p+d)`；
实测量级 ≤ 0.12 分（最坏全相关），典型 ~0。主误差仍是抽样噪声，已界 **≤0.28 分**。

### Decision 2：主读数点取 **p=0.25**，不是计划里的 p=0.5/0.8

`d` 从下界 0.2103 扫到 1.0 时 S 的跨度：

| p | 0.25 | 0.30 | 0.50 | 0.80 |
|---|---|---|---|---|
| 跨度 | **8.94** | 8.69 | 6.44 | 2.41 |
| 每 0.01 的 d 分辨率 | **0.1158** | 0.1135 | 0.0869 | 0.0329 |

⇒ `p≈0.25–0.30` 是灵敏度最优点，比计划的 0.5/0.8 对严格 2–3.7 倍。
主读数用 `p=0.25`（跨度 8.94），第二点用 `p=0.50` 以构成嵌套。

**嵌套性**（种子只由 `(video_id, salt)` 派生、**不含 p** ⇒ 两次抽样共享同一条置换、
小 p 取前缀 ⇒ `kept(0.25) ⊆ kept(0.50)` 逐视频成立）给出一个**干净的 d 估计量**
`r = S(0.25)/S(0.50)`：动态范围 `0.7716 (d=0.2103) → 0.6000 (d=1.0)`，
是计划中 `S(.5)/S(.8)`（`0.889 → 0.750`）的 **2 倍**；对 `d_i` 异质性的稳健性亦约 2 倍
（`cv=1.0`、`d̄=0.2103` 处偏差 0.026 vs 0.058）。嵌套在 R15d 原版里只是隐式成立，
改一个 salt 就静默失效；本版**显式断言**（426/426 已实测零违例）。

**副产物**：比值 `r` 只含 `(p_i, d)`，**不含 `F_ANCHOR`** ⇒ 它反过来是锚点 `34.75`
的一次独立测量。两个估计量在同一份官方分上**必须自洽**，不自洽就说明读数形式有错。

### Decision 3：声明尺寸用**整字段省略**（B3 先例）

后台考证结论（只读，全库 + 全 git 历史）：

- **「缺失 `model_size_mb` 仍拿到官方分」有已证实先例**：≥5 个复赛包，其中就包括
  本探针逐字节复制 bbox 的锚点包 `LFM_V8_B3_SEMIFINAL`（34.75，manifest 无该字段）。
- **「声明 >9000 MB」零先例**：8 个声明 `9503.763` 的包**全部** `uploaded:false`、
  `score:null`。9503.763 本身是 `9,503,762,857 B / 1e6`，即那些包往 MB 字段里填的是
  **文件字节数**，不是参数量 —— 与我记忆里「9B 是参数上限、不是文件上限」这条一致。
- **平台是否校验过该字段：无记录。** 唯一的平台行为线索是静态的：三个无字段、
  教师 66.7 GB 的包照给了分。本地 `aic.contract` 报的 426 条 `model_size` error 是
  **我们自己的工具比平台严**，不是平台行为。

⇒ 缺省不写该字段，与锚点包同构，使平台侧记账与整条阶梯一致。
`--model-size-mb` 若显式给出则强制 `0 < v ≤ 9216.0`（本地口径）。

### 验证（三层，全部已过）

1. **合成数据干跑**：40 视频，嵌套断言过，比值跨度实测 `0.771548 → 0.599925`，
   与解析预期 `0.77156 / 0.60000` 一致。
2. **闭环**：给定真值 `d` 用官方口径精确模拟打分（`F_i = 2·S_IoU,i/(N_pred,i+N_gt,i)`，
   426 视频宏平均，故意用 **cv 很大的异质 `Rbar_i`** 再校准到 `F_anchor = 34.75`），
   再反解 `d`：

   | d_true | 0.21029 | 0.25 | 0.30 | 0.40 | 0.50 | 1.00 |
   |---|---|---|---|---|---|---|
   | 反解 d @p=.25 | 0.21579 | 0.26895 | 0.29719 | 0.39650 | 0.50194 | 0.99966 |
   | 比值 r 反解 | 0.23206 | 0.28620 | 0.30358 | 0.39817 | 0.50478 | 0.99940 |

   ⇒ **`Rbar` 异质性确实被抵消**（故意设成强异质仍能回真值），
   且两个估计量**互相自洽**。单点在 `d≲0.25` 处的 +0.019 偏差来自
   `p_i` 量化残差，比值式在低端更准（0.23206 vs 单点 0.21579，真值 0.21029）——
   低端应**以比值式为主读数**。
3. **真包**：`videos_full_frame_range 426/426`、`missing_bbox_source 0`、
   嵌套逐视频零违例、`bboxes` **58536/58536 逐字节与 B3 一致**、
   独立 validator 双 arm 均 `valid:true`。zip 内只放 `predictions.jsonl`，
   与 `LFM_V8_B3_SEMIFINAL.zip` 的 `namelist` 同构。

### 资产（远端 `/root/autodl-tmp/experiments/R23_RANDPROBE/`）

| arm | p 实测 | n_pred | zip sha256（前 16） |
|---|---|---|---|
| `R23_RANDPROBE_P250` | 0.24996384 | 58536 | `51910acc96f14f70` |
| `R23_RANDPROBE_P500` | 0.4999734696 | 117077 | `98862a8f78526209` |

**未上传。上传由用户执行**（DECISIONS 2026-09-26 起的一贯规矩：不由 Agent 自动提交）。

### 反向推论（记下来，防止日后误读）

若最终比值 `r ∈ [0.7715, 0.6000]`，说明锚点 `34.75` 可信；
若 `r` 落在带外，则**先怀疑锚点与真实 `p`**（`p` 用的是 `p_i` 实测均值，
不是 `keep_frac` 名义值；R22b 已证明 `p` 被**区域粒度量化**，`keep_frac → N_pred`
从来不是四舍五入关系），再怀疑读数形式。

**代价与残余风险**：消耗官方提交额度 2 次（每日 5 次，已确认非总量稀缺）。
残余系统项 ≤0.12 分不可由重跑估计，只能靠"两个估计量自洽"这条交叉检验兜底。


## 2026-10-15：R23 官方复赛回分与后续执行

用户报告已提交 R23，P250=18，P500=26（原报告精度；未独立登录平台核验）。现役最佳仍为35.52。
复赛比初赛难，禁止跨赛段比较，初赛分数不参与当前选模。
撤回“两个随机探针唯一识别 d”的强结论：在逐视频 d_i 异质时，总分只提供加权约束；
rho=p 是随机抽样的期望关系，有限样本不是恒等式，Rbar_i 也不会从宏平均中无条件消失。
18/26 只作为同赛段官方观测保留，不宣称精确密度或已证实的排序天花板。
按用户授权执行 R24：现算 VPB point/block × seeds13/17、公平 CLIP 对照、AHA 真实视觉 smoke。

## 2026-10-15：R24判决

Decision：暂停当前12帧block配方；保持现役可提交版本，不替换CLIP，不开启预留源评估。
Evidence：block−point两seed−.002173/−.001943，未通过≥.005且两seed非负并胜控制的门。
表示对比亦未稳定超越位置控制；不将公开摘要F映射为复赛分数。
AHA按有界失败协议记录缺wandb，未作视觉或质量结论。
Next：先检查时间采样分辨率的oracle损失，避免继续扫当前头；空间瓶颈仍未排除。
报告：reports/r24/summary.md。Status：Accepted。

## 2026-10-15：R24 AHA依赖修复与真实视觉通过

按用户指示安装缺失包，补齐并SHA256校验SigLIP视觉权重。wandb仅作为导入依赖，运行禁用在线日志，无需登录。
FP16已成功加载并精确验证三个评分头，但首次前向出现非有限输出；保留失败证据，不声称定位到具体层。
同脚本改为基座原生BF16后：真实公共TRAIN两不同帧与首帧重复测试通过，视觉嵌入/三个头输出均有限。
不同帧最大响应差2.05078125，重复差0；总20.81秒，峰值18,632,761,344B（17.35GiB）。
Decision：运行可用门通过，质量门未测；不替换比赛包，不扩量蒸馏。先验证离线teacher与人工标签的相关性。
限制：实际加载参数17,346,192,448B，不认定满足比赛权重上限；单帧smoke不代表视频任务质量。
完整环境、命令和失败/通过证据：reports/r24_aha_repair/summary.md。


## 2026-10-15：9B参数规则与AHA直接提交

Decision：按用户明确确认官方总参数≤9B执行，撤回按9GB文件大小否决AHA及强制量化的决策。实际文件bytes仅作工程审计。量化不减少参数数量，32B量化系统仍超限。
Evidence：R25实际合并AHA8.030B＋B3空间0.450B＝8.481B；未量化BF16完整426视频提交包已生成并验证。
Validation：原生完整token评分头、完整因果上下文；省略文本词表投影，第一块与原生整模型前向比对<1e-5。
Status：提交候选，待同赛段官方回分；R23随机18/26不能作为AHA结果，初赛分数不可参考。


## 2026-10-15：R25官方35.18不晋级，先分离位置/预算/空间

Decision：现役最佳记录35.52保留；不因AHA为8B就继续盲目扩模型或蒸馏。
Evidence：R25用户报告35.18，全帧34.75；双校验426/426通过。预算87.51%而非85%，首采样仅28条保留，中段98.77%保留，内容排序优势未证明。
Next：固定公共dev质量门与位置控制；逐视频精确预算控制；相同时间掩码空间控制。位置统计不能替代GT，也不能反推统一密度/分数天花板。
Status：Accepted；raw头诊断受AutoDL SSH关闭阻塞。详见reports/r25_score_analysis/summary.md。


## 2026-10-16：R26公共排序未建立位置控制优势，交付预算/空间隔离包

Decision：保留35.52现役；暂停直接扩展当前generic-query、full-history、等权双头配方。参数容量与文件bytes不是已证实阻塞。
Evidence：26TRAIN/14DEV F@精确.85 AHA=.290883，prefix/train_position=.297467、center=.288381，主要配对CI跨零；因果静态帧也有分数变化。整cell预算超额5870帧是实现事实，不是已证实分数损失。
Next：三个R26包已双校验，分别与正确控制做官方比较；不自动上传，不预测新分数。空间仅几何与固定mask对照，尚无GT归因。后续只在TRAIN选择去趋势/cache/query单变量。
Status：Accepted；AutoDL恢复，公共40源与raw排查完成，预留14未打开。详见reports/r26/summary.md。

## 2026-10-16：中心85+B3晋级36.04，保留B3并执行有限中心窗口对照

Decision: 新安全基线R26_POS_CENTER_EXACT085_B3；停止把当前generic-query AHA排序直接当默认时间分支。保留B3框，执行R27中心75/95及85前/后移5%四臂；不自动扩网格。
Evidence: 同预算同B3中心36.04 > AHA35.07（+0.97）；精确预算35.07 < R25整cell35.18（−0.11）；同R25 mask的B3空间35.18 > 中心框30.23（+4.95）。用户报告，平台上传字节未独立核验。
Limits: 联合指标不能拆成纯时间/空间误差份额；公共DEV不预测官方胜者；未验证统计显著；完整B3作品部署材料待闭环。
Status: Accepted，R27四包已生成并双校验，官方分待回报。

## 2026-10-16：R27扫描收口，R28转空间主体评分

用户报告R27_CENTER075_B3=35.87、R27_CENTER095_B3=35.39，较中心85+B3=36.04分别−0.17/−0.65；本地ZIP SHA一致，平台上传SHA未独立核验。停止预算/位置扫描，EARLY/LATE暂停，保留安全基线。
R28实际完成426视频234159帧空间缓存回放，最大坐标差0；x/y/全画面126/141/159视频，4913使用关键帧全ok_head，无回退。记录路径中的B3_s0头/LFM基座缺失，不能声称原视频部署通过。历史插值/加密/检测先验不重复；B3已是candidate-utility，优先主体评分、视觉表示、真实原生几何标注。
Qwen3.5-4B两分片738tensor映射完整；公共4源×3帧×2ratio工程pilot配置先冻结，独立环境，20min超时，无训练、无官方视频推理、预留14源未打开。读数及运行状态见reports/r28；公共pilot不直接晋级或预测官方分。

## 2026-10-16：R31不直接部署，R32进入提交候选

R31低维事件状态在暴露DEV相对中心+0.013904但CI跨零，且没有AIC对应特征缓存；不把公共诊断收益冒充可部署收益，也不启动大时序训练。采用远端已有AIC冻结`temporal_k0.90`生成R32，配合B3框，作为唯一新的可上传候选；R26中心85+B3仍是安全回滚包。R32已通过独立checker和semifinal contract，尚未上传、没有官方回分。详见`reports/r31/summary.md`及`submissions/R32_VT_K090_B3/manifest.json`。

## 2026-10-16：明日五包提交组合冻结

在R32之外，新增四包进入待上传队列：R33 replay单头、R34 exact-DP单头、R35 两头并集90%、R36 两头共识优先并以中心B3帧补齐精确85%。选择依据是已有AIC 426冻结缓存与明确的时间决策差异；不再做无机制的keep扫描。R35/R36同时加载两条时间头，申报0.450776179B，仍远低于9B。五包均保留B3空间，避免已被R26官方回分30.23否定的中心空间框。官方上传顺序建议：R35、R36、R34、R33、R32；R26中心85+B3保留回滚但不占明日新增名额。所有新包尚无官方分数。

## 2026-10-16：提交名额改为机制候选

停止中心位置与keep比例扫描。提交候选改为冻结视觉状态驱动的时间集合决策；共同控制为B3空间和精确85%预算。R41（内容门控弱监督先验）是主攻，R40（状态覆盖）是竞争机制，R37–R39用于区分峰值、窗口与冗余解释。若官方回分均不超过R26 36.04，则停止当前视觉特征路线。

## 2026-10-16：先做12源盲标，不扩大数据集

Decision：人工先验阶段只使用 AutoDL 已有 `QVH_NATIVE_CANDIDATES` 的12个源视频、每个30秒（约25MB），暂不下载TVSum、SumMe、VideoXum或LIVE-YT整库到电脑。时间、上下文和native构图框必须来自同一原视频，避免把不同标注政策拼成伪GT。

Evidence：当前官方最佳36.04来自中心85+B3，AHA和中心窗口未形成内容增量；历史监督/表示替换也没有稳定迁移。最有决策价值的未知项是高光是否需要过程、声音/语言，以及高光帧是否能稳定构图。12源覆盖动作结果、危险上下文、讲话信息、多人舞台、景观和普通活动，6源双人重叠，足以先做机制筛选。

Protocol：Label Studio默认任务已去掉类别、抽样原因、assignment和QVH query，只保留图像与时间。先验收两个视频导出，再完成其余10个；结果只做路线诊断，不当官方GT、不直接训练提交模型。若标注结果不能区分路线，再补同一候选清单的6源，而不是立即扩大到其他数据集。

追加裁定（2026-10-16）：人工先验协议升为 v1.1。高光判断必须经过“完整视频（含声音）→事件时间线→逐帧相对保留价值→构图可行性”的固定流程；0/1/2/3 与“不确定”按证据区分。当前任务没有目标比例字段，故自由比例 `crop_box` 只用于判断是否存在稳定构图窗口，不能训练空间回归或冒充官方 native GT。首轮 query 选项锁为“未看查询”，双人分歧保留，不先裁判成单一标签。

## 2026-10-16：标注服务常驻与 Smoke 入口

Decision：Label Studio 采用用户级 systemd 常驻，FRP 通过系统级 `frpc.service` 暴露公网 `http://47.108.220.174:3002`；本地服务仅监听 `127.0.0.1:18081`。Smoke 项目先保留 60 个任务供界面、音频播放、必填字段和导出格式验收。

Evidence：本地和公网 `/health` 均返回 HTTP 200；Label Studio 登录、项目/任务 API 和公网 JPEG 访问均已验证。完整 JSON 含 Smoke 子集，正式导入前必须避免重复。

## 2026-10-17：R39–R41不晋级，停止当前内容启发式扫参

Evidence：用户回报R41=35.19、R40=34.15、R39=34.48，相对安全基线36.04均退步。本地三ZIP SHA与打包记录一致，平台/上传SHA未独立核验。共同B3空间框沿用，但198842帧与R26的199037帧并非严格相同，不能用总分唯一归因时间排序。

Decision：保持R26中心85+B3为安全基线，停止这三种配方的超参扫描。R37/R38状态保持待回分，不把它们写成失败。优先完成已准备12源盲标，区分视觉状态变化与真正事件价值、上下文及音频依赖；不从官方总分反推测试GT，也不据此否定所有内容表征。

## 2026-10-17：标注任务内提供完整窗口播放器

原界面只有图片，任务没有MP4字段，无法在标注界面执行“先看过程再打标”。给现有任务原位补充`context_video`和`context_player`，通过HyperText内的原生HTML播放器展示30秒片段，并提供单独打开MP4的链接。补上项目的视频目录读取权限；保留H.264/AAC原件，增加VP9/Opus兼容源。公网Chromium实测两个Smoke视频均能播放、解码声音并跳至15秒，媒体请求206。原标注控件继续绑定image，v1.1判断标准不变；明确30秒窗口与原始长视频全片的区别。数据库备份保留在服务私有目录，不进Git；现有60任务ID、1份草稿和0份正式标注保持一致。后续扩展只导入未存在的300条，保留Smoke成果。验证见`reports/annotation_video_player_20261017.json`。

## 2026-10-17：标注v1.2取消不适用的必填项

用户反馈无高光仍被迫填写信息依赖，以及“未看查询”只有一个选项却必须点击。`query_relevance`属于盲标流程而非逐帧语义，保留为隐藏可选旧控件以兼容已有结果，不再要求手动确认，也不自动替标注者声称未看过查询。`needs_context`改为可选：0/1可留空，2/3请按实际信息依赖填写；缺失按适用性分层统计，不能把2/3漏填补成“当前帧足够”。允许整个窗口无高光，不强制凑出2/3。事件存在与精彩程度分开判断，普通动作仍按实际阶段/类型记录。既有1份正式标注和1份草稿保留，不回填或重写；版本和验证见`reports/annotation_schema_v12_20261017.json`。

## 2026-10-17：Smoke保留，先校准构图语义再扩标

Evidence：60份正式结果四核心/时间图片映射零错误；两源等级2/3共43帧，信息依赖完整。11帧多框呈现分别框人物/食物/字幕的模式，与一个完整保留窗口不符；自动旗标共18题，51/52另有阶段复核建议。详见`reports/r42_smoke_quality_20261017/review.md`。

Decision：保留全部原结果，针对题号修订，不要求60帧重标，不自动合并草稿/框或改标签；修订验收前不扩其余300任务。界面标签也容易诱导对象检测式画框，需强调每帧一个窗口。自由比例框始终不是赛事native crop GT。

Scope：不以正例/3级占比判错，不按摘要去冗余要求压低连续高光；自报当前帧足够不等于静态视觉可观测性已证。时间标签只做两个源的描述性诊断，未训练、无独立确认和官方收益。扩标前还应明确保留价值与固定预算摘要的语义区别，不能直接用序数等级推官方Y/keep。

## 2026-10-17：V1.3完整扩展，先纠正标签生成流程

Evidence：用户修订14道Smoke后，四核心与媒体映射零错误，自动旗标由18题缩至6道多框题。旧标准含摘要数量/去冗余暗示，可能改变逐时刻事件价值监督，不能靠要求相邻高光降级来校准。

Decision：冻结通俗V1.3标准，取消配额与去冗余要求，连续精彩可连续2/3；只对2/3填写信息依赖和一个完整保留窗口，0/1可跳过。甲新增240题，乙240题，实际导入540份任务/360图/12源；6源重叠，新增工作量均为8段30秒视频。6题旧多框补修不再阻塞扩标，不自动修改任何人工答案。

Version / scope：旧60仍为V1.2用户修订，同版本一致性主分析只用4个双方V1.3新标重叠源，另2源作版本敏感性；没有账号访问隔离，按项目分工独立完成。公网播放器、说明与提交路径通过，测试标签未保存，幂等重跑新增0题。自由比例框不当native crop GT，自报依赖不当模型上限；没有新训练、独立确认或官方涨分。证据与命令见`reports/r42_annotation_v13_20261017/summary.md`。

## 2026-10-17：R43 InternVideo2 6B 候选包完成

Decision：生成并保留R43上传候选；用户已上传并回报官方34.95。不自动发布、不把本地合同通过写成官方涨分。

Evidence：InternVideo2 Stage2 6B事件完成度时间排序在远端完成426/426。batch=88在video 379出现CUDA OOM，说明单次探测成功不等于长作业稳定；切换全新进程batch=4后无回退完成剩余视频。总推理参数6,816,814,181<=9B。426/426、199,037条预测、精确round(0.85*N)、B3框逐帧0 mismatch，独立checker与ZIP往返通过。

Artifact：`submissions/R43_IV2_EVENT_COMPLETION_20261017/R43_IV2_EVENT_COMPLETION_20261017.zip`，ZIP SHA256 `f20b91eadd89c2e6a73fd249a0dcba64078203008f80edf5b9a8902947074cb1`；predictions SHA256 `925eb721aed41e320ea8c3217b402df41a8bdbb03adfce8c66888103e37c5525`。用户回报官方分数34.95，平台和上传字节未独立核验。

Scope：该结果只支持路线可运行并满足提交合同；尚不支持InternVideo2带来官方收益。若官方分低于R26=36.04，停止当前提示/融合扫参；若明显超过，再在独立证据和资源约束下分析事件语义增量。

## 2026-10-17：R43回分与人工标签改变下一路线

Evidence：R43官方分34.95，相对R26中心85+B3=36.04下降1.09。426视频、199037帧、round(0.85*N)预算和逐帧B3框均相同；R43平均替换R26约53帧/视频，平均选帧交集88.7%。因此负差来自时间集合排序的净损失，不能归咎空间分支或预算偏差。

人工标签质量审计：540/540正式标注和媒体合同通过，但项目1/2的540条都由同一Label Studio账号完成。四个V1.3重叠源120帧的高光等级完全一致率28.3%，相差不超过1级62.5%；不能将重复结果当双人GT。高光2/3中279/340处于“进行中”，88/340明确需要前后过程或声音/语言。标签应作机制诊断，暂不直接硬标签训练。

Decision：关闭R43事件完成度、提示词和0.75/0.25融合扫参。下一主攻是小型、可识别的三路诊断：

1. 在冻结视觉特征上分别预测“进行中/结果”与通用高光，检查R43是否把目标错误收窄为结果；
2. 加入音频/语言可靠性门控和前后状态差分，使用源级留一验证，不把自报标签当官方F；
3. 用真正第二标注者对四个V1.3重叠源或最小分歧集盲复核，先修正标签可靠性再训练。

Promotion gate：只有源级留一验证中时间目标和模态增量均稳定、且在冻结合同上超过R26的公共诊断门，才生成下一上传候选；否则保留R26安全包，停止当前大模型替换路线。完整质量报告见`reports/r42_annotation_v13_full_20261017/summary.md`。

## 2026-10-17：R44先做可观测性与联合效用诊断

Decision：不重做540帧标注，也不把同一账号的重复结果当双人GT。先在12个原始源上做源级留一的低维诊断，目标分开为通用价值、进行中/结果阶段和声音/前后过程依赖；中心位置作为控制，依次加入静态、局部状态差分和音频可靠性门控。

Evidence：R43只改变时间排序却比R26低1.09；重叠标签等级二次加权κ=0.185、信息依赖κ=-0.015。直接把事件完成度替换中心的必要条件未成立。R44配置与资源/停止门见`configs/R44_OBSERVABILITY_UTILITY_20261017.json`，综合分析见`reports/r43_r42_postmortem_and_r44_plan_20261017.md`。

Promotion：只有源外配对效用稳定为正、未知字段保持mask、B3空间和精确85%预算固定、总推理参数<=9B时，才构造新候选。当前未启动训练、未使用官方标签、未生成R44提交包。

R44_DIAG_001结果：12源留一的简单读出中，局部前后状态描述子使通用等级MAE从1.0668降到0.9695，9/12源改善；进行中/结果和前后/声音依赖目标没有改善，低级音频特征反而变差。该结果只支持继续做低容量局部状态残差，不支持音频模型、事件完成度替换或新提交。详见`reports/r44_diag_001_observability_20261017.json`。

## 2026-10-17：R44 以 R26 为唯一父包继续迭代

**Decision**：接受 `R26_POS_CENTER_EXACT085_B3`（官方 36.04）为 R44 的父包；只改时间排序，不改 B3 空间框、不改每视频精确 85% 预算。采用人工标签校准的低容量局部状态残差 `eta=.20`，而不是重新部署 R43 的 6B 事件完成度模型或高维 embedding 读出。

**Evidence**：源级公共诊断中局部+IV2 等级 MAE 1.0702→0.9992，固定集合代理增量 +.02086，但只有 9/18 视图为正，标签来自同一账号且不是官方 GT。包级 426/426、199037 帧、B3 框逐项不变、独立 checker/ZIP 回归通过。新包尚未官方上传，收益仍未知。

**Fallback / stop**：R26 永久保留为回退。R44 官方回分不超过 36.04 时停止继续扫残差系数；超过时才继续做上传 SHA 绑定的确认与误差分析。


## 2026-10-17：R44 +0.05 不构成稳定突破

Evidence：R44=`36.09`，R26=`36.04`；R44 只替换 6352/199037 帧，B3 框逐帧一致。公共人工标签近邻交换 null 的 `P(ΔScore≥+0.05)=0.2369`，且不是官方 GT。

Decision：把 R44 记录为弱实打实信号与随机波动不可识别的候选；保留为当前最高用户回分包，但继续以 R26 作为安全基线。停止 eta/alpha 扫描和同类微调。

Next mechanism：`R45_BOUNDARY_GUARDED_RESIDUAL_B3` 固定边界保护规则，只允许 R44 相对 R26 的变更在中心窗口两端各 5% 边界带内发生，冻结中间区域。若该机制不能显示可解释收益，转向同域空间 utility readout 或真实音频/事件阶段信息。


## 2026-10-17：R45 采用已生成集合的边界保护过滤

Decision：R45 不重新拟合或重算 R44，直接过滤其已生成的换帧集合。保留 R26 中心窗口两端各 5% 边界带内的 5942 次交换，冻结其它 410 次；这样可以单独检验边界局部性，且不会引入新模型、隐含扫参或新的观测。

Evidence：426/426 覆盖、199037 帧、精确预算、B3 公共帧框一致，ZIP 往返通过；SHA 见 `reports/r45_boundary_guarded_residual_20261017/summary.md`。

Risk：R44 的新增帧有一部分在窗口外，过滤后的 R45 仍保留 93.5% 的 R44 换帧，若官方无收益，不能单独排除残差方向，只能排除“仅靠 5% 边界保护即可修复”的解释。

## 2026-10-18：下一主攻固定为高频状态边界决策

Decision：在没有精确V8 B3 head/base的条件下，暂停空间utility readout，转向4fps局部状态残差。生成两个机制臂：D3仅用子秒局部状态，D4使用同一状态加冻结IV2事件锚；二者都固定`eta=.20`、Ridge `alpha=10`、精确`round(.85*N)`，并各自派生R26两侧5%边界保护包。

Evidence：12源源级留一诊断显示D3总体回归MAE略差于D2，但固定集合代理+0.02283、11/18视图为正；D4回归MAE最佳但集合代理+0.01306。边界代理只在3/12源为正，故边界是风险控制而非已证收益。语义×状态交互不改变集合，停止。

Promotion：候选包必须426/426、199037条、每视频精确85%、B3框逐项等同R26、总参数450283377并通过ZIP往返审计。官方回分若不超过R26/R44波动范围，停止高频残差微调，下一资源投入精确B3 utility恢复或真实音频/事件结构监督。

## 2026-10-18：TimeLens2只保留为教师诊断

Decision：TimeLens2-8B的公开区间代理有明确增量，但精确参数计数为8,767,123,696；与B3的450,283,377合计9,217,407,073，超过官方9B。停止把TimeLens2和B3组合成提交包，也停止不必要的426全量推理。

Next：若要继续，必须先证明合规学生能从公开合法源的区间监督学到可迁移信号，或接受无参数中心空间的性能代价；不得把缓存的TimeLens2区间当作绕开参数限制的办法。

## 2026-10-18：V9 vision-only空间使R47恢复为合规候选

前一条结论只适用于TimeLens2+完整450M B3，参数为9,217,407,073，仍然停止。随后核对远端`LFM_V9_B3_VISIONONLY_SEMIFINAL`完整父集与`vision_only_model/MANIFEST.json`：实际加载的SigLIP2 vision tower、B3 head和YuNet合计87,407,729参数，且已有426视频全帧预测。TimeLens2与该组件合计8,854,531,425，满足当前9B口径。

Decision：保留`R47_TIMELENS2_V9_B3`作为明日最高风险候选，保留`R47_V9_B3_CENTER85`作为纯空间替换对照，保留`R47_TIMELENS2_CENTER_EXACT085`作为严格参数合规的空间降级对照。三者都不宣称官方增益；V9候选的资格依赖已审计vision-only artifact确实是部署空间组件，manifest明确该条件。

Evidence：TimeLens2公共源18视图代理相对中心+0.01691；完整分片覆盖426/426且无fallback；V9父集覆盖426/426、234159原始帧。不存在官方GT回分，不能把代理提升升级为比赛收益。
## 2026-10-18：R48只在低频—高频共识时应用时间残差

Decision：D3的4fps局部状态虽有集合代理信号，但回归MAE略差于D2，说明无条件高频修正会混入镜头变化和噪声。预先固定R48规则：仅当1fps局部读出与4fps局部读出位于同一视频内中位数同侧时，才应用`eta=.20`高频残差；不扫描阈值、eta或alpha。

Evidence：同一12源、540条人工标签的源级留一代理中，D3 `+0.02283`、R48共识门控 `+0.02360`，均为9/18视图正；起始变化门控仅`+0.01376`、7/18。该结果是机制诊断，不是独立确认，也没有官方联合GT。

Action：完成R48的426视频部署包后，只做结构审计和与R26/R46的逐帧差异统计。R48若进入明日额度，只用于检验“高频信号需可靠性门控”的命题；不把公共代理小增量当成涨分承诺。若官方回分未超过现役波动范围，停止时间残差微扰，转向恢复原生B3 utility或真实事件/音频监督。

**Package result**：R48全量输出通过426/426、199037、精确预算、R26 B3框逐项继承和ZIP往返。完整门控包相对R26在415个视频改变集合，平均交换12.65帧；固定5%边界包保留5273次边界交换。两包均只作明日可选验证，未上传、无官方分数；不把R48与D3同时解释为两个独立收益。

## 2026-10-18：时间残差扩展止损与下一优先级

**Decision**：停止R49事件阶段、R50全片轨迹、R51持续性门控和R52光流统计四条时间微扩展；保留D3/R46为唯一有明确公共集合信号的时间路线。下一优先级是TimeLens2事件区间与合规V9 vision-only空间分支的提交级检验，并继续恢复/验证空间utility资产。

**Evidence**：统一12源、重复标签按源/时刻聚合的中心soft代理为`0.716772`，D3为`0.739795`（`+0.023023`，8/12源正）。R49 `+0.002913`，R50完整 `+0.020013`、去位置 `+0.022113`，R51 `+0.006539`，R52 `+0.010928`。R51的反例尤其关键：对称持续运动门控丢失有效单侧启动/结果帧。所有结果仅为同一批人工诊断，不能升级为官方涨分或独立确认。

**Risk / scope**：D3的公共增益可能仍来自12源选择、同账号标签和少量交换；R44官方只比R26高0.05，未证明残差可靠。R50全片特征含可解释但未必迁移的源内轨迹，已因低于D3停止。R48若审计通过最多占一个明日提交名额，不能与D3同时被称为独立突破。

## 2026-10-18：TimeLens2 改为中心锚定 residual

**Decision**：保留 TimeLens2 事件区间作为可观测语义证据，但不再把区间内帧硬置于中心先验之上。R54 使用固定 `rank(center)+.20*(rank(interval)-.5)`，与R47共享同一响应、V9空间父集和精确85%预算。

**Evidence**：12源18视图公共代理：中心 `0.704417`，R47硬区间 `0.721327`（`+0.016911`，6/18正），R54 `0.729414`（`+0.024997`，10/18正）；二值代理R54 `+0.032869`。这是公开诊断，不是官方联合GT。

**Action**：R54形成包级候选，优先级高于R47硬区间；只试一至两包时建议R46 + R54。R47硬区间保留为机制对照，不把公共代理增量写成官方涨分承诺。

## 2026-10-18：低维音频与画质条件化未过机制门

R53_AUDIO的低维音频统计在源级留一中仅`+0.00749`，RGB+audio `+0.01162`，低于D3；R53质量交互从D2的`+0.02235`降至`+0.01498`，预测阶段相对D2只增加`+0.00238`且CI跨零。停止这些低容量微扩展，不据此排除真正的ASR/音频语义模型；若继续音频路线，必须引入能解析语言/声学事件的新输入并重新做独立对照。

## 2026-10-18：R54正、R46负改变主路线

**Decision**：停止4fps灰度局部状态残差族，把R54的TimeLens2语义区间和视频级窗口相位作为主攻；保留V9中心空间控制与R56端点性作为可解释分离候选。

**Evidence**：R46=35.88，相对R26=36.04为-.16，且只改变时间集合；R54=36.22，相对R26为+.18，但同时改变V9空间框。R54换帧主要删除早端、增加晚端，425/426视频近似连续窗口，说明有效信息更接近视频级相位修正。完整包差异和SHA见`reports/r54_r46_official_postmortem_20261018/summary.md`。

**Uncertainty**：R54的+.18不能分解为TimeLens、V9和交互；不能据此宣称TimeLens单独贡献+.18。R47_V9_B3_CENTER85是最高价值隔离包，R54与它共享V9框后可检验时间部分。

**Action**：启动R55_DENSE，将TimeLens2输入由2fps提升到4fps；同时保留R56 endpoint residual作为不扫参数的结构控制。若dense候选高于R54，转向多提示/多观测联合；若低于R54，停止简单加密采样，转向显式窗口相位和多提示事件边界。

## 2026-10-18：增加固定后段相位竞争控制

R54的选集变化几乎是连续窗口后移，且公开样本固定后移2秒的 proxy 高于R54。增加 `R57_FIXED_LATE2S_V9_B3` 作为单一结构控制，使用V9空间、精确85%预算和固定2秒偏移。若R57接近/超过R54，主线转为视频级相位预测；若R57明显低于R54，保留TimeLens区间语义的必要性。该判断需要官方回分，不能由公共proxy直接推出。

## 2026-10-18：拒绝窄结果prompt

TimeLens2第二个“窄事件/结果”提示词在12个公开片段上低于generic prompt（`.720168` vs `.721327`，4/18正），停止该提示词全量推理。后续多提示必须先通过同一公共18视图的结构门，再消耗426全量算力。


## 2026-10-18：允许大幅波动的主攻路线

用户要求优先验证可能产生数分波动的结构性方案。保留R26作为安全锚、R54作为已核实最高分；提交探索臂改为4fps TimeLens端点集成（R59）和空间分歧/稳定性极端对照（R63/R64）。这些臂改变的是观测粒度或联合决策，区别于已停止的低级RGB残差与音频微交换。

**判别预测**：若R59高于R54，说明2fps确实遗漏事件边界；若R59下降，停止简单加密采样。若R63上升而R64下降，空间分歧可能标记有效内容转移；若R64上升，V9分歧更像构图不稳定；两者都下降则停止该联合启发式。

**更正（2026-10-18，过程记录）**：R57=`34.76`、R59=`35.04`均低于R54=`36.22`，分别反驳全局后移和低分辨率4fps端点替换。随后发现原R63/R64对约90%的几何平值使用stable rank，按帧号注入了位置偏置，故不再把它们当有效机制实验。过程中的高风险候选曾改为`R67_JOINT_V9_DISAGREEMENT_TIEAVG_G1`：center/interval保持R54 stable rank，仅geometry disagreement使用tie-aware平均秩并过滤`<=1e-6`噪声；gamma=0精确复现R54，gamma=1仅交换2427帧、均值位置`.519774`。其后官方回分为`36.16`，低于R54，R67归档为负结果；R63/R64不上传。最终结构性高风险候选见下方R69发行决定。

旧FPS错误包R59/R60及浮点边界包R61/R62不具备实验效度，已标记作废。


**最后名额复核（2026-10-18）**：R67已回分`36.16`，低于R54 `0.06`，因此无标签V9/R26空间分歧不能继续作为高光价值信号。R63/R64的stable-tie实现无效，旧R68变体也不占名额。若保留一个高风险试验，改用R69“TimeLens2完整视频直接构图框 + R54时间集合”：它改变空间决策路径，仍严格`<=9B`且不使用测试标签；只在完整解析和合同审计通过后上传。


**R69发行决定（2026-10-18）**：直接 TimeLens2 空间 query 全量 426/426 解析成功。最终发行包为 `R69_TIMELENS_DIRECT_CROP_V1`：保留 R54 时间集合和精确 85% 预算，直接采用完整视频 query 的目标比例 crop（每视频一个全局框），不采用旧的 `CENTER_MAXWIN` 中心投影变体。包为 426/426 视频、199037 条预测、8,767,123,696 参数、0 fallback、0 帧集合差异、129443 个选中帧框替换；完整 checker/contract/ZIP 回归通过。预测 SHA=`aad8d3455490c2d3d43d42bf98c93117b405993452f985fb840126109452dfcf`，ZIP SHA=`f5d6dcadbd511f286c98360a72cfb568146056eb77d8cee6d62194b9e7839871`。官方分数待上传后回传。R54/R26 保持回退，R63/R64/R68 及旧 R69_CENTER_MAXWIN 变体不上传。

## 2026-10-18：R70 候选效用读出判别实验

**Decision**：在R69直接构图包之外，启动R70固定候选集合的utility readout实验。它针对“视觉信息存在、B3读出未利用”的假设，不改变时间集合，不把缺失的B3 checkpoint身份补写成可用。

**Evidence**：R46公开native crop审计在同一129候选上观察到稳定oracle gap（LIVE confirmation约`.247`、LIVE dev约`.234`、RV约`.162`），说明候选内仍有可达空间效用；R67仅用空间分歧改变时间选帧后从36.22降至36.16，关闭了无标签分歧启发式。两项证据不能把oracle gap折算官方分数，但共同支持先测候选读出。

**Protocol**：固定V9 vision-only tower、processor、129 legal windows和目标比例；RV/LIVE native crop表按source划分训练与留出；输入沿用V9的2305维候选特征，GPU训练小head，source-macro IoU分别比较RV、LIVE和center。两个留出池同时过门后才重提取426并制作空间单变量包；否则关闭R70。

## 2026-10-18: R70 inference resource path and package

**Decision:** use one process with GPU-batched candidate pooling/head and one bounded CPU prefetch worker for R70 official inference. Do not use four parallel CPU-bound processes.

**Evidence:** the original path copied each grid to CPU, rebuilt 129 masks per frame, and called the head per frame; the repaired path is numerically identical on controlled grids, completed 426 videos with zero fallback, and reached 97–100% GPU utilization samples. Full output took 174.3s. The resulting R70 package keeps R54 time selection and changes only spatial boxes. Official score is unknown until upload.

**Action:** retain `submissions/R70_SPATIAL_UTILITY_R54TIME_BATCH32/` as the next spatial candidate; do not upload automatically. If the score falls, interpret that as public native-crop readout transfer failure, not as a throughput failure.

## 2026-10-18：R69/R71/R72 决策

**Decision 1 — 分段构图进入高风险候选，保持单包边界。** R69 分段响应在真实高光 run 上确实产生时间变化（188 个视频多 crop），且只改变空间框、时间集合完全继承 R54；因此比继续无标签空间分歧更有机制针对性。它尚无 native 联合 GT 或官方分数，不能把响应变化写成 IoU 收益。上传前统一 canonical 目录 `R69_SEGMENTED_CROP_R54TIME`、ZIP 内文件、预测 SHA 和 contract，避免与 `R69_SEGMENTED_CROP_R54_TIME` 重复命名包混淆。

**Decision 2 — 逐视频预算不升级主线。** 公共 10 视窗 soft 略正而 binary 略负，且只改分母/帧集合；证据不足以区分密度预算收益与标注代理噪声。保留 R71 为最后风险备选，不扫 floor/cap/slope/median，不替代 R54/R26。

**Decision 3 — 原始 B3 恢复路线关闭。** 当前库存没有原始 head/base 的可追溯证据；V7 不能重标为 B3。缓存 B3 预测只支持历史包复现。若未来重训，必须新建 checkpoint/实验 ID 并报告训练数据、seed、参数；不能宣称“恢复原 B3”。TimeLens2+B3 组合超过 9B，禁止形成最终提交。

## 2026-10-18：R69 门控与 R70 topsoft 联合路线

**Decision 1 — 明日分段构图使用稳定性门控包。** R69 分段响应的无门控版本有大段间跳变；固定 `0.12/0.12` run 级门控将总变差降低 46.5%，只回退 52/427 runs。该结构控制优于继续扫 segment 数或 context；官方回分前不再改门限。

**Decision 2 — 新空间目标保留 topsoft。** 在相同公开 source-disjoint split 上，topsoft 同时保持 RV/LIVE 正增量，平均略高于 MSE；pairwise/listwise 没有更强。先保留 topsoft 单变量空间包，等官方回分，不把微小 public delta 写成比赛收益。

**Decision 3 — 启动空间效用驱动时间重排。** 用 topsoft top-candidate utility 作为预期 crop IoU proxy，以固定乘法规则加入 R54 时间价值，构造单一联合候选。它与旧 R67 空间分歧不同：使用有 native crop 监督的候选 utility，且不靠空间框差异注入位置秩。官方回分将决定是否继续 utility 校准。

## 2026-10-18：联合 utility 的两种决策形式

保留两个可审查的联合形式：R72 的 additive `.20` residual 作为 R54 兼容控制，R70 topsoft 的乘法 `T*(.5+.5U)` 作为 expected-utility 结构候选。两者都不扫权重；若官方只能测试一个，优先保守 R72 以降低对 R54 的破坏，若目标是寻找结构性突破则优先乘法候选。两包均无官方回分，暂不替代用户指定的明日 2/3/5。

## 2026-10-18：空间效用时间重排的识别修正

**Decision**：R70/R72/R73 第一轮路线尚未闭环。R70 `max_j utility` 代表最佳合法候选，而R72/R73最终输出的是V9 B3框；二者在官方8848关键帧上均值相差-.03886，约43.5%关键帧不近似相同。因此不能把R72/R73称为“最终空间效用驱动时间”的严格实现。

**Evidence**：公开168源诊断中，`max_score-center_score`与实际边际空间增益Spearman=.377，`max_score`仅=.128；但绝对选中IoU相关性相反，说明边际目标和绝对目标不能混写。R73增量utility中44.7%关键帧为零平值，stable rank把帧号写进决策；R74 midrank修复后swaps 5004→2418。

**Action**：未来联合时间包优先使用部署框对齐的`output utility`，再用乘法期望效用形式（R77）作为结构候选；不要在官方回分前扫eta/底座/插值。下一训练优先在公开native crop上直接监督V9部署框IoU，而不是继续预测129候选最大值。R76/R77均为未上传诊断包，明日2/3/5保持冻结。

## 2026-10-19：直接部署框 utility 的判定

**Decision**：部署框 native IoU 直接读出路线完成公共诊断并生成R81，但不把公共提升写成官方涨分。50,561参数的topsoft-logits+输出位置头在156源dev上row Spearman=.4747，明显优于原始候选分数的直接排序；LIVE source-macro bootstrap仍跨零，停止继续损失/宽度/系数扫描。

**Artifact rule**：可用A0 surrogate与历史B3/V9输出不一致，不能用它重跑官方空间框。R81只读取已审计V9 B3全帧框和已完成的R70 topsoft关键帧logits，因此保留输出身份；训练目标的B3 logits来源和SHA必须继续绑定。若该身份审计失败，R81降级为诊断，不上传。

**Action**：R81作为下一结构候选保留，明日2/3/5不变；官方回分后最多用一个额度检验R81，不再做无机制扫描。


## 2026-10-19：R82 不确定性惩罚进入候选

R81 的直接部署框 IoU 目标在公共外部源上有可迁移排序信号，但单头 source-macro 只有 `.16869`。五头集成的 `mean-.5*std` 在同一外部源上达到 `.19120`，源级 paired bootstrap 95% CI 不跨零；这支持“域转移时高估风险”这一竞争解释。

这个 beta 是在外部结果后比较的，所以不把 R82 记为科学确认，只把它作为高风险上传候选。R82 的空间框逐帧继承 R54/V9 B3，参数 `8.856303335B` 合规，checker 和 ZIP 通过。下一步只允许一次官方验证；R54=36.22 保持安全锚。

## 2026-10-19：本地验证不作为官方分数代理

**Decision**：将本地验证分为合同检查、同目标机制诊断和官方确认三层；不再用公共代理的正负或大小直接排序 AIC 提交包。

**Evidence**：R70=36.42 与公共 native crop utility 同向；R71=36.43 而十个公共视窗 binary 代理为负；R82=36.20 而外部源级 utility 排序 bootstrap 为正；R69=24.95 而平滑性指标改善 46.5%。这四次回分覆盖了代理同向、冲突和反向三种情况。完整表见 `reports/r83_official_local_validity_20261019/summary.md`。

**Implication**：R70 保留为一次空间读出正证据，R71 保留为一次预算机制正证据；R82 的直接 utility 时间重排关闭，R69 的稳定性作为评分目标关闭。历史 PHD2 代理曾与平台脱钩，因此不能仅凭 PHD2 AP 选择新包。

## 2026-10-19：启动低容量视频级政策路线

**Hypothesis**：约 50–150M 的模型若在公开 PHD2/GIF 同源标签上学习视频级事件中心、宽度和边界政策，可能比逐帧大模型排序更接近官方评测偏好。

**Experiment boundary**：只用公开训练标签，按源视频留出；不读取官方测试标签、官方回分或由其反推信息。输入为全片粗粒度视觉/运动统计，输出一个归一化 interval policy，再生成固定 85% 集合；空间框固定继承 R54/V9 B3。

**Falsifier**：若源级留出不能同时超过中心 85% 和现有 PHD2 frame-rank，或只提升 AP 而集合效用不升，则关闭路线，不扩大数据、不扫区间公式。若通过，才制作一个单变量官方包。

## 2026-10-19：R83 interval policy 形式关闭

R83 interval policy 在 5 折源级 PHD2 留出上没有超过 center（`0.025424` vs `0.025424`），逐帧 BCE 反而为 `0.025281`。由于 8 槽只产生两个 7/8 连续窗口，这个 probe 只能关闭该受限形式，不能声称所有 100M 小模型路线失败。

基于 R70=36.42 与公开 RV/LIVE native crop 同向、R82=36.20 与外部 utility 反向，下一主攻改为低容量空间策略：在 RetargetVid/LIVE native crop 上训练连续坐标或候选集合读出，固定 R54 时间集合；两个 source-disjoint 留出池同时超过 R70 topsoft 才进入官方包。

## 2026-10-19：R84 低容量空间读出进入高风险候选

R83 interval policy 在 PHD2 上退化为中心后，R84 改用有 native crop 监督的空间坐标一致性读出。它在公开 LIVE/RV 留出上有 `+0.02547` source-macro 增量，但低于 R70 的 `+0.07006`；因此不替代 R70，只保留一个结构不同的官方验证候选。

R84 包使用 TimeLens2 + V9 vision tower + 623,361 参数 head，总计 `8.853590257B`，不使用 YuNet；426/426、199037、0 fallback，checker/contract/ZIP 往返通过。ZIP SHA=`526d3e0b41c329806c7746e0f7c1404b0ec8d3d2470495e8470c0f25983934f4`，尚未上传。若 R84 官方明显低于 R70，关闭坐标一致性空间读出；若接近或超过 R70，只允许研究固定域门控，不扫融合权重。

## 2026-10-19：R85——转向PHD/GIF目标域迁移

**Decision**：不再把“直接换高光模型”当路线。下一主攻是用PHD/GIF源级监督适配已有AHA，并以HL-CLIP式saliency pooling作为外部结构对照；第一阶段固定R54/R70空间输出，只判断时间策略是否获得目标域集合效用。

**Evidence**：公开榜单182条记录，≥40分42条、≥45分4条、median 36.44；R71=36.43约第92名。历史R25 AHA零样本35.18、R43 InternVideo2事件完成度34.95、QV跨域和MrHiSum迁移负结果只否定零样本或监督错配。PHD/GIF目标监督仍未被AHA/HL-CLIP适配使用。

**Reversal**：目标域适配在源级留出集合效用上不能超过PHD2 frame-rank和中心控制，或提升完全由位置先验产生，则关闭该迁移路线。没有真实负视频池之前，不训练或提交empty-video gate。
# 2026-10-19：R86/R87 路线决策

**R86**：保留为本轮新的可上传候选。它使用 T5_CROPHEAD_V3 的公开 native-crop 源外确认结果，迁移到当前 V9 观察点，固定 R54 时间集合和参数总量；不把公共 IoU 增量换算成官方分数承诺。LIVE confirm3 `+0.0062447`（95% CI `[+0.0034429,+0.0092864]`），RetargetVid confirm2 `+0.0148418`（95% CI `[+0.0096558,+0.0200412]`）。

**R87**：先修 AHA 连续帧运行时合同，再谈 PHD/GIF 适配。AHA 基座和 adapter 已加载，但当前克隆环境在第二帧开始产生 NaN；静态 cache、SinkCache 和 EOS 过渡均未解除。不能把此结果当作目标域迁移负结果，也不制作官方包。

## 2026-10-19：低层构图先验保留为高风险空间候选，低层 bank 不替代语义时间模型

R89 的公开 native-crop Confirm 源宏增量为 +0.004104，开发增量为 +0.002497，两个留出集同向，因此允许生成官方候选；增量不足以折算为官方涨分。当前 1 秒差分消融没有增益，不能声称运动模式已验证。

R90 的 TVSum 源级 top85 代理显示 32D motion/quality/composition bank 比静态 ResNet 低 0.01002，加入短差分后仍低 0.00971；静态+bank 与静态等价。决定暂停“低层 bank 直接替代时间模型”，下一轮只研究静态语义的低维事件状态、镜头结构和可迁移的时序残差。

R91 作为独立高风险候选保留：冻结 ResNet18+TVSum 连续读出（总系统 98,584,754 参数）与审计过的 V9/B3 空间全帧框组合，精确继承 R54 每视频预算。它是唯一能直接验证“100M 轻量语义可否替代 TimeLens”的当前包；没有官方回分前不把 TVSum 代理升级成比赛收益。
# 2026-10-19：PDF 传统处理方案的路线裁定（R92）

**决定**：保留“空间形状多视图 + 目标域时间监督”作为新机制实验；不重复 R90 的低维低层 bank、未配准帧差和标量拼接。首个版本固定 R54/R70 空间输出与视频预算，只改变时间表征和读出。

**依据**：R89 在公开 native crop 上只有约 `+0.0041` IoU，说明低层构图有小幅信号但不能折算官方 F；R90 的 32D bank 及简单时序统计显著弱于静态 ResNet，关闭的是标量 bank 路线。PDF 提出的图像图、多视图 gated fusion 和配准运动残差尚未在本项目测试。

**判别实验**：R92-A RGB 基线、R92-E RGB+低频+高频+配准残差 gated fusion、R92-F 运动视图源内时间置换；PHD2/GIF 源级留出，真实 GIF 区间 union，隔离 `gif_hard` 占位字段。E 相对 A 的源级集合效用需达到 `+0.005` 且 bootstrap 95% CI 下界大于 0，F 的增量需消失，才允许做一个固定空间的官方时间候选。

**限制**：外部视频质量/显著性论文的结果不能直接换算 AIC 分数；没有官方联合 GT，不能声明已突破40分。ROI utility 暂不用于时间排序，R82 的官方负回分保留为警告。

**R92 回决**：实际四视图 gated fusion 没有超过 RGB，运动视图时间置换也没有下降；关闭“冻结 ResNet18 + PHD2/GIF 8帧弱标签 + 低/高/配准残差 gated fusion”这一具体组合，不生成提交包。R89 的低层空间先验不受此回决影响。

**R93 决定**：将研究转向可观测信息路径。4fps TimeLens2 dense intervals 改变了事件边界观察，固定端点 residual 生成一个非重复官方候选；官方回分前冻结规则，不扫 eta/prompt。

## 2026-10-19：研究1 PDF H1/H2 判别结果

**决定**：不把 U²-Netp 或当前片段内 shot 结构制作成官方提交包。两项新增通路均未达到预注册独立增量门，继续扫阈值、融合权重或 U² 参数没有机制依据。

**证据**：R94 LIVE dev U²−同容量零输入 control `−0.000463`，95% CI `[−0.003289,+0.002383]`；历史 confirm3 `+0.000834`，CI 跨零。R95 448 源 shot−control `−0.000364`，95% CI `[−0.002038,+0.001271]`。R94 control 相对 H0 的 `+0.007883` 说明小候选读出仍有信号，但不能归给主体模型。

**批判边界**：PDF 正确地把前景/shot 特征视为待验证信息，且不把公开 crop IoU 换算成 AIC Score；但 100M 方案若依赖缓存的 V9 空间链仍不是完整端到端参数账本，R95 的 embedding 95 分位也不是完整视频真实切镜。下一路线改为目标域事件状态监督与完整上下文，固定 R54/R70 空间，先建立源级联合集合效用门。


## 2026-10-19：R96 后续主攻冻结

R94 U²-Netp 与 R95 shot 代理均未过独立增量门；R96 在人工诊断标签上观察到短上下文 AP 方向性增量，但源级 CI 跨零。下一实验固定 R54/R70 空间与预算，使用 PHD/GIF 源级区间及 R42 状态/依赖掩码训练一个小双向上下文状态头。未标注帧不写硬负，R42 不当官方GT；必须在源级固定预算集合 F 同时超过中心和 R54 控制后才允许制作候选。


## 2026-10-19：R97 关闭区间状态辅助

R97 的公开 GIF 区间前/中/后辅助头在固定85%集合F上相对同结构控制为 -0.000780，CI跨零且点估计退步。关闭这条具体监督，不扫状态权重。R96 的短上下文方向性证据仍保留；若继续，必须更换可观测监督/联合集合目标，而不是在同一8点区间状态上继续训练。

## 2026-10-20：三路线判定与提交策略

**Decision**：R103 当前邻帧 V9 token 读出关闭；R102 AIKU transfer 保留为后续目标兼容 temporal student 的弱正向初始化；R104/R105 AViMoS 纯视觉 saliency 是唯一显示目标域 native crop 增量的新增路线，但当前公开 blend 存在明显源级过拟合，暂不把未经完整官方推理复核的 saliency blend 放进主提交。

**Evidence**：R103 actual `0.5677629 < 0.5684209`；R102 `+0.003413`；R105 actual `+0.0035215`、permuted `0.5734110`，同时 tune/hold blend 由 `0.5885/0.5536` 暴露迁移不稳。

**Submission set**：截止前优先保留三份完整合法包 `R70_TOPSOFT_SPATIAL_R54TIME`（官方已知36.42）、`R71_ADAPTIVE_BUDGET_R54_V9`（官方已知36.43）和 `R89_LOWLEVEL_COMPOSITION_R54TIME_B3`（native crop 两个留出池同向、ZIP/checker通过）。R93 作为可替换的观测密度风险包；不把本地代理提升写成官方涨分。
# 2026-10-20：批准三条跨采样率事件拓扑提交候选

R106/R107/R108 是新的解码机制，不是已有 R93/R88/R91 的重打包。它们把 TimeLens 区间从二值保留掩码改成边界壳层、生命周期相位和跨采样率不一致性，固定空间与预算用于官方可归因验证。R107 的公开机制诊断最强，R108 的假设最激进，R106 作为稳健壳层对照；三包均未上传、官方分数未知。
# 2026-10-20：R107/R91 回分后的提交决策

R107=36.43与R71持平，说明当前事件生命周期壳层在官方任务上没有可识别收益；R106/R108与其共享观测合同，默认降级。R91=33.87，关闭TVSum摘要监督+ResNet18+1秒插值的轻量时间路线。下一批未知包优先R93（新观察密度）、R88（修正后的T5/B3空间合同）、R89（两个公开留出池同向的低层构图）；若只剩一个极端风险名额，选R108。
# 2026-10-20：四次官方回分后的路线切换

R107的36.43证明生命周期解码最多持平当前最好包；R108的35.62关闭采样率分歧选帧；R93的36.32说明当前低分辨率4fps端点路线只有有限收益；R91的33.87关闭TVSum+ResNet18轻量时间迁移。时间路线回到连续中心锚，下一预算优先R88修正空间读出和R89低层构图；不再继续R106/R108同族公式或TVSum小模型。

## 2026-10-20：DINOv3 v1 运行时合同

**Decision**：按 `dinov3_video_highlight_training_plan.md` 先执行冻结 DINOv3 的匹配表征实验；时间 T-0/T-1 与空间 S-0 分开，不在首轮同时更换时间头、候选框、预算和后处理。官方 DINOv3 源码不直接 vendoring，运行时接收显式源码路径并把 commit 写入实验 manifest。

**Evidence**：本地 ViT-B/16 权重严格加载，参数 `85,669,632`，CLS/4 storage/patch 输出合同符合官方模型；公共 QVHighlights20 smoke 的 frame ID、PTS、CPU/GPU 输出维度通过。第一次 GPU smoke 发现 FP16 输入类型错误，修复后复测通过。完整审计见 `reports/dinov3_v1/`。

**Boundary**：这只证明运行时和数据合同可用，不证明 DINOv3 在 AIC 上有收益。PHD2/GIF 和 R70 native-crop 训练资产尚未在当前主机恢复；新主机可用后先做小批特征提取和匹配控制，再决定是否投入三 seed 训练。

用户随后提供 `dinov3-main.zip`，决定以该归档作为远端首选源码来源。归档 SHA256=`300754433a43da44653996789e9dcb01557a83d5dcd0fc4e3cf850b9bb227dca`；解压后同一 ViT-B/16 权重加载成功，CPU smoke 与临时网络 clone 输出逐元素一致。源码归档不提交 Git，仓库只记录其 SHA；DINOv3 License 文件随远端解压目录保留。
