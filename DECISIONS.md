# 技术决定

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
