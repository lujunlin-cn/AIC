# 2026-09-25 表示泛化研究记录

本轮完成了评估实现修正、90次实际nested-CV训练、两批14条真实SumMe跨数据集评测、RetargetVid人工crop评测和固定候选部署验证。**目前仍不能宣布DeiT-S替代A0。** TVSum摘要选择有积极均值信号，但配对区间跨0，两批SumMe均未复现。所有 `official_f_video`、`competition_score` 均为null。

## 1. 可重复的评价与数据协议

原 `local_protocol_v1.json` 的7条已是 `comparison_holdout_v1`，不再作为pristine lockbox。原文件及历史结果不覆盖，状态说明见 `splits/local_protocol_v1_status.json`。没有随机重切50条来声称获得新测试集。

实际MAT含可靠category，10类×5视频。旧报告“没有category”的说法已更正。`tvsum_nested_cv_v1.json` 固定2次重复、每次5外折；每折30 train、10 inner-dev、10 outer，按category/source-group划分。TVSum全体已开发暴露；外层隔离阻止本次训练的直接泄漏，不能消除项目历史的模型选择偏差。

三种表示使用相同frame sampling、真实frame indices、continuous label、TemporalUNet、soft BCE、batch1、AdamW LR0.001/weight_decay0.0001、20epochs、gradient clip1。Seed20260925/26/27。Checkpoint仅按inner-dev continuous BCE最小值；threshold只在inner-dev从0.20至0.50、步长0.05选择，平局选较小值。无平滑。600组配对外层记录的indices/timestamps/labels/mask精确一致。

90次训练合计登记1032.39秒，单次最长14.73秒；使用物理GPU2/4/5，启动前检查。每次保留best/last、optimizer、完整config、选择历史、外层概率、manifest/hash和environment/command。单次训练的预算保护远低于12小时。该trainer保存checkpoint，但未提供resume CLI；不能声称已验证恢复训练。

### 指标定义

- **Binary proxy**：固定 `(mean20−1)/4 >=0.5` 的人类目标，预测阈值独立。不是AIC高光标签。
- **TVSUM_RANKING_V2**：平均ties的Spearman；Kendall tau-b；线性relevance NDCG和NDCG@floor(15%×N)，预测ties用期望DCG；固定GT的AP；top15_mean_relevance不再错误命名AP。常量序列相关性未定义记null。
- **TVSum author-style 15%-budget summary F1**，协议 `TVSUM_SUMMARY_V1_FIXED`：原frame score，确定性60帧分段，尾部余数合并最后一段，segment mean，floor(0.15N)容量、0/1 knapsack；预测min/max归一化，人类评分保留原量表。分别与20 annotator summary算F1后取平均。作者脚本的随机pred_seg示例没有照搬。该固定分段不是可靠shot detection，不能直接与采用KTS的论文表格混比。

作者TVSum代码固定revision `7bf3fb8ca032f5f4b42e22ba41f5e2490b819f96`。Python与**未修改** `solve_knapsack.m` / `knapsack.m` 使用GNU Octave10.3交叉验证10个合成、tie、tail、真实annotator案例，summary mask逐帧完全一致。证据：`tvsum_matlab_crosscheck_001.json`。

## 2. 不重训的历史DEV多指标复核

16条相同DEV，raw score、无新增阈值搜索；历史checkpoint本身曾使用DEV，故只作诊断。

| Model | Binary proxy F1 | Spearman | NDCG@15% | Summary F1 | FP16完整bundle bytes |
|---|---:|---:|---:|---:|---:|
| A0_006 | .151875 | .344299 | .631290 | .195647 | 25,685,169 |
| A1 feature-level shift | .144745 | .332453 | .638598 | .198463 | 25,685,169 |
| DeiT-S | .136273 | .255772 | .582222 | .167939 | 46,618,447 |
| ViT-B | .159280 | .264713 | .609570 | .186469 | 174,986,447 |
| Internal layer1 TSM | .134263 | .360480 | .673576 | .207584 | 完整部署bundle未在本轮重新导出 |

DeiT的历史 `.140241` 含median9，这里的 `.136273` 是raw policy，不能混用。TSM这组开发诊断不能推翻其既有不稳定fold结果，不重启该方向搜索。

## 3. 公平nested CV：均值、方差与配对差异

以下为30个fold×seed宏平均值的mean±sample SD。SD描述重复实验波动，不是30个独立数据集的抽样标准误。

| Representation | Binary proxy | Spearman | NDCG@15% | Summary F1 |
|---|---:|---:|---:|---:|
| A0 | .16084±.03398 | .43216±.09656 | .65285±.05610 | .21521±.02452 |
| DeiT-S | .16869±.03367 | .41439±.06788 | .66350±.04574 | .23073±.01541 |
| ViT-B | .17301±.03553 | .44127±.05957 | .66934±.03396 | .23064±.02127 |

Summary median分别 `.21748/.23002/.23036`。三seed summary均值：A0 `.21368/.21630/.21565`，DeiT `.22208/.23305/.23706`，ViT-B `.23318/.22982/.22890`。DeiT每seed均值高于A0，但视频间差异仍大。

配对bootstrap先对同一视频的2repeat×3seed平均，再抽50个source-video，20000次、seed20260925；不把同一视频6次当6个独立样本。区间以现有trained folds为条件，不覆盖所有重新抽训练总体的不确定性；多指标属于探索，无多重检验校正。

| Paired delta | Binary F1 [95% CI] | Spearman [95% CI] | Summary F1 [95% CI] |
|---|---|---|---|
| DeiT−A0 | +.00786 [−.01712,.03693] | −.01777 [−.07939,.04530] | +.01552 [−.00653,.04070] |
| ViT-B−A0 | +.01217 [−.01216,.03858] | +.00911 [−.04039,.06150] | +.01543 [−.00265,.03577] |
| ViT-B−DeiT | +.00432 [−.01258,.02170] | +.02688 [−.00295,.05609] | −.00009 [−.01095,.01125] |

**答案：DeiT的优势不能解释为稳定更好的全局frame ranking。** NDCG@15%的小正delta也跨0；summary高分段选择存在积极信号，但不足以确认胜出。ViT-B对DeiT没有稳定额外summary价值。

## 4. Per-video与旧7条为何异常友好

完整逐观测数据、逐视频CSV、category统计、FP/FN和quantile在 `representation_generalization/`。Category只使用真实MAT字段，不自动杜撰sports/vlog等分类。

- 旧7条分布：PR3条，VT/GA/MS/FM各1条，其余5类没有覆盖；不是category均衡样本。
- 狗耳清洁 `xxdtq8mxegs` 的历史F1从A0 `.29091` 到DeiT `.78689`，单条贡献了全部平均delta约40%；教师flash mob `xmEERLqJ2kU` 的提升再贡献约23%。这两条合计约63%，解释了小holdout被少数样本主导。
- 旧policy也不同：A0 raw/.40，DeiT median9/.35；历史选择率mean约 `.0858` 与 `.1470`。因此不能把旧差异全部归因于backbone语义，或全部归因于category。
- 在新的外层预测中，狗耳清洁仍有summary增益 `.2550→.4850`；其timeline显示DeiT对约70–90秒实际清洁操作更集中。自行车锁教学 `JgHubY5Vw3Y` 为 `.0498→.4181`。
- 相反，轮胎更换 `XzYM3PfTM4w` 为 `.3115→.1958`；两轮平衡车 `xwqBXPGE9pQ` 为 `.3000→.2049`。这些是内容/排序差异的案例证据，尚不能笼统断言“CNN擅长运动、ViT擅长语义”。
- Category summary均值DeiT在GA/BT较强（约`.320/.320` 对A0 `.234/.214`），但VT/VU/BK/FM较弱。每类仅5条，不做category条件后验阈值。

Nested CV预测选择率mean：A0 `.3562`、DeiT `.2419`、ViT-B `.2603`，GT proxy约`.0557`。空输出分别0/300、3/300、0/300；F1=0分别37/300、51/300、48/300。**主要失败并非普遍空输出，而是过选、低precision及部分视频错排。** Precision约`.099/.116/.113`，recall约`.621/.472/.510`。旧零fold不能简单归因于threshold，也不能把所有非空预测解释成有效排序。

可视化：`figures/*_timeline.png` 与 `*_frames.jpg`；timeline固定outer repeat0/seed20260925，只用于解释，不选最有利seed。

## 5. 真正raw OOD：SumMe

通过Zenodo record4884870 `datasets.tar` 的严格HTTP Range索引，定位嵌套 `SumMe.zip`，ZIP CRC核验后恢复raw视频；不再等待失败LFS。先按压缩字节大小选8条，选择在读取标签前冻结。也取得25条MAT、作者evaluator与README；预提GoogleNet特征只作备用，没有用来证明backbone优劣。

首批严格纳入6条：Fire Domino、Jumps、Playing_on_water_slide、St Maarten Landing、car_over_camera、paluma_jump。Cooking decoded1280 vs GT1286、playing_ball3119 vs3120排除，不stretch label。

所有镜像MP4的PTS存在非单调；PyAV与OpenCV前30帧像素完全一致。`ANNOTATED_CFR_DECODE_ORDINAL_V1` 显式按decoded ordinal+标注FPS采样、要求container FPS和GT相符、完整decoded count相符。它基于数据集frame-index annotation约定，不是对原始源视频时间轴完全等价的证明。比赛推理仍拒绝破损PTS，不自动用平均FPS蒙混替换。

SumMe native GT为 `user_score>0`（原数组非简单0/1），连续ranking使用 `gt_score`。预测仍固定60frame/15% summary，报告作者mean human F1和max human F1两列。Python与原封不动 `summe_evaluateSummary.m` 的18例误差≤1.12e-16；不把常见论文max协议和作者mean协议混称。

| Historical deployed bundle | Mean human F1 | Max human F1 | Spearman | NDCG@15% |
|---|---:|---:|---:|---:|
| A0_006 | .28779 | .53584 | .10555 | .42839 |
| DeiT-S | .21340 | .45172 | .11065 | .32101 |
| ViT-B | .17264 | .38958 | .12810 | .27911 |

三种raw bundle均通过最终阶段JSONL validator，分别999/1826/3642条预测；这些JSONL使用历史DEV固定threshold，native summary评测使用独立预算选择，不能混成同一输出的指标。首批总延迟41.87/41.97/42.94s包含重复probe/decode/计分/写出、并行CPU竞争，不是纯backbone速度；峰值VRAM176,241,152 /253,321,728 /660,336,128 bytes。

为了排除“只比历史某个checkpoint”的混杂，又从与TVSum缓存一致的FP32 backbone提取SumMe特征，对**全部30个CV heads/模型**零训练、零OOD调参复核。不是ensemble，不选SumMe最佳head。

| Matched CV heads平均 | Mean human F1 | Max human F1 | Spearman | NDCG@15% |
|---|---:|---:|---:|---:|
| A0 | .25752 | .50864 | .11696 | .45092 |
| DeiT-S | .15345 | .42593 | .05486 | .27331 |
| ViT-B | .12961 | .36055 | .02950 | .21449 |

按6个source先平均30heads再bootstrap，DeiT−A0 mean-F1 delta `−.10407 [−.20425,−.01098]`，ViT-B−A0 `−.12791 [−.23934,−.02745]`。仅6条、size-biased、mirror对齐假设，区间不应解释成普遍CNN胜过ViT，但足以否决“已证实DeiT可升级Primary”的当前主张。

### 第二批外部复核：完成，不再作为未来未见数据

固定压缩大小ranks9–16、约357MB压缩数据，在15分钟下载上限内完成；模型/阈值/head选择/摘要协议全部冻结。8条全部满足帧数/FPS要求：Air_Force_One、Base jumping、Bike Polo、Bus_in_Rock_Tunnel、Car_railcrossing、Kids_playing_in_leaves、Scuba、Statue of Liberty。按视频大小取样的偏差仍在，不能声称随机代表全部SumMe。

| 第二批8条 | A0 | DeiT-S | ViT-B |
|---|---:|---:|---:|
| 全部30个CV heads平均 native mean F1 | .178832 | .119547 | .154113 |
| 全部CV heads Spearman | .113081 | −.019109 | .043656 |
| 全部CV heads NDCG@15% | .391472 | .262238 | .321645 |
| 历史FP16完整bundle native mean F1 | .146885 | .121240 | .149049 |

第二批CV-head DeiT−A0 mean-F1 delta `−.05929 [−.10725,−.00786]`；ViT-B−A0 `−.02472 [−.06575,.02136]`。完整bundle中ViT-B仅比A0高.00216，不能宣称稳定胜出。三种bundle JSONL全部通过validator，分别974/2852/7019条预测；总耗时181.93/189.79/194.97s，峰值VRAM176704000/272421888/698634752bytes。该耗时包含原视频解码、计分和并行CPU竞争，不是纯模型速度。

合并两批14个**互不重复**视频；汇总程序检查两批每个head的run_id/checkpoint SHA完全相同，不容许用后批改模型。

| 14条合并（各source先平均全部30heads） | Mean human F1 | Max human F1 | Spearman | NDCG@15% |
|---|---:|---:|---:|---:|
| A0 | .212554 | .474688 | .114744 | .416952 |
| DeiT-S | .134077 | .376074 | .012593 | .266983 |
| ViT-B | .143610 | .382902 | .037589 | .275723 |

DeiT−A0 mean-F1 `−.07848 [−.13175,−.02734]`，ViT-B−A0 `−.06894 [−.13133,−.01452]`；DeiT仅2/14条summary均值更高。Spearman配对区间依然跨0。单一历史完整bundle合并mean-F1为 `.207271/.160737/.159159`，与CV-head平均方向一致。这是对**这些冻结模型及这个子集**的负面泛化证据，不是否定所有ViT预训练/微调方案。

### 固定预算非学习对照与泄漏筛查

新增 `SUMME_BASELINES_001`，不使用每视频真实摘要数量。Constant-first、uniform-segment、32次固定随机sample-score均使用预定15%预算；随机种子固定、不选最佳seed。Native mean-F1分别 `.125784/.099874/.137548`。随机选择先在与A0相同的采样位置产生分数，经相同插值和knapsack；常量/均匀选择不声称有效rank correlation。

A0 CV-head平均−随机均值 `+.07501 [+.02407,+.13841]`；完整A0 bundle−随机 `+.06972 [+.01324,+.13357]`。DeiT和ViT-B对随机的配对区间均跨0。因此A0保留不仅来自相对模型比较，也有此有界OOD预算任务的正信号；不外推为AIC联合分数。

`CROSS_DATASET_DUPLICATE_001` 检查50TVSum×14SumMe=700对：SHA256无重复，9均匀位置/视频的63-bit pHash没有触发“至少3个SumMe采样帧最近Hamming≤6”的预定复核条件。稀疏筛查可能漏短片段、重构图，不能证明预训练或所有近重复均不存在。视频manifest、GT SHA、原帧数、排除理由、cohort均保存。

两批SumMe已经被用于本轮模型比较；今后若根据本结果设计新模型，它们应称外部comparison benchmark，不能重新包装成未见lockbox。后续确认仍需要新增source。

## 6. 空间证据与可部署性

详细见 `spatial_benchmark_status.md`。20条DHF1K原视频，12,365帧，2ratio，6 human raters，6固定方法。Center `.48190/.73951`，saliency `.47639/.72116`，subject_proxy `.48342/.74839`，proxy+EMA `.48357/.74841`，真实YuNet face `.48548/.75126`，face+EMA `.48634/.75214`。

最大均值增益仍不稳定：face+EMA−center双ratio均值delta `.00854 [−.01054,.02855]`。人脸只检出约11.8%的视频宏平均帧；无脸时回退gradient subject proxy。多人片段020会追右边人脸而人工crop关注中央互动；003单人讲解则改善。均值结果不能升级为成熟主体模块。

`dense_v1` 会在所有原帧上更新observer，包括未选中的帧；切镜重置后EMA，最后只写入选frame。保留 `legacy_sampled_v1` 原行为，新增模式显式选择，未删除旧结果。8个真实E2E检查/3600预测通过validator，部署crop与benchmark对应坐标误差0。A0+YuNet实际25,917,758 bytes，ViT-B+YuNet175,219,036 bytes。

固定发行入口 `python -m aic.release` 与 `releases/20260925_v1/manifest.json` 已完成：离线加载前核对SHA256和全部实际bytes，不覆盖既有输出，不允许临时改threshold。持久化FP16权重加载为FP32标准kernel；不能把文件压缩说成实际半精度计算加速。A0/DeiT/YuNet原文件已本地备份，二进制不进入Git。

`ENGINEERING_RELEASE_001` 在CUDA统计先于设备初始化时报错，保留失败。修复后002使用DHF1K001/003/020，A0三条均空，DeiT只003有48帧；仅说明固定阈值的输出行为，DHF1K无temporal GT不能把空输出判成FN。003再用历史DEV两条AwmHb44_ouw/37rzWOQsNIw：A0 center1263帧、DeiT center2297帧、A0 face+EMA1263帧，三者全通过validator且均非空。A0两种空间模式选帧完全相同。总耗时33.01/34.72/157.78s；face dense观察耗时明显更高，应保留center默认。实际原视频共16339帧、约545秒，这不是单网络前向速度。

仍使用最大合法crop宽度，没有训练可变crop size、复杂主体关联或神经crop head；这不是“A3完成”。没有共同temporal+spatial GT，oracle只能等待对应标签，禁止乘两个benchmark的分数。

## 7. 当前决定与必须回答的问题

1. DeiT是否有threshold-free优势？没有稳定全局排序优势；summary/top-budget有TVSum均值信号。
2. 15% summary是否更好？nested mean +1.55pp，但配对CI跨0，结论不确定。
3. 旧7条为何友好？非均衡组成、少数大提升视频、不同score/threshold/postprocess共同影响；不能作单因果归因。
4. paired nested差异？见第3节，DeiT−A0 binary+.00786、summary+.01552、Spearman−.01777，均跨0。
5. ViT-B额外semantic value？有部分rank均值信号，无稳定超DeiT summary/OOD证据。
6. 新OOD？已取得真实SumMe，两批共14条严格帧数/FPS匹配，全部3×30heads及3个完整bundle完成；另2条不匹配被排除。
7. 跨dataset结论？“DeiT优于A0”在两批均未复现；A0在本SumMe子集比固定随机预算有正配对证据，不能否定整个ViT路线。
8. RetargetVid接入？已完成annotations/evaluator/raw和真实IoU。
9. center/saliency/proxy IoU？分别见第6节，纯saliency均值退化，其余微增且不确定。
10. 最大瓶颈？数据覆盖与任务对齐最缺证据；ranking/domain shift、calibration、subject choice都有实测失败，无联合GT不能算谁损失最大比赛分。
11. Best S是否DeiT？DeiT是challenger，不是已证实Best S；A0保守保留。
12. Engineering fallback？仍是A0_006 raw/.40 + center，完整权重25.69MB。
13. 明天开放提交先比较哪三种结构？A0+center（控制）、DeiT-S+center（表示差异）、A0+真实face/association/EMA（空间差异，25.92MB）。ViT-B+center保留第四个M档参照。只按事先冻结配置运行，不按单个测试视频人工修正。

VLM未运行，没有teacher human-correlation证据。Feature Bank v1、TSM与temporal smoothing不重开。没有AIC联合GT，所有候选只具有本地证据，不能填官方成绩或size加权比赛排名。

## 8. 下一轮队列与复现

1. 先扩充不同来源的真实highlight训练/验证覆盖并验证原视频时轴；已有14条SumMe只做外部comparison。VideoXum/YouTube Highlights仅取有界小子集，不从本轮结果反复调SumMe。
2. 独立空间train/dev/test协议后，针对020类多人选错对象，单独检验subject observation/association；先确认IoU再上更大检测器。
3. 若跨域排名仍差，设计一个有界loss/head控制，inner-dev选择、outer+OOD确认；不增加第三第四backbone。
4. 官方输入/evaluator到位先跑三种结构差异候选、联合oracle；此项暂不可得不阻塞以上工作。
5. 仅在本地teacher快速就绪时做人类ranking pilot，不批量伪标签。

复现：

```bash
python scripts/benchmark_representations.py --jobs configs/RG_DEV_001.json --gt-root /data/aic/datasets/TVSum/summary_gt_v1 --output /data/aic/experiments/<new_id> --device cuda:0
CUDA_VISIBLE_DEVICES=2 timeout --signal=TERM --kill-after=3m 11h50m python scripts/run_nested_cv.py --config configs/RG_NCV_001.json --model A0
# 另两种表示在检查后使用物理4/5；额外seed使用 RG_NCV_SEEDS_002.json
python scripts/analyze_nested_cv.py --root artifacts/RG_NCV_001 --output reports/representation_generalization
python -m scripts.analyze_ood_spatial --ood artifacts/SUMME_CV_OOD_001 --output reports/representation_generalization/summe_cv_analysis.json
```

远程解释器固定 `/opt/miniconda3/envs/cv/bin/python`，`PYTHONPATH=.`，资产只在 `/data/aic`。原run输出只读，新运行需新ID/输出目录，禁止覆盖历史。

## 9. 追加受控实验：线性头能否改善跨域退化

`RG_LINEAR_001`：只把Temporal U-Net替换为每timestep线性D→1 scorer。相同backbone cache、BCE、AdamW LR.001/weight_decay.0001、20epochs、batch1、2×5nested fold、3seed和inner checkpoint/threshold选择。A0/DeiT各30次真实训练；600条paired outer输入的labels/mask/frame indices/timestamps完全一致。新run_id与独立checkpoint目录，不覆盖控制组。

| Representation/head | TVSum Binary F1 | Spearman | NDCG@15% | Summary F1 | SumMe14 Mean F1 |
|---|---:|---:|---:|---:|---:|
| A0 U-Net | .16084 | .43216 | .65285 | .21521 | .21255 |
| A0 linear | .15167 | .32930 | .62380 | .21324 | .14771 |
| DeiT U-Net | .16869 | .41439 | .66350 | .23073 | .13408 |
| DeiT linear | .15207 | .33129 | .62391 | .21781 | .12230 |

Linear−U-Net：A0 Spearman `−.10286 [−.14801,−.05926]`；DeiT summary `−.01292 [−.02128,−.00475]`。外部14条A0 summary `−.06485 [−.10564,−.02562]`，DeiT `−.01178 [−.03868,.01062]`。因此当前固定预算线性头不能改善跨域退化，保留U-Net。不是对所有MLP/TCN/不同正则或训练预算的否定。

SumMe此时已经被表示比较使用，本次只能叫exposed external comparison；使用已从raw提取的同backbone缓存，只验证head容量改变，不声称新的独立backbone泛化。全30heads平均，不选OOD最佳head。线性头没有被升级为候选，不导出完整参赛包；head bytes不能当完整模型大小。

A0：head 513 parameters，持久化head文件 2933 bytes；30次合计训练 51.48s、单次最长 2.93s。

DeiT_S：head 385 parameters，持久化head文件 2677 bytes；30次合计训练 54.74s、单次最长 2.62s。

## 10. 工程收口与复现证据

当前新增实现提交：`e5b86e2`（标注ordinal reader）、`476b0d6`（dense spatial）、`d088c31`（native核对/有界OOD）、`bc6ca2d`（hash-audited发行入口）、`11a0a12`（OOD非学习/重复筛查）、`7e10f6c`（线性头预注册）、`f776342`（配对head分析）。代码源码逐文件哈希核对66项一致；远程Git HEAD较旧且工作树有历史文件，未reset/clean，使用显式同步与hash验证。

`PROVENANCE_AUDIT_001/002`保留事后环境/源码/配置/命令文件哈希，明确不是启动时记录，历史config不改写；linear另有启动时source SHA与实际implementation commit。90次原始U-Net训练加60次linear合计150次，所有checkpoint保留best/last，单次最长14.73s；没有为了占满GPU重复训练。

`RELEASE_CLEAN_001`从Git `f776342` archive解出独立代码目录，使用既有固定cv环境、HF/Transformers offline模式，在历史DEV视频37rzWOQsNIw上运行3候选。A0center332帧、A0faceEMA332帧、DeiTcenter1080帧，JSONL与原运行逐项完全一致，全部valid。它是干净代码副本复现，不冒称全新隔离依赖安装或官方环境验收。二进制权重和代码归档均在本地/远程保存，Git只保存源码/清单/小结果。

完整本地Pareto表及独立评价轴图在 `representation_generalization/local_pareto.json`、`.csv`、`figures/local_pareto.png`。表里明确区分CV-head平均性能和历史部署bundle实际尺寸，没有拼接成官方单一分数。

本地72 tests passed（3.35s）、远程72 tests passed（9.03s）。远程pytest8.4.2仅装到 `/data/aic/envs/pytest_v1`，训练cv环境不改写；`CUDA_VISIBLE_DEVICES=` 的CPU回归未占用未授权GPU。217条registry记录ID唯一，official字段全部null，原protocol manifest hash未变。
