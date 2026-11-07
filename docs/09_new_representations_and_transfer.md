# 路线 09：AHA、DINOv3、Valen/Jev、AIKU 与 AViMoS 迁移

整理日期：2026-11-07。总入口：[index.md](index.md)。这些方向都曾被具体审计或实测，但成熟度不同；“模型能加载”不算任务实验成功。

## AHA：原生因果评分头直接迁移

路径为 `AHA因果视觉上下文 → relevance/informativeness → 时间排序 → B3空间`。R12阶段做过资产与smoke；R24先缺wandb、FP16非有限，补齐权重后原生BF16真实视觉smoke通过。

R25真正完成426视频候选，总8,480,646,545参数，复赛 **35.18**，低于35.52。R26修正cell广播超额为精确85%后 **35.07**；相同空间下中心85为 **36.04**。因此负结果不只是运行时失败。

R29公共26train/14dev的head+position相对position Δ=−.003262，CI跨零；作者论文核查还发现relevance来自YouTube replay、informativeness含局部阶段启发式，泛化query/等权融合/growing cache并非完整论文推荐合同。R87改BF16与causal cache后finite_fraction=1，但32源macro AP仅.0952，没有做成功迁移提交。

可保留结论：当前两标量直接替换失败；未否定完整表示、合法query、目标域训练或其他因果观察方式。

## DINOv2/v3：视觉表征与候选空间token

更早V5_H3已训练冻结DINOv2 ViT-B/14配33格候选头，比较纯几何G、视觉V和几何+视觉VQ，三头均完成2000步。VQ在confirm2 +.0190、LIVE新验证+.0113，CI为正，但RV dev CI跨零，未同时通过当时双集门，因此没有出正式包；不能说项目第一次尝试DINO是在DINOv3。

| 实验 | 已执行内容 | 结果 |
|---|---|---|
| runtime/source audit | 用户权重、源码归档、ViT-B/16严格加载，85,669,632参数 | CPU/GPU、CLS/storage/patch、原始帧PTS合同通过；只证明可运行 |
| PHD2 temporal V1 | 5,886片段、三seed union/soft TCN、位置/差分对照 | 当前读出未超过位置控制；后来发现同用户重叠导致soft>1，旧union/soft对比作废 |
| spatial V1 | 129窗patch、CLS广播、空间shuffle | patch优于破坏控制，表明有空间信号，但头容量/归一化与R70不匹配，不可据此关闭主干 |
| matched spatial V2 | 相同R70 1.511M头、MSE、归一化、源级训练选模 | DINO=.508787、V9=.564663、中心=.497588；DINO−V9=−.055876，CI[−.080370,−.033693]，当前替换配方失败 |
| PHD2 label repair | 同用户重叠区间先合并 | 修复863行soft标签、2143个旧帧值>1；union未变。修复不等于修复后所有时间臂已重跑 |

129窗平均仅15.665种patch mask提示观察量化可能限制精细选择，仍是待检机制，不是已有解决方案。

## Valen-2B / Jev：原生候选选择与合同迁移

Valen空间SFT在RV601–700这100个已暴露源上，source-macro IoU=.675226，旧head=.656760，Δ=+.018466，CI[+.010735,+.026371]，71/100源正。四候选oracle=.762726，129候选oracle=.833979。该结果支持候选内选择可学，但这些源不新鲜，不能称独立confirm，更没有官方Score。

UTILITY_T006仅把目标温度.12→.06，11,276训练行、244步。168源dev Δ=+.000791，CI跨零；暴露RV池Δ=+.001296，CI跨零。温度更尖没有可靠实用收益，不把此前SFT正信号一并撤销。

Valen时间监督做过数据审计；Jev移植了真实PTS、半开区间、等预算`left/keep/right`、低置信回退与顺序稳定性合同，并通过4项单元检查。**截至现存记录，Jev该合同没有完成两域真实T0或时间LoRA正结果。** 不与Video-ORA-9B权重相加。

## AIKU：摘要监督预训练时间读出

资产实际取得：6000train/1500test组、1024D预提取特征、gtscore/gtsummary/change_points，但缺原视频、特征提取器与来源映射。R102实际使用600组训练，在PHD2目标域重置输入投影后迁移时间干。

448源固定85%F：target-only=.415304，transfer=.418717，Δ=+.003413，CI[+.00080,+.00610]。是弱正迁移证据；原始AIKU特征不能从比赛视频产生，pilot也不是完整可部署学生。

## AViMoS：显著性预训练转native crop辅助

R104只用RGB，100train/30dev视频，475,521参数decoder：CC .2584→.4202、NSS 1.6222→2.6970，证明学到眼动结构。

R105在R70候选上加入窗口内质量、质心距离、熵和面积等4个显著性特征。168源IoU：control=.5747144，actual=.5782359，source-permuted=.5734110。Δ=+.0035215，有机制线索；但简单rank blend在调参半池.5885、留出.5536，低于留出baseline .5630，显示过拟合。未形成官方增益候选。

## 与 GPT 讨论的问题

1. 哪个方向值得补独立源确认：Valen候选选择、AIKU时间迁移、AViMoS辅助，还是DINO更细空间读出？优先依据是什么？
2. 如何将正向局部机制接到真实视频→最终路径，避免仅在不可部署特征或暴露源上比较？
3. 新表征若不能超过匹配V9控制，是否应改变监督或观察分辨率，而非继续更换head/loss造成混杂？

## 证据入口

- `reports/r25_aha_submit/summary.md`、`reports/r29/aha_transfer_check.md`、`reports/r86_r87_observation_and_transfer_20261019/summary.md`。
- `reports/dinov3_v1/ablation_reassessment_20261020.md`、`reports/dinov3_v2/README.md`。
- `reports/valen_route/valen_sft_verdict_20261020.md`、`reports/valen_route/utility_t006_results_20261020.md`。
- `reports/jev_videoora_transfer_20261021.md`、`reports/r102_aiku_transfer_20261020/summary.md`、`reports/r104_avimos_saliency_20261020/summary.md`、`reports/r105_avimos_aic_utility_20261020/summary.md`。
