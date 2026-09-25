# 2026-09-25 自主研究 sprint 记录

本轮在本地和远程实际执行，代码提交为 `6b9a00b`、`bdf0f3f`、`7aef227`、`3424baa`；本地 `pytest -q` 为 46 passed。远程训练使用物理 GPU 2、4、5、6、7，未使用 0、3，物理 1 保留既有进程。

## 评估链和标签

TVSum v1.1 的 MATLAB v7.3 文件字段包括 `video`、`nframes`、`length`、`user_anno` 等。50 条记录的 `user_anno` 都是 `(20,nframes)`；例如 `AwmHb44_ouw` 为 `(20,10597)`，`fps=29.97002997`，`duration=353.586567s`。20 是评分者数，第二维已经对应原视频帧。作者 README/TSV 的评分采集以约 2 秒片段为单位，发布时把每个片段分数重复到帧数；作者 15% summary evaluator 是背包选段和逐评分者比较协议。

因此当前标签保存为每帧 20 个原始评分、`labels=(mean(scores)-1)/4`、`frame_indices=0..nframes-1`、`mask=true`，并标记 `annotation_type=summary_importance_2s`、`label_protocol=tvsum_per_frame_summary_mean_norm_v1`。机器可读审计在 `reports/tvsum_annotation_audit_v3.jsonl`。审计显示旧脚本的 `linspace(0,nframes,nframes+1)` 对全部视频是恒等映射，和直接转置的标签最大差小于 `5e-8`，所以标签时间轴错位不是近零 F1 的主因。连续重要性、固定 `>=0.5` 二值 temporal proxy、作者 15% summary protocol 现在分别命名；任何一个都不是 AIC 联合 GT。

`frame_f1` 现在把固定 target threshold 和 prediction threshold 分开；prediction threshold 存在 config 中，只能从开发数据选择。训练日志保存每视频 F1、宏平均、micro 诊断、正例比例、选中比例、空预测率、precision/recall、分位数、MAE 和 Spearman。原来的变长 batch 也确实存在污染：旧 GroupNorm/pooling 实测 5 步补到 12 步时有效 logit 最大差 0.3392，8 步补到 12 步最大差 0.5733。新增 `lengths` 后逐真实长度计算，单独输入、不同 padding、和长视频组 batch 的有效 logit 回归差为 0。

同一个真实视频在远程 PyAV 15.1 下的 raw→sampled RGB→backbone→temporal 和已有 cache→temporal 路径，frame/timestamp 完全一致，feature 最大差约 `2.5e-5`，aux 完全一致，probability 最大差约 `2.1e-7`；FP16 导出单独通过加载/推理检查。当地 PyAV 18.1 与远程解码的特征漂移明显更大，因此正式候选固定远程环境。

## 简单基线和修复版模型

16 个 val 视频上的简单基线（预算只由训练 target rate 得到）为：all-negative `0.00000`、all-positive `0.08715`、训练均值常数 `0.00000`、均匀预算 `0.05225`、随机预算 `0.06708`、linear ridge `0.03383`。linear ridge 的 mean Spearman 为 `0.2866`，但固定 0.5 阈值仍不如 all-positive，说明分数排序和校准必须分开处理。

修复版 A0/A1 的同一 proxy-v2 val 结果如下；数值是 temporal proxy video-macro F1：

| run | fixed 0.5 | dev threshold | tuned macro | 说明 |
|---|---:|---:|---:|---|
| A0_004 | 0.01316 | — | — | lengths-aware，batch=2 |
| A0_005 | 0.04828 | 0.40 | 0.15010 | lengths-aware，batch=1 |
| A0_006 | 0.05661 | 0.40 | **0.15188** | proxy-v2，BCE，batch=1 |
| A1_004 | 0.09607 | 0.30 | 0.12685 | feature-level embedding shift |
| A1_005 | 0.12108 | 0.35 | 0.14474 | proxy-v2，feature-level shift |
| A0_012 | 0.11679 | 0.40 | 0.14143 | 只换 SmoothL1 |

接近零 F1 的实际原因主要是固定 0.5 下分数校准偏低和空预测：旧 A0_001 空预测率约 0.938、A1_002 约 0.750。阈值在开发集选择后，A0/A1 都明显上升。padding 是另一个真实实现问题，但标签映射不是主因。A1 在 fixed 0.5 下看起来更强，校准后 A0_006 在同一 split 更强；这不是对所有 TSM 形式的否定。SmoothL1 对照低于 BCE；本轮没有足够证据支持 ranking loss 或更简单 temporal head。

## Split 稳定性

在 43 个视频上建立 5 个 source-group fold，同 seed、batch=1、15 epoch。用固定 threshold 0.30 的 fold macro F1：

- A0：`[0.08529, 0.20472, 0.11486, 0, 0.16796]`，mean `0.11457`，std `0.07900`，median `0.11486`。
- A1：`[0.09380, 0.21172, 0.09111, 0, 0.14274]`，mean `0.10787`，std `0.07764`，median `0.09111`。

每 fold 单独调阈值的数值 A0 `0.14983±0.03692`、A1 `0.15422±0.03665` 是验证集内调参的乐观诊断，不能作为最终 lockbox 结果。一个 fold 完全为零且标准差很大，当前不能宣布 A1 稳定优于 A0。

## Feature Bank、canonical TSM、ViT

Feature Bank 审计发现第 11、23–28、31 维恒零，(0,8)、(0,9)、(5,22)、(7,16) 重复或近重复。保持相同 32D head 的分组消融：motion-only `0.02724`、quality-only `0.01892`、composition-only `0.01613`、audio-zero `0.00000`。这淘汰当前 bank 定义和融合方式，不说明真实 waveform audio 无效。

历史 A1 已准确改名为 feature-level embedding shift。新增 `temporal_shift_feature_map` 在 ResNet 中间 `[B,T,C,H,W]` 上执行 canonical TSM；GPU probe 的 mean shift difference `0.12085`，参数为零，平均 batch 2.524ms→2.973ms（约 +17.8%）。尚未全量 recache/end-to-end train，因此不能把该 probe 当 internal-TSM 收益。

B0 使用 frozen torchvision ViT-B/16 ImageNet-1K 768D embedding 和相同 Temporal U-Net。fixed 0.5 macro F1 `0.06760`，开发 threshold 0.30 为 `0.15928`；FP16 bundle `174,986,447` bytes，约 M 档。它已完成真实 raw video→ViT→temporal→saliency crop→JSONL validator，单视频输出 validator valid、2266 个预测。B0 是有信息量的 M 档参照，不是已经确认的最终提交模型。

## 空间和 VLM

`center`、`saliency`、`subject` 已接入 raw inference；A2 会从同一 sampled RGB 构造同版本 aux。真实视频三种模式都通过 validator，接触表显示 saliency/subject 的中心轨迹确实不同于 center。当前只测合法性、速度、加速度、jerk 和 shot reset；TVSum 没有 crop GT，所以没有空间 IoU 或 AIC F_video 结论。本轮没有启动 VLM teacher pilot：在 temporal 评估链修好后，B0 和分组实验已先提供更高信息价值，VLM 仍留在下一轮小样本任务。

## 当前答案和下一轮队列

- 当前 engineering fallback：FP32/FP16 完整 ResNet18 + Temporal U-Net，center crop，FP16 25,685,169 bytes；A0/A1/A2 raw pipeline、JSONL 和 validator 可运行。
- 当前最有性能证据的 temporal candidate：A0_006 + dev-only threshold 0.40；B0 的单 split proxy 略高但 M 档、split 未验证，暂不替换 fallback。
- A1 feature-level shift 保留为有价值对照；canonical internal TSM 仅完成正确性/吞吐，不应继续复用 final embedding cache。
- 所有 AIC 联合 `F_video`、bbox IoU、size-weighted competition score、空间瓶颈判断，因缺官方联合 GT/evaluator/test index 仍为 null/unverified。

按证据排序的下一轮：

1. 取得官方样例/index/evaluator 和合法 crop GT，先跑 center fallback，再做 temporal oracle/spatial oracle。
2. 用完全不调参的 lockbox 固定 threshold，增加 A0_006/B0 多 seed、video bootstrap CI 和 domain failure analysis。
3. 将 canonical internal TSM 接入可导出的 raw-video encoder，做小规模 recache 后再决定是否全量训练。
4. 清理 Feature Bank 恒零/重复维度，使用 residual/gated fusion 重新做单组消融；真实音频必须从 waveform 解码后再测。
5. 在本地开源 VLM 上做小样本 teacher signal pilot，只有与人工 proxy/可核验标签有信息增益才生成缓存伪标签。
