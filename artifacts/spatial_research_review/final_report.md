# P2 候选构图效用研究：CPU 执行与 GPU 就绪报告

**状态：CPU 研究与代码准备已完成，真实新特征 GPU 前向及小读出训练尚未执行；本轮 GPU 使用为 0 秒。**

依据用户给定 `plan/AIC_spatial_research_review.md`。研究父链是 P2；新任务明确取代此前 Scale-only 常数/训练路线。原 P1/P2 权重、提交包、时间预测与历史实验保持不动。

## 1. 已执行研究与发现

- E1：用原完整公共缓存先复现 732 个 P1 输出，再按 P2 原 QP=.6 重放 122 条 A/B 路径。61 个已完成源中剔除已知字节暴露的 video2793 后，保留 60 源/360 个物理帧/720 个 A/B 观测；31 Train、17 Dev、12 Confirm 源。
- Utility训练扩展：以纯CPU增补369个Train源，合计400个LIVE Train源/4800个A-B观测；按源哈希和固定6时间帧抽样，不根据GT质量选择，额外Train帧无P2中心且不需要生成。实际P2中心/部署诊断仍只用上述已核验60源，二者不混报。原始完整public-label manifest及增补manifest的SHA已锁定。
- E2：已准备 GAICD Journal 通用候选 MOS ranker。Train 2,099 个图像 ID，181,360 个有效候选，6,058,417 个严格 MOS 差>.25 的逻辑 pair；只在同图比较，最多均匀取128 pair/图/步。该阈值是预注册设计，并非 Val/Test 拟合。
- E3：重新核验原论文 §3.3 与实际 schema，恢复原两阶段人工筛选的偏序。Train 2,281 条 subject annotation / 2,203 个图像源，3,981,607 个有向 pair；Dev 275,032、Confirm 270,469。原文件显式 pair 字段仍为0，恢复来自原人工筛选关系，不是 best_box、GT IoU 或列表先后。
- SACD 前两阶段在全部 2,590 条非 Test annotation 中严格嵌套，100候选的整数格点坐标为 floor(W/5)、floor(H/5) 步长。第二阶段实际保留至少16个，不强迫原始数据等于16。所有 subject 保留 image-level split、mask 和 annotation SHA；没有 Test annotation 内容读取。
- E4：保留129个最大合法窗口的研究思想，但不冒充旧B3。新特征来自 P2 已有 Qwen3.5视觉塔的真实 pre-merger token，复原2×2 merge组顺序，再做分数面积ROI pooling。训练 regressor/soft-target utility，而非自由坐标或 Scale token。
- E5：GAICD Train 的离散 MOS 峰簇已生成；SACD 只有偏序，不伪造连续 MOS 场。峰稳定性/连续场校准尚未验证，保持 null。
- E6：P2 中心 ±1/3/5% 以及源外 Val 残差匹配扰动已完成，A/B、Train/Dev/Confirm 分开。Val 残差只作诊断，不构造 Train 标签。没有折外 Position checkpoint，因此不声称真正 cross-fit Position 已完成。
- E7：已实现主体mask条件特征与等容量zero-subject对照，分别训练线性/MLP；无人工遮挡/互动真值的分层为 unknown。
- E8：镜头内三帧中位数CPU对照已完成。独立 Confirm A/B的绝对IoU增量均为0，没有采用该平滑控制；速度/加速度/jerk均记录，未把更平滑认定为更准确。受限DP工具保留，当前机制证据不足时不启用。

**SACD / GAICD 的数百万 pair 都是少量原始人工评定的逻辑展开，不是数百万独立样本。Bootstrap 独立单位是视频源/图像；SACD 同图所有 subject 一起重采样。**

## 2. P2 与129候选 GT-visible 上界

下表为720观测中 Confirm 的源级、整数合法窗口指标。Oracle读到了公开GT，且不是完整QP后的学习型模型收益；不能折算为官方F1。B是转置LIVE，不是原生B。

| 路线 | Confirm源数 | P2真实路径IoU | 129窗口GT可见上界 | Oracle−P2及95%源级CI |
|---|---:|---:|---:|---|
| A | 12 | 0.542441 | 0.673454 | +0.131013 [0.087870, 0.180035] |
| B | 12 | 0.519207 | 0.672435 | +0.153228 [0.093051, 0.224858] |

这只支持先做小读出实验，不支持宣称视觉模型已能关闭Oracle Gap。该批源曾暴露给历史诊断；本次split对新readout互斥，但不是项目从未见过的lockbox。

## 3. 已就绪代码和训练合同

配置 `configs/SPATIAL_RESEARCH_REVIEW_20261108.json` 固定12个读出实验：E4四个、E2四个、E3/主体消融四个。先线性再同容量MLP，对IoU回归与soft target、MOS回归与pairwise分开。所有head为FP32/AdamW/lr=3e-4/weight_decay=1e-3/batch16；E4每个600步、其余400步，每100步仅按Dev source-macro选checkpoint。所有温度、tie margin、seed先于Confirm锁定。

输入共18,557条，13,736个独立(dataset,source)组，15,881份独立图像字节。GNMC只保留Train/Validation，4组跨split字节alias从Train隔离，同源重复合并后Train 6,983个图像源。未测P2静态基线的GNMC记录仍为null，不把center伪装成P2。

完成真实权重子集加载入口、冻结/eval视觉塔、variable-K/mask collator、source/dataset均衡采样、Train-only归一化、checkpoint/RNG恢复、feature SHA/输入合同、原P2视频插值/整数导出/QP回放与一次Confirm。GPU运行自动先做新视觉路径singleton/batch smoke，既有9B Smoke不重复。CPU fixture不作为真实模型集成证据。

部署只可选择一个读出头：线性含buffers后为8,999,781,863元素；MLP32含buffers后为8,999,889,435。禁止把独立LFM/V9编码器加到P2上，也不能把12个诊断head都集成部署。

## 4. GPU启动及预算

已同步独立目录 `/root/autodl-tmp/videoora42_clone/spatial_research_review_20261108/`。开启GPU后执行：

```bash
cd /root/autodl-tmp/videoora42_clone/spatial_research_review_20261108
bash scripts/run_spatial_research_gpu.sh
```

链条：CPU媒体/磁盘检查 → 真实共享视觉特征与新路径smoke → 12个小读出训练 → 锁定best-Dev → Confirm → 原P2视频QP评价 → 主体条件消融 → `gpu_results_report.md` 与晋级/阻塞报告。`--resume` 已在shell中启用。断开SSH可按运行说明使用nohup。没有自动提醒、上传或替换最佳包。

持久账本累计上限21,600秒，加载/预热计入；进程共享排他锁与lease。超时杀本阶段进程组，未闭合尝试阻止重启。CPU分析不计GPU时间。6小时是硬上限而非实测耗时承诺；目前用量0。磁盘guard在GPU启动前要求剩余feature最坏体积×1.15再加2GiB余量，不清理历史数据。

## 5. 验证与证据

- compileall通过；新研究45 tests、已有Scale 267 tests、全仓库435 tests通过。中断恢复测试验证相同RNG/权重，测试不降低现有断言。
- 真实数据语义验证通过1,693,899个整数候选检查，无source或image-byte跨split泄漏。
- 证据：`validation_cpu.json`、`semantic_validation.json`、`prepared.json`、`data_summary.json`、`references/SACD_pair_definition_evidence.json`。远端结果见 `remote_validation.json` 与 `media_audit.json`。

## 6. 未完成项和禁止晋级原因

1. 当前机器为无卡配额，真实新Qwen feature路径GPU smoke、12个readout的训练/指标均尚未执行，结果保持null。既有真实9B smoke是可复用历史证据，不是新特征路径的已通过证明。
2. 当前克隆未恢复RetargetVid原视频、旧R70特征/旧B3头；本地及迁移清单亦只有RetargetVid标注，不能当可运行的视频训练集。没有伪造旧特征，也没有用V7/V9替代B3。
3. 原生B视频验证、第二个独立视频源域尚缺；GNMC是静态辅助，转置LIVE不能填这个空缺。因此链条跑完也可能因真实证据不足停在部署门，而不会擅自生成890视频包。
4. E1的多人/遮挡/互动人工分层、真正折外Position、SACD偏序连续场校准仍无法验证。E9缺同源事件监督，暂缓；没有9B/Position/SFT/DPO训练。
5. 新readout的官方收益为null；当前P2=44.65、C1=42.23、C3=42.87均为负责人提供的半决赛参考回分，不填成本轮新C0 receipt。原报告B_MIX“无回分”与最新STATUS中的用户回分42.88冲突，本次以最新项目记录保留42.88，未用其分数训练或选公共标签。

## 7. 方法来源

SACD: Yang et al., DOI 10.1007/s41095-021-0263-3，原论文§3.3及下载副本SHA。仅恢复可以由原筛选流程证实的阶段偏序，未使用best_box。

实验设计与资源检查参考本地experimental-design/get-available-resources工作流。Kassis et al. (2026), Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents, DOI 10.48550/arXiv.2609.00065，当前arXiv元数据核验见references/methodology_reference.json。该方法参考不是实测提分证据。

**结论：可启动受预算约束的小型候选效用研究；尚不能宣布超过P2，也不能跳过原生B/双视频域/官方回分门。**
