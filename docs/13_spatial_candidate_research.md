# P2 候选构图效用研究运行说明

依据 `plan/AIC_spatial_research_review.md`，分支 `newday`。本轮仅准备并验证研究链，GPU 开机后显式启动。P2 原权重、原提交包和官方时间集合不修改。

## 已锁定的实验

| 实验 | 输入/监督 | 实际入口 |
|---|---|---|
| E1 | CPU 重放 P2 QP=.6，中心误差/GT 边缘代理分层 | `run_spatial_research.py prepare` |
| E2 | GAICD Journal MOS；线性/MLP32 × regression/pairwise | `run_spatial_research.py run` |
| E3 | SACD 同主体原始阶段筛选偏序；线性/MLP32 | 同上，`E3_*_subject` |
| E4 | 129 整数合法候选，LIVE/GNMC IoU；线性/MLP32 × regression/soft target | 同上，`E4_*` |
| E5 | GAICD Train MOS 离散峰簇 | `run_spatial_research.py prepare` |
| E6 | 固定 P2 中心的尺度标签扰动敏感性 | `analyze_spatial_review_cpu.py` |
| E7 | SACD subject-mask 特征与等容量 zero-subject 对照 | 同一运行链、`E3_*_zero_subject_control` 与最终成对报告 |
| E8 | 镜头内有限中位数控制、绝对 IoU 与速度/加速度/jerk；DP 工具独立保留 | `analyze_spatial_review_cpu.py` |

E9 缺同源事件/crop 监督，保留为长期研究，不启动。E1 的遮挡、多人互动真值不可恢复时明确为 unknown，不用通用显著性或 mask 数量冒充人工场景标签。

## 模型和参数

不复刻缺失 B3，不使用独立 LFM/V9 encoder。新特征明确命名 `P2_QWEN35_PREMERGER_ROI_V1`：从 P2 的原 Qwen3.5 视觉塔取 1152 维 pre-merger token，恢复 merge-group-major 空间顺序，池化 crop 内/外/差，共 3456 维，加 12 个几何/subject 特征，总计 3468 维。

只训练读出头，P2 Base/A/B Position Adapter 均不更新。原完整推理元素数含 buffers 为 8,999,771,458；加入一个线性头和 normalization buffers 后 8,999,781,863，加入一个 MLP32 后 8,999,889,435。诊断模型不能全部集成部署；最终只允许一个合格头。新视觉塔前向 GPU smoke 尚待 GPU 开机，不能把既有 9B smoke 或 CPU 单元测试当成已通过。

## 数据及语义

- LIVE：复用完整公共轨迹缓存，先复现 732 个 P1 采样输出，再重放 P2 QP=.6。排除已知字节暴露源后保留 31 Train/17 Dev/12 Confirm 视频源，合计 720 个 A/B 观测。B 明确为转置 LIVE；这些 holdout 对新 readout 源互斥，但在项目历史中已用于其他诊断，不是从未打开的 lockbox。
- LIVE utility 训练与上述实际 P2 中心诊断分开：CPU 从已核验 clean public manifest 增补369个纯 Train源，每源固定6个时间帧/两种视图，合计400个Train源/4800个A-B观测。该任务监督是每个合法候选的crop IoU，不需要为额外Train帧生成P2中心。source hash抽样和帧号不读取GT质量；400是首轮资源内pilot覆盖设计，不是显著性样本量门。输入SHA及可重建脚本为`prepare_spatial_live_train.py`和`additional_live_train_audit.json`。
- GNMC：只读 Train/Validation 标签，按图像 SHA256 合并同源，隔离跨 split 字节重复源；真实 16:9 GT，不制造 9:16 标签。P2 静态基线未测量的记录保留 null，不用 center 当 P2。
- GAICD：只使用 Journal，保留原候选与 MOS，严格差值大于 .25 才形成 MOS pair。该 .25 是预注册设计值，不是从 Val/Test 拟合的“最优阈值”。通用 ranker 不要求固定中心，和旧 Scale-only 的严格固定中心子集分开。
- SACD：原始 json 没有显式 `ranking_pairs`；原论文 §3.3 的第一、二阶段人工筛选提供可恢复的同主体偏序。精确重建 100 个原始网格窗口，第一阶段入选对落选、第二阶段入选对其上一阶段落选构成有向 pair。每阶段内不排序，不增加传递闭包；不使用 `best_box` 或第三阶段手工修框基础窗口选择。保留 mask/subject/crop ID/annotation SHA。几百万 pair 是少量人工筛选结果的逻辑展开，**不是几百万独立评分/独立 source**。
- 不读公开 Test 标签，不读官方隐藏标签；负责人已完成许可确认，本轮不重复许可研究。

## GPU 开机后的单条命令

在已同步的独立研究目录运行：

```bash
cd /root/autodl-tmp/videoora42_clone/spatial_research_review_20261108
bash scripts/run_spatial_research_gpu.sh
```

如需 SSH 断开后继续：

```bash
nohup bash scripts/run_spatial_research_gpu.sh > artifacts/spatial_research_review/gpu_run.log 2>&1 < /dev/null &
```

脚本依次执行媒体 SHA/尺寸审核、真实共享视觉特征提取和新路径 smoke、12 个已注册的小型读出实验、锁定 best-Dev checkpoint、一次 Confirm、原 P2 视频 QP 部署评价、主体条件对照和晋级门报告。失败即停止，已完成特征及 `last.pt`/`best.pt` 保留；同一命令 `--resume` 恢复，不覆盖旧模型。没有 SFT、DPO、9B 重训、全量官方推理或上传步骤。

依赖与当前远端一致：Python 3.12.3、torch 2.8.0+cu128、transformers 5.5.4、safetensors .6.2、numpy 2.2.6、scipy 1.16.2、Pillow 11.3.0。不自动升级模型环境，不重新安装 torch。

GPU 账本为 `artifacts/spatial_research_review/gpu_budget_ledger.json`。单个研究输出目录共用排他锁，模型加载/预热均计时，累计最多 21,600 秒，超时终止整个阶段进程组。未闭合的历史尝试阻止重新启动，不允许重置预算。CPU 工作不记为 GPU 时间。不要修改锁定配置或代码后复用旧输出目录。

## 结果和停止条件

重点查看 `readout_results.json`、各实验 `confirmation.json`、`E7_subject_conditioning_ablation.json`、`research_readiness.json`。Oracle 始终标注 GT-visible；pointwise integer 指标与真正 P2 视频后处理指标分开。MOS、subject ranking、crop IoU 保留各自语义，source bootstrap 使用 `(dataset_source, source_id)`，同图的 subject 共同重采样。

当前 RetargetVid 原视频/旧 R70 feature/head 未在克隆恢复，原生 B 视频验证仍缺失；因此即便小 readout 在 LIVE 上正向，也不能自动清除双域/原生 B 部署门。研究链能完整跑完并给出结果或拒绝晋级的原因；**GPU 开机并不补齐缺失的数据域，也不保证提分**。完整 890 视频新包仍待真实源外证据和后续授权，原 P2 最佳包始终保留。

## 依据和方法来源

- SACD 原论文：Yang et al., DOI `10.1007/s41095-021-0263-3`，§3.3；下载副本及 SHA 在 `artifacts/spatial_research_review/references/`。以原始阶段筛选恢复偏序，不声称恢复了全部论文 pair 或原 SAC-Net 实现。
- 实验设计及有效资源检查使用本地 `experimental-design`、`get-available-resources` 工作流；方法参考 Kassis et al. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*, DOI `10.48550/arXiv.2609.00065`。这不是模型提分证据。
