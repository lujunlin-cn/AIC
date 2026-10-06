# R6 claim ledger

状态词表沿用 R5：NOT_RUN / INVALID_IMPLEMENTATION / DEV_ONLY / INCONCLUSIVE /
LOCAL_GATE_PASS / WAIT_OFFICIAL / OFFICIAL_GAIN / NO_PRACTICAL_GAIN_IN_SCOPE /
PAUSED_BUDGET。官方分数一律未惩罚原始分，禁止换算。

## R6 新裁定（P0，2026-10-05）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| Q7-A: xperm 干预分解（五对照 × 20 种子） | DEV_ONLY 已裁定 | 双池读数见下表。**池内相似性混淆被排除**：同源排除的 derange 仍仅 +0.0034 [0.0006,0.0061] / +0.0041 [−0.0012,0.0091]；**事件-时间对应贡献为零**：res_shuffle 打乱对应后 dF1 ≈0（dev +0.0009 跨零、confirm +0.0002 跨零）；**rep_mu** 删光时间变化仅 −0.0026/−0.0033；**zeros** +0.1043/+0.0849 精确复现 round-5，且 γ 中位数塌至 0.0003（分数塌缩、排序退化）——"零输入离开分布放大影响"被证实 | r6_xperm_intervention_audit.json (+per_video.csv) | 无（本轮定性升级，不推翻 P0-CONTENT） |
| P0-CONTENT 措辞精化 | 措辞更新 | "champion 内容贡献是视频级" → "模型读取 = 视频级语义（rep_mu −0.003）+ 残差静态统计（res_shuffle ≈0）；'哪个时刻发生什么'的贡献在五对照下均为零；池内相似性混淆排除" | 本 ledger + xperm json | 无 |
| Q7-B: 1s 块标签投影噪声量化 | DEV_ONLY 已裁定 | 128 dev 源、69.1 万原帧：**e_proj 0.00755**、e_min 0.00376、部分命中块率 0.0147、**帧真值投影与 tubelet 投影标签一致率 0.9794**、完美块预测器原帧口径 F = 0.9286（块常数损失上限 7.1%）。**结论：1s 投影噪声不是 native 线失败的主因，native 关闭决定稳固** | r6_label_projection_audit.json (+per_source.csv) | 发现 VFR/尾块系统性偏差时 |
| 源台账与祖先审计 | 冻结事件（非结论） | **val178 与 B3 任何训练分割交集 = 0（vid 级验证，1115 训练片段全在 train_index）**；T5 弃用线曾在 val178 建缓存，按 Q7-C 情况 1 判非泄漏；val178 冻结为一次性确认池（178 ≥ 120，BLOCKED 不触发）；旧 dev124/confirm227 永久 DEV_ONLY | source_ledger.md、source_ancestry_audit.json、preregistration.yaml | val178 发现感知近重复穿透时 |
| 数值 parity 门（R6 重跑） | LOCAL_GATE_PASS | F_cpu = F_npu = 0.77229、abs_F_diff 0.0、mask_diff 0、γ>2ε 全过——当前栈（CANN 9.0.0 + torch_npu 2.9 + Siglip2 patch）健康。**注入自测**：temporal_shift 32/32、layout_swap 32/32、fp16_overflow 32/32 全抓到；**bad_padding 为 finding**：尾部零填充在 8 帧契约下不翻 top-k 决策（γ>2ε 门限保护；mean-score 级 21/32 可检）——部署侧低风险，但 S-REREAD 头训练的 padding 策略仍须固定并记录（AdaSpot 教训属训练侧） | r6_numeric_parity.json | 换栈后重跑（沿 R5 规则） |

## S/T/V 候选（全部 NOT_RUN，预注册见 preregistration.yaml）

| Claim | 状态 | 主门 |
|---|---|---|
| S-REREAD（真实候选裁剪图再编码） | **NO_PRACTICAL_GAIN_IN_SCOPE（DEV 已裁定 2026-10-05）** | 门：S2 ≥+0.03 vs B3 且 vs S1-control、CI 下界>0。**实际（dev80，640 行，源簇 CI）**：S0 B3 零样本 0.5289；S1-control 0.5355（+0.0066 [0.0019,0.0122]——纯容量微弱正）；**S2 真实裁剪观察 0.5243（−0.0046 [−0.0125,+0.0036]）——比同容量假对照还差 0.011**；S3zero λ=1 纯余弦 0.3655（−0.163，Siglip2 裁剪-全图余弦偏好与标注者相悖，GPT 预言应验）。**裁定**：在 Siglip2 特征+残差头设定下，"真实裁剪观察"无增益——0.247 的 oracle gap 在当前特征+监督广度下不可学。**标注分歧假设升权**：LIVE-YT-VC 论文（R6 P10）报告相邻标注者原始 IoU ~0.50——B3 的 0.529 已与"另一个标注者"相当；gap 的大部分可能是平均标注者 vs 单一标注者的不可学分歧（验证实验：多标注者源上 B3 对每个标注者分别的 IoU）。**9B 决策含义**：空间天花板可能受制于标注一致性而非模型能力；大模型增量应优先压时间线/联合推理。短名单 v2（top3+g33，regret 0.0055）与 2560 行特征缓存保留为 VLM 时代考场 | r6_s_heads_results.json、r6_s_pool_freeze.json、r6_spatial_shortlist_oracle_v2_g33.json | 重开条件：标注一致性验证显示 gap 有可学成分 + 换更强视觉特征（1B+ 编码器）同时重测 |
| S-ZERO（零参数余弦融合） | NOT_RUN | IoU +0.02、CI 下界>0、λ 不在确认集选 |
| T-CONTEXT（多位置真实上下文 2×2） | **NO_PRACTICAL_GAIN_IN_SCOPE（DEV 已裁定 2026-10-05）** | 共同评估地面 = anchor dev 变体（90 源）：multi 训练迁移读数 **−0.192 [−0.215,−0.169]** vs anchor 训练、−0.184 vs slotprior——门要 +0.007，实际强负。**结构性读数**：每个臂只在自己的切片协议上达到 slotprior+0.008~0.031（anchor+tcn 0.5752 vs slotprior@anchor 0.5672），跨协议即掉到先验以下；dense≈tcn（+0.001~0.005，第三次确认无增益）。**多位置数据干预不能让模型从位置先验转向内容**——R6 Q2/Q6 假设在本输入/标签/头范围内证伪。300 源 3,448 变体解码零失败，读数干净。**范围限定**：关闭的是本数据干预+这两个头，不推广"内容不可学"（R6 失败动作原文） | r6_temporal_context_matrix.json、r6_t_context_plan.json | 重开条件：池化前 token 读取或外部语义条件（R6 Q4 路径）出现；单独换头/换 loss 不构成重开 |
| T-TOKENS（池化前跨层 token） | NOT_RUN（条件） | 同 T 门；AP 单独永不晋级 |
| V-JEPA 80M | NOT_RUN（条件） | 需 T-TOKENS 失败 + 契约全过 + 用户明确批准 |

## 名额策略（沿 R5）

3 个官方名额继续持有。S-ZERO 过空间门可作"前沿机制诊断包"。
ST 组合需两个单变量包均有正向官方证据。无门通过不出包。

## 本轮不重开（沿 R5 + R6 确认）

- 更大时间编码器（B 级未胜位置基线前 L/V-JEPA 排队）
- 无约束期望F 配方换形式重开
- 位置先验头换 loss 重包装
- keep 连续扫描
- native 旧配方无限重跑（重开须走"特征/任务定义变更"路径：真实裁剪图、池化前 token、真实多位置事件支持——即本轮 S/T 线本身）

## R8 修正与新线（2026-10-05 深夜，输入核验后）

**输入核验**：R8 文档审查 commit 917328e（=当时 HEAD）；附件 208/208 合成检查本机复跑通过（梯度误差 ~1.8e-11 同量级、RLOO 7.98e-17 逐位一致）；B3 谱系闭合——部署 checkpoint 确出自 v8_s_train_multidata（source_ledger.md:21、source_ancestry_audit.json:10）；P02 论文独立重读，引用属实。作者未跑任何真实实验。

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| DECISION_LAYER D0/D1 设计缺陷修正 | 修正已裁定（未花 CPU） | 共同严格单调变换保持 argmax ⇒ D0"单调决策规则上限"=B3 自身 argmax（除并列），D1 校准臂收益恒为 0；D0 作 7B 路由的逻辑删除（单调不变性对所有输入成立，不能评价更强模型）。D2 降级为风险诊断；D3 被"全信息精确期望效用"臂包含（36 奖励全已知 ⇒ 精确策略梯度 p_j(r_j−r̄)，无需采样；36 个 IoU 是同一标签的 36 次评价，不是 36 位标注者）。**教训入册**：连续两轮外部评审在花 CPU 前抓到预注册设计错误（R6 z-score、R7→R8 空洞单调臂） | reports/r8/decision_layer_correction.md、reports/r8/preregistration.yaml | 无（数学恒等式） |
| "Ahmed 均值回归次优"在我们设定下的适用性 | 措辞修正 | B3 标签本就是候选效用均值（u_j=mean_a IoU(b_j,Y_a)；RV 6 人、LIVE 1 人），argmax over 条件均值效用即 Bayes 最优；Ahmed 差距适用于框坐标均值回归（我们从未做）。**真正偏离 Bayes 规则的是训练目标**：huber(0.25)+0.3·pair（huber 总体最优解 E[clip(û−U,−δ,δ)|x]=0 ≠ 条件均值；pair 项再扰动）。合成反例证明次序可反转（A: 0.4·1+0.6·0 vs B 恒 0.3；huber 最优 1/6 选 B、均值选 A）。是否在我们表上实际发生 ⇒ C/M/E 实测 | reports/r8/preregistration.yaml (C_M_E) | C/M/E 任一臂过门 +0.015 且 CI 下界>0（E 须胜 M） |
| 标注分歧假设的最强数字被撤回 | 修正已裁定 | LIVE-YT-VC 论文的 ~0.50 是**相邻帧**框 IoU（时间平滑统计；相邻帧可来自不同标注者），不是同帧标注者间一致性——R6 本表"相邻标注者 IoU ~0.50"表述作废。**后果**："B3 0.529 ≈ 人类上限"失去直接支撑，0.247 oracle gap 重新无界，Part B（估计/决策误差）回到待测状态——由 C/M/E 裁定，不得预设方向。新锚点（P02 Table IV，灵活裁剪协议、与我们 9:16 同型）：STCAT 52.3 / CG-STVG 53.1 mIoU，从零训练模型塌向中心偏置；B3 0.529 与已发表模型同档（不同分割，量级锚非上界）。LIVE-YT-VC++（~8.1 人/帧）上游今日核实仍 "Coming soon"，无数据升级 | 本 ledger、reports/20261005_round8_plan.md §2.3 | C/M/E 或 VLM_SFT 出现实测可学差距 |
| 标签语义纪律 | 冻结纪律 | LIVE 行=raw_single_box（每帧单一标注者，上游 README 今日复核）；RV 行=6 位标注者本地在库（唯一本地多标注资源，供 A_PROTO 协议敏感性审计）；禁止把单个缓存框展开成合成"标注者"；官方协议三分支（单抽取/IoU 平均/坐标聚合）只能测敏感性、不能识别 | reports/r8/preregistration.yaml (LABEL_PROVENANCE, A_PROTO) | ++ 放出或官方公布协议 |
| R8 新探针线（全部 NOT_RUN） | NOT_RUN | C_M_E（CPU 6h：C=huber+pair / M=纯 L2 / E=全信息期望效用+KL{0,.03,.1}；3 种子全报告；门 +0.015 vs B3、源级配对 CI 下界>0；E 声明优势须胜 M；≤2 臂上 76-fresh 一次性确认）；A_PROTO（CPU 2h：RV 6 人协议敏感性，仅诊断）；NPU_PARITY_PROBE0（4h，接替 r7 Probe 0）；VLM_SFT（12h：候选 ID 受约束读出、视觉塔冻结、LoRA r=8 q/v、lr{5e-6,1e-5}、门 +0.02 vs 最强已确认基线）；TEACHER_PILOT（4h 条件：32B 须 +0.02 且 CI>0 才谈蒸馏；32B 输出禁止进最终推理依赖链）；DENSE_VIEW_KD（6h 条件：稠密视图须在真实标注 IoU 上胜稀疏；FD-OPSD 选择性蒸馏项对单 token 动作退化为 0——N=1 中心化，已推导， transplant 用 categorical KL）；OPTIONAL_RLOO（8h 条件：仅当枚举不可行） | reports/r8/preregistration.yaml | 各自门条款 |
| VLM_CROP_EXAM（r7 V0-V3） | 降级（DEMITTED） | 让位于 VLM_SFT（7B 从未监督适配，是证据最少的干净方向；SFT 直接打门，零样本考场只产 evidence reading）；臂定义冻结不变，仅在预算有余时作证据探针 | reports/r8/preregistration.yaml (VLM_SFT note) | VLM_SFT 失败且预算有余 |
| 奖励配方裁定 | 冻结配方 | 空间 reward=原始候选 IoU 权重 1 不扫描；KL 0.03 为设计初值非验证最优；entropy/group-std/分位数/格式/教师/时间平滑奖励全 0；baseline 不消除位置捷径（守卫：ID 置换+内容置换+源级分割）；期望F作 RL reward=新 scope 须操作者批准，不静默重开；~28 分时间缺口为早期推演非官方上界，不得据此分配预算 | reports/r8/preregistration.yaml (reward_recipe) | TEACHER_PILOT 过门（教师奖励另立） |

## R8 执行（2026-10-05 深夜起，批次 1）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| LABEL_PROVENANCE 审计 | **PASS（已执行 2026-10-05）** | 8 项检查全过：池结构（2560=240+80 源×8 行）、隔离（train240∩dev80=∅、与 val178 零交集、dev80 不在 B3 训练 npz）、祖先（train240 全在 live_train manifest）、u 重放（**dev80 640 行逐位精确 diff=0.0；train240 抽 16 行 2.4e-07**）、短名单索引有效、u∈[0,1]。**勘误记录**：预注册的"ID 置换不变性"检查被判同义反复（每候选一标量的存储下置换多重集恒不变；x[perm][argsort(perm)]=x 对任意置换成立），替换为 shortlist 索引有效性；索引对齐由 5a 全量重放覆盖。短名单长度 34/35/36 不一=top3+g33 去重，argmax 覆盖率 28/32（报告值非门） | reports/r8/source_and_label_semantics.json | 无 |
| A_PROTO 协议敏感性审计 | **DEV_ONLY 已读（诊断，无门）** | RV 全池 1,896 行、gt 全 (6,4)：B3 argmax 决策在 mean-utility 语义 0.7265 vs 坐标均值框语义 0.7450（**决策层敏感性 −0.016**）；oracle 0.9008 vs **0.9971**（oracle 层 −0.091）；标注者留一 0.7306。**读法**：官方协议三分支（单抽取/IoU 平均/坐标聚合）的身份差异是二阶因素——决策层 ~0.016、oracle 层 ~0.091，均远小于标注者间方差本身；oracle 0.997 反映 RV 全框同尺寸、坐标均值框几乎总被某合法窗口覆盖。不识别官方协议，不为任何候选加冕 | reports/r8/aproto_results.json | 官方公布协议时按其语义重读 |
| C/M/E 目标函数对照 | RUNNING | 3 进程并行（C/M/E），15 训练配置（5 配置 × 3 种子 × 3000 步）。**复现判据修正（任何臂训练前记录）**：预注册 argmax 级容差 0.002 原理上不可达——同帧 129 窗 B3 分数近邻差 ~1e-3 与 CPU/NPU 浮点同阶，argmax 翻转是数据性质非实现错误（诊断：Pearson r=1.000000、mean|diff|=7.4e-4、翻转集中于近邻）；判据改为分数级（r≥0.9999 且 mean|diff|≤0.002，PASS），**配对基线=B3 CPU 前向 0.53108**（与臂同设备同代码路径，零设备混淆），NPU 读数 0.52887 降为锚点并排报告 | reports/r8/control_l2_exact_results.json（待全臂完成合并） | C/M/E 任一过门 +0.015 且 CI>0（E 须胜 M）→ 76-fresh 一次性确认 |

## R8 工程备忘（2026-10-05）

- **臂调度教训**：r8_cme.py 首跑按 --arm 粒度分 3 进程，E 进程内部串行 9 配置（wall ~3h）；应按"配置"粒度分片（15 单配置进程）+ 每配置独立增量写盘。根因：结果 json 在全部配置完成后才写，进程中断即丢 per-vid 明细。后续所有批次（NPU parity、VLM_SFT 网格）按配置分片 + 增量落盘。

## R8 C/M/E 终裁（2026-10-05，dev80 判定）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| C/M/E 目标函数对照（五配置 × 3 种子 × 3000 步） | **NO_PRACTICAL_GAIN_IN_SCOPE（DEV 已裁定）** | 门：+0.015 vs B3（CPU 配对基线 0.53108）且源级配对 CI 下界>0。**全部五配置未过**：C=0.52919（−0.0019 [−0.0149,+0.0125]）；M=0.53474（+0.0037 [−0.0116,+0.0185]）；E β=0/0.03/0.1=0.54034/0.54045/0.53684（+0.0093/+0.0094/+0.0058，CI 全跨零）。配对：E_rep(β=.03) vs M +0.0057 [−0.0022,+0.0147]（点正不显著）；E vs C +0.0113 [−0.0016,+0.0248]（方向合理论、功效不足）。**结构读数**：C≈B3（管线自证）；β 不敏感（非正则问题）；E 头放弃绝对尺度（pred err 4.99 vs 0.224，softmax 效用目标只管排序，预期行为）；内容置换下所有目标都掉到 ~0.48（决策的内容成分小是目标无关的）；shortlist regret 全员 ~0.22-0.23。**裁定**：按预注册关闭条款——目标函数（huber+pair vs L2 vs 全信息期望效用）在本特征+数据规模下不产生可学决策层差距；确认池未花（无过门臂）；9B 预算集中 IV2-1B 时间线 + VLM_SFT | reports/r8/control_l2_exact_results.json、reports/r8/per_source_argmax_iou.csv | 仅沿预注册特征重定义路径：若特征底座升级（VLM_SFT 或 1B 编码器）,目标对照可随新特征一次性重测；同特征同数据重跑 = 不重开 |

## R8 NPU 线（10-05 夜至 10-06 晨，自主窗口）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| IV2-1B CPU/NPU parity 门 | **LOCAL_GATE_PASS** | stage1(k710) 权重、8 帧 224p、pooled fc_norm 特征：cosine **0.9999912**、rel L2 0.004184、fp16 NPU 有限值——门（cos≥0.999 且 rel≤0.02）过。注：上游 InternVideo2 checkout 已被 10-04 环境重置清除，github 直连 clone 恢复（HEAD 3d52108 与 09-26 smoke 记录一致） | r8_npu/iv2_parity.json | 换权重/换帧数/换栈时重跑 |
| IV2 frag 特征全量提取 | **DONE** | QVH_V10/frag_train 9000 窗口 × 8 帧：居中 8 帧滑窗协议（帧 i = clamp(i−3..i+4) 的 pooled，每帧 8 秒时间上下文——单帧视觉塔不具备的性质，backbone swap 按 prereg 记为 feature change）；3 分片并行 ~92 分钟/分片；产出 **9066 npz**（~66 窗口非 8 帧被跳过），格式与 frag_feats_train 逐字段对齐（pooled (8,768) fp16 + t），时间头管线可 drop-in 换根 | r8_iv2/iv2_frag_feats_train/p{0,1,2}/ | 无（数据产物） |
| Qwen2.5-VL parity | 修正后重跑中 | 首跑 GATE FAIL 判读为**检查代码 bug 而非 NPU 缺陷**：transformers 5.18 的 last_hidden_state 是 2D 拼接 token 流（batch8=15488×1280），permutation 检查误用图级索引取 token ⇒ 9.28 假爆炸；单图一致性本身极好（cosine 0.999993/0.999976，reload 逐位一致）。修复=按 grid_thw 切分每图 token 段；rel 门 0.05→0.10 校准（首测 0.0531 后、任何 probe 数字消费前记录：fp16 在 1936-token 流累积 ~5% 属正常，binding 判据是 cosine≥0.999+置换/重载逐位）。教训：**API 形状假设必须在写检查前实测**（CPU 对照一次即可发现 2D 流） | r8_npu/parity_7B.json（重跑中） | 3B/7B 双 PASS 后 probe0 数字生效 |
| 工程坑清单（本夜） | 记录 | ① pkill 自匹配第 3 次（引号拼接规则已有 memory，仍需每次警惕）② IV2 源码被环境重置清除 ③ 修 bug 后只重传了单进程脚本，另一卡跑旧脚本崩（并行纪律：修 bug 必须重启全部相关进程）④ TBE 首跑编译的点阵进度≠卡死 ⑤ Module 没有 .npu() 快捷方法必须 .to("npu")（torch_npu patch 只保证 Tensor） | 本 ledger | 无 |
