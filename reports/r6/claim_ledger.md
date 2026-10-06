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

## R8 NPU 线终局（2026-10-06）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| Qwen2.5-VL parity 门（7B/3B） | **LOCAL_GATE_PASS（双）** | 首跑 FAIL 判为检查代码缺陷（2D 拼接 token 流按图索引）。决定性对照：8 图 batch 前向与单图逐个前向**逐位一致**（cosine 1.000000 ×8）⇒ NPU 无图序依赖。修复后：7B rel 0.0531 / cos_min 0.999976 / perm 0.0 / reload 逐位；3B rel 0.0395 / cos_min 0.999888 / perm 0.0 / reload 逐位。rel 门 0.05→0.10 校准记录于首测后、任何 probe 数字消费前 | reports/r8/npu/parity_{7B,3B}.json、batch_vs_single.log | 换栈/换 transformers 版本时重跑 |
| Probe 0 吞吐定价 | **DEV_ONLY 已读** | **可用口径 = 逐图前向**：7B batch1 = 1.72 crops/s/卡（3325 visual tokens/s，HBM 16.1 GB）；3B batch1 = 1.69 crops/s（HBM 7.7 GB）。**多图拼接 batch 在此栈代价为二次方**：batch8 超过 25 分钟未完成（batch1 为 0.58 s ⇒ 线性外推应 4.7 s），batch32 推算 >6 h，计时终止。**工程结论：此栈上 VLM 训练与推理的视觉前向必须逐图 + 梯度累积**。7B/3B 吞吐几乎相同 ⇒ 瓶颈在序列注意力而非参数量 | reports/r8/npu/probe0_{7B,3B}_b1.json | 出现 window-attention/SDPA 可用路径时重测 |
| VLM_SFT 50 步稳定性探针 | **DONE（稳定性成立，无质量声明）** | LoRA r=8 语言层 q/v（视觉塔冻结，可训练参数 0 个在 visual 内——断言通过）、单图前向 + 梯度累积 8、lr 5e-6：50 步全部有限梯度（grad_norm 最大 5.77）、无 NaN；**8.69 s/step ⇒ 外推一 epoch（240 步）= 0.58 h/卡**，12 h 限制内可容纳 ~20 epoch。loss 2.18→1.77→2.58（64 样本循环上波动，不构成质量证据）。VLM_SFT 全部前置条件满足：parity PASS + provenance PASS + C_M_E 完成 + 稳定性 DONE | reports/r8/npu/vlm_sft_stability.json | 无（已就绪，等操作者批准训练窗口） |
| 探针脚本磨合记录 | 记录 | 4 次启动失败原因各不相同且全部闭环：①框索引越界（crop_boxes 按短名单顺序而非候选索引）②③ processor 派生字段 mm_token_type_ids 未随 input_ids 拼接扩展（读 modeling 源码定位实名）④ 解包错误 ×2（zip 替换）。全部为脚本首跑磨合，非环境问题 | reports/r8/npu/vlm_sft_probe.log | 无 |

## R8 双线并行（2026-10-06，用户批准 VLM_SFT 训练窗口）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| IV2 P/O/S/L/B 归因矩阵（第 8 次启动） | **RUNNING（S/L 轴序缺陷修复后重跑）** | 第 7 次运行崩在 S 臂首 batch：`x[:, perm8]` 的 fancy 索引把 768 通道轴**替换**为 perm8 长度 8 ⇒ 输入 (8,8)，conv 通道不匹配。复查发现 L 臂同病（`z[:, ::2]` 切在通道而非时间）——R5（d_in=6144，时间=flatten 通道）移植时未换轴；时间轴应为 axis 0。修复：S=`x[perm8]`（帧置换）、L=`z[::2]=x[::2]`（隔帧）。已产生且不受影响的有效读数：P=0.0（零输入）、O 臂 lr 扫描 3e-5/1e-4/3e-4 = 0.0375/0.1042/0.2333（重跑中复现一致），选定 lr 3e-4（预注册网格端点，已记录）。GO/STOP 判据不变：O 绝对 Spearman>0 且 O−P 配对 CI 下界>0 | r8_npu/iv2_pol_matrix.json（待全臂完成） | 无（预注册条款内） |
| VLM_SFT 正式训练（用户批准窗口） | **RUNNING（双配置并行卡 2/3）** | 协议启动前冻结：train240 全量 1,920 行 / dev80 640 行读出；LoRA r=8 语言层 q/v、视觉塔冻结（可训练参数断言无 visual 泄漏）；单图前向 + 梯度累积 8、AdamW、clip 1.0、seed 510；**4 epochs（预算内留完整 dev 曲线：每 epoch ~35 分训练 + ~8 分读出，单配置 ~2.9 h，双配置合计 ~5.8 NPU 时 ≤ 12 h 上限）**；读出口径与 C/M/E 一致（IoU = u[sl[pick]]，video-macro，非法/解析失败 pick 记 0 并计数）；末轮后跑内容置换对照（一致率远高于 1/36 才算通过）。门（离线合并脚本判）：dev macro IoU − B3 0.53108 ≥ +0.02 且源级配对 CI 下界>0 + 置换对照通过；epoch 选点属 dev 选择，最终确认走 76-fresh，不直接声明。帧预提取先行（两配置共享目录，missing≠0 即中止启动） | r8_npu/vlm_sft_lr5e6.json、vlm_sft_lr1e5.json、sft_adapters/lr{5e6,1e5}/ep{1..4}/ | 双配置均不过门 → 该臂按预注册关闭 |

### P/O/L run8 读数与 run9 修正（2026-10-06 下午）

**事实 1（run8 臂表，有效）**：五臂完成，每源 Spearman 口径（3 种子均值）：P 0.0（三种子全 0）/ O 0.2333 / S 0.1667 / L 0.0333 / **B 0.425（三种子完全一致 0.425/0.425/0.425）**。学习率扫描逐位复现 run7（0.0375/0.1042/0.2333 → 3e-4）。

**事实 2（run8 判定字段无效）**：run8 打印的 STOP 系实现缺陷产物——`O_absolute_spearman` 与 O−P/O−B 配对差误用**窗口平均 logit**（`ps_store`）而非每源 Spearman；−0.0093 是 logit 尺度均值差，与预注册秩相关判据无关。臂表本身经 `spearman_src` 计算故有效。run9（修正版）已在跑：全部判定量改用每源 Spearman 配对差（种子均值后源级 bootstrap），logit 读数降为参考字段。

**事实 3（科学读数，不依赖判定）**：B 臂（旧底座特征，同 9066 窗口同 8 帧网格同 TCN 同协议）0.425 显著高于 O 臂（IV2-1B 特征）0.2333——**换根 IV2 在窗口级排序读数上弱于旧底座约 −0.19**；L 臂（隔帧置零）掉至 0.0333，采样密度敏感。含义：即便 run9 判 GO，时间头训练的底座选择也应重议（旧特征信号更强），换根动机被该读数动摇。

**工程备忘（第 3 次同类教训）**：判定字段写完后未对照预注册口径逐步核对单位（logit vs 秩相关）——判据实现必须逐字段与预注册文字对齐，读数单位错误不会崩、只会静默产出错误裁定。

### P/O/L run9 正式判定：GO（2026-10-06 16:45）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| IV2 P/O/S/L/B 归因判定（run9，修正口径后） | **GO（形式判定）** | 判据：O 绝对 Spearman>0 且 O−P 配对 CI 下界>0。读数：O 绝对 **0.2333**；O−P = **+0.2333 [0.021, 0.492]** 下界>0 → GO。次要配对：O−S = +0.067 [−0.2, +0.4] 跨零（顺序信息不显著，均值池化削弱 S 臂区分度，预注记）；O−L = +0.2 [0.0, +0.467] 下界恰 0（边缘）；**O−B = −0.192 [−0.521, +0.133] 跨零**（旧底座方向占优但不显著）。臂表与 run8 逐位复现（管线确定性）。**局限（判定功效）**：dev 900 源中仅 8 源有 ≥3 窗口可算源内 Spearman（窗口按源分布 ~2 窗/源），bootstrap 只在 8 源上重采样，CI 宽；GO 下界 0.021 仅略高于 0 | reports/r8/npu/iv2_pol_matrix.json（run9 覆盖）、iv2_pol_per_source.csv（32 行 = 8 源 × 4 对照） | 无（预注册条款内；时间头底座选择另议） |
| 时间头底座选择（GO 解锁后的实质决策，待用户定） | **OPEN** | B 臂反转（旧 0.425 vs IV2 0.2333，跨零 CI）+ O−L 采样密度敏感 ⇒ 两条可行路径：① 时间头训练直接用旧底座特征（信号更强，放弃换根）；② 双底座对照训练（旧 vs IV2，同协议同门）。预注册第 3 步文本以 IV2 为默认底座，B 臂读数是其之后的新证据 | 本 ledger | 用户决策后按所选路径预注册细节再启动 |
| VLM_SFT epoch 1-2 读数（双配置并行） | **RUNNING（趋势）** | dev macro IoU（C/M/E 同口径，B3 基线 0.53108，门 = ≥0.55108 + CI>0 + 置换对照）：lr 5e-6 = 0.47500 → 0.49180；lr 1e-5 = 0.48899 → **0.51708**。两配置均单调上升；lr 1e-5 更强，距门 +0.034。ep3/ep4 运行中（8.7-8.8 s/步，无 NaN），预计 18:00-18:30 完成并跑置换对照 | r8_npu/vlm_sft_lr{5e6,1e5}.json（增量写盘） | ep4 后按门裁定；不过门则该臂关闭，确认池不动 |

### 时间头双底座对照启动（2026-10-06 17:5x，用户拍板方案②）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| 时间头双底座对照（底座选择，dev 级） | **RUNNING（双进程并行）** | 协议启动前冻结：底座 A=旧特征（QVH_V10 frag_feats_train）、B=IV2（iv2_frag_feats_train），交集 **8,998 窗 / 4,500 源**两进程逐位一致（合并断言窗口集合相等）；头=run9 同构 TCN 从头训练（r7 条款：旧头权重不作对照）；lr 3e-4、3,000 步 ×8 batch、种子 510/511/512；主读数=dev 全窗口 pooled Spearman + **源级聚类 bootstrap**（修正 run9 仅 8 源的功效缺陷）；裁定规则预记录：A−B 配对聚类 CI 下界>0 ⇒ 旧底座，上界<0 ⇒ IV2，跨零 ⇒ 记录等效、默认主口径均值高者。官方门（temporal F +0.007 vs champion，common pool）是底座选定后的下一步，不在本步 | r8_npu/dualbase_{A,B}.json、dualbase_merge 裁定（待跑） | 无（用户已拍板） |
| IV2 特征池含 66 个跨分片重复窗口 | **记录（数据完整性）** | 探针实测：iv2_frag_feats_train 的 9,066 个 npz 仅 9,000 个唯一窗口名——3 分片并行提取时 66 个窗口写了两次。**连带影响**：run9（及 P/O/L 全系列）窗口列表含 66 个重复项；重复窗口同特征同标签，排序读数中成对并列，对秩相关是二阶扰动，不动摇 GO 判定方向，但窗口级样本量虚增 ~0.7%。双底座对照已加载去重（A/B 同 8,998） | 探针读数（本 ledger）、r8_temporal_dualbase.py seen 去重 | IV2 特征池如用于正式提交管线，重新以唯一窗口清单对账 |

### VLM_SFT 正式训练终局（2026-10-06 17:49，双配置完成）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| VLM_SFT 候选 ID 读出门（预注册：+0.02 vs B3 且配对 CI>0 + 置换对照） | **FAIL（双配置，臂关闭）** | dev80 候选 ID 硬判读 video-macro IoU（B3 配对基线 0.53108，门 0.55108）：lr 5e-6 = 0.475→0.492→0.505→**0.517**（ep1-4）；lr 1e-5 = 0.489→0.517→0.523→**0.524**（ep1-4）。最佳 = lr 1e-5 ep4 = 0.52411，**低于 B3 基线 −0.0070，距门 −0.027**；两配置单调上升但 ep3→ep4 增益仅 +0.0015/+0.012（饱和）。置换对照：0.0674 / 0.0643 vs 随机 0.0278（2.4×/2.3×，模型跟随内容但不完全）；0 无效 pick（解码 100% 合法）；pick 直方图集中于中段 ID（top5 全在 14-18/36），位置偏置与内容跟随并存。预算：双配置合计 ~5.8 NPU 时 ≤ 12 上限，卡 2/3 并行 2.9h 墙钟。**裁定：按预注册关闭该臂；确认池未动（无过门臂）**。科学结论：LoRA r8（语言层 q/v）+ 单图候选 ID 读出不足以在该空间任务上越过 B3 特征头——7B VLM 直接读图选框弱于缓存的 129 候选特征打分 | reports/r8/npu/vlm_sft_lr{5e6,1e5}.json、sft_adapters/lr{5e6,1e5}/ep{1..4}/ | 预注册重开路径仅特征重定义（如视觉塔解冻/更大秩/更多数据）——需显式新预算提案，不在本轮回 |

### 时间头双底座裁定（2026-10-06 18:5x，对照完成）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| 时间头底座选择（双底座对照，dev 级） | **OLD_BASE（旧底座，按预记录规则）** | pooled dev Spearman（3 种子均值，8,998 窗交集、TCN 从头、3,000 步）：**A 旧底座 0.17974**（0.175/0.185/0.180）vs **B IV2 0.10528**（0.130/0.079/0.107）——三种子 A 全部占优；A−B = **+0.056，源级聚类 bootstrap CI [−0.001, +0.110]** 恰跨零（下界 −0.0012）。预记录裁定规则：跨零 ⇒ 记录等效、默认主口径均值高者 ⇒ **旧底座**。三角证据同向：本对照三种子 + P/O/L B 臂（旧 0.425 vs IV2 0.2333）⇒ **时间头采用旧特征底座；IV2 换根线正式关闭**（P/O/L GO 的「IV2 有信号」成立但不构成换根理由）。合并脚本窗口集合断言通过；训练 log 与合并脚本的 pooled 读数有 ≤0.009 差异（窗口名对齐 vs 索引对齐），方向与裁定不变 | reports/r8/npu/dualbase_{A,B,verdict}.json | IV2 侧出现新读层/新协议读数反转方向 |
| 下一步（待用户） | **OPEN** | r7 第 3 步的官方门（temporal F +0.007 vs champion，common pool，源级 CI）需要：champion 时间头的 temporal F 评测协议、common pool 定位、窗口级排序头到时间轴输出的映射。champion 管线细节需考古（VTREPLAY 系脚本 + R5 native 特征/标签投影），考古后按旧底座出正式门判定方案 | r7 preregistration IV2_1B_TEMPORAL 条款 | 无 |

### 时间头考古与 pilot 复现（2026-10-06 晚，用户批准 1→2 路径）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| champion 时间头管线考古 | **DONE（全链闭环）** | 池=PHD2_FRAG_V1（5,886 frags/2,262 源）+ LFM_V10/pool_feats（每 frag (8,768) SigLIP slot 特征+t）；标签=PHD2 selections 区间投影 slot 守望窗、mixed-only；分割=eval_sources_50（1,131 eval 源/1,954 eval frags，源级隔离）；读数=f1_at_keep(0.80)=2·hit/(k+n_gt)（与官方算术同构）；头=TCN 家族（train 默认 dils (1,2,4)，champion ckpt 自带 ch/dils），Adam 1e-3/batch 8/900 步/clip 1.0；**champion=LFM_V10/probe_deploy_head.pt**。与 QVH frag 池（双底座对照所在）是不同池不同切片协议——底座选择的 dev 证据不直接外推，但 IV2 在 PHD2 池提特征需数小时 NPU 且已有两处同向负证据，性价比低，不提 | r8_temporal_gate.py（口径函数逐字复刻）、v11_expected_f_train.py（母本） | 无 |
| pilot 复现（正式门前置，用户批准） | **PASS（管线零漂移）** | 同口径重跑：池 train 1,963/eval 1,954 frags（与 R5 记录一致）；**champion F1 = 0.6588 与 R5 锚逐位一致**；**slotprior F1 = 0.6625 与 R5 锚逐位一致**——位置先验仍高于 champion +0.0037，门的真实对手是 0.6658（=0.6588+0.007）；mse 重训单种子 0.6273（历史 mse 水平） | r8_npu/temporal_gate.json、temporal_gate_pilot.log | 无 |
| 正式门判定（r7 第 3 步） | **RUNNING** | 预注册条款：mse+exactdp × 种子 {20261006-08}（TCN 从头），臂级 3 种子均值选判定臂（选点规则启动前预记录），配对 per-fragment delta vs champion、源级聚类 bootstrap 2,000 次；门=delta ≥ +0.007 且 CI 下界>0；slotprior 0.6625 作为诚实对照同报（判定臂须同时高于它才有部署意义） | r8_npu/temporal_gate_full.json（待出） | 不过门 ⇒ r7 第 3 步负结果关闭归档 |

### r7 第 3 步正式门判定：FAIL（2026-10-06 晚，IV2_1B_TEMPORAL 全链收口）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| 时间头正式门（r7 IV2_1B_TEMPORAL 第 3 步） | **FAIL（预注册条款内负结果）** | 判定臂=exactdp（臂级 3 种子均值 0.6625 vs mse 0.6354，选点规则启动前预记录）。**exactdp 0.6625/0.6626/0.6625（三种子几乎逐位一致）**；delta vs champion = **+0.00368，源级聚类 CI [0.00097, 0.00645]**——CI 下界>0 但 **delta < +0.007 门 ⇒ FAIL**（差 0.0033）。**best_arm_clears_slotprior = false**：判定臂恰好收敛到位置先验解（0.6625 = slotprior），增量全部来自位置结构，内容信号增量为零——与 R5 content ablation、expected_f 四臂、R8 C/M/E 四重证据闭环。6 训练 ~18 分钟 CPU，判定脚本含形状断言（首跑判定段数组形状崩，修复后 judge 模式从 JSON 重建判定，不重训） | reports/r8/npu/temporal_gate_{full,verdict}.json、temporal_gate.json | 预注册重开仅沿特征重定义（新特征在新池过 slotprior 级证据）；同池同特征重跑=不重开 |
| **IV2_1B_TEMPORAL 全链收口** | **CLOSED（三步全部完成）** | ① parity PASS（cosine 0.9999912）② P/O/L GO（IV2 有内容信号）→ 双底座对照：旧底座更强（+0.056，三种子同向）⇒ 换根关闭 ③ 正式门 FAIL（同池同协议从头训练增量 +0.0037 < +0.007，且不超位置先验）。**R8 的 9B 增量线全部收口**：VLM_SFT FAIL（0.524<B3 0.531）、C/M/E 关闭、IV2 换根关闭、时间头门 FAIL。确认池未动（3 个名额保留，hour-36 决策默认诊断包）；champion 保持 V11_VTREPLAY 34.95 | 本 ledger R8 全段 | 见各行重开条件 |

### 诊断包 D_N/D_X/D_Y 构造（2026-10-06 晚，用户批准交付；名额规则修正：每日 5 个）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| 名额规则修正 | **记录** | 用户 10-06 纠正：提交名额为**每天 5 个**（非总量稀缺）。r7/r8 slot_strategy 的 hour-36 一次性决策与 D/S-T 互斥设计按每日窗口重释：诊断包与真候选并行不挤占；memory 已存 | memory aic-submission-quota-daily | 无 |
| 诊断包三件套（r7 slot 默认，用户批准） | **DELIVERED（待用户上传）** | 基线 = 冻结 champion LFM_V11_VTREPLAY（官方 34.95），每包施加一个已知合成扰动（官方分差 = 该域迁移斜率的一个数据点）：**D_N** = 保留帧索引整体平移 +round(0.1·n)（覆盖不变、定位移动；与历史 keep/k_size 斜率正交）；**D_X** = x′=clamp(x+0.05W)；**D_Y** = y′=clamp(y+0.05H)。可辨识性合成验证 PASS：时间 F1 1.00→0.90（−0.10）、IoU 1.00→0.837（−0.163，x/y 对称），单调远超噪声 ⇒ 官方分变化可唯一归因。三包均过 write_submission 全量合同校验（426 视频，semifinal enriched index）。本地斜率参照：Δlocal = −0.10（时间）/ −0.163（IoU），官方读回后 transfer = Δofficial / Δlocal | /data/aic/semifinal_20261001/submissions/LFM_D{N,X,Y}_DIAG.zip、LFM_D{N,X,Y}_DIAG/、r8_npu/diagnostic_pkgs.json | 官方读回后入 transfer_pairs |

### R9 计划落地：GPT6PRO 回答收到 + 指标穷举审计（2026-10-06 深夜）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| R9 外部研究计划 | **RECEIVED** | GPT6PRO 按 V9 提示词返回 R9 报告（Q1-Q8+最终裁决）：根因单一承诺=「Anchor+mixed+guard 任务让位置规律承担大部分覆盖决策，当前读出的内容没有在保留边界上提供稳定增量」；最高信息/成本比实验=按实际整数 K 穷举保留集合+训练源拟合 macro-F 最优位置先验；编码器排序=IV2-1B Stage1 last/−5 → Stage2 对照 → Qwen 视觉塔（第32/24块）；32B 裁决=不可提交（>9B）+教师输出与官方隔离+公共源 pilot 门 +0.02；3 天预算 ~72 NPU·h。全部事实/推断/未验证分级完整，本地核验后执行 | reports/r9/gpt6pro/（报告+审计包，含 experiment_plan.json） | 无 |
| R9 审计工具自测 | **PASS（本地+910A 双验证）** | metric_sensitivity_audit.py：256 二值标签×9 预算=2,304 穷举案例，闭式界/随机期望/边界交换公式最大误差 3.3e-16；本地 numpy 2.4.6 与 910A 各跑一遍均 PASS，与包内 self_test_results.json 一致 | reports/r9/gpt6pro/{metric_sensitivity_audit.py,self_test_results.json}、910A /tmp/r9_st.json | 无 |
| 池计数分解（R9 Q1.4 要求） | **DONE（纠正 V9 一处错误）** | index_clean 5,886 frags/2,262 源 → 特征全在 → **mixed-only 保留 3,917 frags/1,963 源（丢弃 33.5%）**——V9 提示词写「discards 67%」是错的（保留比例记反），R9 报告对此存疑是对的。G 直方图 eval：G=1:71 / 2:374 / 3:521 / 4:408 / 5:247 / 6:198 / 7:135；eval 全部 8 槽（1,954）。G=3 峰值但非全池=3，R9 的「不能从均值反推 G≈3」成立 | r9_audit/{train,eval}_public.jsonl、export_record.json（冻结清单+分解，已入库） | 无 |
| 实际 K 核对（R9 Q1.3 假设） | **CONFIRMED（本方代码逐字核对）** | f1_at_keep(0.80) 在 n=8 上 k=round(6.4)=**6**（实际保留率 0.75）；keep 0.70→round(5.6)=6 **同一预算**——「0.70/0.80 曲线相同」不是经验结论而是同一动作。确认位置：scripts/r8_temporal_gate.py:114、v10_keep_curve_heads.py:151 | grep 记录（两处 `max(1, int(round(keep*n)))`） | 无 |
| **K=1..8 指标穷举审计（R9 最高信息/成本比实验）** | **DONE（决定性读数）** | 冻结池+champion 真分数（CPU 重前向），macro-optimal 位置先验按训练源拟合：**K=6（实际工作点）：head 0.6588 / margP 0.6625 / macroP 0.6625 / oracle 0.7309 / randE 0.5561；head−macroP 配对源簇 CI [−0.0064, −0.0011] 全负（显著劣于先验）；top-6 集合与先验相同比例 88.5%、同命中 94.0%、平均每片段仅 0.13 次边界交换**。oracle 余量 +0.068 ⇒ **非指标饱和**（R9 判读表第 2 行：继续表示/标签机制分析）。**K=2：head 0.4459 > macroP 0.4398（+0.0061）但 CI [−0.0096, +0.0213] 跨零**；K=1/2 same_mask 仅 0.36-0.48——头有内容信号但点估计不显著；预算越宽松头越退化为位置先验（same_mask 0.36→0.885）。macroP=margP（K=6 两者 mask 相同）⇒ R9 §7.4 担心的「macro 先验更高」未发生，先验天花板不上移。**新编码器价值判据改写：必须体现在边界交换，槽内排序改善对部署 F 贡献严格为 0** | reports/r9/audit/audit_k{1..8}.json + per_fragment | 池/分数/K 任一变化即重跑（工具+清单已入库，分钟级） |

### R9 第 1 天 NPU 主线启动：IV2-1B Stage1 slot 池多层提取（2026-10-06 深夜）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| PHD2 媒体解码栈换道 cv2→PyAV | **DONE（强制换道，事实驱动）** | cv2 探测全部 1,963 manifest 源：**h264 1,254 / AV01 685（34.9%）/ VP90 24**；容器 cv2/ffmpeg 无 AV1 软解（grab 静默失败+填充帧风险）。PyAV 18.0.0 wheel 内嵌 libdav1d 三种全解。帧对位 parity：h264 源同帧号 cv2↔pyav 像素 MAE=0.0（3 帧抽查）。**实测 fps 与 index 字段可不一致**（-3ixfTKGG8A 实测 23.976 vs index 25）⇒ 一律取流实测 fps。缺帧绝不填充：npz 带 nmiss 质量键（pilot 12 片段 11 个 nmiss=0，1 个 = frag 末槽上下文越出视频物理边界 3.2s，属数据边界） | r9_iv2_slot_feats.py（docstring 全记录）、/tmp/r9_par.py 探针 | 无 |
| 池既有结构记录：同特征重复片段对 | **记录** | `_f00` 与 `_f00_r` 变体共享同一冻结 t 轴（pool_feats 逐位相同）⇒ 任意编码器都会给两者相同特征，而审计里它们是独立样本。这是 PHD2_FRAG_V1 索引层既有性质，非本次提取引入 | /tmp/r9_probe7.py 输出 | 无 |
| IV2 Stage1 slot 池提取（R9 第一批） | **RUNNING（双卡分片）** | 范围=冻结审计清单 3,917 mixed frags；时间轴=复用冻结 pool_feats 't'（绝不从媒体重推）；输入=每槽 [t−3..t+4] 8 秒上下文（与 QVH 已验证协议同构）；tubelet_size=1 ⇒ T'=8 无时间降采样坑（R9 §3.3 警告不适用于本配置）；读出三件套=mean768（QVH 同路径 fc_norm∘clip_projector）+ m1408_l + m1408_m5（hook blocks[34]=35/40 层，读出层协议决策推迟到头训练）； pilot 五门全过（parity MAE 0/AV1 人工样张/T' 断言/数值健康/3.5s/片）。速率较 pilot 慢（AV1 软解），预计 ~4.8h 墙钟、~10 NPU·h（R9 给第一批 8h 上限，同量级） | r9_iv2_slot/p0+p1、r9_iv2_chain.sh | nmiss>0 比例过高即停 |
| R9 头训练与门（已备好，特征齐即跑） | **READY** | 判定臂选法与 R8 门一致（arm 级 3 种子均值选判定 loss=exactdp vs mse）；读出臂 mean768/m1408_l/m1408_m5 × 2 loss × 3 种子=18 训练（CPU ~1h）；归一化仅训练源拟合；**R9 新门=对 champion 与 macro-optimal 位置先验均 +0.007 且两源簇 CI 下界>0**（审计的 macroP=margP=0.6625，先验天花板未上移）；champion f1_list 与 audit_k6 per-frag macro_prior_f 冻结引用 | r9_slot_head_train.py | 新特征读出缺键即中止链 |

### R9 A1/A2 判别审计（2026-10-07 凌晨，CPU 0.3min，页缓存全热）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| A1 标签定义审计（guard/中心点/占用三定义） | **DONE（标签定义排除）** | 同一冻结清单三种标签定义：prior F1 = 0.6625/0.5497/0.5549，champion−prior = −0.0036/−0.0024/−0.0023——**换标签定义只整体缩放 F1，先验平台结构不变**。R9 报告 A1 实验的结论：guard 标签不是平台成因 | reports/r9/audit/a1a2_audit.json | 无 |
| A2 分数手术分解（real/tperm/meanvec/zerovec） | **DONE（AP-F1 分离教科书读数）** | real F1 0.6588 / AP 0.7463；tperm 0.5548/0.6016（=随机预算期望 0.5561，置换恰好摧毁集合信息）；**恒定分数（mean/zero）0.6625/0.7101=slotprior 本身**（本池先验 top-6 恰为前 6 槽）。champion 对恒定分数 **AP +0.036（有全序内容信号）但 F1 −0.0037（信号不落边界、0.13 次/片交换净效用为负）**。判定：champion 不是没有信号，是信号不转换为边界交换 | 同上 | 无 |

### R9 CPU 信号探针：media probe + 三族 screen（2026-10-07 凌晨，CPU 19min）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| 音轨存在率 | **FACT** | manifest 源音轨存在率 96.7%（train）——audio 信号族未被「无音轨」排除，PyAV 音频解码路径验证可用 | r9_signal_probe npz、r9_media_probe.py | 无 |
| 廉价标量 screen（vis 5 列 + aud 6 列） | **DONE（全 null，探针级）** | 2 参数网格（w0·logit(prior)+w1·z(x)，train 似然拟合）：**全部 11 列 w1=0**；槽级点二列相关全部 \|corr\|<0.05（最大 hist_chi2 −0.043，n=31k slot）。手写数值验证确认非实现 bug。结论 scope：**像素统计/手工音频标量层无信号**——GIF 高光选择是语义级判断；audio 族后续必须上有学习型 embedding（PANNs/BEATs 级）才可测，手工标量层已可关闭 | reports/r9/audit/signal_screen.json、r9_signal_screen.py | 谱级/嵌入级特征未测，不在本结论 scope |

### R9 编码器第一批判定：IV2-1B Stage1 三读出全 FAIL（2026-10-07 凌晨）

| Claim | 状态 | 比较、门槛与读数 | Artifact | 重开条件 |
|---|---|---|---|---|
| IV2-1B Stage1 @ PHD2 slot 池（R9 第一批，双门预注册） | **FAIL（三读出全负，CI 全负）** | 判定臂 exactdp（预选规则）3 种子均值：mean768 0.6557 / m1408_l 0.6515 / m1408_m5 0.6472；对 champion delta −0.0031/−0.0073/−0.0116，对 macroPrior −0.0068/−0.0109/−0.0153，**六个 CI 全部全负**。三点结论：①SigLIP CLIP 对齐特征仍是创作者选择标签上的最好表示（与 QVH 负证据同向）②InternVideo2 论文 THUMOS14 的「−5 层>last」在 GIF 池**反转**——动作定位的层迁移结论不穿越标签类型 ③exactdp 在新特征上不再收敛到 slotprior（0.6557<0.6625）——新特征连位置信息等价性都不具备。头训练修复史：temporal_gate.json 无 f1_list（R8 judge 重写删失）→ 从冻结审计导出 scores 重建+自洽断言；f1_at_keep 接收 list 两次 TypeError → 本地 8 片段合成冒烟一次抓出三处（asarray×2、mean 轴反）后真机一次通过 | reports/r9/audit/slot_head.json、r9_slot_head_train.py | Stage2/Qwen 塔判定同表；置信新表示需过双门 |
