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
| S-REREAD（真实候选裁剪图再编码） | NOT_RUN | IoU +0.03、配对 CI 下界>0、胜 B3 与同容量对照、短名单 regret ≤0.01 |
| S-ZERO（零参数余弦融合） | NOT_RUN | IoU +0.02、CI 下界>0、λ 不在确认集选 |
| T-CONTEXT（多位置真实上下文 2×2） | NOT_RUN | 原帧时间 F +0.007 双对照、CI 下界>0、正例视频退化 ≤0.005 |
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
