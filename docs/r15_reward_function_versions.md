# RL 奖励函数版本史（v1 → v6 + 反事实基线 + oracle）

> 覆盖：2026-10-10（R13 裁决）至 2026-10-14（run10）。
> 代码基准：`rtx6000/r15_grpo_reward.py`（奖励）、`rtx6000/r15_common.py:177`（advantage）。
> 事实标注 **[事实]** = 官方分或代码可查；**[推断]** = 机制解释；**[未验证]** = 待验证假设。

---

## 0. 一页总览

| 版本 | 主干公式 | 引入动机 | 结局 |
| --- | --- | --- | --- |
| **v2** | `0.85·spearman(s, g) + 0.15·range` | 排序代理 | 弃用：与官方 F_i 差两个自由度 |
| **v3** | `0.7·R_F + 0.2·R_num + 0.1·fmt`，经 shape_reward | 与官方 F_i 同构 | 保留为组件（R_F 项一路沿用） |
| **v4** | `0.55·R_F + 0.15·R_TW + 0.2·R_num + 0.1·fmt` | 全零组零梯度（smoke 56% zero_adv） | R_TW 沿用至今 |
| **v5** | `0.9·spearman(s, target) + 0.1·fmt` | 段内排序直接优化 | 判负：把弧线先验当敌人清零 |
| **v6** | `0.5·R_F + 0.3·R_TW + 0.2·pair_core` | v5 净值负 → 保留先验 keep 结构 | **现行主干**（run8/9/10） |
| **cf 基线** | `adv ← adv − λ·(cf_r − mean cf_r)` | 输入无关输出（模板）收割奖励 | cf_table 标量版=数学空操作；cf-swap list 版生效 |
| **oracle** | `adv_gt = 2·w_q`（reward ≥ 组均时） | GT 轨迹轨迹级 REINFORCE | run10 改 mean→sum 聚合 |

---

## 1. v2 — 排序代理（2026-10-10 之前）

```
r_v2 = 0.85 · spearman(scores, gif_hard) + 0.15 · range_norm
```
（`r15_grpo_reward.py:147`，legacy 保留供 A/B）

**修改项**：无（基线版本）。

**决策理由（弃用）[事实]**：官方分 = F_i = 2·S_IoU/(N_pred + N_gt)，而 v2 的
Spearman 与 F_i 之间**差两个自由度**——阈值选择（选几帧）与视频宏平均等权。
奖励-指标错配存在，RL 优化方向与评测口径不同构。

**证据**：`reports/r13/rl_research_verdict.md:84-92`（Q4 裁决）。

---

## 2. v3 — 官方口径同构化（2026-10-10）

```
r_raw = 0.7·R_F + 0.2·R_num + 0.1·format
r     = shape_reward(r_raw)          # TempSamp-R1 非线性整形
```

**组件定义 [事实]**（`r15_grpo_reward.py:63-94`）：

- `R_F = 2·s / (k + n_gt)`，s = 命中 GT 帧的 IoU 之和（无框时 IoU≡1）。
  **官方语义**：[预测中只有落在 GT 帧的选中帧贡献 IoU，落在非 GT 帧的选中只抬高
  N_pred（分母杠杆）]。
- `R_num = exp(−|k − n_gt| / min(n_gt, 3))`，TempR1 的 TAL 数量奖励。
- `shape_reward`：τ=0.8，τ 以上 log 压缩、τ 以下指数扩张（放大次优 rollout 间的差距）。

**决策理由 [事实]**：v3 与官方 F_i 段级同构，**奖励-指标错配为零**。
Spearman 移出奖励改作 holdout 门（红线 0.60）。format 保留 0.1 低权重
（SFT 后 JSON 合法率已 95%+，维持即可，不作训练信号）。

**遗留预案 [推断]**：若实测组内 reward 全同率 >50%，回升 Spearman 0.15
作连续梯度垫片——**该预案在 run3 实际被触发（见 v4）**。

---

## 3. v4 — 时间维连续梯度（2026-10-11）

```
r_raw = 0.55·R_F + 0.15·R_TW + 0.20·R_num + 0.10·format
```

**新增项**：`R_TW = exp(−W1 / (n_gt/n + 0.15))`，W1 为归一化时间轴上
预测分数质量与 gif_hard 质量的**一维 Wasserstein 距离**
（`cdf_w1`：等分桶 CDF 的 L1 平均，双方 +ε 平滑）。

**决策理由 [事实]**：smoke 实测 **56% 的组 zero_adv**（全零组，无组内相对信号，
GRPO 组内基线把梯度全抹掉）。点估计类奖励（R_F 的 topk、R_num）在离散 GT 上
粒度太粗。W1 项给近邻 miss 错误**距离成比例的梯度**。

**外部来源**：TimeLens2（arXiv 2607.17423）Table 8/9，报告 75.8% 全零组被救活；
我们本地近邻 miss 梯度密度 21.9% vs 远距 5.7%（`aic-timelens2-rl-recipe`）。

**设计细节 [事实]**：空段（n_gt=0）时参考质量取**均匀分布**——惩罚集中高分，
与 R_F/R_num 的空段语义一致（空段 + 空 topk 得满分 1.0，空段 + 有选择得 0）。

---

## 4. v5 — pairwise 段内排序（2026-10-13，run5/run7）

```
r_v5 = 0.9 · spearman(scores, target) + 0.1·fmt      （不可解析 → −0.1）
```

**修改项**：弃用 R_F/R_num/R_TW 三点估计，主干换成对 target 的秩相关。

**决策理由 [事实]**：点估计奖励**可被输入无关的先验排序（"模板山"）部分收割**。
run5/7 池（823 段 × 8 帧）审计量化了这一点——平均弧线模板对任意 target 的
spearman **0.7497**，任意两条 target 交叉相关 0.5827，run1 输出全坍缩（48 段
1 个唯一输出）时 spearman 仍有 0.7927 ≈ 0.75 弧线基线。**RL 前几轮收的全是弧线，
不是逐视频信号。**

**结局 [事实]**：run7 官方 **35.04**（母包 ckpt_030 = 35.44，**−0.40**）。
三因子归因（`aic-r15-official-verdict-rl-redesign`）：

1. swap 拆先验的**净值必为负**（先验本身是官方域有效排序力，拆掉净损 0.29 量级）；
2. 域错位（PHD2 输入依赖信号，教师 0.325 灰区）——Mr.HiSum 32.74 的第 5 次应验；
3. 粒度错位（8–12 秒窗口内 12 帧段内排序 vs 全视频时间轴 top-80%）。

**关键教训 [推断]**：**弧线先验不是敌人，是 4B 自己缺的资产**。正确的解法不是
拆它（swap），而是把它从外部喂进去（VPB 注入，run8 起）。

---

## 5. v6 — 保留先验的 keep 结构（2026-10-13，run8/9/10，现行）

```
r_raw = 0.5·R_F + 0.3·R_TW + 0.2·pair_core
       （pair_core = 0.9 · spearman(scores, target)，解析失败 → 0）
```

**修改项**（相对 v5）：

| 项 | v5 | v6 | 理由 |
| --- | --- | --- | --- |
| R_F（keep 命中，官方同构） | 弃用 | **0.5 恢复为最大权重** | 保留弧线先验的 keep 结构，只在其上优化排序与命中 |
| R_TW（时间容差） | 弃用 | **0.3** | 连续梯度，防全零组 |
| pair_core（段内排序） | 0.9 | **0.2** | 保留排序信号但降权，避免重蹈拆先验覆辙 |
| R_num | 弃用 | 弃用 | 被 R_F 内含（k 与 n_gt 的失配已在分母体现） |

**决策理由 [事实]**：官方裁决后重构——v5 把弧线先验当敌人清零，净值为负；
v6 的任务从「从零学排序」降级为「在 SFT 先验上修 top-k 边界帧」，
detmpl 换 spearman 是这层设计的直接证据。

**官方验证 [事实]**：run8 无注入包**六轮 RL 首次总排序超过基线**
（spearman 0.4499 > base 0.4387，+0.011；均在 ±0.03 噪声带内，
但方向为正且与 run5/run7 的持续落后相反）。run7 = 35.04 官方分待判。

---

## 6. 反事实去先验化基线（2026-10-13，三次迭代）

```
orarl_advantage(r_on, r_gt, cf_r, cf_lambda):
    mu = mean(r_on);  adv = [r − mu]
    if cf_r is list:  adv = [a − λ·(c − mean(cf_r))]     # 逐样本
    elif cf_r scalar: adv = [a − λ·(c − cf_mean)]        # 标量
    adv = [a · 1.5 if a > 0 else a]                      # 正优势放大
    adv = [a − mean(adv)]                                # 再中心化
```

### 6.1 cf_table 标量版 → 数学空操作 [事实]

标量 cf_r 是**组内常数**；减去它后被末尾的再中心化**精确抵消**。
run5 发车时验证发现，cf 项在标量形式下恒等于恒等映射。代码保留仅作兼容。

### 6.2 cf_lambda 去中心化 list 版 → 生效 [事实]

`cf_r` 改为**逐样本的输入换位反事实**：把 THIS 输出解码到 OTHER 段的 target 上
算奖励（`--cf-swap-k 4` 取组内 4 个其他样本）。它在组内**有方差**，能穿过再中心化。

**设计意图 [推断]**：任何**输入无关**的输出（如固定模板）在 swap 下得分不变 →
advantage 归零 → **结构性消灭模板梯度**。这是对 v5 弧线坍缩的直接修法。

**结果 [事实]**：run7 官方 35.04（−0.40）——**swap 版未能救回 v5**。
归因：swap 拆的是**有效先验**（净值负 0.29 量级），不是无效模板。
**教训**：去先验化只有在先验本身有害时才成立；我们这里先验在官方域是资产。

### 6.3 cf_lambda 参数

run5 注释写 cf_lambda 0.5，实际发车脚本统一 `--cf-swap-k 4 --cf-lambda 1.0`。

---

## 7. oracle 轨迹级 REINFORCE（2026-10-13 夜，run10）

```
r_gt ≤ mu  →  w_q = 0,  adv_gt = 0
r_gt > mu  →  w_q = min(1, ((r_gt − mu)/(r_gt + ε))²),  adv_gt = 2·w_q
```

**修改项 [事实]**：`seq_logprob_loss` 的 oracle 分支聚合 **mean → sum**
（`--oracle-agg sum`，`r15_common.py:98`）。

**决策理由 [事实]**：run9 的 4 位小数 VPB target 使 oracle 尾巴长 **209.8 token**
（run8 为 83.0）；`mean` 聚合把 oracle 推力除以 L，**梯度被目标格式决定的长度稀释**。
改 `sum` 后 gnorm 涨 ~700×，但 **AdamW 的实际步长不变**（1e9 不裁剪 + m/√v
归一化吸收了尺度）——**自我修正记录**：gnorm 差 ≠ 步长差。

---

## 8. 全局改动（非奖励本体但同期）

| 项 | 修改 | 理由 |
| --- | --- | --- |
| `--vpb-inject` | VPB-B 头 12 点分数向量作为文本 token 注入 prompt | 零训练喂感知先验；detmpl ×3.2、uniq 0.79→0.96 |
| 池 v3 | 823×8 → **2508 段 × 12 帧**（sft_data_v2） | 池错配修复：v2 是 8 帧老体系 |
| 池 v4 | PHD2 203 源视频**整轴 12 点段**（官方推理形态） | 粒度错位修复：官方 = 全视频时间轴 top-80% |
| 难例 mining | 组内 swap 方差大的段优先 | 有效训练样本 = 输入依赖分量大的段 |
| grad-ckpt | peak 显存 43G → 24.8G | 根治 |

---

## 9. 血泪教训（供后续复用）

1. **奖励与评测指标同构是第一位**。v2 排序代理 → v3 官方 F_i 同构，是所有后续版本的地基。
2. **点估计奖励需要连续项垫片**。v3 的 56% zero_adv → v4 加 W1。
3. **先验是资产还是敌人，先问官方域**。我们错把官方域有效先验（弧线 0.75）
   当敌人拆（v5 + swap），净值 −0.40；正解是从外部补先验（VPB 注入），不是拆。
4. **组内常数项在再中心化下必为零**。cf_table 标量版三次迭代才发现的数学空操作。
5. **模板化坍缩要在训练中主动监控**：detmpl / uniq / pair_r 三指标内嵌探针，
   连续超阈自动停并存档。
6. **gnorm 暴涨不等于步长变大**。AdamW 的尺度不变性会吸收；要看 ratio 不看绝对值。

---

## 10. 版本号与代码位置对照

```
r15_grpo_reward.py:
  v2  → reward_v2()      :147   legacy
  v3  → reward_v3()      :97    0.7R_F + 0.2R_num + 0.1fmt
  v4  → reward_v4()      :226   0.55R_F + 0.15R_TW + 0.2R_num + 0.1fmt
  v5  → pairwise_reward():283   0.9spearman + 0.1fmt
  v6  → reward_v6()      :257   0.5R_F + 0.3R_TW + 0.2pair
  cf  → orarl_advantage():177   r15_common.py
  oracle → seq_logprob_loss()  :98   r15_common.py

发车脚本（--reward-mode 标记）:
  run5/6/7 → --reward-mode pair
  run8/9/10 → --reward-mode v6 --vpb-inject --cf-swap-k 4 --cf-lambda 1.0
```
