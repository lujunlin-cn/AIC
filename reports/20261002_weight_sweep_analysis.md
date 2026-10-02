# V9 KD 权重扫描 —— 六臂源级配对 bootstrap（2026-10-02）

对照: A0_ctrl（纯 GT 池 46,944 行，无 PHD² KD）
机制臂: A1_raw（KD pool weight=1.0，21,264 行 PHD² 教师点位高斯目标）
权重扫描: w0.15 / w0.30 × seed0/seed1（KD pool weight 降到 0.15 / 0.30）

口径: 源级配对 bootstrap（4000 次重采样源视频），矩形 IoU；SIG = CI95 不跨 0。

## 各池结论

### phd2_val（PHD² 域内，33 源 —— 我们的"主场"）
| arm | mean | vs A0_ctrl |
|---|---|---|
| A1_raw | 0.7636 | **+0.0326 [+0.0165,+0.0482] SIG** |
| w0.15_s1 | 0.7507 | +0.0154 SIG |
| w0.15_s0 | 0.7489 | +0.0165 SIG |
| w0.30_s1 | 0.7489 | +0.0160 SIG |
| w0.30_s0 | 0.7473 | +0.0141 SIG |
| A0_ctrl | 0.7333 | — |

→ 所有权重的 KD 臂在域内都显著为正（CI 不跨 0）；但 **A1_raw（weight=1.0）显著高于所有降权臂**（A1 vs w*: +0.016~+0.018 全部 SIG）。**域内增益随权重单调上升，weight=1.0 最优。** 降权没有任何域内收益，只是白白丢掉一半增益。

### confirm（227 源，跨域确认池 —— 最接近"复赛镜像"）
| arm | mean | vs A0_ctrl |
|---|---|---|
| w0.30_s1 | 0.6337 | **+0.0204 [+0.0084,+0.0329] SIG** |
| A1_raw | 0.6312 | **+0.0178 [+0.0076,+0.0282] SIG** |
| w0.30_s0 | 0.6184 | +0.0051 |
| w0.15_s0 | 0.6142 | +0.0009 |
| A0_ctrl | 0.6133 | — |
| w0.15_s1 | 0.6066 | −0.0067 |

→ **w0.30_s1 在 confirm 上反而超过 A1_raw**（+0.0204 vs +0.0178，都 SIG），且 w0.30_s1 vs w0.15_s1 = **+0.0271 SIG**、vs w0.15_s0 = +0.0195 SIG。在 confirm 上"高权重的特定 seed"有真实增益；但 seed 间不稳定（w0.30_s0 只有 +0.0051 不显著，w0.15_s1 甚至负）。**confirm 增益对 KD 权重与 seed 都敏感，不像 phd2_val 那样单调稳健。**

### rv_dev / rv_rot_dev / live_dev（GT 域池）
- rv_dev：所有臂挤在 0.58~0.59，无任何 SIG 差。**KD 不伤 rv_dev。**
- rv_rot_dev：A0_ctrl 0.5476 最高，A1_raw 0.5315 最低，但所有 pair CI 都跨 0（20 源检验力不足）。**无显著退化证据，但也无增益。**
- live_dev：A1_raw +0.0142 SIG vs A0_ctrl；w0.30_s0 +0.0138 SIG、w0.30_s1 +0.0150 SIG；w0.15_s1 +0.0049 不显著。**live_dev 也偏好中等偏高 KD 权重。**

## 判定
1. **"降权能保住域增益又不伤 GT"的假设不成立**：phd2_val 上权重降一半增益掉一半（w*≈+0.016 vs A1 +0.033），GT 池却并未因此更好（rv_dev/rv_rot_dev 降权臂同样无显著差异）。权重和域增益是单调关系，不是 trade-off。
2. **confirm 上 w0.30_s1 反超 A1_raw 是单 seed 现象**（w0.30_s0 只有 +0.005），不能据单 seed 推荐降权——与此前"单 seed confirm +0.0178 高估、pooled 仅 +0.009"是同一陷阱。
3. **结论：KD pool weight=1.0（A1_raw 配置）保持域内最优且不显著伤 GT；继续把 KD 流推到更大规模（教师全量 5,795 点位）比调权重更有杠杆。** 权重扫描这条线关闭。

产物: /data/aic/experiments_910a/LFM_V9/weights_cmp_analysis.json

---

## 附:QVHighlights V10 时间轴管线落地（用户批准后启动）

**结构事实（官方 release 标注，区别于本地旧 QVH 资产 schema）**
- `saliency_scores` = `[n_clips][n_annotators]`，标注者给每个 clip 0–4 的整数高光价值分；**clip 宽度随视频变化**（150s/11clip≈13.6s/clip，非固定 2s）。取**逐 clip 标注者均值**为密集回归目标——正是 Moment-DETR 以降所有 QVHighlights 模型的标准显著性目标。
- `relevant_windows` = 秒级高光区间（train 12,803 条），转 `exp(-d/τ)` 软标签与 PHD² TCN 同形 → 一个头可同时吃两数据集。
- 7,100 train 视频 ≈1 query/视频，saliency 是 query 无关的视频级 → **源级 bootstrap 以 vid 为独立单元**，与现有分析器兼容。

**产物（910A `/data/aic/experiments_910a/QVH_V10/`）**
- `labels_{train,val,test_with_gt}/` = 每 vid 1Hz 网格 npz：`t / saliency(0-1) / soft(τ=3s) / n_annot`。train 7,100 / val 1,519 / test 1,529 全量就绪。
- `frames_{sp}/` = PyAV 1Hz 抽帧（bucket 目录是**小写首字符**，文件名保留原始大小写——大写 vid 需 `vid[0].lower()` 命中，这是第一版 missing 80% 的根因）。
- `feats_{sp}/` = vision-tower pooled `[T,768]`（不存 grid：全量 grid 3.6TB 不可写，pooled 1.6GB 可行；QV 进的是时间轴，不吃空间网格）。

**脚本**: `v10_qvh_labels.py`(标签)、`v10_qvh_extract.py`(抽帧)、`v10_qvh_feats.py`(pooled 特征)、`v10_qvh_tcn.py`(变长多尺度 TCN，dils 1,2,4,8,16 覆盖 5–120s 高光，对齐 QV SOTA 的全局-局部机制)。

**待办**：特征抽取占 NPU 逻辑卡 0,1（教师占 2–5，不冲突）；val/test 先抽（验证集），train 待确认。QV 入训须在 `_registry/holdout_exposure_v1.json` 同类台账登记训练域新成员。

---

## ⚠️ 平台证据：PHD² KD 在官方复赛集上是负迁移（VISIONONLY 34.61 解读）

**头等价链**：VISIONONLY 的头 = A1_raw（GT 池 + PHD² KD weight 0.5）；B3 头 ≈ A0_ctrl（同 GT 池 rv_train+rv_rot_train+live_train，无 KD）。两者预测集同为 234,159 全帧（除 TEMPORAL 删 21.6% 帧）。

| 头 | 本地 confirm | 本地 phd2_val | 官方 raw |
|---|---|---|---|
| A0_ctrl ≈ B3（无KD）| 0.6133 | 0.7149 | **38.76** |
| A1_raw（+PHD² KD 0.5）| 0.6312 (+) | 0.7515 (+) | **34.61** |

**判定**：本地 confirm/phd2_val/live_dev 三池全显示 KD 为正（+0.014~+0.037），但官方集上 KD 头低 4.15 raw——**本地验证集对 PHD² KD 方向性失效**。这是第二个"本地验证集方向性错误"案例（空间学习 0/6 是第一个），且更危险：KD 本是"补域差"的机制，结果在官方域反而降分。

**可能机制（待证）**：
1. PHD² 教师点位的分布与官方 GIF GT 的判据不同——教师学的是"视觉主体中心"，官方高光可能偏"动作峰值/剪辑意图"，把候选窗拉向教师偏好的位置反而偏离 GT。
2. KD weight 0.5 过大，把 GT 监督的信号稀释（权重扫描显示 confirm 上 w0.30_s1 反超，但那是单 seed）。
3. **rv_rot_dev −0.016 的小负是被低估的预警**——竖源几何是官方 28% 的构成，KD 恰在这受伤。

**行动含义**：① P1-4 五臂裁决需重新解读——本地池正迁移不能作为出包依据；② VISIONONLY 的 k=0.95 杠杆本身仍成立，但**应挂 B3 头而非 A1 头**重新打包（同预测集 + k 系数，不掺 KD 风险）；③ KD 权重扫描方向应反转——往更小权重甚至关停 KD 验证。
