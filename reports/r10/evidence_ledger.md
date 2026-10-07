# R10 证据台账（可学习信号普查与策略空间审计）

计划来源：reports/r10/gpt6pro/（GPT 6 PRO，2026-10-07，repo SHA ab60493）。
执行纪律：双门判定（+0.007 vs champion 且 vs macro-optimal prior，源簇 CI 下界>0）；
SCREEN（训练/开发源）→ CONFIRM（新合法源，Holm α=0.10，家族 p=max(p_LL, p_F_vs_prior, p_F_vs_champ)）。

## 0. R10 计划核验（2026-10-07）

| 项 | GPT 断言 | 本地核验 | 结论 |
|---|---|---|---|
| self-test | 2,055 项 PASS，max_err 1.11e-16 | 本地复跑一致 | ✔ |
| 评价源数 | 1,954 片段 / 984 源（非 50） | audit_k6.json + 冻结 manifest grep 一致（979 train 源） | ✔ `eval_sources_50.json` 文件名是遗留误导 |
| AP 强位置对照 | prior 0.7444 vs champ 0.7463（+0.0019） | a1a2_audit.json A1.guard 一致 | ✔ +0.036 只对恒定分数（0.7101）成立，对位置先验仅 +0.0019 |
| oracle/prior/交换 | gap 0.068446、交换 0.12794/frag、same_mask 0.8854 | audit_k6.json 一致 | ✔ |

## 1. 先验层检查点（CPU，2026-10-07）

- 效用先验 b_s=E_train[2Y/(K+G)] 的 top-6 集合与概率先验 p_s 完全相同（[0..5]），eval F1 均 0.662454。
  **结论：先验层无免费增益；效用读出的意义只剩条件于 X 的残差（q_s=b_s+h(X)）。**
- eval 与 train 源零重叠（复核通过）。

## 2. 家族 A（学习型音频）执行记录

- 编码器：`MIT/ast-finetuned-audioset-10-10-0.4593`（transformers 原生，AudioSet mAP 0.459）。
  PANNs CNN14 GitHub release 直连不可达（9 字节错误页）；AST 在 R10 预注册编码器表内（R2 用的即 AST）。
  权重经 hf-mirror 秒级到位：/data/aic/pretrained/ast_audioset。
- 冒烟（合成音频，CPU）：extractor 对 2s 窗口自动 pad 到 1024×128；527 类头就位；
  2s 单独 vs 嵌入 10s 的 pooler cosine 0.896 → 协议固定为逐窗独立前向。
- 提取脚本 scripts/r10_audio_extract.py：媒体定位逐字复用 R9（冻结 t 轴=源内绝对秒，PyAV seek 2s 提前量）。
  **两个解码 bug 及修复**：
  1. `Container.seek(..., stream=st)` 把 timestamp 按 stream time_base（1/44100）解释——微秒整数被读成 582.75s；
     去掉 stream= kwarg，与 R9 视频路径一致（微秒）。
  2. resample 输出 to_ndarray 形状 (1, 355)（平面×样本），`mean(axis=1)` 把 355 样本塌成 1——
     改 `reshape(-1)`（resampler 已混 mono，展平恒安全）。

## 3. ⚠️ 官方域音频可用性（用户提供，2026-10-07，GPT 6 PRO 审计官方 174 集）

- **77% 有真音轨（134/174，全 AAC 立体声：127×44.1kHz + 7×48kHz）；23%（40/174）完全无音频流。**
- 抽样 10 个响度实测全部真声（mean −8.7~−23.2 dB，峰值近 0 dB，0 静音假音频）。
- PHD2 训练池音轨存在率 96.7%（R9 探针）→ **存在率域偏移：96.7% vs 77%**。

由此固化的预注册约束（家族 A 全程有效）：

- **构造性回退**：`score_s = a_s + avalid_s · g(x_s)_s`。avalid=0 槽残差硬置零、落回位置先验；
  禁止让头从 ~3% 缺失样本学习缺失行为（官方 23% 上无训练支撑）。
- **has_audio 不进特征**（PHD2 近常数、官方强变量，域漂移特征）；只作回退开关与覆盖记账。
- **增益定价按覆盖折扣**：官方域期望增益上限 ≈ 0.77 × 池内增益（若残差仅在可用片段有效）；
  pilot 报告按 H_coverage 口径并记此折扣。
- 官方部署输入合同：无音频流 → 全 frag avalid=0 → 输出=位置先验（与 R9 champion 同姿势）。

## 3.1 家族 A pilot 判定（2026-10-07，64 源 / 132 frags / 5 折源分组 CV）

软覆盖审计（AST 527 类，R10 4.4 口径，模型估计非标注）：
- Speech mean σ=0.30（33% 槽 >0.5）、Music mean σ=0.41（47% 槽 >0.5）。
- **Cheering mean σ=0.0005、Crowd σ=0.0002——池内音频是 BGM+人声，不是体育欢呼/人群声。**
  「高光=欢呼」机制在本池基本排除；PHD2 是创作者 GIF（配乐），非原始转播。

主判定（`pilot_head.json`，feature=ast_pool，λ∈{0,.25,.5,1,2} 折内选择）：
- **R1（prob residual）与 R2（util residual）λ 网格 5/5 折全选 0.0**——残差被完全关闭，
  输出=折内先验 0.6675（delta_vs_prior 精确 0，CI [0,0]）。
- 置换对照=先验=真实（0.6675）：无分布泄漏可捡（132 frags 的 TCN 本可记住噪声选正 λ）。
- pilot 子集上 champion 0.6928（子集效应，全池 champion−prior 仅 −0.0036）。
- 判定状态（R10 8.4 口径）：**NO_PRACTICAL_GAIN(AST pooler 读出 @ pilot scope)**；
  家族 A 保持 open（64 源不足以关模态），但不再为此花 NPU。
- 二次免费检查（进行中）：ast_mean（768 非池化读出）、ast_527（527 类语义分数作特征）——
  同一 npz 内零提取成本；两者也 null 则 AST embedding 线整体 PAUSED_BUDGET，转 ASR/OCR。

工程注：132 frags 太小，本 pilot 的职能是「方向+方差校准」，不是功效确认（R10 8.3：
64 源 MDE≈0.022σ⁻¹ 量级）；连正 λ 都选不到这一事实，比 delta 数值本身更有信息量。

## 3.2 家族 S（ASR 语义）覆盖审计与 pilot 判定（2026-10-07）

覆盖审计（whisper-base，32 片段先行→pilot 132 全量，修复 2 个对位 bug 后）：
- **93.8% 片段有 ≥4 词转录、平均 54.6 词/8s、2.5 段/8s；多语言确认（英/韩/印尼/西/泰…）。**
  覆盖不是瓶颈（R10 4.4 失败条件「语音覆盖低」排除）。
- 特征（预注册，语言无关形态学/密度，规避词典翻译）：词密度、断句密度、拉长感叹
  （重复字母 ≥3，跨语言感叹形态）、数字词（比分念白）、全大写强调、槽内重复词。

pilot 判定（`r10_asr/asr_head.json`，同协议 5 折源分组 CV）：
- R1 prob：+0.0019 vs prior（λ 1/5 折正）。
- **R2 util：+0.0057 vs prior（CI [0, 0.0150]，λ 2/5 折正）**——与 AST-527 R2（+0.0046）
  同量级同方向；均按 POLICY_ONLY_CANDIDATE 语义记账。
- 置换对照=先验（0.6675）——泄漏通路不存在。
- **功效修正：σ_influence=0.0151，仅为 R10 8.3 假设（0.03–0.05）的一半以下**
  （本池先验强：88.5% frag top-6=先验集合，源级残差方差小）。
  由此 64 源 pilot MDE=0.0060（80% 功效单端点）——**pilot 并非无判定力**；
  +0.0057 贴线。全池 M=984 时 MDE=0.0015，可做实确认或否决。

## 3.3 当前证据格局与全池收窄决策（2026-10-07）

| 读出 | delta vs prior | CI | λ 正折数 | 状态 |
|---|---:|---|---:|---|
| AST pooler（R1/R2） | 0 | [0,0] | 0/5 | NO_PRACTICAL_GAIN @pilot |
| AST token-mean（R1/R2） | 0 | [0,0] | 0/5 | NO_PRACTICAL_GAIN @pilot |
| AST-527 语义分数 R1 | +0.0054 | [−0.009, +0.021] | 2/5 | 弱方向 |
| AST-527 语义分数 R2 | +0.0046 | [−0.001, +0.012] | 2/5 | 弱方向 |
| ASR 形态特征 R1 | +0.0019 | [0, +0.006] | 1/5 | 弱方向 |
| ASR 形态特征 R2 | +0.0057 | [0, +0.015] | 2/5 | 弱方向 |

**共同主题：语义/语义密度通路一致地弱正（+0.002~0.006），声学 embedding 对位一致 null。**
决策：全池 AST 提取与全池 ASR 转录并行铺开（纯 CPU，~192 核预算内），
对有方向的读出在全池（1,963 源）上收窄 CI 至门以下判定（R10 SCREEN 阶段完成态）。

## 3.4 全池 ASR 判定（2026-10-07，3,917 frags / 1,963 源 / 5 折源分组 CV）

- 转录全池完成（whisper-base，57 min，0 错误，0 无音轨缺失——冻结池 100% 有音轨）。
- **R1 与 R2 的 λ 全部 5/5 折选 0.0——两个读出在全池被完全关闭，输出=折内先验 0.6608**
  （champion 0.6588；+0.00364 CI [0.00093, 0.00643] 全正是「先验优于 champion」的已知事实，非信号）。
- pilot 的 +0.0057（σ_ψ 0.0151）为小样本噪声；全池残差方差=0。
- 判定：**NO_PRACTICAL_GAIN(ASR 语言无关形态特征，全池 scope)**。家族 S 的剩余通路
  （预注册事件词典/小型多语文本编码器语义特征）未测，但形态学/密度层已全池否证。

## 3.5 家族 T（OCR 状态）覆盖关闭（2026-10-07，pilot 104 锚点）

- 引擎偏离记录：PaddleOCR 在昇腾容器 SIGSEGV（其 NPU 探测与 CANN 9.0.0 冲突）、
  EasyOCR 权重在 GitHub release 不可达 → tesseract 4.1（R10 4.2 内的备选，偏离已记录）。
- **104 锚点（13 frag × 8）：数字文本覆盖 0.0%，平均 5.1 字符/锚点；64.4% 锚点有零星文本
  （水印/台标类，非秒级事件信号）。**
- 按 R10 Q2 覆盖审计规则：本池可读比分/计分 UI ≈ 不存在 → 家族 T 的 H_coverage 上界≈0，
  **NO_PRACTICAL_GAIN(coverage scope)**——「比分变化→高光」机制在本池无输入。

## 4. 状态表（R10 关闭规则口径）

| 探针 | 状态 | 备注 |
|---|---|---|
| AUDIO_LEARNED | SCREEN | AST 三读出 pilot 完成（pool/mean null，527 弱正）；全池收窄中 |
| OCR_STATE | NOT_TESTED | 排后 |
| ASR_SEMANTIC | SCREEN | 32→132 片段覆盖审计 + pilot 判定完成（R2 弱正 +0.0057）；全池转录中 |
| COMPENSATED_MOTION | NOT_TESTED | |
| REGION_TOKENS | NOT_TESTED | ≤6 NPU·h 有界 |
| SPATIAL_IDENTITY | NOT_TESTED | |
| VLM_EVENTS | NOT_TESTED | |
| AV_INTERACTION | NOT_TESTED | 条件于音频缓存 |
| TRAIN_BANK_RECURRENCE | NOT_TESTED | |
| ONE_RESERVED_TIER3 | NOT_TESTED | |
