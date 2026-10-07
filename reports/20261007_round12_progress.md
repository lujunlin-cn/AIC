# R12 进度报告（2026-10-07 深夜）：NPU 重启后探针矩阵 + P0-a 对接审计进行中

前置：`reports/20261007_round12_r45_plan.md`（v1）、`20261007_round12_frontier_addendum.md`（v2）、
`20261007_round12_frame_ranking_frontier_v3.md`（v3，commit 9410741）。
执行地：910A（ssh -p 44000，整机重启完成后）。

---

## 1. NPU 健康判定：整机重启未恢复（R12-01）

用户指令「批准 R12，等 910A NPU 恢复再做后面操作」——本节即恢复性验证结论。

### 1.1 重启后探针矩阵（Qwen2.5-VL-7B 金标准，单卡 generate）

| 探针 | 卡 | 注意力 | 结果 |
|---|---|---|---|
| probe1 | 0（默认） | eager | **OOM**：卡 0 被两个非我方进程占满（各 16.3GB；`/tmp/mm_probe.py` + `r11_v1_event_roles.py` warmup，疑似另一会话的 7B 复验）。未触碰他方进程。 |
| probe2 | 6（空卡） | SDPA（默认） | **崩溃**：`scaled_dot_product_attention` 报 `can not cast format when output is input` + ERR01007 OPS not supported |
| probe3 | 6（空卡） | eager | 模型加载 56.8s 正常、基本算子真实执行（add-kernel 警告=有算力活动）；但 generate 24 token **>14 min 零产物**，AICore 三次采样 **0%**，裸点输出——**与 R11 判定八 TEFUSION 编译死循环完全同签名**，900s timeout 收尾无 PROBE_RESULT。 |

### 1.2 判定与根因指向更新
- **NPU 多模态 generate 线：重启后仍不可用**。纯文本/基本算子执行健康（加载、小算子、
  前向启动都正常）。
- **判定八的「host 侧残留态损坏」假设被削弱**：整机重启理应清零 daemon/共享内存/缓存库，
  故障依旧 → 根因更可能位于 **CANN 9.0.0 算子编译层固有缺陷**（特定算子的 TBE 编译任务
  投递后永返 size=0），而非状态残留。升级排障价值上升：华为工单/CANN 版本降级实验
  （9.0.0 ↔ 8.3.RC2 配平矩阵）值得一试，但属用户裁定范畴，未执行。
- 新坑入册：**910A 上 Qwen2.5-VL 必须显式 `attn_implementation="eager"`**（默认 SDPA 直接
  OPS 崩溃，报错指向 sdpa_attention.py:158）。此前记忆只写了 FP16 eager，未写默认值会崩。

### 1.3 过程事件（如实记录）
- 卡 0 被他方会话进程占满（未触碰，只换空卡）。
- 指纹库 32 分片曾被外部信号整批扫掉（日志止于 22:29-32、零异常、无 OOM，疑似另一会话
  broad pkill 误伤）——**setsid 重启后恢复**，断点续跑设计验证有效。CPU 批处理启动一律
  加 setsid 入册。

## 2. P0-a：426 × PHD² 对接审计（CPU，进行中）

### 2.1 素材链（全部就位）
- 官方 426：`/data/aic/semifinal_20261001/intake/index.enriched.jsonl`（426 行，duration/fps/
  sha256 全）+ `keyframes/keyframes/<id>/*.png`（~11 帧/集，帧号命名）→ **指纹已算完**
  （`official_kfs.jsonl`，426 行）。
- PHD² 侧：raw/youtube **13,993 源在盘**（399GB，文件名=YouTube ID）+ selections
  train/test（110,611 + 10,222 vid，**键=YouTube ID**，per-user GIF 区间 t0/t1/is_last）。
  media.jsonl 的 media_path 未回填（不阻塞，直接扫盘）。
- PHD² 指纹库：32 分片 × 每 2s 一探针（上限 400/视频）→ 断点续跑中（重启后 4,463+
  基础上继续），完成后 13,993 视频全覆盖。

### 2.2 匹配设计（p0a_match.py，已冒烟）
- 逐官方视频 × 全指纹库 Hamming 矩阵（63-bit pHash），候选=命中数最多者；
- 复验：按候选估 offset，回源解码逐帧复验（中位 Hamming ≤14 且内点率 ≥0.6）；
- 对上后 join selections GIF 区间（平移进官方时间轴），报 window 内区间数。
- ⚠️ 诚实预期管理：V10 当年 174 集审计（map174_stage2-5，本日重读档案）的失败形态是
  「时长匹配成功（71/174 有媒体时 71 scored）、首帧 mae≈25 相似、**中后段发散（70-80）**、
  GIF 区间落在窗口外（cover_share=0）」→ 判「非连续裁剪」。本次用 pHash 内容匹配
  （比当年时长匹配强一代），但若 426 仍是非连续裁剪，复验会整批不过——两种结局都
  写进判定，不预设。

### 2.3 判定预期（预注册式，两种结局都合法）
- 结局 A（对上子集）：官方域训练数据解锁 + **官方窗口切割函数可估计**（offset 分布）→
  按「GIF 时刻 ± pad」重切 PHD² 训练池 → 复赛镜像池（治「本地指标无平台预测力」的
  池协议病根）。
- 结局 B（整批不过）：与 V10 结论一致化——426 非连续裁剪自 PHD² 在盘源；对接线永久
  关闭，监督解锁改走 QV/user-vote/EMO 包。
- 产出物：`/data/aic/experiments_910a/LFM_V11/r12_p0a/{candidates,matched}.jsonl +
  summary.json`；判定报告待 bank 完成后出。

## 3. 前沿 v3 已入库（本日早些时候，9410741）
- TRINITY（ECCV 2026）Mr.HiSum SOTA 表：冻结 CLIP + Transformer(d5,h512,1D-RoPE)+BCE
  40.98/69.06 最优；AHA 低于专用监督基线 → R12-AHA 降级、R12-GATT 改指 TRINITY 配方、
  新增 R12-EMO（情感峰值辅助视角，批次按视角隔离——MixBatch 反例 58.74→42.19 已录）。
- 全文：`reports/20261007_round12_frame_ranking_frontier_v3.md`。

## 4. 下一步（等用户 NPU 通知后）
1. **NPU 快速复验**（10 分钟）：eager 探针重跑一次；若仍死锁，建议裁定走华为工单或
   CANN 8.3.RC2 双栈实验（动驱动层，风险自担）。
2. **P0-a 收口**（CPU，不依赖 NPU）：bank 完成后跑 p0a_match → 出结局 A/B 判定报告 +
   `EXPERIMENTS.md` 入账。若结局 A：窗口切割函数 → 复赛镜像池构建排期。
3. **R12 机制包**（需 NPU 或按 CPU 预算逐包裁剪）：TRINITY 式 GATT 头 CPU 可训（特征在盘），
   Mr.HiSum pairwise ranking 管线激活（P0-b）不依赖 NPU——可在 NPU 等待期先动。
