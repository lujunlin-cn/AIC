# V5 研究周期总结（2026-09-29）

指南：`AIC_48.67分复盘与突破60分研究指南_V5_20260929.md`。母本统一 INTERP（`QWEN32B_DT_INTERP_V4_FINAL`，正式 **48.67**，zip `501cebce…`）。时间 mask、face 分支、B0 EMA、后处理全程冻结。执行平台：V100 远端（`/home/supie/AIC`，分支 `teacher-v5-20260929`）。运行台账：`registry.jsonl`；周期记录：`EXPERIMENTS.md` / `DECISIONS.md`。

## 提交顺序（供平台提交，均不预报分数）

1. **`QWEN32B_INTERP_XRERANK_V5_FINAL`** —— 本轮唯一候选包。
   zip sha256 `027fd12a51b128d87014e2aae006c93c179b5c2fe4f8eecb143a60b62fa9a150`（995,981 B）
   predictions sha256 `4f7cfe52b830ad1929dd266f1df032802273d4c78a51c4db3a8aeac85e94b934`
   174 视频 / 78,992 帧；成绩**待评**。
2. 无第二包：H3 gate2 在 rv_dev 上 CI 未过，按预注册不出包（见下）；P2 负结果不出包。

当前正式最佳保持 **INTERP 48.67**；新包未回分前不改变任何正式结论。

## P0：INTERP 上的左右方向视觉复核 —— 正结果，已打包

把 x 轴（axis0）行的 DENSE 自由轴点替换为 E4 式"看裁剪图"复核点（候选集 3–5 个同尺寸窗、
正序倒序两问一致才采纳、否则保留 DENSE 点），y 行与非空间行构造性零变化。这是 E4 在 V4
被否决的正确重启：E4 的选择器当时拖累 y 轴，P0 用管线级拼接把收益限制在它真正有效的 x 行。

| 集 | 全部 | axis_x | axis_y |
|---|---|---|---|
| dev2 (70v) | +0.0093 [+0.0036,+0.0161] | +0.0187 [+0.0073,+0.0323] | 0（构造） |
| confirm2 (100v) | +0.0094 [+0.0040,+0.0163] | +0.0188 [+0.0079,+0.0326] | 0（构造） |

- E4 sanity 复现：dev2 0.009302 vs 发表 0.0093；confirm2 0.007228 vs 0.0072（±0.0001）。
- 预注册 `configs/V5_P0_XRERANK_PREREG.json`（sha `ba5f1d17…`），全部规则通过。
- 包身份守卫（全硬失败型）：DENSE 点路径复现母包 174/174；rerank 仅改 x 行 x 分量；
  y 与非空间行零变化；keys==TEMP、bbox==spatial 父包逐帧 0 差异；guard_tripped=1 与母包相同
  （TEMP 移除继承守卫）。vs INTERP 实际差异 21 视频 / 10,273 帧（vid19 2,796、vid20 4,337）。
- 复现：`reports/20260929_v5_p0_xrerank_package.md`（评估 + 官方拼接 + mask combo 完整命令）。

## H3：真正读取空间视觉特征的候选评分器（G / V / V+Q 匹配消融）

**gate1 通过、gate2 在 rv_dev 上 CI 未过 → 按预注册不打包**；视觉特征有真实增量的科学
问题得到肯定答案。预注册 `configs/V5_H3_VISUAL_PREREG.json`（sha `eb9ee068…`，出结果前
写定）；训练 cuda:2 seed 1、2000 步（G 561s / V 991s / VQ 5,981s）；数据 rv 160/40/200 +
live 1,416/148/172 单元（全部 axis0，14,607 GT 关键帧），冻结 DINOv2 ViT-B/14
（sha `d73036b5…`）特征 2,136 单元全缓存。

| 门 | 规则 | 结果 |
|---|---|---|
| gate1 视觉价值 | best(V,VQ) − G > 0（rv_dev+live_dev 均衡 dev 关键帧 IoU） | **+0.0110**（G .6194 / V .6272 / VQ .6304）✅ |
| gate2 rv_dev | mean>0 且 CI_low>−0.002 且 axis1≥−0.002 | VQ mean +.0110 ✓ 但 **CI_low −.0112 ✗**（40 视频过窄） |
| gate2 confirm2 | 同上 | **VQ +.0190，CI_low +.0071，axis1 +.0172** ✅（三项全过，CI 下界>0） |

管线级配对（选择器替换 INTERP 点后走完整管线，视频配对 bootstrap 5000，Δ = 选择器 − 母本）：

| 集（视频数） | G | V | VQ |
|---|---|---|---|
| rv_dev (40) | +.0113 [+.0056,+.0175] | +.0117 [−.0118,+.0370] | +.0110 [−.0112,+.0361] |
| rv_confirm2 (200) | +.0077 [+.0038,+.0116] | +.0147 [−.0003,+.0307] | **+.0190 [+.0071,+.0313]**，y 轴 +.0172 |
| live_dev (148) | +.0040 [+.0020,+.0064] | +.0135 [+.0019,+.0250] | +.0145 [+.0033,+.0255] |
| live_val 未见过 (172，仅报告) | +.0017 [+.0002,+.0033] | +.0116 [+.0014,+.0220] | +.0113 [+.0020,+.0207] |

- 关键帧级（VQ）：selector−mother 在 rv_confirm2 +.0226、live_val +.0120；oracle 差距仍大
  （confirm2 oracle .830 vs selector .688），选择器只吃到 oracle 增益的 ~16%。
- **y 轴信号**：VQ confirm2 axis1 +.0172 —— 全项目首个在上下方向为正的定位信号
  （H1 在 y 轴 −1.73、E4 y −.0034）。
- 失败原因判定：rv_dev 只有 40 视频，CI 宽度 ~±.022 是结构性窄表问题，非信号缺失
  （confirm2 200 视频同号且 CI 下界 >0；live_val 全新 172 视频同号）。
- 值得注意：**纯几何头 G 在四个集全部为正**（含 rv_dev CI 下界 +.0056）——33 网格候选 +
  Huber 头的候选重评分本身已胜过母本；预注册 gate2 只考核视觉头（V/VQ），G 不在晋级范围，
  但它证明"候选重评分"这个框架有效，视觉特征在其上处处再加增量（除窄表 rv_dev 的噪声内）。
- 按预注册**不出包**；下轮若重启：先扩 rv_dev 划分（需重新预注册，不得事后钓标志）。
- 复现：`PYTHONPATH=/home/supie/AIC /opt/miniconda3/envs/cv/bin/python
  scripts/v5_h3_visual_scorer.py --tables $E/V5_H3_VISUAL/tables --visual $E/V5_H3_VISUAL/features
  --output $E/V5_H3_VISUAL/train --cuda cuda:2 --seed 1 --steps 2000`；
  消融表 `reports/20260929_visual_scorer_ablation.csv`，完整指标 `reports/h3/metrics.json`。

## P2：Qwen 点名主体 + 检测器定位 —— 负结果，关闭

confirm2 24 视频（axis×motion 四层分层）。Qwen3-VL-32B 从 ≤6 个采样关键帧点名 1–4 个主体
短语，GroundingDINO-tiny 在全部 DENSE 关键帧上落位，qwen 顺序 + 覆盖率 ≥60% 选身份，
自由轴框中心替换 DENSE 点，缺检回退，走完整 INTERP 管线，配对 bootstrap 5000。

| 变体 | Δ (ground−dense) | CI95 | better/worse |
|---|---|---|---|
| v0 无条件替换 | −0.0657 | [−0.1063, −0.0317] | 7/16 |
| v1b 限幅 ≤0.75 窗宽 | −0.0385 | [−0.0621, −0.0200] | 5/18 |

四分层全负。机制：单帧 top-1 开放词表框中心的方差大于 DENSE 蒸馏点；限幅只截跳变不改偏置。
与 P0 对照结论："再看一眼"在**候选裁剪空间**有效（+0.0093）、在开放词表框空间无效。
重启前置命题：跟踪+平滑后的检测中心须先离线胜过 DENSE 点。详见
`reports/20260929_v5_p2_ground_pilot_negative.md`。

## 数据侧：PM-400 / AVE-PM 实际就绪情况

- **直链全灭**：PM-400 `video_links.csv`（97,681 行）中抽 12 条 × 3 组请求头（含 AVE-PM
  过滤列表条目），2026-09-29 实测全部 HTTP 502；主仓 `main/` 路径 404。协议与输出留档
  `artifacts/pm400_links/`、远端 `/data/aic/external_datasets/PM400/raw/links/`。
- **可用通道**：社区 AVE-PM GDrive 缓存（id `1zXOeVE9…`，68,067,442,732 B），后台下载中，
  15:01 时 17.6 GB（26%），~4.6 MB/s，ETA ~17:50。下载完成前 `avepm_cache.tar.part` 为部分
  文件，**不交付、不解读**。
- **补标任务已建**：`ext_data/configs/portrait_reframe_pilot_v1/`
  （`ANNOTATION_SCHEMA.md` + `sampling_manifest_v1.json`）。320 源 / 86 类全覆盖，
  215/51/54（train/dev/confirm）按 `sha256("portrait_pilot_v1:"+pm_video_id)` 划分，
  源视频（pm_video_id）绝不跨划分；240 片段带 AVE-PM 事件上下文，字段
  `context_only_not_crop_gt: true` —— **事件/类目/BGM 仅作补标上下文，不冒充人工裁剪 GT**。
  状态 `pending_annotation`，合并策略已在规范中冻结（逐标注者行不合并，打分前冻结合并规则）。
- 竖屏→16:9 补标自由度：仅 y 轴（w=W，h=round(W×9/16)），每源 ≤12 帧、2 s 间隔、
  0.5 s 起步，转场帧打 `skip_transition`。

## 174 视频独立校验与守卫汇总

- P0 包：validator / 独立 checker（逐视频 keys、逐帧 bbox、mask 复算）/ 解压往返全过；
  键集与 TEMP 完全一致（78,992 帧）；官方索引 sha `9a59316e…` 校验一致。
- H3 若出包，走同一 `mask_combo_release` + 独立校验流程（keys==mask 父包、bbox==spatial
  父包、guard_tripped 与母包核对）。
- P2 无官方产物（负结果，不出包）。

## 预算与可中断性

- P0 官方 rerank 55 x 视频 ~20 min（GPU 4-7，vLLM）；H3 训练三头各 ~10 min（2000 步）+
  评估 ~30 min，cuda:2 单卡，可中断（特征已全部缓存为 2,136 个独立 npz）。
- 教师查询零新增：P0 复用 E4 缓存 rerank 产物；H3 不查询教师；P2 只用了 24 视频的命名
  与检测（pilot 预算内）。
