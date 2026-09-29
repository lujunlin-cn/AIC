# 冻结数据版本 v1（2026-09-29）

三个版本都已冻结：只读（444/555），带 SHA256SUMS，并登记在 `_releases/INDEX.jsonl`。后续任何修改都只能发布新版本号。refresh_all 只刷新工作注册表，不会改动 `_releases/`。

| 版本 | 远端路径 | SHA256SUMS 的 sha256 | 文件数 |
|---|---|---|---|
| spatial_crop_v1 | `/data/aic/external_datasets/_releases/spatial_crop_v1` | `faa14d8b216a691e6f385dd79fc0f0a00b20c7695885aca3bd5a7e312256a893` | 8225 |
| temporal_evidence_v1 | `/data/aic/external_datasets/_releases/temporal_evidence_v1` | `5c65c70707b27eaf26bafa41d298d413b81ebb540a2305c55eb111f33f471216` | 2188 |
| mrhisum_feat_subset_v1 | `/data/aic/external_datasets/_releases/mrhisum_feat_subset_v1` | `3f60c686c2bbd512c8be8fe076d1d54eb46250bdf56efc3867fedc42cf24692c` | 1522 |

- 台账：`_registry/holdout_exposure_v1.json`，sha256 为 `64bc3e33…fb7ff7`。本地副本是 `configs/holdout_exposure_v1.json`。
- 代码快照：每个版本自带 `code/` 目录，并记录 code_tree_sha256。空间和时序两版是 `3568d0c3…d2fd9a`，Mr.HiSum 版本是 `70ede85f…ba6b22b`，因为它构建得晚一些。
- 媒体不复制进版本，只记录路径和 sha256，可用 `Release(..., verify="all")` 逐个校验。
- 质量报告：`_releases/<name>.quality.json`，本地拷贝在 `reports/releases/`。所有文件都已按 SHA256SUMS 复核，bad=0。

## 划分与台账

- 暴露记录直接从实验产物读取，覆盖 SPATIAL_SUBJECT_002、MAX_WINDOW_* 的 per_video 划分、T0/T1_T2 的 keyframes.jsonl（confirm2 取自 split 字段，不做推测）、T5 tables、T3/T4 records、V4_E1_OBS025 的 rv_subset（未登记的进行中实验也计为暴露），以及 QVH bounded。每个来源文件的 sha256 都写在 evidence_sha256 里。
- 共有 2284 个源视频组被标记为暴露，按角色分为 fit、diagnostic、selection、confirmation 四类。
- 新的确认保留集随台账一起冻结：LIVE 227 个组，YTH 47 个组。用盐 `confirm_reserve_v1` 按哈希分桶选出，比例 0.15。
- `official_split` 保持原样不动。`project_split` 在 aic_split 之上叠加台账得到，每一行都写了 `project_split_reason` 和 `exposure_roles`。
- 验证用过的视频，以及同源视频在其他数据集里的副本，按组键 `yt:` / `dhf1k:` / `liveytvc:` 整组排除在训练之外。
- 版本内的划分如下：

| 版本内划分 | 来源 |
|---|---|
| train | project_split=train，扣掉 dev 桶 |
| dev | 未暴露训练组里 sha256("release_dev_v1:"+组) 分桶 < 0.10 的部分 |
| confirmation | 冻结的保留集 + 未暴露的官方 val |
| exposed_eval | 已用于选择或确认的组：只评测、永不训练，也不算新鲜确认 |
| excluded | quarantine 以及对齐等问题，每行都写明原因 |

- 构建时会检查同一组是否跨划分，结果为 0。
- `Index.select` 现在默认按 project_split 过滤。实验侧应该读取冻结版本，而不是工作注册表。

## spatial_crop_v1（RetargetVid + LIVE-YT-VC）

| 数据集 | 原始媒体 | 有效人工标注 | 独立源视频 | train | dev | confirmation | exposed_eval |
|---|---|---|---|---|---|---|---|
| RetargetVid | 200 | 2400 个文件（6 名标注者 × 2 种比例），1,472,208 个框 | 200 | 0 | 0 | 0 | 400 单元 / 200 视频 |
| LIVE-YT-VC | 1800 | 1800 × 30 帧，54,000 个框 | 1800 | 1115 | 124 | 227 | 334 |

- 就绪状态：两个数据集都满足 schema 通过、样例可读、原视频可训练。但 RetargetVid 在 v1 中没有训练划分。
- 每个单元的 `boxes_xywh` 形状为 [A,K,4]，并带 `valid[A,K]`，保留每名标注者和两种比例，同时保留原始 `ltrb_raw`。
- LIVE 的插值轨迹放在 `derived/` 下，共 1566 条，只取单场景视频。只有显式传 `include_derived=True` 才会返回，字段名统一加 `derived_*` 前缀。
- 人工框的字段带 `gt_source="human"`。LIVE 的 `gt_is_max_window=False`。
- 坐标约定为 `xywh_halfopen_v1`（x=l，w=r−l，不裁剪）。原始 CSV 和标注文件按字节原样复制，打包数据与原始数据的最大差为 0。
- 窗口打分器 `aicext/window_scorer.py`：
  - 输入：W、H、目标比例、候选框。
  - 输出：每名标注者的 IoU[C,A]、均值、最小值、最大值、valid 掩码、候选是否合法（legal），以及是否为最大窗口（is_max_window）。
- IoU 约定有两种：`halfopen`（默认）和 `inclusive_plus1`，后者与 RetargetVid 上游及教师脚本一致。边界处理有三种：none、clip_v1、clip_exp_v1。三者分别计算，互相对照：

| 检查 | 结果 |
|---|---|
| 复现 T5 目标（inclusive_plus1，LIVE 用 clip_exp_v1） | rv_confirm2、rv_dev、live_confirm3、live_dev 四组的 max_abs_diff 都是 0 |
| +1 约定与半开约定的 IoU 差 | LIVE 均值 0.0012，RV 均值 0.0050 |
| +1 约定改变候选 argmax 的帧比例 | LIVE 27.3%，RV 5.0% |
| 最大窗口能达到的 IoU 上限 | RV 中位数 0.999；LIVE 中位数 0.871，p5 0.340 |
| 真实 PTS 时间线 | 2000/2000 条长度等于帧数，首帧为 0，严格递增，rotation=0；抽样重新解码，PTS 完全一致 |
| 解码与 batch | 首、中、末帧解码正确；padding 位置的 valid 全为 False；RV 与 LIVE 混合 batch 正常 |
| 显著性辅助检查（只用来看框与帧是否对齐） | 框内/框外比值中位数 22.1，94.7% 大于 1 |
| 越界框 | LIVE 有 92 个框超出画面，保留原值，边界处理按版本号区分 |

GAICD 不在这个版本里，作为单独的图像辅助数据：期刊版和会议版不重复计数，MOS 也不转换成 highlight。

## temporal_evidence_v1（YouTube Highlights）

- 实际交集：
  - 作者列表 550 个视频，其中 419 个有媒体。
  - MTurk 标注 413 个视频，其中 315 个有媒体。
  - 最终发布 316 个视频，其中 233 个带 MTurk 标注。
- 排除 234 个视频，原因如下：
  - 无媒体：131 个（已删除或私有，不再重试）。
  - 帧数不一致：84 个。
  - 疑似错位：14 个。
  - 镜像告警：5 个。
- 帧映射规则：标签帧 f 对应解码后的显示帧 f。只保留"最后一个标签帧与解码帧数之差 ≤ 3"的视频。

| 划分 | 视频 | 其中带 MTurk | 领域 |
|---|---|---|---|
| train | 164 | 101 | dog 28、gym 22、parkour 11、skating 11、skiing 49、surfing 43 |
| dev | 17 | 10 | skating 为 0 |
| confirmation | 48 | 35 | 六个领域都有 |
| exposed_eval | 87 | 87 | T3/T4 用过的验证代理 |

- 监督分为几个通道，彼此不合并：

| 通道 | 含义 | 可靠性 |
|---|---|---|
| `votes_max` / `votes_mean` | 原始投票，按约 2 s 的片段给出 | 数值保持原样 |
| `human_code=2` | 有人选中（votes>0） | 显式正例，共 2465 个片段 |
| `human_code=1` | 无人选中（votes=0） | 弱信号，不能当可靠负例（8805 个片段） |
| `human_code=0` | 未标注（NaN） | 无监督 |
| `preference_pairs` | 同一视频内按投票排序的相对偏好 | margin=1 时共 39,478 对 |
| `auto_code` | 自动 match 标签，分四档：3 matched、2 borderline、1 unmatched_weak、0 unlabelled | match=−1 不是负例 |

- 上游只公开了每个片段的软合计，没有逐个 worker 的投票、worker 数和 worker ID。
- 时间线：316/316 条都通过检查。解码抽样正确，padding 位置的 valid 全为 False。

## mrhisum_feat_subset_v1（纯特征，不算原视频数据）

- 这是固定子集，包含构建时已完成的全部 1462 个分片。
- 各划分的视频数：train 25,069，dev 2,808，confirmation 4,008。
- 排除 7 个视频：1 个与内部评测集重叠，6 个的特征没有抽取出来。
- 抽查 train 前 2000 个视频，标签长度都等于特征步数。
- 就绪状态：schema 通过、样例可读、可用于训练特征头；不能当原视频训练数据用。
- gtscore 来自 Most-Replayed 的观众行为聚合，属于弱信号；gt_summary 是算法生成的。

## 最小加载示例

```bash
source /home/supie/AIC/ext_data/scripts/env.sh && cd /home/supie/AIC/ext_data
CUDA_VISIBLE_DEVICES="" $AIC_EXT_NICE $AIC_EXT_PY examples/load_releases.py
```

2026-09-29 09:47 实跑输出（节选）：

```
spatial batch ['LIVE_YT_VC', ...] (4, 1, 30, 4) (4, 1, 30) [False, False, False, False]
scorer liveytvc:video10 (65, 1) best cand 29 0.6804 {'coord': 'xywh_halfopen_v1', 'iou': 'halfopen', 'gt_boundary': 'none', 'scorer': 'window_scorer_v1'}
LIVE (1, 30, 4) gt_is_max_window False ceiling 0.6804 derived (180, 4)
temporal batch (4, 1590) [ 109   62 1590  676] (4, 1590)
pairs (124, 2) codes (array([1, 2]), array([60, 63]))
```

测试：`tests/test_release_scorer.py` 与原有测试共 16 项，在远端全部通过。

## 已知限制

- RetargetVid 的 200 个源视频全部用于验证过，因此只能进 exposed_eval，v1 里没有 RV 训练集，也没有新鲜的 RV 确认集。
- LIVE 的确认保留集（227 个）曾参与 T5_CROPHEAD_V3 的 live_train 拟合。对 T5 派生模型来说，它不算新鲜数据；每个组都记录了这一点。
- LIVE 的框不是最大窗口，上限中位数只有 0.871，不能当成最大窗口真值使用。
- +1 约定会改变 27% LIVE 帧的 argmax。换约定比较分数时，必须同时注明 iou 和 gt_boundary。
- YTH 的人工子集很小：train 只有 101 个视频带 MTurk，dev 缺少 skating。
- 84 个 YTH 视频疑似是 60 fps 重传版，标签只覆盖约前一半帧。v1 没有尝试修复。
- T3/T4 使用的 124 个 YTH 验证代理视频里，有 37 个存在对齐问题：32 个帧数不一致，5 个疑似错位。清单见 `reports/releases/yth_proxy_alignment_flags.json`。建议实验侧复核这部分结论。
- 与官方测试集（173 个）做了字节级比对，命中为 0。但这只能排除字节完全相同的文件，排除不了重编码、裁剪或内容层面的重叠。
- Mr.HiSum 只提供特征，不计入原视频数据量。

## 后台采集（截至 2026-09-29 09:46）

| 数据 | 状态 |
|---|---|
| DAVSOD 训练集 + 验证集 | 已完成：107 个视频，全部通过校验，并已接入工作注册表 |
| DAVSOD 测试集（Easy-35 / Normal-25 / Difficult-20） | 代理链路正常（generate_204 通过），但 Drive 返回 "Quota exceeded"；`batch4_fetch` 按有界 Range 退避重试，最多 200 次 |
| ClipShots a/b/c | 在测试集之后排队；a 已下载 64 MiB 的 `.part`，可断点续传。三个分卷是同一个 gzip 流，b、c 无法单独使用 |
| 接入 | `post_batch4` 等下载结束后，自动接入 DAVSOD 和 ClipShots，再执行 refresh；不会改动已冻结的版本 |
| 第二优先级 | Mr.HiSum 后续特征、PHD²/LaSOT 全量排在 batch4 之后 |
| SA-V | 暂缓 |
