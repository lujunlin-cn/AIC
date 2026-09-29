# ext_data：外部公开数据采集与预处理

本目录是这一轮外部数据工作的代码与报告，和主仓库 `aic/`、`scripts/` 分开，不改动另一个 agent 的产物。大数据只放在远端数据盘，本目录只放代码、配置和报告。

- 远端代码目录：`/home/supie/AIC/ext_data`，由本地仓库同步过去
- 远端数据根目录：`/data/aic/external_datasets`，可用环境变量 `AIC_EXT_ROOT` 覆盖

## 目录约定

```
<EXT_ROOT>/<Dataset>/
  raw/          原始包：archives/ 下是下载原件，解压后的媒体按数据集自带的组织方式保留
  annotations/  上游仓库快照（upstream_repo/，已固定 commit），以及解析后的逐条标注 JSON
  processed/    本数据集的索引：media.jsonl、annotations.jsonl、anomalies.jsonl、
                dataset_card.json、timelines/*.npz（真实 PTS）；viz/ 下放可视化样例
  splits/       预留
  logs/         events.jsonl、validation.json、YouTube 队列与失败清单
<EXT_ROOT>/_registry/
  media / annotations / anomalies / source_ledger  各一份 .jsonl 和 .parquet
  status.json、overlap_report.json、validation_summary.json、registry_status.md
```

## 环境

所有命令都在远端执行，先执行：

```bash
source /home/supie/AIC/ext_data/scripts/env.sh
cd /home/supie/AIC/ext_data
```

env.sh 做了这些设置：

- 解释器用 `/opt/miniconda3/envs/cv/bin/python`。
- 所有任务都用 `nice -n 15 ionice -c2 -n7` 降低优先级，并设 `OMP_NUM_THREADS=2`。
- 下载前检查磁盘，至少保留 700 GB 空闲（`AIC_EXT_MIN_FREE_GB`）。
- 设 `AIC_EXT_PROXY=http://127.0.0.1:18890`。这是本地工作站开的反向隧道，只用于 Google Drive、Box、HF、YouTube 等远端无法直连的域名（域名表见 `aicext/download.py` 的 `PROXY_HOSTS`）。
- yt8m、hf-mirror.com、GitHub、作者服务器都直连。

长任务用 `bash scripts/run_bg.sh <session> <log> <完整命令>` 启动。每个任务一个 tmux 会话，结束后保留 10 分钟，方便查看退出码。

## 复现命令

每条命令都可以断点续跑。下载写入 `.part` 并按 Range 续传，校验大小和 sha256 后才改名；解压以 `.extracted` 标记；探测结果按 path|size|mtime 缓存。

注意：每次跑完 ingest 都必须执行一次 `bash scripts/refresh_all.sh`，依次做划分、注册表、校验。因为 ingest 会把 aic_split 重置为空。refresh 只刷新工作注册表 `_registry/`，不会改动 `_releases/` 下已冻结的版本。实验应该读取冻结版本，而不是工作注册表。

| 数据集 | 下载 | 接入 / 预处理 |
|---|---|---|
| DHF1K + RetargetVid | `python scripts/fetch.py --dataset DHF1K --gdrive 1UEFQmRdDbtVT-ePjMZVrv9oVV0ra631s --dest $AIC_EXT_ROOT/DHF1K/raw/archives/video.rar --expected-size 4032563381`，annotation.rar 同理（1zs63yz_MvIvKLV2Gv7SGnYtK3os_TDcl，3888600981） | `python scripts/ingest_dhf1k_retargetvid.py --workers 4` |
| LIVE-YT VC | Box 列目录并逐文件下载，已集成在脚本里 | `python scripts/ingest_live_yt_vc.py --stage all --workers 3`（阶段：list / labels / media / index） |
| Mr.HiSum | 作者 Drive 上的 h5 和 metadata；YT-8M 帧级特征从官方镜像流式下载 | `python scripts/ingest_mrhisum.py --stage features --workers 3`，然后 `--stage index` |
| YouTube Highlights | HF 镜像 tar（已记录 sha256）；缺失视频走 yt-dlp 队列 | `python scripts/ingest_youtube_highlights.py`；`python scripts/run_yt_queue.py --dataset YouTubeHighlights` |
| GAICD / DAVSOD / ClipShots | `python scripts/fetch_many.py --manifest configs/batch2_downloads.json [--only GAICD]` | `ingest_gaicd.py`、`ingest_davsod.py`、`ingest_clipshots.py --stage all` |
| PHD² | CSV 已在上游仓库快照里；YouTube 走 yt-dlp 队列（需要 `~/.local/bin` 下的 yt-dlp + deno） | 子集选定 `python scripts/select_phd2_subset.py --budget-gb 420`（tier0 = 测试集 is_last 官方指标视频，tier1 = 训练侧按"每小时 GIF 数"密度排序，预算内约 1.2 万个视频），下载 `bash scripts/run_bg.sh phd2_subset $AIC_EXT_ROOT/_logs/phd2_subset.log $AIC_EXT_PY scripts/run_phd2_subset.py`（预算感知、断点续跑，停在实际落盘字节达标的时刻） |
| LaSOT | 类别 zip 从 hf-mirror 下载，逐成员比对官方 LaSOTTesting.zip 的 CRC | `python scripts/ingest_lasot.py --categories mouse electricfan` |
| SA-V | 全量需要在浏览器里同意条款（阻塞） | `python scripts/ingest_sav.py`（只接入官方仓库自带的样例） |

其他命令：

```bash
bash scripts/refresh_all.sh                                         # 划分 → 注册表 → 全部校验
VALIDATE_DATASETS="GAICD ClipShots" bash scripts/refresh_all.sh     # 只校验指定数据集
python scripts/make_report.py --out $AIC_EXT_ROOT/_registry/registry_status.md
python -m pytest -q -p no:cacheprovider tests
```

## 冻结版本（releases）与暴露台账

- 台账 `_registry/holdout_exposure_v1.json`（本地副本：`configs/holdout_exposure_v1.json`）：
  - 从实验产物中读取每个源视频组的暴露角色，分 fit、diagnostic、selection、confirmation 四类，并冻结新的确认保留集。
  - 生成命令为 `python scripts/freeze_holdout_ledger.py --version v1 ...`。这个脚本拒绝覆盖已有台账，新的台账只能用新版本号。
- `project_split` 由 aic_split 叠加台账得到，每行附 `project_split_reason` 和 `exposure_roles`，`official_split` 不变。`Index.select` 默认按 project_split 过滤。
- 冻结版本位于 `_releases/<name>/`：
  - 内容包括 RELEASE.json（来源、规则、台账 sha、代码树 sha）、manifests、原始标注的字节级副本、packed npz、timelines、`code/` 快照和 SHA256SUMS。
  - 构建后设为只读，并追加到 `_releases/INDEX.jsonl`。
  - 构建：`python scripts/build_release.py`。质检：`python scripts/release_quality.py`，结果写到 `_releases/<name>.quality.json`。
- 版本内的划分：
  - train / dev：dev 是未暴露训练组中的 release_dev_v1 桶，占 10%。
  - confirmation：冻结的保留集加未暴露的 val。
  - exposed_eval：已用于验证，只能评测，不能训练。
  - excluded：每行附排除原因。
- 读取器在 `aicext/release.py`：`Release(name, verify=)`、`SpatialCropUnits`、`TemporalEvidenceUnits`、`FeatureSubsetUnits`、`score_unit(_frame)`、`preference_pairs`。
- 窗口打分器在 `aicext/window_scorer.py`：
  - 输入 W、H、目标比例和候选框。
  - 输出每名标注者的 IoU、聚合值、valid 掩码，以及候选的 legal / is_max_window 标记。
  - IoU 约定为 halfopen 或 inclusive_plus1，边界处理为 none、clip_v1 或 clip_exp_v1。
- 示例：`examples/load_releases.py`。当前版本的说明见 `reports/20260929_data_releases_v1.md`。

## 训练侧读取（adapters）

每种监督类型对应一个读取器，不会合并成统一的"highlight 标签"。所有读取器都返回 numpy。

- 取 DataLoader：用 `TorchDataset(adapter)`，配合 `collate_pad(batch, pad_keys)`。
- 变长数组：`collate_pad` 把所有轴补齐。补齐位置的 `valid` 为 False，并额外返回 `<key>_shape`。

| 类型 | Adapter | 数据集 |
|---|---|---|
| crop_box_dense | `DenseCropAdapter` | RetargetVid（6 名标注者 × 1:3 / 3:1） |
| crop_box_sparse / crop_box_derived | `SparseCropAdapter`（`annotation_type=` 选择） | LIVE-YT VC（30 个稀疏帧；插值结果单列，不是人工真值） |
| saliency_map / fixation_points | `SaliencyMapAdapter(kind=)` | DHF1K（连续显著图与二值注视点分开） |
| salient_object_mask | `MaskSequenceAdapter(variant=object_level\|instance_level)` | DAVSOD |
| image_crop_candidates | `CropCandidatesAdapter` | GAICD（候选框 + MOS；未评分项 valid=False） |
| temporal_segment_label | `SegmentLabelAdapter(field=, variant=)` | YouTube Highlights（match 为自动标签，mturk 为人工） |
| temporal_score_1d | `GridScoreAdapter(mode=features\|video)` | Mr.HiSum（gtscore；gt_summary 是算法派生的） |
| temporal_user_selection | `UserSelectionAdapter` | PHD²（用户未选的片段是未标注，不是负例） |
| shot_boundary | `ShotBoundaryAdapter(subset=)` | ClipShots（cut 与 gradual 分开；only_gradual 中非转场帧无效） |
| object_track_box | `ObjectTrackAdapter` | LaSOT（目标身份、遮挡和出视野逐帧保留） |
| object_masklet | `MaskletAdapter(source=manual\|auto)` | SA-V（manual 与 auto 分开） |

## 契约

- 时间：
  - 帧索引从 0 开始，按显示顺序。帧时间 = pts × time_base − 首帧 pts，全帧解码后存到 timelines npz。
  - 采样取每个网格时间点之后的第一帧真实帧，规则与 `aic.video` 相同。
  - 区间标签按 `[t0, t1)` 落到真实帧时间上。
  - 网格分数锚定在 `anchor + k·step`。Mr.HiSum 的 anchor 为 0.0，避免约 1 s 的偏移。
  - 帧目录数据集（DAVSOD、LaSOT）用来源时间线；没有来源时间线时用名义时钟，并在 `timeline_method` 里注明。
- 坐标：
  - 读取器统一输出 xywh，单位为编码像素（未经旋转），外加按 (W,H,W,H) 归一化的一份。
  - 每条标注的原始约定写在 `coord_format`，转换方式写在 `coord_target`，有歧义的地方写在 notes 或 dataset_card。
- 有效性：
  - 未标注的位置一律 `valid=False`，不填 0。
  - 派生结果（插值、缩放）单独成为一条标注，并记录 `derived_from` 和 `derivation`。
- 划分（`build_splits_and_ledger.py`）：
  - 按同源组划分，组键为 `yt:<id>` / `dhf1k:NNN` / `<dataset>:<id>`；同一源视频的剪辑、标注者、比例和派生版本都在同一组。
  - 保留 DHF1K 官方划分；其他数据集中评测性质的官方划分进 val，训练性质的进 train；没有官方划分的按哈希分，10% 进 val。
  - 与内部评测集（TVSum、QVHighlights val、SumMe）同组的一律 quarantine。
  - AIC 官方评测集只做字节级比对，从不接入。

## 报告

- 本轮进度与交付：`reports/progress_20260928.md`
- 冻结版本 v1：`reports/20260929_data_releases_v1.md`，质量报告在 `reports/releases/`
- 注册表的即时状态：由 `scripts/make_report.py` 生成，另有一份拷贝在 `reports/registry_status.md`
