# 外部数据注册表状态（自动生成 2026-09-28T19:26:59+08:00）

数据根目录：`/data/aic/external_datasets`；schema 错误 0 条。

| 数据集 | 就绪状态 | 媒体（就绪/总数） | 标注条数 | aic_split | 校验 | 占用 |
|---|---|---|---|---|---|---|
| ClipShots | 仅标注（媒体未就绪） | 0/5917 | 5917 | {"train": 5417, "val": 500} | pass | 0.0 GiB |
| DAVSOD | 部分就绪 | 61/61 | 244 | {"train": 61} | pass | 8.3 GiB |
| DHF1K | 原视频可训练 | 1000/1000 | 1400 | {"train": 600, "val": 100, "test_nolabel": 300} | pass | 15.1 GiB |
| GAICD | 图像可训练 | 3336/3336 | 4572 | {"val": 700, "train": 2636} | pass | 0.9 GiB |
| LIVE_YT_VC | 原视频可训练 | 1800/1800 | 3366 | {"val": 178, "train": 1622} | pass | 8.0 GiB |
| LaSOT | 仅小样本 | 40/40 | 40 | {"val": 8, "train": 32} | pass | 5.1 GiB |
| MrHiSum | 仅特征头可训练 | 0/31892 | 63784 | {"train": 27883, "val": 4008, "quarantine": 1} | pass | 2.2 GiB |
| RetargetVid | 原视频可训练 | 200/200 | 2400 | {"train": 100, "val": 100} | pass | 0.0 GiB |
| SA_V | 阻塞 | 1/1 | 2 | {"train": 1} | pass | 0.1 GiB |
| YouTubeHighlights | 原视频可训练 | 419/550 | 963 | {"train": 354, "val": 196} | pass | 18.5 GiB |

## 各数据集要点

### ClipShots
- 版本：labels 8d033e2591c23b00806179676e0a0b7461468281; videos https://drive.google.com/drive/folders/1AAhTbNroSFsygHBXa88emCU7f50MxI8t
- 来源：https://github.com/Tangshitao/ClipShots
- 可训练类型：shot_boundary
- 阻塞/缺口：videos: Drive parts not yet complete/extracted
- 下载状态：{"pending": 5917}；官方划分：{"train": 3657, "test": 500, "only_gradual": 1760}
- 标注类型：{"shot_boundary": 5917}
- 异常：{"odd_transitions": 72, "zero_frame_num": 1}（明细 `ClipShots/processed/anomalies.jsonl`）

### DAVSOD
- 版本：Drive packages (updated 2021-02-18)
- 来源：https://github.com/DengPingFan/DAVSOD
- 可训练类型：salient_object_mask, fixation_points, saliency_map
- 阻塞/缺口：missing ['DAVSOD-ValidationSet.zip', 'Difficult-20.zip', 'Easy-35.zip', 'Normal-25.zip']
- 下载状态：{"verified": 61}；官方划分：{"train": 61}
- 标注类型：{"salient_object_mask": 122, "fixation_points": 61, "saliency_map": 61}
- DataLoader 批次形状：[{"masks": [4, 4, 72, 128], "frame_idx": [4, 4]}]

### DHF1K
- 版本：author release: video.rar + annotation.rar (Google Drive)
- 来源：https://github.com/wenguanwang/DHF1K
- 可训练类型：saliency_map, fixation_points
- 下载状态：{"verified": 1000}；官方划分：{"train": 600, "val": 100, "test": 300}
- 标注类型：{"fixation_points": 700, "saliency_map": 700}

### GAICD
- 版本：journal + conference packages (Google Drive, author READMEs)
- 来源：https://github.com/HuiZeng/Grid-Anchor-based-Image-Cropping-Pytorch
- 可训练类型：image_crop_candidates
- 下载状态：{"verified": 3336}；官方划分：{"test": 500, "train": 2636, "val": 200}
- 标注类型：{"image_crop_candidates": 4572}
- DataLoader 批次形状：[{"boxes_xywh": [8, 90, 4], "mos": [8, 90], "valid": [8, 90]}]

### LIVE_YT_VC
- 版本：LIVE-YT-VC base release via Box; repo 2a91e9492c092c966b10fdc9b369b057d92e8ffb
- 来源：https://github.com/steven413d/LIVE-YT-VideoCropping
- 可训练类型：crop_box_sparse, crop_box_derived
- 下载状态：{"verified": 1800}；官方划分：{"none_published": 1800}
- 标注类型：{"crop_box_sparse": 1800, "crop_box_derived": 1566}
- 异常：{"out_of_bounds_boxes": 72}（明细 `LIVE_YT_VC/processed/anomalies.jsonl`）
- DataLoader 批次形状：[{"boxes_xywh": [4, 30, 4], "valid": [4, 30], "frame_idx": [4, 30]}]

### LaSOT
- 版本：LaSOT (1,400 seq, 70 categories); sample categories only
- 来源：http://vision.cs.stonybrook.edu/~lasot/
- 可训练类型：object_track_box (sample)
- 阻塞/缺口：full set = 70 category zips, ~248 GB; only a sample downloaded this round
- 下载状态：{"sample_only": 40}；官方划分：{"test": 8, "train": 32}
- 标注类型：{"object_track_box": 40}
- 异常：{"repacked_archive": 2}（明细 `LaSOT/processed/anomalies.jsonl`）
- DataLoader 批次形状：[{"boxes_xywh": [4, 32, 4], "visible": [4, 32], "valid": [4, 32], "frame_idx": [4, 32]}]

### MrHiSum
- 版本：mr_hisum.h5 + metadata.csv (author Drive), repo aeb667f71239887034c2974062a341d312762867
- 来源：https://github.com/MRHiSum/MR.HiSum
- 可训练类型：temporal_score_1d (feature head on YT-8M features)
- 不可用于：['VideoMAE / raw-video fine-tuning (no raw media)']
- 阻塞/缺口：YT-8M features for 23839 videos pending
- 下载状态：{"metadata_only": 31892}；官方划分：{"train": 27892, "val": 2000, "test": 2000}
- 标注类型：{"temporal_score_1d": 31892, "temporal_summary_derived": 31892}
- 异常：{"duplicate_source_video": 6}（明细 `MrHiSum/processed/anomalies.jsonl`）
- DataLoader 批次形状：[{"features": [8, 285, 1024], "gtscore": [8, 285], "valid": [8, 285]}]

### RetargetVid
- 版本：github bmezaris/RetargetVid@43673dd83b279c4aedeeea22f32d03582ac45194
- 来源：https://github.com/bmezaris/RetargetVid
- 可训练类型：crop_box_dense
- 下载状态：{"verified": 200}；官方划分：{"train": 100, "val": 100}
- 标注类型：{"crop_box_dense": 2400}
- DataLoader 批次形状：[{"boxes_xywh": [4, 6, 63, 4], "valid": [4, 6, 63], "frame_idx": [4, 63], "time_s": [4, 63]}]

### SA_V
- 版本：SA-V (CC BY 4.0); indexed: official example only
- 来源：https://ai.meta.com/datasets/segment-anything-video-downloads/
- 可训练类型：object_masklet (1 example video only)
- 阻塞/缺口：full download requires clicking through the SA-V license on the Meta download page, which issues an expiring link list (manual browser step); once links.txt is provided, scripts/fetch_many.py can take a manifest built from it
- 下载状态：{"sample_only": 1}；官方划分：{"train": 1}
- 标注类型：{"object_masklet": 2}

### YouTubeHighlights
- 版本：author repo 29083d2dc8ee951986f6fb436f1be8f92c8937fd; media hf:jhanglee/youtube-highlights-full@b28d0207471f8baf445dccb10c5e074a717f3b1d
- 来源：https://github.com/aliensunmin/DomainSpecificHighlight
- 可训练类型：temporal_segment_label
- 阻塞/缺口：131 author-list videos unavailable (not in mirror; yt-dlp: removed/private, see logs/missing_media.json)
- 下载状态：{"verified": 419, "missing": 131}；官方划分：{"TraiLooseL": 299, "TestLooseL": 162, "TestTightL": 34, "TraiTightL": 55}
- 标注类型：{"temporal_segment_label": 963}
- 异常：{"missing_media": 133, "misaligned_suspect": 14}（明细 `YouTubeHighlights/processed/anomalies.jsonl`）
- YouTube 队列：{"gave_up": 130, "done": 2, "failed": 1}（失败清单 `YouTubeHighlights/logs/yt_failures.json` / `yt_queue.jsonl`）

## 重叠与泄漏

- 跨数据集同源组：{"DHF1K & RetargetVid": 200, "DAVSOD & DHF1K": 60, "ClipShots & MrHiSum": 12, "DAVSOD & RetargetVid": 5, "MrHiSum & TVSum(in-house)": 1}
- AIC 官方测试集字节级命中：0（official test IDs are platform-native (0..173) without YouTube IDs; only SHA-256 byte identity can be checked; they are never ingested as training data）
- 官方 train 被移出 train：{"MrHiSum->val": 8, "MrHiSum->quarantine": 1}
- 官方评测项因同组被划入 train：{}
- 组内 split 冲突：0
- 划分策略：{"eval_like_official_splits": ["TestLooseL", "TestTightL", "test", "test_difficult20", "test_easy35", "test_normal25", "testing", "val", "validation"], "val_fraction_hash": 0.1, "inhouse_eval_groups_quarantined": 1, "inhouse_train_groups_seen_externally": 0}

