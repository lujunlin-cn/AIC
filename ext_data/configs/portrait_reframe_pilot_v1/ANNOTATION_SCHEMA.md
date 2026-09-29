# portrait_reframe_pilot_v1 — 真实竖屏 → 16:9 补标任务规范

状态：**pending_annotation**（2026-09-29 建立任务；媒体依赖社区 AVE-PM 缓存的后台下载，标注未开始）。
采样清单：`/data/aic/external_datasets/PM400/pilot/sampling_manifest_v1.json`（本地副本随本目录提交）。

## 任务定义

对每个 PM-400 竖屏源视频（9:16）抽取关键帧，标注一个**合法 16:9 裁剪窗口**——
即"真实竖屏源转 16:9"的定位监督。这是官方 119 个 y 轴视频（横屏→9:16）之外
缺失方向的同构任务：窗口不可缩放、不可旋转，只有一个自由度。

## 合法窗口约束（唯一自由度）

- 窗口宽度 = 源宽 W（全宽），窗口高度 = round(W × 9 / 16)。
- 自由度仅剩竖直位置 y ∈ [0, H − h]（整数像素，ltrb：[0, y, W, y+h]）。
- 不允许缩放、不允许裁出画面、不允许旋转。

## 关键帧选择（标注者看到什么）

- 每源最多 12 帧：t = 0.5 s 起，每 2 s 一帧，超出源时长即停。
- 标注者可为某帧标记 `skip_transition=true`（镜头转场帧不标注）。
- 首版不做自动转场检测；这是刻意的简单基线。

## 输出格式（每人每帧一行 JSON）

```json
{"pm_video_id":"v0200fg10000…","frame_idx":3,"t_sec":6.5,"window_y":412,
 "annotator_id":"<anon>","skip_transition":false,"confidence":"sure|unsure"}
```

多标注者保留各自行，不预先合并；合并策略（均值/中位数/最大窗）在标注完成前
单独冻结，不允许看到分数后再选。

## GT 地位（红线）

- **人工窗口标注完成后才是 GT**。PM-400/AVE-PM 自带的 86 类动作标签、
  事件起止（event_start/end）、BGM 标记一律只作为**上下文参考字段**随行携带，
  不得当作裁剪 GT、不得当确认集、不得进任何"人工监督"统计。
- AVE-PM 事件区间不等于高光区间；事件之外不等于应删帧。
- 新竖屏伪标签（若有）只能进弱监督训练池。

## 划分与组规则

- 源组键 = `pm_video_id`；同一源切出的所有 AVE-PM 片段共享组，不得跨
  train/dev/confirm。
- pilot_split 由 `sha256("portrait_pilot_v1:"+pm_video_id)` 分桶（2/3、1/6、1/6），
  见采样脚本 `sample_portrait_pilot.py`；不允许人工调整单个视频的划分。
- 320 源 / 240 片段引用为本轮起始设计，非最终训练集规模。

## 许可与用途登记

- PM-400 数据 CC BY-NC-SA 4.0，非商业研究用途（bytedance/Portrait-Mode-Video
  DATA.md 原文）；下载即视为同意。用途：视频裁剪/重构帧算法研究。
- 原始抖音直链已于 2026-09-29 实测全部失效（12 链接 × 3 种请求头全 502，
  含 AVE-PM 官方 filtered 清单）；媒体获取走 AVE-PM 社区缓存
  （Google Drive id `1zXOeVE9xaMprd1O_Xec65Wn7B5VWelLT`，68,067,442,732 B，
  已于 2026-09-29 启动后台下载，tmux 会话 `avepm_cache`）。
