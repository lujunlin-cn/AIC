# portrait_reframe_pilot_v2 — 真实竖屏 → 16:9 补标任务规范（v2）

状态：**pending_annotation**。v1（`../portrait_reframe_pilot_v1/`）冻结保留；v2 只修正确性
问题与标注协议，不改划分、不改 320 源清单（沿用 v1 的 `sampling_manifest_v1.json`，
组键/分桶规则不变）。

v2 相对 v1 的更正（V6 指南 6.1 点名）：

1. **方向表述更正**：官方 174 视频中 **119 个是竖屏源转 16:9**（上下移动窗）——本任务与之
   **同方向**，是它的原生域监督补充；v1 误写为"官方 119 个 y 轴视频（横屏→9:16）"。
   横屏→9:16 是另外 55 个视频的方向，与本任务相反。
2. **坐标口径**：正式窗口高度是**连续值** `h = W × 9 / 16`（精确 16:9），不得取整；
   标注者拖动的把手按显示像素吸附，但记录层写连续值。行格式同时给
   `window_y`（整数像素，标注输入）与 `window_bottom_cont = window_y + W*9/16`（连续，正式框）。
   显示舍入只发生在渲染层，永不进入正式框。
3. **双标注者协议**：每源 ≥2 名独立标注者（互不可见）；两人窗中心差 > 0.1×h 时触发
   第三人或保留分布，**不伪造一致 GT**。保留每标注者原始行（v1 已定，v2 重申）。
4. **时间段标注**：在预先固定的一部分完整片段上（`sampling_manifest_v1.json` 中
   `temporal_subset: true` 的片段），标注者按播放上下文标"应保留 / 可删 / 不确定"
   时间区间（`keep|trim|unsure`）；**事件区间之外不自动标负**（AVE-PM 事件≠高光）。
5. 每源标注上限、抽帧、skip_transition、许可与红线（人工标注完成后才是 GT；事件/类目/BGM
   仅上下文）**全部沿用 v1**，此处不重复。

## v2 输出行格式（每人每帧一行 JSON）

```json
{"schema":"portrait_reframe_pilot_v2","pm_video_id":"v0200fg10000…","frame_idx":3,"t_sec":6.5,
 "source_w":720,"source_h":1280,"window_w":720,"window_h_cont":405.0,
 "window_y":412,"window_bottom_cont":817.0,
 "annotator_id":"<anon>","annotator_round":1,
 "skip_transition":false,"confidence":"sure|unsure"}
```

## 时间段行（仅 temporal_subset 片段）

```json
{"schema":"portrait_reframe_pilot_v2","pm_video_id":"…","kind":"temporal",
 "span_start_sec":2.0,"span_end_sec":9.5,"label":"keep|trim|unsure",
 "annotator_id":"<anon>","note":null}
```

## 质检（交付时随附 `validate_annotations.py`）

- 结构校验：schema 字段、窗口约束 `0 ≤ window_y ≤ H − h`、比例常量、必填字段、有限数。
- 覆盖统计：每源标注者数、每帧行数、skip_transition 率、confidence 分布。
- 一致性：同源同帧两名标注者的 |y1−y2| 分布；>0.1×h 的帧占比（触发第三人的名单）。
- 时间段：重叠/矛盾区间计数（同标注者 keep 与 trim 重叠 → error）。
- 输出：机器可读 QC 报告（JSON）+ 人工可读摘要；**不做任何合并**，合并规则在标注完成后、
  打分前按本文件第 3 条单独冻结（v1 承诺沿用）。

## 与 E1 的接口

- 标注行直接进 E1 数据接口：`window_y / h_cont` → 归一化自由轴位置；annotator_round 用于
  学习曲线（先 60–80 源，再补全 320 源）；`source_group = pm_video_id` 沿用 v1 分桶。
- 教师伪标签（若有）只进弱监督池，行上 `label_origin: teacher` 与人工行
  `label_origin: human` 永远分开计数。
