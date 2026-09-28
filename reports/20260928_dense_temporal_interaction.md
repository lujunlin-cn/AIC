# DENSE × TEMP 2×2 组合（DT_V3）

日期：2026-09-28 晚。依据《AIC_46.84分复盘与32B教师下一轮执行指南_20260928.md》§4。

## 2×2 表（平台分，官方返回前 DT 留空）

| 时间 \ 空间 | N0 原空间 | DENSE 空间 |
|---|---:|---:|
| 全选 | 42.83（NOFACE，`f73c399f…`） | 43.76（dense，`972b3ed1…`） |
| TEMP 掩码 | 46.84（TEMP，`631d0104…`） | **47.88**（DT_V3，`c702de92…`，用户 22:33 告知） |

47.77 = 46.84 + 43.76 − 42.83，只是“两模块无交互、分差可加”的算术参照，不是预测分。

## 官方返回（2026-09-28 22:33）

| 量 | 值 |
|---|---:|
| S_DT − 46.84（相对 TEMP） | **+1.04** |
| S_DT − 43.76（相对 DENSE） | +4.12 |
| interaction = S_DT − 47.77 | +0.11 |
| DENSE 效应：全选时 / TEMP 掩码下 | +0.93 / +1.04 |
| TEMP 效应：N0 空间 / DENSE 空间 | +4.01 / +4.12 |

按预先写死的规则，`S_DT > 46.84`，**DT_V3 晋级为新的已评分母本**（最高已知成绩 47.88）。

读法：DENSE 的收益在 TEMP 保留帧上没有丢失，两个模块在平台上近似可加，+0.11 的交互量级很小。每格只有一次提交，没有平台噪声估计和逐视频分数，不把 +0.11 解释成协同。与"DENSE 改动 91.6% 落在保留帧上"的预测变化统计一致，但这个统计本身不构成证明。距用户给出的历史 60 参照还差 12.12 分。

对后续实验的影响：T5 窗口头的母本是 TEMP、观察协议是 N0（按预注册保持不变）。若它通过确认，官方包仍按 TEMP 单变量出，与 DT 的叠加必须作为另一次有记录的组合，不能并入同一个包。

## 组合契约与校验

`DT_V3 = DENSE 全帧 bbox 按 TEMP 的 (video_id, frame) 集合过滤`。脚本 `scripts/mask_combo_release.py` 不调用教师、不插值、不平滑、不重新切镜。

| 检查 | 结果 |
|---|---|
| 母本 SHA（DENSE / TEMP / N0 predictions） | `e26c68fd…` / `730d341e…` / `fcdd98dc…`，均与冻结记录一致 |
| 掩码来源 | 由 `temporal/removal` 回复 + `keep_frames` 重算，逐视频 == TEMP 帧集合 |
| TEMP 自身 bbox == N0 | 0 差（再次确认 TEMP 只删帧） |
| `keys(DT) == keys(TEMP)` | 174/174 视频一致，0 差 |
| `bbox(DT) == bbox(DENSE)` | 78,992 帧 0 差 |
| 视频行 / 预测数 | 174 行，78,992 帧（= TEMP），空视频 0；不补帧 |
| 视频顺序、比例、边界 | 顺序 == index；targetRatioWH == 母本；项目 validator + 独立 checker + 解压后 checker 均 valid |
| 独立解压复核 | `scripts/verify_combo_zip.py` 直接解压三个 ZIP 比对（不走打包代码），valid |

坐标格式：三个母本与 DT 的 bbox 都是浮点（例如 `117.5`、`422.675…`），平台已接受并评分；按“禁止改变 bbox”要求保留原值，不取整。

## DENSE 的改动落在哪些帧

| 量 | 帧数 |
|---|---:|
| DENSE 相对 N0 改变 bbox 的帧（全 87,781 帧） | 69,920 |
| 其中落在 TEMP 保留帧 | 64,026（91.6%） |
| 其中落在 TEMP 删除帧 | 5,894（8.4%） |
| DT 相对 TEMP 改变 bbox 的视频 | 166 / 174 |

TEMP 删除 10.01% 的帧，DENSE 改动在删除帧里的占比（8.4%）略低于这一基数，说明 DENSE 的空间改动没有集中在被删帧上。**这只是预测变化统计**：没有官方逐视频标签，不能据此推断 DENSE 的收益是否保留在 TEMP 帧上；需要 DT 官方分回答。

## 官方返回后的判读规则（预先写死）

- `S_DT > 46.84`：DT 晋级为新的已评分母本，后续时间精修、窗口头都改用 DT 作为母本重新声明。
- `S_DT ≤ 46.84`：保留 TEMP 为母本；DENSE 的 +0.93 可能主要来自被 TEMP 删除的低 IoU 帧，或在保留帧上无效。不再对 DENSE 做平台扫描。

## 复现

```bash
cd /home/supie/AIC
OMP_NUM_THREADS=2 /opt/miniconda3/envs/cv/bin/python -m scripts.mask_combo_release \
  --frozen configs/QWEN32B_NOFACE_DENSE_TEMP_V3.json \
  --index /data/aic/official_test_20260926/intake/index.enriched.jsonl \
  --output <新目录>
S=/data/aic/official_test_20260926/submissions
/opt/miniconda3/envs/cv/bin/python scripts/verify_combo_zip.py \
  --index /data/aic/official_test_20260926/intake/index.enriched.jsonl \
  --candidate <新目录>/QWEN32B_NOFACE_DENSE_TEMP_V3_FINAL.zip \
  --mask-zip $S/QWEN32B_NOFACE_TEMPORAL_V2_FINAL/QWEN32B_NOFACE_TEMPORAL_V2_FINAL.zip \
  --spatial-zip $S/QWEN32B_NOFACE_DENSE_V2_FINAL/QWEN32B_NOFACE_DENSE_V2_FINAL.zip \
  --expect c702de926450717ff3c18c4a11aabeee9f3ad3a49ab28f0f512768b14722cf67 \
           631d01042c42e26877dde16fc619fbc9041bd68b6e44b0c412ee5285004b01f4 \
           972b3ed176621398025a8e930d817ae66d2e6090821c5d314581d0a4c1e288c2
```

组合只读已有预测，重跑应得到相同的 predictions SHA `eab552b3…`；ZIP 内含文件修改时间，重跑的 ZIP SHA 会不同，复核时以 predictions SHA 为准，上传以本次 `c702de92…` 为准。从原视频端到端独立复现 DT 需依次重跑 N0 观测缓存 → `point_d2` 加密点（`max_window_release --frozen configs/QWEN32B_NOFACE_DENSE_V2.json`）→ `qwen_temporal_removal`（TEMP 命令见 `QWEN32B_OFFICIAL_V2/temporal/command*.txt`）→ 本脚本；本轮为节省 32B 计算使用了缓存，未做端到端重跑。
