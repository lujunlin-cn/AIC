# V5 P0：INTERP 上的 x 轴视觉复核（官方包 QWEN32B_INTERP_XRERANK_V5_FINAL）

日期：2026-09-29。母本：INTERP = QWEN32B_DT_INTERP_V4_FINAL（正式 48.67，zip 501cebce…）。
预注册：`configs/V5_P0_XRERANK_PREREG.json`（sha ba5f1d17…，出结果前写定）。

## 结论

**P0 通过预注册晋级规则，官方候选包已打包，成绩待评。**
在 INTERP 管线上，把 x 轴（axis0）行的 DENSE 自由轴点替换为 E4 式"看裁剪图"复核点
（两次顺序一致才接受，否则保留 DENSE 点），y 轴与非空间行逐字节不变。

## 证据（RetargetVid，6 标注者均值 IoU，视频宏平均，配对 bootstrap 5000 次）

| 对比 | 集 | 全部 | axis_x | axis_y |
|---|---|---|---|---|
| XRERANK_INTERP − INTERP | dev2 (70 视频) | **+0.0093** [+0.0036,+0.0161] | **+0.0187** [+0.0073,+0.0323] | 0（构造保证） |
| XRERANK_INTERP − INTERP | confirm2 (100 视频) | **+0.0094** [+0.0040,+0.0163] | **+0.0188** [+0.0079,+0.0326] | 0（构造保证） |
| RERANKNF − DENSENF（sanity） | dev2 | +0.00930（E4 发表值 0.0093） | — | — |
| RERANKNF − DENSENF（sanity） | confirm2 | +0.00723（E4 发表值 0.0072） | — | — |

- 晋级规则：dev2 axis_x CI 下界 > 0 ✓（+0.0073）；confirm2 axis_x CI 下界 > 0 ✓（+0.0079）；
  全视频均值 ≥ −0.002 ✓。sanity gate（|复现−发表| < 0.004）✓。
- x 行有改动的关键帧：均值 6.7 个/行，170 行中 135 行至少 1 个改动。
- 复现命令：
  `PYTHONPATH=/home/supie/AIC /opt/miniconda3/envs/cv/bin/python scripts/v5_p0_xrerank_eval.py
  --cache $E/OBS_CACHE_RETARGET_ALL200 --annotations /data/aic/datasets/RetargetVid/annotations_all
  --d2 $E/QWEN_RV200_POINT_D2/points --rerank $E/V4_E4_RERANK/rv_points_rerank --splits dev2,confirm2
  --output $E/V5_P0_XRERANK/eval`
  （E4 缓存复用，零新教师查询；dev2 与训练 001–080 有重叠故只作 sanity，confirm2 为既有验证。）

## 官方包

- `QWEN32B_INTERP_XRERANK_V5_FINAL.zip`，995,981 B，
  **sha256 `027fd12a51b128d87014e2aae006c93c179b5c2fe4f8eecb143a60b62fa9a150`**
  predictions.jsonl sha256 `4f7cfe52b830ad1929dd266f1df032802273d4c78a51c4db3a8aeac85e94b934`
- 174 视频 / 78,992 帧（与 TEMP 完全同键集）；模型规模披露与 N0/TEMP/DT_V3/INTERP 相同
  （Qwen3-VL-32B FP16 33.36B 参数，JSONL 省略 model_size_mb，preliminary 格式）。
- 中间产物：`QWEN32B_INTERP_XRERANK_V5_SPATIAL`（全帧 spatial 层，
  predictions sha e353698e…，zip sha 65ee4de5…）。
- 身份守卫（全部硬失败型，全过）：
  1. DENSE 点走同一代码路径逐字节复现 INTERP spatial 母包（174/174 视频）——证明管线复现正确；
  2. rerank 点仅改动 x 行的 x 分量（y 分量与 invalid 标志逐点相等）；
  3. y 轴与非空间行对母包零变化（21 个有改动的视频全部是 x 行）；
  4. 最终包 keys == TEMP 逐视频 0 差异；bbox == spatial 父包逐帧 0 差异；
     mask 从冻结移除回复重算；validator / 独立 checker / 解压往返全过；
     guard_tripped=1 与 INTERP 母包相同（同一移除数据的继承守卫，非新触发）。
- 相对 INTERP 母包的实际差异：**21 视频 / 10,273 保留帧**（视频 19: 2,796 帧、20: 4,337 帧，
  其余 ≤350；y 行 0）。
- 复现：
  `python scripts/v5_p0_splice_spatial.py --index .../index.enriched.jsonl
  --rerank $E/V5_P0_XRERANK/off_points_rerank --output .../QWEN32B_INTERP_XRERANK_V5_SPATIAL
  --combo-config configs/QWEN32B_INTERP_XRERANK_V5_FINAL.json`
  然后 `python -m scripts.mask_combo_release --index ... --frozen configs/QWEN32B_INTERP_XRERANK_V5_FINAL.json
  --output .../QWEN32B_INTERP_XRERANK_V5_FINAL`。

## 局限

- rerank 点来自 E4 在 RV 验证集上验证过的同一生成物；官方测试视频上无法本地验证，
  但生成参数（候选集、两次顺序一致、prompt）与 RV 完全一致，属同分布外推。
- 增益集中于 x 轴（55 个官方 x 视频），对 119 个 y 视频无影响（构造保证）。
- 官方成绩待评；本包不预测分数（包身份规则）。
