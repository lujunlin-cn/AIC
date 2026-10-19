# AIC 输入索引与提交契约（本地校验器）

更新：2026-10-19。本文是本地工程协议。新提交使用 `current` 合同；
`semifinal` 是 `current` 的别名。尚未取得同版本官方可执行 checker，
本地校验通过不等于官方验收或模型质量通过。

依据：[当前赛题页面](https://www.aicomp.cn/tracks/tracks-6/4267.html)、
仓库 `01_赛事规则与评分标准.md` 和
`reports/r29/official_rules/evidence.json`。当前计分为 `Score=100*F_video`。

## 输入索引

首选使用 enriched JSONL，每行一个视频样本：

```json
{"video_id":"sample-001","video_path":"/data/aic/test/sample-001.mp4","width":1920,"height":1080,"frame_count":300,"targetRatioWH":[16,9]}
```

`video_id` 必须唯一；`width`、`height`、`frame_count` 是正整数并对应实际解码视频；`targetRatioWH` 必须是两个有限正数。紧凑索引可以只给 `video_id`、`video_path`（或由 `video_id + extension` 推导）和 `targetRatioWH`，再运行：

```bash
python -m aic.index --help
```

```python
from aic.index import enrich_index
enrich_index("compact.json", "/data/aic/test", "index.jsonl")
```

相对 `video_path` 以传入的 `video_dir` 为根解析。生成的 enriched index 带有 `index_schema_version: aic.input_index.v1`、解码帧数、尺寸、FPS、旋转和 `coordinate_convention`。代码不自动旋转视频，坐标按 coded pixels 记录；正式规则若指定其他旋转约定，应重新生成索引并锁定版本。

`aic.contract.load_index` 同时接受 enriched JSONL、JSON 数组和单对象 JSON；不会跳过空行、重复键、NaN/Infinity 或重复 `video_id`。

拿到赛事评测集后，先固定原始视频、解码帧数和 PTS，再运行一次性 intake：

```bash
python scripts/prepare_eval_set.py \
  --compact-index official_index.jsonl \
  --video-root /data/aic/eval/videos \
  --output-index /data/aic/eval/index.intake.jsonl \
  --output-manifest /data/aic/eval/intake.manifest.json
```

该步骤不读取标签、不训练、不改阈值；它会检查每个视频的实际解码尺寸/帧数、PTS 单调性、重复路径和 SHA256，并将 `targetRatioWH` 原样带入 enriched index。后续候选批量推理应只使用 intake index，并保存对应 manifest 的哈希。

## 输出 JSONL

每个 index 视频必须且只能有一行：

```json
{"video_id":"sample-001","targetRatioWH":[16,9],"model_params_b":0.45,"predictions":[{"frame":12,"bboxes":[100.0,80.0,640.0]}]}
```

`frame` 是原始解码序号，从 0 开始；`bboxes` 是单个 `[x,y,w]`。
高度由目标比例换算，坐标必须在原视频 coded-pixel 可行域内。
`predictions: []` 合法。提交按 `frame` 升序，不得重复帧。

所有视频必须提供一致的 `model_params_b`，单位为 10 亿参数。
本地当前合同接受 `0 <= P <= 9`，拒绝负数、字符串、bool、NaN 和 Infinity。
0 只允许表达实际无学习参数的算法；不能用 0 隐去编码器或缓存生成模型。
冻结骨干、视觉塔、projector、未合并 LoRA 和全部学习型辅助模型均计入。
实际共享参数只计一次；量化和重复前向不改变参数数。

`model_size_mb` 仅为可选存储信息，当前合同不以文件大小决定资格。
若保留该字段，各行应一致且数值非负有限。
checker 不加载模型；必须另外保留实际参数审计、权重 SHA256 和原视频复现记录。

批量验证：

```bash
python -m aic.contract validate \
  --index index.jsonl --submission submission.jsonl --stage current
```

Python API：

```python
from aic.contract import validate_submission_file
report = validate_submission_file(
    "submission.jsonl", "index.jsonl", stage="current",
    actual_model_params_b=450_000_000 / 1_000_000_000,
)
report.raise_for_errors()
```

独立 checker 不导入 `aic.contract`：

```bash
python scripts/independent_submission_check.py \
  --index index.jsonl --submission submission.jsonl \
  --contract current --parameter-count 450000000
```

此处参数数是示例，必须替换为当前全部推理模型的实际计数。
新包不能使用默认 `legacy` 或 `--allow-missing-size` 作为当前合规依据。

ZIP 的项目约定为根目录只有一个 `predictions.jsonl`。完整检查入口：

```bash
python scripts/check_current_package.py \
  --index index.jsonl --zip candidate.zip \
  --parameter-count 450000000 --expected-videos 426 \
  --output package_check.json
```

`426` 是当前交接记录的视频数，必须与当阶段冻结索引一致。
入口同时执行两套 checker，记录索引、ZIP 和包内 JSONL 的 SHA256。
可使用 `--expected-zip-sha256` 核对冻结包的身份。
程序不读取 GT，不运行推理，不推算官方分，也不上传文件。

权重文件字节清单仍可单独保存，用于部署复现：

```bash
python -m aic.contract size model.pt detector.onnx
```

## 本地 evaluator

```python
from aic.evaluation import evaluate_submission
evaluate_submission(prediction_rows, public_gt_rows, index, stage="current", strict=True)
```

当前 evaluator 使用同 `video_id + frame` 的精确匹配、连续 crop IoU、
重复预测第一条有效记录和每视频宏平均。当前得分为 `100*F_video`，不乘历史大小系数。
只在合法公共监督或合成数据上运行该入口，不读取官方测试标签。
`strict=False` 仅用于诊断无效预测的分母影响。
输出始终标记 `official_evaluator_verified: false`；公共分数不能预测官方回分。

## 历史回放

为了复算已有历史报告，显式 `final` / `replay` 仍保留 2026-09-24 的
`model_size_mb` 与大小系数。`preliminary` 也保留历史行为。
这些名称是兼容接口，不代表当前赛事阶段应使用旧规则。
新调用应显式选择 `current` 或 `semifinal`，不能省略模式。
历史 ZIP 和已回分原件不在本次修复中改写。
