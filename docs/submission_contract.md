# AIC 输入索引与提交契约（本地校验器）

本文记录当前代码对即将取得的 AIC 评测集的边界。它是本地工程协议，尚未与赛事方同版本官方 evaluator 交叉验证。

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
{"video_id":"sample-001","targetRatioWH":[16,9],"model_size_mb":50.0,"predictions":[{"frame":12,"bboxes":[100.0,80.0,640.0]}]}
```

`frame` 是原始解码序号，从 0 开始；`bboxes` 是单个 `[x,y,w]`，高度由目标比例换算，坐标必须在原视频 coded-pixel 可行域内。`predictions: []` 合法。提交输出按 `frame` 升序，不能重复帧；所有视频的 `model_size_mb` 必须一致，复赛/最终阶段必须提供，范围为 `0 < M <= 9216`。模型大小应来自实际解压并加载的全部权重文件；MB 与 MiB 的赛事边界口径仍待确认。

批量验证：

```bash
python -m aic.contract validate \
  --index index.jsonl --submission submission.jsonl --stage final
```

Python API：

```python
from aic.contract import validate_submission_file
report = validate_submission_file("submission.jsonl", "index.jsonl", stage="final")
report.raise_for_errors()
```

权重清单与未压缩字节审计：

```bash
python -m aic.contract size model.pt detector.onnx
```

## 本地 evaluator

```python
from aic.evaluation import evaluate_submission
evaluate_submission(prediction_rows, gt_rows, index, stage="final", strict=True)
```

该 evaluator 使用同 `video_id + frame` 的精确匹配、连续 crop IoU、重复预测第一条有效记录、每视频宏平均和大小系数。`strict=False` 仅用于诊断无效预测的分母影响，输出始终标记 `official_evaluator_verified: false`；不能把诊断分数上传为正式结果。
