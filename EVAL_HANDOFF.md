# 评测集接入手册

评测集拿到后，先保留平台提供的原始压缩包、原始 index 和文件名，不在原目录覆盖或重命名。推荐把视频放在远程 `/data/aic/eval_incoming/<eval_set_id>/videos`，把原始索引放在同目录；代码和输出仍放在 `/home/supie/AIC`、`/data/aic/predictions`。

## 1. 预检并冻结输入

如果平台给的是只包含 `video_id`、路径和 `targetRatioWH` 的 compact index：

```bash
python scripts/prepare_eval_set.py \
  --compact-index /data/aic/eval_incoming/SET/raw_index.jsonl \
  --video-root /data/aic/eval_incoming/SET/videos \
  --output-index /data/aic/eval_incoming/SET/index.enriched.jsonl \
  --output-manifest /data/aic/eval_incoming/SET/intake.manifest.json
```

该步骤会完整解码每个视频，核对实际宽高、帧数和 display PTS 单调性，记录 FPS、旋转、音轨、coded-pixel 坐标约定和 SHA-256。它不会读取训练标签、调阈值或生成预测；输出存在时拒绝覆盖。

如果赛事已经提供 enriched index，仍应把它和视频逐条用 `aic.index`/`probe_video` 核对，再生成同样的 intake manifest。任何帧数、尺寸、PTS 或目标比例不一致都先停止推理。

## 2. 先跑工程 fallback

启动前只选择授权且空闲的物理 GPU 1、2、4、5、6、7，并记录 `nvidia-smi`。推荐先运行 A0：

```bash
python scripts/run_eval_candidates.py \
  --manifest releases/20260925_v1/manifest.json \
  --candidate A0_center \
  --weights-dir /data/aic/weights/engineering_release_20260925 \
  --index /data/aic/eval_incoming/SET/index.enriched.jsonl \
  --intake-manifest /data/aic/eval_incoming/SET/intake.manifest.json \
  --output-dir /data/aic/predictions \
  --device cuda:0 --stage final --run-id SET_A0_001
```

`cuda:0` 是 `CUDA_VISIBLE_DEVICES` 映射后的逻辑编号；例如使用物理 GPU 2 时设置 `CUDA_VISIBLE_DEVICES=2`。脚本每次建立新的 run 目录，核对 release 权重 SHA、intake index SHA 和完整模型大小，不会覆盖旧结果。

A0 成功并通过本地 validator 后，再对 `DeiT_center` 和 `A0_face_ema` 各运行一次独立目录。三者的输出 JSONL、report、command 和 manifest 都保留。官方联合 `F_video` 与 competition score 在获得同版本官方 evaluator 前保持 `null`。

## 3. 首轮检查表

- 原始数据压缩包和 index SHA 已记录；不把测试视频放入训练或 teacher 流程。
- intake manifest 的 `pts_verified=true`、视频数量和目标比例与平台材料一致。
- A0 输出逐行 JSONL 可解析，视频覆盖完整，frame 唯一且升序，`[x,y,w]` 合法，`model_size_mb` 一致。
- report 中 `loaded_weight_bytes` 等于 release manifest，`official_f_video` 和 `competition_score` 仍为 `null`。
- 记录每视频 selected-frame 数量、空输出率、端到端耗时、峰值显存和失败视频；不根据未公开标签调阈值。
- 只有官方评测器和规则明确后，才另建官方结果目录；不要覆盖本地工程证据。

无真实评测集时，可用 `python scripts/release_batch.py --smoke ...` 或现有短视频 smoke 检查入口，但 smoke 不代表比赛分数。
