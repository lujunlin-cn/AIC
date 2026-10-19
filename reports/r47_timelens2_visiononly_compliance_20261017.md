# R47 TimeLens2 + VISIONONLY B3 合规核验（2026-10-17）

## 结论

TimeLens2-8B 可以与 VISIONONLY B3 组成不超过 9B 的推理组件组合；不能与完整 LFM2.5-VL-450M B3 组合。计算如下：

| 组件 | 参数 |
|---|---:|
| TimeLens2-8B | 8,767,123,696 |
| SigLIP2 NaFlex vision tower（只加载 vision tower） | 85,843,200 |
| V8 B3 candidate-utility head | 1,511,425 |
| YuNet 2023mar | 53,104 |
| **合计** | **8,854,531,425** |

完整 LFM B3 的组件为 `448,718,848 + 1,511,425 + 53,104 = 450,283,377`，与 TimeLens2 合计 `9,217,407,073`，超过 9B，不能写入 R47 的实际推理合同。

## 证据

远端审计快照 `/root/autodl-tmp/aic_audit_20261015/final_df331ee/configs/LFM_V9_B3_VISIONONLY_SEMIFINAL.json` 声明只调用 `model.model.vision_tower(...)`，不执行 text、generate 或 projector。对应导出脚本 `scripts/v9_export_vision_weights.py` 只读取 `model.vision_tower.*` 的 197 个 safetensors 张量；导出参数 85,843,200，权重字节 171,686,400。该快照同时明确排除 language model 354,483,968 参数和 projector 8,391,680 参数，因为它们不在推理路径。

R47_TRUE 的空间父表是远端
`/root/autodl-tmp/evac/data/aic/semifinal_20261001/submissions/LFM_V9_B3_VISIONONLY_SEMIFINAL/predictions.jsonl`
（SHA256 `95b9219676a5eece74fbce7bc108d08ba2b84759d0f754bf5660cad9a44c8e8a`），
426 行、426 个唯一视频、234,159 个全帧预测。此前用
`/root/autodl-tmp/experiments/R44_RESIDUAL_001/b3_allframes.jsonl`（SHA256
`2e1dd604...`）生成的同名试包实际继承完整 R26 B3，已废弃，不能称为
VISIONONLY。TimeLens2 输入目录 `/root/autodl-tmp/aic_video` 与打包时的
`/root/autodl-tmp/vpb_artifacts/video_426` 均为 426 个视频；抽查 0、99、290、425
的 SHA256 完全一致，因此区间秒轴和 V9 原始帧轴使用同一媒体。

## 部署边界

R47_TRUE 转换器只改变时间集合：读取 TimeLens2 的 JSON 区间，按原始 FPS 映射到帧号，使用 `round(.85*N)` 精确预算；V9 框从上述全帧父表逐项复制，已逐帧核对 199,037 项完全相等。任何实际重跑空间推理的实现必须使用导出的 vision-only 权重和 V8 B3 head，不能加载完整 LFM safetensors。仅有缓存框时，报告应把缓存来源和实际加载组件分开记录，不能用“文件里没有语言模型”替代推理路径审计。

## 当前证据级别

12 个公开人工标注源、18 个视图上的固定 TimeLens2 区间转换代理为 `0.7044166 -> 0.7213273`（`+0.0169107`，6/18 视图改善）。这是公共诊断，不是官方分数；R47 426 视频分片推理完成并通过包级审计后，才可列为明日提交候选。
