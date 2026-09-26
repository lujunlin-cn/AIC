# 2026-09-26 正式评测集冻结推理与上传包

生成时间（UTC）：2026-09-26T11:25:16.948809+00:00

正式评测集仅用于 intake、解码核验和冻结推理；未训练、微调、标注、逐视频调参、第三方上传或 teacher 伪标签。

## Test-set inventory

- 原始 ZIP：`基于视频大模型的通用视频高光剪辑.zip`
- ZIP bytes：`757234263`
- ZIP SHA256：`130bdceb2574ef3ca853f6937fdf5f24c3fb1f054c92477f0c2d075e53efe0b6`
- 解压目录（远程）：`/data/aic/official_test_20260926/extracted/`
- intake manifest：`/data/aic/official_test_20260926/intake/intake.manifest.json`
- 视频/索引数量：174 / 174；ID 集合：0..173
- 目标比例：16:9=119，9:16=55
- intake index SHA256：`9a59316e97f5a80fc2c22f43f38ebd72130d20109d17954efa3dde9f86d404d5`
- intake labels_present：`false`
- 解码损坏视频：`0`（174/174 通过完整解码、PTS 单调和帧数核验）
- 与公开 174-entry `test_index.json`：video_id 和 `targetRatioWH` 完全一致（公开 index SHA256 `2780cbaf79259799a36eab3e851975f2403cf4741b30a5bd3596c5db7df691a6`）。

## Frozen candidates and outputs

| ID | Model / spatial | threshold | weight bytes | predictions | empty rate | runtime (s) | validator | ZIP | ZIP SHA256 |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| SUB_A | ResNet18 + repaired Temporal U-Net / center | 0.4 | 25,685,169 (25.685169 MB) | 1809 | 0.862069 | 742.213 | project + independent + unzip regression valid | `submissions/20260926/AIC_SUB_A_A0_CENTER.zip` | `2cb5aa29ff5f059cf8f50d26cebd915188461cf8214de5225be7724b79a139a3` |
| SUB_B | DeiT-S/16 + repaired Temporal U-Net / center | 0.35 | 46,618,447 (46.618447 MB) | 8133 | 0.557471 | 742.764 | project + independent + unzip regression valid | `submissions/20260926/AIC_SUB_B_DEITS_CENTER.zip` | `5585c5f311a5e012dc920c8e13d9d700c27d36789db51b8691ebccdd51503968` |
| SUB_C | DeiT-S/16 + repaired Temporal U-Net + YuNet / true_face_smooth (dense_v1) | 0.35 | 46,851,036 (46.851036 MB) | 8133 | 0.557471 | 1959.029 | project + independent + unzip regression valid | `submissions/20260926/AIC_SUB_C_DEITS_SPATIAL.zip` | `c3d73d3ca5ed52dccc1052345d5e0d1e7e4653c4dd82c96894344cbfc2f87691` |


Canonical upload archives are the root-level ZIPs above. The per-candidate directories retain audit copies with identical `predictions.jsonl` payload hashes; ZIP container hashes differ only because of archive metadata/name.

每个 ZIP 已执行：`unzip -l`、解压到新临时目录、payload byte-for-byte 比对、项目 validator、独立 checker；三者均 `valid=true` 且 errors=0。ZIP 根目录唯一文件为 `predictions.jsonl`。

## Reproduction

- Git commit：`0430bbf`（候选冻结配置） plus runner fix commit `0bee50e` and independent checker `3c16f06`。
- Remote code：`/home/supie/AIC`；weights：`/data/aic/weights/engineering_release_20260925`。
- Environment：`/opt/miniconda3/envs/cv/bin/python`；sample_fps=2.0；batch_size=16；FP16 persistent weights；stage=final。
- Physical GPUs: SUB_A=2, SUB_B=4, SUB_C=5; GPU 0/3 were not used.
- `official_f_video=null`, `competition_score=null` because no official joint evaluator/labels were available in this workflow.

## Upload order

1. `SUB_A` — stable engineering fallback and format-chain check.
2. `SUB_B` — temporal representation comparison against A0.
3. `SUB_C` — same DeiT temporal prediction with frozen dense face spatial observer; tests spatial contribution.

No candidate received test-set-specific tuning. Test videos were not sent to any third-party service.

## Upload readiness and zero-score plan

- `SUB_A`, `SUB_B`, `SUB_C`: `READY_TO_UPLOAD=yes`; each archive contains only the required root `predictions.jsonl`, and all three validation layers returned zero errors.
- If the platform reports zero, inspect in order: ZIP root/file name, JSON vs JSONL interpretation, video_id coverage, targetRatioWH, zero-based global frame numbering, rotation/coded-pixel convention, `[x,y,w]` geometry and boundary, ZIP encoding, and `model_size_mb` handling. Do not retrain first.
- The local validator is `local_rules_2026-09-24_not_official`; no official joint ground truth was available, so official score fields remain null.
