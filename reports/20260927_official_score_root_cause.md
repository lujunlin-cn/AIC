# 官方分数差异根因分析：G / E01 / Stage2

日期：2026-09-27

## 结论

`G=17.46` 与 `E01=17.46` 的首要原因高度可能是上传了并行推理的单个 shard ZIP，而不是合并后的 174 视频 FINAL ZIP。远程产物中，`SUB_G_shard0/upload.zip` 和 `SUB_E_shard0/upload.zip` 都只含 87 个视频；它们的 `video_id` 是偶数序列 `0,2,...,172`。对应的 FINAL ZIP 才包含全部 174 个视频。17.46 约为完整 34.4 分的一半，和“87/174 视频被计分、缺失视频按零分处理”的表现一致。

因此当前证据不支持“ K710 foundation 只有 Stage2 一半效果”。相反，Stage2 full package 得分 `34.42`，此前 E 的 full package 得分 `34.43`，两者几乎相同；K710 的 17.46 应先按提交包覆盖错误排查。

## 三组官方反馈

| 平台标签 | 本地候选/推断 | 官方分数 | 本地包覆盖 | ZIP SHA256 |
|---|---|---:|---:|---|
| E01 | 很可能是 `SUB_E_shard0` | 17.46 | 87/174，偶数 ID | `bc80b8b35512331fb585c3801a678b3e0d4405f2775a930545d5c346da41feb4` |
| G | `SUB_G`；需核对平台实际上传文件 | 17.46 | FINAL 为 174/174；shard0 为 87/174 | FINAL `a5be193b72e0f18d4a91aabc94f3fd237aa229803db74abc6086100527f61366`；shard0 `b4c5dcac2734e87722cc8f08125acde752418c05a6a8a66796a96eeddf5b9817` |
| Stage2 | `SUB_F_INTERNVIDEO2_STAGE2_NATIVE_V1_FINAL` | 34.42 | 174/174 | `a7622bae7b78712549a344f5b5d61fd1762c395a734d82a2c85173bc344a6537` |

此前 InternVideo2 Stage1-1B K700 的合并 FINAL 包也曾获得 `34.43`。该结果与 Stage2 的 `34.42` 一致说明完整覆盖包能够达到约 34.4 分量级。

## 本地包审计证据

远程 `/data/aic/official_test_20260926/submissions` 中的包结构如下：

| 包 | JSONL 行数 | 唯一视频数 | 视频 ID | predictions | ZIP 字节数 |
|---|---:|---:|---|---:|---:|
| `SUB_E_shard0/upload.zip` | 87 | 87 | 偶数 `0..172` | 43,132 | 526,175 |
| `SUB_E_shard1/upload.zip` | 87 | 87 | 奇数 `1..173` | 44,505 | 528,687 |
| `SUB_E_INTERNVIDEO2_NATIVE_V1_FINAL/upload.zip` | 174 | 174 | `0..173` | 87,637 | 1,053,182 |
| `SUB_F_INTERNVIDEO2_STAGE2_NATIVE_V1_FINAL/upload.zip` | 174 | 174 | `0..173` | 87,781 | 1,054,654 |
| `SUB_G_shard0/upload.zip` | 87 | 87 | 偶数 `0..172` | 42,936 | 523,747 |
| `SUB_G_shard1/upload.zip` | 87 | 87 | 奇数 `1..173` | 44,649 | 530,068 |
| `SUB_G_INTERNVIDEO2_K710_NATIVE_V1_FINAL/upload.zip` | 174 | 174 | `0..173` | 87,585 | 1,052,204 |

每个 shard 自身都能通过“该 shard 的 index”校验，但不能作为完整测试集提交。合并程序只有在两个 shard 的 ID 集合并集等于 174-entry index 时才生成 FINAL 包。

## 为什么不是 K710 表征本身导致分数减半

三套完整推理都固定了 threshold `.35`、2 FPS、Temporal U-Net、YuNet `true_face_smooth` 和相同 JSONL exporter。K700 与 K710 的完整 JSONL 做逐视频比较时：

- 174 个视频中只有 3 个视频的 selected-frame 集合发生变化；
- 变化总量为新增 144 帧、删除 196 帧；
- 所有共同 selected frame 的 bbox 完全一致，原因是使用同一个 YuNet spatial path；
- K710 与 Stage2 的差异也集中在 1 个视频的 196 帧删除；
- 三者的完整包都覆盖 174 个视频，且都没有空输出。

这种输出差异规模无法解释宏观分数从约 34.4 变成 17.46。相反，87/174 覆盖正好提供了一个直接的分数减半机制：缺失的一半视频没有有效预测，按视频宏平均时贡献为零。

## 其他可能性及当前证据

1. **平台使用了旧文件或错误 ZIP**：如果用户确认上传的是 FINAL ZIP，仍需检查平台保存的文件名、ZIP 内部行数和 SHA256。远程同时存在 shard 与 FINAL，人工选择时很容易选错。
2. **平台对缺失视频的处理**：平台未公开缺失 ID 的具体计分实现，但 17.46 与 34.42/34.43 的近半关系是强信号。不能把 17.46 当成模型 raw score，直到覆盖问题排除。
3. **模型大小系数**：G、E/K700 和 Stage2 都属于 500M–9B 档，系数均为 `.90`；系数不会产生约 2 倍差异。初赛平台分数还可能直接显示 `100*F_video`，但无论采用哪种显示，size tier 都不能解释 17.46。
4. **空间预测错误**：K700/K710/Stage2 使用同一 YuNet 路径；共同帧 bbox 完全相同。空间差异不是当前分数减半的解释。

## 纠正步骤

上传前必须只从 `*_FINAL/upload.zip` 选择文件，并执行：

```bash
unzip -l upload.zip
# 必须只包含 predictions.jsonl
wc -l <(unzip -p upload.zip predictions.jsonl)
# 必须是 174
sha256sum upload.zip
```

对应本轮 K710 正确包：

```text
/data/aic/official_test_20260926/submissions/SUB_G_INTERNVIDEO2_K710_NATIVE_V1_FINAL/upload.zip
SHA256: a5be193b72e0f18d4a91aabc94f3fd237aa229803db74abc6086100527f61366
JSONL: 174 lines / 87,585 predictions
```

不要把 `SUB_G_shard0` 或 `SUB_G_shard1` 作为官方完整测试集提交。若平台允许重新提交，优先用这个 FINAL 包复测 G；在重新得到平台结果前，不对 K710 的模型能力下结论。

## 当前判断

- Stage2 full：已验证约 `34.42`。
- K700 full：已验证约 `34.43`。
- K710 full：模型质量暂时未知；已生成并验证 FINAL 包，但 `17.46` 很可能对应单 shard 上传。
- E01：`17.46` 与 E shard0 的 87 视频结构一致，暂按“单 shard 误提交”处理；若 E01 实际是另一文件，需用平台侧上传文件 SHA 重新归因。

这份分析没有使用测试视频内容进行调参，也没有改变任何 checkpoint、threshold 或 spatial policy。

## Correction (2026-09-27)

The user subsequently confirmed that the complete 174-video K710 FINAL package scored **34.38**. The earlier 17.46 was therefore a confirmed single-shard upload/coverage result. See `reports/20260927_official_score_root_cause_v2.md` for the final comparison against K700=34.43 and Stage2=34.42.
