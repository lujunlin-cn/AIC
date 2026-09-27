# G / E01 / Stage2 / K710 官方分数复核报告（v2）

日期：2026-09-27

## 结论

K710 的完整 174 视频包已经获得官方平台分数 **34.38**。因此此前 `G=17.46` 的原因已经可以确认：它不是 K710 模型质量分数，而是并行推理阶段的单 shard 包（87 个视频）被提交或被平台读取。`E01=17.46` 也呈现同样的单 shard 痕迹。

完整覆盖后的三组 foundation 结果为：

| 完整候选 | foundation | 视频覆盖 | 官方平台分数 |
|---|---|---:|---:|
| E / K700 | InternVideo2 Stage1-1B K700 | 174/174 | 34.43 |
| Stage2 | InternVideo2 Stage2-1B | 174/174 | 34.42 |
| G / K710 | InternVideo2 Stage1-1B K710 | 174/174 | 34.38 |

三者最大差异只有 `0.05` 分，约为 K700 分数的 `0.145%`。当前证据说明：在固定 native-QVH temporal head、threshold、采样和 YuNet spatial 的条件下，K700、Stage2、K710 的官方能力处于同一窄区间，不能声称 Stage2 或 K710 带来了实质性官方提升。

## 1. 17.46 的根因已被完整包反馈验证

远程同时保留了 shard 和合并包：

| 包 | JSONL 行数 | 视频 ID | predictions | ZIP SHA256 |
|---|---:|---|---:|---|
| `SUB_E_shard0/upload.zip` | 87 | 偶数 `0..172` | 43,132 | `bc80b8b35512331fb585c3801a678b3e0d4405f2775a930545d5c346da41feb4` |
| `SUB_G_shard0/upload.zip` | 87 | 偶数 `0..172` | 42,936 | `b4c5dcac2734e87722cc8f08125acde752418c05a6a8a66796a96eeddf5b9817` |
| `SUB_E_INTERNVIDEO2_NATIVE_V1_FINAL/upload.zip` | 174 | `0..173` | 87,637 | `342d36e229f2502d863e8944a4906030050e16282cc94f5cb81138bc9a0d419a` |
| `SUB_G_INTERNVIDEO2_K710_NATIVE_V1_FINAL/upload.zip` | 174 | `0..173` | 87,585 | `a5be193b72e0f18d4a91aabc94f3fd237aa229803db74abc6086100527f61366` |

单 shard 分数 `17.46` 与完整包约 `34.4` 的近半关系，与缺失 87 个视频在视频宏平均中贡献零分一致。K710 完整包得到 `34.38` 后，这个解释从“高度可能”升级为“已由复测结果确认”。

因此，`G=17.46` 和 `E01=17.46` 应记录为 submission packaging / coverage failure，不能记录为模型能力失败。

## 2. 完整包官方结果的真实比较

### 分数差异

- K700 → Stage2：`34.42 - 34.43 = -0.01`
- K700 → K710：`34.38 - 34.43 = -0.05`
- Stage2 → K710：`34.38 - 34.42 = -0.04`

这些差异远小于此前 DeiT-S 到 InternVideo2 的跃升，也没有达到可以支持新 backbone 替换的强证据。三者同属 500M–9B 档，实际加载权重分别约为 2.0495GB、2.8275GB、2.0495GB；若平台应用同一 L-tier 系数，系数不会改变三者的相对排序，也不会制造 17.46 这种二倍差异。

### 本地输出差异

三套完整候选都使用：

- threshold `.35`
- 2 FPS
- 相同 Temporal U-Net 家族
- 相同 native-QVH 96/24 训练协议
- 相同 YuNet `true_face_smooth`
- 相同 JSONL exporter

逐视频审计显示：

- K700 与 K710 只有 3 个视频的 selected-frame 集合不同；新增 144 帧、删除 196 帧。
- K710 与 Stage2 只有 1 个视频发生 196 帧的集合差异。
- 所有共同 selected frame 的 bbox 完全一致，因为三者共用 YuNet。
- 三者均 174/174 覆盖、0 个空视频。

因此完整包之间的官方分数窄差是符合输出相似度的；17.46 则不可能由这些少量 frame-set 差异解释。

## 3. 为什么 local DEV 排名没有转化成明显官方差异

native-QVH DEV 的 ranking 结果为：

| 模型 | Spearman | NDCG | F1 |
|---|---:|---:|---:|
| K700 / Stage1 | 0.23474 | 0.96763 | 0.76609 |
| Stage2 | 0.21433 | 0.96962 | 0.76543 |
| K710 | 0.19818 | 0.96576 | 0.76543 |

本地 ranking 顺序与完整官方分数顺序大致一致，但绝对差异很小，且 DEV 是 query-conditioned QVHighlights 人类显著性协议，不是 AIC 联合 temporal + crop GT。它可以用于淘汰明显较差的表示，却不能预测 0.01–0.05 的官方细差。

当前合理解释是：

1. **标签语义存在差异**：QVHighlights 的 query-conditioned saliency 与 AIC generic highlight/crop 不是同一个目标。
2. **阈值造成输出饱和**：固定 `.35` 后，三个 head 输出的二值选择几乎相同，representation 的连续分数差异被离散化抹平。
3. **空间路径被锁定**：三者都使用 YuNet，人脸/主体观察能力没有随 foundation 改变；非人脸主体仍是共同瓶颈。
4. **Temporal head 容量与 2 FPS 采样限制了上限**：更换 video encoder 没有改变 head、采样分辨率或 postprocess，因而 foundation 差异无法充分传递到最终帧集合。

这不是说 K700、Stage2、K710 的视觉特征完全等价，而是当前提交协议没有提供足够的自由度让它们产生可分辨的官方收益。

## 4. 当前模型结论

- **Best official full score**：K700 `34.43`，但与 Stage2 `34.42`、K710 `34.38` 实质持平。
- **Stage2**：没有在当前冻结协议下证明超过 K700；其 local NDCG 较高，但官方只低 `0.01`。
- **K710**：完整包并不失败，官方 `34.38` 仅比 K700 低 `0.05`；此前 `17.46` 不能用于否定 K710。
- **E01/G 的 17.46**：确定属于提交覆盖链路问题，应从模型排行榜分析中剔除。
- **模型大小**：三者都是 L-tier，继续更换同量级 foundation 的收益空间目前很小，且没有尺寸系数优势。

## 5. 下一步优先级

1. 将 `*_shard0`、`*_shard1` 标记为内部中间产物，平台上传目录只暴露 `*_FINAL/upload.zip`。
2. 每次上传前强制检查 ZIP 内 JSONL 行数为 174、唯一 ID 为 174、ID 集合等于正式 index，并记录上传文件 SHA256。
3. 暂停继续 K700/Stage2/K710 的同协议 frozen backbone 横向提交。
4. 优先做有单变量含义的后训练：在 train/dev 上比较 temporal head adaptation、partial backbone finetune、4 FPS temporal observation 和真实 subject/crop head；正式测试集不参与选择。
5. 如果需要下一次官方诊断提交，选择改变 temporal 或 spatial 机制的候选，而不是再提交一个只换 foundation 权重的近同输出模型。

本报告只使用正式测试集的工程元数据、预测文件结构和官方反馈，不使用测试视频内容调参，也不把官方平台分数反推为未经提供的 raw `F_video`。
