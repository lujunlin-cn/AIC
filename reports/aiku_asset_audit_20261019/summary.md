# AIKU × brAIn Video Summarization Challenge 数据审计

审计对象：`/home/hajimi2025/aikubrainvideosummary.zip`，SHA256：
`0e179afc43461b92ff4719f2e719c62464320164da772eaf81f0f1923d769272`。

## 文件与规模

| 文件 | 解压大小 | 观察到的语义 |
|---|---:|---|
| `train.csv` | 14,307,563 B | `video_id,gt_score`，6,000 行 |
| `train.h5` | 5,003,075,036 B | 6,000 个 group，每组预提取视觉特征和标签 |
| `test.csv` | 24,009 B | 1,500 个 `test_video_*` ID |
| `test.h5` | 1,247,702,336 B | 测试预提取特征，未读取标签 |
| `baseline.ipynb` | 19,868 B | 1024D 特征上的 Transformer + MSE 示例 |
| `sample_submission.csv` | 28,520 B | 每个测试视频输出一个分数数组 |

压缩包没有原始视频、音频、视频来源 ID、特征提取器权重、FPS/PTS 或许可说明。`train.h5` 每个 group 都有：

```text
features       [T, 1024] float32
gtscore        [T]       float32
gtsummary      [T]       float32, 0/1
change_points  [S, 2]    int64
```

全训练集共有 1,214,730 个特征步；`T` 范围 121–300，中位数 200；`S` 范围 25–61，中位数 41。特征值范围约 `[-2.0005, 2.0005]`。

## 标签检查

- `train.csv` 的 6,000 个 `gt_score` 数组与 H5 `gtscore` 逐视频对应，前三个样本最大误差小于 `3e-8`。
- `gtscore` 全集范围 `[0,1]`，均值 `0.3375`。
- `gtsummary` 是二值摘要标签，正例比例 `179,924 / 1,214,730 = 14.81%`。
- 每视频 `gtscore` 与 `gtsummary` 的 Spearman 相关均值 `0.3981`，按每视频正例数取 score top-k 的重叠率均值只有 `0.4793`。因此 `gtsummary` 不是简单的 `gtscore` 阈值或 top-k 派生标签。
- `change_points` 是片段边界，但压缩包没有明确说明边界的来源、帧率或原始时间轴。

## 代码事实与可迁移边界

官方 baseline 将 1024D `features` 直接送入一层 Transformer 编码器和回归头，训练目标是逐步 MSE；它不包含原始视频编码器。该数据可以作为一个摘要时间监督 benchmark，但当前无法证明 1024D 表征与 AIC 当前 V9/TimeLens 表征同源，也无法从它重算 AIC 输入。

因此本轮不训练 AIKU→AIC 迁移模型。最小可行下一步是确认 Kaggle/赛事资料中的特征提取器、原视频来源和标签生成协议；在这些信息缺失时，任何迁移只会是“跨特征空间 head benchmark”，不能称目标域预训练或部署候选。

官方测试 H5 仅用于合法推理时存在，当前审计没有读取任何测试标签，也没有将测试数据用于训练或调参。
