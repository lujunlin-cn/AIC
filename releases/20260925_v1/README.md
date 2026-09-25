# 固定工程候选 20260925 v1

这是可复现的本地工程候选，不是已获官方分数或确认最优的参赛版本。
`manifest.json` 固定所有实际加载权重的 SHA256、bytes、采样、阈值和空间模式。
运行入口先核验文件哈希，拒绝旧输出路径，不会自动下载权重或调参。

| Candidate | Temporal | Spatial | 完整加载 bytes |
|---|---|---|---:|
| A0_center | ResNet18 + Temporal U-Net，raw .40 | center | 25,685,169 |
| DeiT_center | DeiT-S/16 + 相同 head，raw .35 | center | 46,618,447 |
| A0_face_ema | A0，raw .40 | YuNet + proxy fallback + association + shot reset + EMA | 25,917,758 |

DeiT raw .35 与历史 median9/.35 是不同 policy，不借用后者的 holdout 分数。
所有阈值沿用历史 DEV；没有 SumMe 或新空间 GT 调参。
FP16 指持久化权重文件，加载后标准 FP32 运算；不宣称实际 FP16 kernel 推理。

## 运行

完整权重已备份到本地 `artifacts/engineering_release_20260925/weights`，不进入 Git。
远程原文件位置也记录在 manifest；三种候选合计备份只需一份 A0、一份 DeiT 和一份 YuNet。
基准环境是远程 `/opt/miniconda3/envs/cv/bin/python`，Python 3.12、PyTorch 2.9.1+cu128、torchvision 0.24.1+cu128、PyAV 15.1、timm 1.0.20、OpenCV 4.13。
环境完整快照随 `environment.txt` 保存。V100 使用标准 PyTorch kernel，不要求 FlashAttention/BF16。

在 repo 根目录执行；GPU 启动前运行 `nvidia-smi`，选择空闲授权物理卡：

```bash
CUDA_VISIBLE_DEVICES=2 timeout --signal=TERM --kill-after=3m 11h50m \
  /opt/miniconda3/envs/cv/bin/python -m aic.release \
  --manifest releases/20260925_v1/manifest.json \
  --candidate A0_center \
  --weights-dir /data/aic/weights/engineering_release_20260925 \
  --index input_index.jsonl --video-root /path/to/videos \
  --output /data/aic/predictions/A0_center_001.jsonl \
  --report /data/aic/predictions/A0_center_001_report.json \
  --device cuda:0 --stage final
```

`input_index.jsonl` 每行是实际元数据，不从文件名猜测：

```json
{"video_id":"example","video_path":"example.mp4","width":1920,"height":1080,"frame_count":900,"targetRatioWH":[9,16]}
```

另外两个候选只替换 `--candidate` 和新输出路径。Index 的宽高/总帧数必须与真实解码一致；PTS 仍严格检查。
SumMe 镜像专用 annotation-ordinal 解码不进入此部署入口。
输出包含全部被选原始帧；dense spatial 在全部原始帧更新轨迹后才提取被选帧，检测器 bytes 一并填报。

## 验证边界

正式赛事 evaluator/index/运行时限制未取得；本地 validator 通过不等于官方已验收。
`official_f_video` 和 `competition_score` 保持 null。
当前 face observer 只处理人脸，不是完整 person/object detector；没有可变 crop size 学习。
模型来源、实验指标与局限见 `reports/20260925_representation_generalization.md`。

代码快照 `f776342` 的独立目录复现已在历史DEV视频37rzWOQsNIw上通过：三个候选分别332/1080/332帧，JSONL与原运行精确一致。该验证使用既有固定cv环境，不是重新安装全部依赖。详见 `reports/representation_generalization/clean_release_reproduction.json`。
