# 远程训练环境快照

核查时间：2026-09-25（Asia/Shanghai），主机 `supie@119.62.14.24:22022`。

```text
Python       3.12 ( /opt/miniconda3/envs/cv/bin/python )
PyTorch      2.9.1+cu128
torchvision  0.24.1+cu128
PyAV         15.1.0
GPU          Tesla V100-SXM2-32GB
```

实验只使用物理 GPU 2、4、5、6、7；物理 0、3 禁止，物理 1 有既有进程。V100 路径使用 FP16/AMP；没有依赖 BF16、FP8、FlashAttention 或专用 Triton kernel。每次训练命令由 `timeout 42600s` 包裹，实验目录在 `/data/aic/experiments/<run_id>`。

本地 PyAV 版本与远程不同会造成解码特征漂移，因此原始视频正式推理和一致性审计固定上述远程环境。
