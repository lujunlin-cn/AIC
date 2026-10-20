# DINOv3 v1 asset audit and runtime smoke

2026-10-20 本地完成了计划的资产核验和最小端到端 smoke。DINOv3 ViT-B/16 权重严格加载成功，参数量为 `85,669,632`，权重 SHA256 为 `73cec8be7427c8655ceced13ce62f6e20a1fa90d1b4d4a550df17a1144081a7c`。官方源码使用 commit `6876159a11b4df116f30f667f8c9888617df0751`，运行时通过 `dinov3.hub.backbones.dinov3_vitb16` 加载本地权重。

新增的 [dinov3_runtime.py](../../scripts/dinov3_runtime.py) 只暴露规范化 CLS 和二维 patch tokens，4 个 storage/register tokens 单独保存，不混入空间网格。[dinov3_extract_public_features.py](../../scripts/dinov3_extract_public_features.py) 从显式公开视频清单解码，保存原始解码帧号、PTS、视频尺寸、预处理和权重身份；它兼容后续时间特征缓存和空间候选实验。

用本地已有 QVHighlights20 公共视频选取原始解码帧 `0,10,20` 做 CPU 与 GPU smoke，得到 `CLS [3,768]`、storage `[3,4,768]`、patch `[3,14,14,768]`，帧号和 PTS 完全一致。首次 GPU smoke 发现 FP16 模型与 FP32 输入冲突，已修复为按模型参数 dtype 投入输入，并重新通过两端 smoke。详细机器记录见 [data_audit.json](data_audit.json) 和 [smoke.json](smoke.json)。

这一步没有开始 T-0/T-1/S-0 训练：本地没有 `/data/aic` 的 PHD2/GIF 媒体，也没有当前 checkout 可直接读取的 R70 RetargetVid/LIVE native-crop 表。新 AutoDL 克隆完成后，首个动作是核实 GPU、恢复公开源资产、克隆同一 DINOv3 commit，然后先提取一小批 DINOv3 cache，再运行 T-0 与 S-0 的匹配控制。
