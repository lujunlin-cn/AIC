> **2026-10-15 生效更正：按用户确认的官方口径，总推理参数量 ≤9B。文件字节数不决定参赛资格，量化不减少参数量。本文旧文件体积口径已被本更正替代；文件 bytes 仍作为存储、部署和复现记录。**

旧smoke报告保留。后续R25实际合并推理系统总参数8.481B，合规判断及提交包见 `../r25_aha_submit/summary.md`。

# R24 AHA 依赖修复与真实视觉测试

2026-10-15，AutoDL40206 RTX4080 32GB。用户授权安装缺包并继续运行。

## 结果

**BF16真实视觉smoke通过**。总耗时20.806秒（加载19.673秒），峰值GPU allocated 18,632,761,344字节（17.35GiB）。
两个不同真实画面分别取公共VideoXum TRAIN视频 `dXSlCw8AXv0` 的20秒和150秒，再重复首帧。
同一文本提示、batch1、eager attention、冻结模型；无比赛测试数据，无预留源评估。

- 三个评分头与适配器中的权重逐元素一致，最大误差均为0。
- 视觉嵌入及三个评分头输出均为有限值。
- 不同帧最大评分差2.05078125；重复帧最大差0。
- 实际参数8,632,726,048，加载参数字节17,346,192,448。此为运行内存计量，不替代比赛实际权重文件计量；不能认定该大模型满足提交上限。

仅证明视觉链路与评分头可运行，未验证高光评分质量、人工相关性、视频时序能力或蒸馏收益。不替换比赛包。

## 修复与失败记录

wandb是实验日志工具。安装0.17.9；运行设置WANDB_MODE=disabled，无需账号。
一并补齐deepspeed0.15.4（DS_BUILD_OPS=0）、bitsandbytes0.45.3、OpenCV-headless4.10.0.84等。
NumPy降至1.26.4后将SciPy配套为1.13.1。完整版本见environment.txt。
`pip check`剩余decord0.6.0平台支持提示；本次真实解码使用ffmpeg，未覆盖decord路径。

SigLIP视觉权重补全至3,511,950,624字节；完整SHA256：
`ea2abad2b7f8a9c1aa5e49a244d5d57ffa71c56f720c94bc5d240ef4d6e1d94a`。
下载脚本校验Range边界、完整长度与SHA后原子发布。快照 `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`。

原R24缺wandb的失败仍保留在reports/r24。修复后FP16已加载且三个头身份正确，但首次视觉前向非有限输出；见aha_fp16_failure.json/log。
改用基座原生BF16后通过。不宣称已经定位FP16非有限值的具体层或全部原因。

## 复现与证据

脚本：scripts/r24_aha_visual_smoke.py、scripts/r24_resume_siglip.py；SHA见code_sha256.txt。
配置：configs/R24_AHA_REPAIR_20261015.json；安装及运行命令见prepare_command.txt。
每次模型加载与前向单独20min timeout，未进行训练。
远程隔离目录 `/root/autodl-tmp/experiments/R24_AHA_REPAIR_20261015`，不覆盖旧实验。
通过结果见aha_smoke.json/aha.log；环境安装、import、GPU、视觉塔校验证据均在本目录。

下一步先固定公共train/dev小批质量协议，与人工标签及无teacher控制比较，再决定是否扩量离线蒸馏。
