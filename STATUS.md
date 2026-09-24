# 项目状态

更新：2026-09-25（Asia/Shanghai）。阶段：Phase 0 审计完成，Phase 1 实施中；TVSum 训练闸门阻塞。

## 当前最佳

- Current Best / Best <=100MB / Best <=500MB / Best Overall：尚无合法训练实测模型。
- Current Temporal Best / Current Spatial Best：尚无实测结果。
- SAFE_BASELINE：未锁定，不能把合成测试、随机模型或 TVSum 闸门前实验当有效竞赛系统。

## 环境审计

- 本地初始仅有 01、02、AGENTS；无 Git、代码、历史实验与数据。
- SSH 成功；远程 `/home/supie/AIC` 与 `/data/aic` 初始为空。
- 远程驱动 535.309.01，报告最高 CUDA 12.2；系统 Python 3.10.12 无 PyTorch。实际可用 `/opt/miniconda3/envs/cv`（Python 3.12、PyTorch 2.9.1+cu128、torchvision 0.24.1、PyAV 15.1）。
- 8 张 V100-SXM2-32GB；授权 1、2、4、5、6、7。00:15 检查 GPU 1 有既存进程，优先 GPU 2；0、3 不使用。启动前必须重查。
- `/data` 可用约 1.6TB；远程根盘约 148GB；本地约 750GB。
- 两端 ffmpeg、ffprobe、Git、rsync 可用；远程 tmux 可用。

## 当前瓶颈与运行任务

- Current Bottleneck：无已确认许可的联合训练数据、比赛样例/官方 evaluator、可信联合 GT。
- Running Experiments：无合法训练；已完成契约集成、TVSum provenance/解包和 A0 代码审计。
- Latest Failure：TVSum 包内 WebscopeReadMe 的 Yahoo DSA 条款与 CC-BY README 冲突；A0_001 已用闸门前 TVSum 代理完成但被判无效并移入 `/data/aic/quarantine/TVSum_A0_001_blocked`。
- Next Experiments：取得赛事方/权利人书面许可，或准备许可明确的训练集；随后从 clean environment 重建特征缓存和 A0。

## 时间与外部依赖

- 01 中初赛结果截止为 2026-09-30 11:30，报名截止为 11:00；官方文本时区待平台确认。
- 尚未取得比赛样例、索引、官方 evaluator 与测试视频；不推定已报名。
- TVSum 无 composition GT；其时间代理指标不得声称是实际比赛 F_video。
- TVSum manifest 的 `license_gate=blocked`；`download_status=verified` 只表示字节/元数据验证，不是训练授权。
- MB 字节基数、旋转/VFR 官方约定等保留为待核对，代码采用显式本地约定。
