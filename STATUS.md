# 项目状态

更新：2026-09-25（Asia/Shanghai）。阶段：Phase 0 审计完成，Phase 1 实施中；A0 时间代理已完成，准备 A1。

## 当前最佳

- Current Best / Best <=100MB / Best <=500MB / Best Overall：A0_001 时间代理模型已完成；尚无官方联合 F_video。
- Current Temporal Best：A1_001 TVSum proxy F1=0.016484（A0=0.011019，A2=0.000000；单 seed）；Current Spatial Best：尚无实测结果。
- SAFE_BASELINE：尚未锁定；当前 A0 仅在 TVSum 时间代理上验证，缺少空间 GT 和官方测试评测。

## 环境审计

- 本地初始仅有 01、02、AGENTS；无 Git、代码、历史实验与数据。
- SSH 成功；远程 `/home/supie/AIC` 与 `/data/aic` 初始为空。
- 远程驱动 535.309.01，报告最高 CUDA 12.2；系统 Python 3.10.12 无 PyTorch。实际可用 `/opt/miniconda3/envs/cv`（Python 3.12、PyTorch 2.9.1+cu128、torchvision 0.24.1、PyAV 15.1）。
- 8 张 V100-SXM2-32GB；授权 1、2、4、5、6、7。00:15 检查 GPU 1 有既存进程，优先 GPU 2；0、3 不使用。启动前必须重查。
- `/data` 可用约 1.6TB；远程根盘约 148GB；本地约 750GB。
- 两端 ffmpeg、ffprobe、Git、rsync 可用；远程 tmux 可用。

## 当前瓶颈与运行任务

- Current Bottleneck：缺少比赛本地视频、官方 evaluator 和联合空间 GT；已核对公开 baseline 紧凑索引契约。
- Running Experiments：A1_001 保留为当前时间控制；A2 Feature Bank 已完成并拒绝，下一步转向空间候选基线/误差分析。
- Latest Failure：此前的许可证暂停已由用户授权解除；旧 A0 资产从 quarantine 恢复。
- Next Experiments：A1 TSM → A2 Feature Bank；拿到比赛视频后先用 `scripts/build_video_index.py` 做真实帧数/PTS审计。
- Remote smoke：在授权 GPU 外的 CPU/临时目录用合成视频跑通 compact index → enriched index → center-crop JSONL；1 行、5 原始帧、validator valid。该结果不作竞赛指标。

## 时间与外部依赖

- 01 中初赛结果截止为 2026-09-30 11:30，报名截止为 11:00；官方文本时区待平台确认。
- 尚未取得比赛样例、索引、官方 evaluator 与测试视频；不推定已报名。
- TVSum 无 composition GT；其时间代理指标不得声称是实际比赛 F_video。
- TVSum manifest 保留 `license_gate` 和条款哈希作 provenance；本项目按用户授权继续使用可下载数据。
- MB 字节基数、旋转/VFR 官方约定等保留为待核对，代码采用显式本地约定。
