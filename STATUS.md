# 项目状态

更新：2026-09-25（Asia/Shanghai）。阶段：Phase 0 审计完成，Phase 1 实施中。

## 当前最佳

- Current Best / Best <=100MB / Best <=500MB / Best Overall：尚无实测模型。
- Current Temporal Best / Current Spatial Best：尚无实测结果。
- SAFE_BASELINE：未锁定，不能把合成测试或随机模型当有效竞赛系统。

## 环境审计

- 本地初始仅有 01、02、AGENTS；无 Git、代码、历史实验与数据。
- SSH 成功；远程 `/home/supie/AIC` 与 `/data/aic` 初始为空。
- 远程驱动 535.309.01，报告最高 CUDA 12.2；系统 Python 3.10.12 无 PyTorch。
- 8 张 V100-SXM2-32GB；授权 1、2、4、5、6、7。00:15 检查 GPU 1 有既存进程，优先 GPU 2；0、3 不使用。启动前必须重查。
- `/data` 可用约 1.6TB；远程根盘约 148GB；本地约 750GB。
- 两端 ffmpeg、ffprobe、Git、rsync 可用；远程 tmux 可用。

## 当前瓶颈与运行任务

- Current Bottleneck：尚无训练数据、提交实现、可信联合 GT、官方同版本 evaluator。
- Running Experiments：无训练；正在实现原帧映射/validator/evaluator/A0，并取得 TVSum。
- Latest Failure：系统 Python 缺少 torch（环境审计项，并非训练失败）。
- Next Experiments：先合成 contract integration，再 TVSum A0_001 时间代理验证。

## 时间与外部依赖

- 01 中初赛结果截止为 2026-09-30 11:30，报名截止为 11:00；官方文本时区待平台确认。
- 尚未取得比赛样例、索引、官方 evaluator 与测试视频；不推定已报名。
- TVSum 无 composition GT；其时间代理指标不得声称是实际比赛 F_video。
- MB 字节基数、旋转/VFR 官方约定等保留为待核对，代码采用显式本地约定。
