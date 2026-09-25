# 近期任务

## P0

- [x] 固定 TVSum GT/预测阈值，增加 per-video、macro/micro、连续排序诊断。
- [x] 完成 50 个 TVSum MAT/TSV 记录的 shape、nframes、fps、duration、2 秒收集语义审计。
- [x] 修复 padding/GroupNorm 变长 batch 一致性并保留回归测试。
- [x] 验证远程同环境 raw-video 与 cache 路径；固定 PyAV 15.1 推理环境。
- [x] repaired A0/A1、简单 baseline、5-fold source-group stability、Feature Bank 分组、B0 frozen ViT probe。
- [ ] 获取官方样例、evaluator、初赛索引和真实联合 GT；拿到后先跑 center-crop fallback。
- [ ] 锁定官方输入上的可提交 Safe Engineering Baseline，重新在 train/dev 选 threshold。

## P1

- [ ] 真实 backbone-internal TSM 做小规模 recache/end-to-end 对照；当前只完成中间 feature map correctness/throughput probe。
- [ ] 继续验证 B0 的多 split/multi-seed；若收益覆盖 M 档 size coefficient，再集成完整报告。
- [ ] 使用合法 crop GT 做固定 temporal frame 集合的 center/saliency/subject IoU 对照；没有 GT 时保持 null。
- [ ] 解决 Feature Bank 重复/恒零维度后，使用 residual/gated fusion 重开单组实验。
- [ ] 进行一次本地开源 VLM 小样本 teacher signal pilot；不上传比赛测试视频，不直接启动批量伪标签。

## P2

- [ ] 按官方联合分数做 temporal/spatial oracle、size coefficient 和 latency Pareto。
- [ ] 候选多 seed、video bootstrap CI、跨场景失败案例和最终干净环境复现。
- [ ] 备份 fallback checkpoint/config/environment，并准备复赛材料。
