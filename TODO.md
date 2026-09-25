# 近期任务

## P0

- [x] 固定 TVSum GT/预测阈值，增加 per-video、macro/micro、连续排序诊断。
- [x] 完成 50 个 TVSum MAT/TSV 记录的 shape、nframes、fps、duration、2 秒收集语义审计。
- [x] 修复 padding/GroupNorm 变长 batch 一致性并保留回归测试。
- [x] 验证远程同环境 raw-video 与 cache 路径；固定 PyAV 15.1 推理环境。
- [x] repaired A0/A1、简单 baseline、5-fold source-group stability、Feature Bank 分组、B0 frozen ViT probe。
- [x] 冻结 `splits/local_protocol_v1.json`（TRAIN/DEV/LOCAL LOCKBOX）并加入 hash/分区回归校验。
- [x] 在冻结 lockbox 上完成 A0/A1/B0/DeiT-S/internal-TSM 对照，保存 bootstrap/per-video 诊断。
- [x] 完成零参数 temporal postprocess dev sweep、lockbox 单配置评估和 raw inference 接口。
- [x] 完成 DeiT-S/16 S-tier frozen feature、bundle 导出和 raw-video JSONL validator。
- [x] 完成 canonical internal TSM layer1 cache、correctness、5-fold 和 lockbox 对照。
- [ ] 获取官方样例、evaluator、初赛索引和真实联合 GT；拿到后先跑 center-crop fallback。
- [ ] 锁定官方输入上的可提交 Safe Engineering Baseline，重新在 train/dev 选 threshold。

## P1

- [x] 真实 backbone-internal TSM 做小规模 recache/end-to-end 对照；证据不足，已降低优先级。
- [ ] 继续验证 B0 的多 split/multi-seed；若收益覆盖 M 档 size coefficient，再集成完整报告。
- [ ] 对 DeiT-S challenger 做第二独立 split、额外 seed 和可用 OOD 数据验证。
- [ ] 获取 SumMe/YouTube Highlights 可用 raw 子集，建立 dataset-native OOD ranking/summary 指标。
- [ ] 使用合法 crop GT 做固定 temporal frame 集合的 center/saliency/subject IoU 对照；没有 GT 时保持 null。
- [ ] 解决 Feature Bank 重复/恒零维度后，使用 residual/gated fusion 重开单组实验。
- [ ] 进行一次本地开源 VLM 小样本 teacher signal pilot；不上传比赛测试视频，不直接启动批量伪标签。

## P2

- [ ] 按官方联合分数做 temporal/spatial oracle、size coefficient 和 latency Pareto。
- [ ] 候选多 seed、video bootstrap CI、跨场景失败案例和最终干净环境复现。
- [ ] 备份 fallback checkpoint/config/environment，并准备复赛材料。
