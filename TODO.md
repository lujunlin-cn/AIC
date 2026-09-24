# 近期任务

## P0

- [x] 完整阅读 01 / 02 / AGENTS，检查本地与远程。
- [x] 初始化可追踪代码状态与隔离可复现环境。
- [x] 实现、测试原帧/PTS、旋转、letterbox 逆映射。
- [x] 完成逐帧 JSONL、严格 validator、官方公式本地 evaluator 数学测试。
- [x] raw-video dummy 推理输出公共居中 crop 参考（逐原始帧）并通过 validator。
- [x] 下载 TVSum 原包，记录许可/来源/hash/split/帧对齐。
- [x] A0_001 特征缓存、训练、时间代理验证与真实文件审计。
- [x] A1 TSM：相同数据、split、时序头和训练预算，单独比较 TSM。
- [x] A2 Feature Bank：motion/quality/audio/composition 总体对照；结果退化，已停止无结构调参。
- [ ] A3a 固定时间帧集合下比较学习空间候选与最大合法居中 crop（待合法 crop GT）。
- [ ] 取得官方样例/evaluator/初赛索引，核对未知契约并生成实际提交。
- [x] 核对公开 `TempSamp-R1` baseline commit/README/test_index；实现紧凑索引到实际视频元数据的审计转换。
- [ ] 锁定有证据的 SAFE_BASELINE，备份权重、配置、环境与代码。

## P1

- [ ] 联合 GT 缺口：RetargetVid/DHF1K 空间诊断；不得伪造目标比例 GT。
- [ ] 合法 VideoXum 训练子集及同源去重（先核实原始媒体授权）。
- [ ] A1 TSM 受控对照，再依次 A2 motion/quality/audio/composition。
- [ ] A3a/b/c 固定时间帧集合下比较，保存失败案例。

## P2

- [ ] A 可靠后才做 B0/B1/B2 与一次 B-sem；同数据/预算/后处理。
- [ ] Teacher 合规与质量 pilot 完成后再进入 C。
- [ ] 候选三 seed、视频 bootstrap、压缩、干净环境复现与材料准备。
