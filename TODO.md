# 近期任务

## P0 — 本轮收口

- [x] 重新读取协议和实际本地/远程状态；保留旧失败和checkpoint。
- [x] 原7条降级为 comparison_holdout_v1，不伪造 TVSum 新 lockbox。
- [x] ranking/author-style 15% summary实现和 MATLAB/Octave 数值交叉验证。
- [x] 真实 MAT category + repeated nested source-group CV：90次训练、3seed、2×5fold。
- [x] 视频级配对 delta CI、逐视频/类别/FP/FN/过选分析、timeline/contact sheets。
- [x] SumMe替代 raw 路径恢复、严格帧数筛选、原生 evaluator 交叉验证。
- [x] 三个历史完整bundle OOD + 三条表示各30个匹配CV heads的OOD。
- [x] RetargetVid annotations、20条DHF1K原视频、6方法真实IoU、native前置clamp核对。
- [x] dense spatial raw inference/EMA/真实face observer/权重计量/JSONL一致性验证。
- [x] 完成有界 SumMe 独立第二批8条固定配置评测；两批14条分开报告并合并，全部CV heads与完整bundle均完成。
- [x] 加入固定预算常量/均匀/32次随机非学习对照，视频级paired CI；700对SHA/稀疏pHash筛查。
- [x] 冻结3个候选的完整权重SHA/bytes、离线运行入口；本地备份、真实阈值raw JSONL验证。
- [x] 完成A0/DeiT线性head容量控制60次nested训练及已有OOD复核，结果拒绝当前线性替换。
- [x] 最终报告、registry、provenance、状态与干净代码副本复现；本地回归通过，Git/远程同步留审计。

## P1 — 按当前证据排序

- [ ] 扩大可靠 raw OOD 覆盖并核对源视频版本/近重复；模型参数冻结，不能用新增 OOD 反复调参。
- [ ] 取得 AIC 样例/index/同版本 evaluator；先复现 A0 center；有联合GT才做oracle和官方大小加权。
- [ ] 独立空间DEV/评估划分后，针对多人选错主体做一个受控 observation/association 实验；保持center控制和全权重计量。
- [x] `SPATIAL_GROUP_003` 完成固定 top-3 人脸群体中心对照；相对单脸/center 均未形成稳定增益，降级该具体假设，保留后续真正主体关联研究。
- [ ] 完成 DHF1K/RetargetVid 021–030 固定参数确认，若原视频与标注恢复成功则只做一次性 transfer check，不再用其调参。
- [ ] 继续 `YTH_ACQUIRE_001` Range tar 索引；先取得所有小 metadata，再决定是否仅恢复有界人工标注视频 subset。
- [x] 记录 YTH/DHF1K 新来源连接 blocker；不把未取得原视频写成 OOD 或空间泛化证据，后续改用稳定镜像/预提取 benchmark 或继续已有固定候选。
- [ ] 基于 native summary/ranking 错误提出单一 loss/head 假设，通过nested inner选择；不进行DeiT超参扫。
- [ ] 仅在teacher能快速本地部署时，50–200训练/开发clip pilot，用human correlation测质量；暂无收益证据，不蒸馏扩量。
- [x] 保存fallback发行配置、环境、权重SHA和完整提交入口；严格官方环境仍待公布，不能声称已官方验收。

## P2 — 当前明确低优先级

- [ ] Feature Bank v2仅修复恒零/重复后做独立残差门控组；真实音频需实际波形。
- [ ] 只有新OOD机制证据时才重开canonical TSM；保留代码/cache/tests。
- [ ] 官方联合数据到位后研究size-score Pareto、压缩；不把proxy加权冒充competition score。
