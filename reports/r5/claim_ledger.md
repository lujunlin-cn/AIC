# Round-5 claim ledger

状态词表：NOT_RUN / INVALID_IMPLEMENTATION / DEV_ONLY / INCONCLUSIVE /
LOCAL_GATE_PASS / WAIT_OFFICIAL / OFFICIAL_GAIN / NO_PRACTICAL_GAIN_IN_SCOPE /
PAUSED_BUDGET。每条 claim 附 scope、比较、门槛、区间、artifact、重开条件。

## 已裁定（round-5 之前遗留）

| Claim | 状态 | Scope 与依据 | Artifact |
|---|---|---|---|
| 期望F目标（pooled 表示、独立动作期望二值F、本训练配方与导出器） | NO_PRACTICAL_GAIN_IN_SCOPE | 干净确认集 GROUP B 平局 [-0.003,+0.001]；官方 EXACTDP 与母本平；R5 §5.1 关闭范围不扩大到一切决策优化 | round4_confirm_eval_lr1e4.json |
| native 旧结果（AP/ord-rep/占用标签/8窗） | INVALID_IMPLEMENTATION | seek 单位错+无归一化，缓存已打标记；不作为模型路线负结果 | round4_native_input_audit.json |
| 旧 500 源确认集 | DEV_ONLY（降级为回归/开发审计集） | 其结果已知并影响了 round-5 设计 | fresh_confirm_access_log.jsonl |
| "raw 38.83/38.60/6.2 gap/63 上限" | 撤除 | 非法 score/k_size 换算；真实口径 34.95/34.74/45 均为未惩罚原始分，差 10.05 | score_ledger.json |

## Round-5 新裁定（本 ledger 生命周期内）

| Claim | 状态 | Scope、比较与门槛 | 区间 | Artifact | 重开条件 |
|---|---|---|---|---|---|
| A5: B3 已是候选效用头（loss 路线为重复实现） | DEV_ONLY 事实裁定 | v8_s_train_multidata.py:226,319 直接回归标注者平均 IoU（huber） | — | b3_objective_and_candidate_audit.md | 发现 B3 部署权重与训练目标不一致时 |
| S0: 锁定源上存在 ≥0.02 的评分器差距 | LOCAL_GATE_PASS（存在可利用差距，仅该公共域） | live_confirmation 124 视频 3,720 帧，oracle−head，源簇 bootstrap 95% CI 下界 >0.02 | gap 0.247 [0.227, 0.268]；70.2% 帧 gap≥0.05；head−center +0.071 | round5_spatial_oracle.json、spatial_candidate_oracle.csv | 换更多未用源复测失败，或差距由标注噪声解释 |
| A8: native 帧契约（正确 seek 单位）可解码验证 | LOCAL_GATE_PASS（工程契约，非模型结论） | 12 源抽验，流单位 seek + 就近匹配 | 16/16 唯一帧，mean err 0.004-0.02 s（round-4 为 24-1608 s） | native_frame_contract.json | 全量构建时出现 ≤2 唯一帧窗口 |
| A7: 新确认集 500 源冻结 | 冻结事件（非结论） | 排除官方 895 + dev 暴露 2,262 + round-4 触碰 508；池 10,337 | sha16 f43281bf93290cea, seed 20261005 | fresh_confirm_manifest.json | 发现感知近重复穿透（已知 ID 级限制） |

## 待裁定（进行中）

| Claim | 状态 | 前置 |
|---|---|---|
| P0-CONTENT: champion 内容消融 | **DEV_ONLY 已裁定（2026-10-04）**：champion 非"全盲"（real−zeros 双池 CI 正：dev +0.104 [0.095,0.115]、confirm +0.085 [0.070,0.100]），但内容贡献是**视频级**非片段级——real−xperm 仅 dev +0.0031 [0.0002,0.0061]、confirm +0.0041 [−0.0017,0.0097]；**纯位置先验 slotprior 追平 champion**（dev 上 −0.0036 [−0.0064,−0.0008] 反超、confirm −0.0041 [−0.0091,0.0007] 打平）；饱和 exactdp 三 delta 全 0.0 | 预注册判据"双池含零→内容不敏感"未触发，但机制裁定更强：时间排序=位置先验，champion loss/校准调优路线关闭，主推力压改切片+内容输入（R5 §1.4） |
| Anchor/Neutral 切片协议依赖 | NOT_RUN（Neutral 920 片段帧提取 920/920 完成，CPU 特征 16 分片进行中） | 特征完成后重训头对读 |
| T 候选（native 重建）过时间主门 | NOT_RUN | 契约已冻结（A8 LOCAL_GATE_PASS）；P/O/S/B/L 矩阵排队 |
| S1 方向带池化（SEGMENTS=4, 2305→5377 维） | NO_PRACTICAL_GAIN_IN_SCOPE（2026-10-04）：与 B3_s0 同种子同配置，dev 终值 live −0.005，confirmation head −0.010（0.5253 vs 0.5349）、oracle 差距未收窄（0.257 vs 0.247） | 空间攻击面收敛到候选几何/更强骨干/TTA/监督广度；重开条件=换骨干或候选几何重设计（会同时移动 oracle，需重测差距） |
| S 候选过空间晋级门 | NOT_RUN | S0 方向更新=候选几何/骨干/TTA（loss 与特征细分均已排除） |

## 本轮不重开（按 R5 指令）

- 更大时间编码器（B 级未胜位置基线前 L/V-JEPA 排队）
- 无约束期望F 配方换形式重开
- 位置先验头换 loss 重包装
- keep 0.72/0.75 连续扫描
