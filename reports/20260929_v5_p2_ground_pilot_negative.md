# V5 P2：Qwen 点名主体、GroundingDINO 落位的分工路线（pilot 负结果）

日期：2026-09-29。母本管线：INTERP（DENSE 点 + qwen_centres_interp + NOFACE + B0 EMA）。
切片：RetargetVid confirm2 按（axis × motion）分层抽 24 视频（配额 6-7/层，seed 20260929）。
脚本：`scripts/v5_p2_ground_pilot.py`。产物：`/data/aic/experiments/V5_P2_GROUND/`（远端），
本地留档 `artifacts/p2_ground/eval.json`。

## 结论

**P2 路线在当前形态下为负结果，关闭；不晋级、不打包。**

| 变体 | 替换规则 | Δ（ground − dense） | CI95 | better/worse |
|---|---|---|收敛|
| v0 | qwen 顺序首身份 + 覆盖率 ≥60% 门，无条件替换有检测的帧 | **−0.0657** | [−0.1063, −0.0317] | 7/16 |
| v1b | 同上，但替换限幅 \|Δc\| ≤ 0.75×窗口 | **−0.0385** | [−0.0621, −0.0200] | 5/18 |

by_stratum（v1b）：x_fast −0.042，x_slow −0.061，y_fast −0.029，y_slow −0.021；
四个分层全部为负，无子群收益。

## 机制诊断（为什么输给 DENSE 点）

1. **单帧 top-1 检测框中心太噪**：GroundingDINO-tiny 单帧框中心抖动远大于 DENSE 蒸馏点
   （后者本身来自 32B 教师全帧推理 + 平滑）；替换等于用高方差信号覆盖低方差信号。
2. **覆盖率门挡不住"主体命中但框不准"**：覆盖率门只保证身份频繁出现，不保证单帧框中心
   靠谱；小目标（滑雪者）框中心对裁剪 IoU 的梯度方向经常是错的。
3. **限幅缓解但不逆转**：限幅只截断大幅跳变，不改变平均方向错误；收益视频从 7→5 变少
   说明部分原始收益正来自大幅替换，混杂在噪声里。
4. **Qwen 点名 → 短语 → 检测器**这条信息通路上，"重要对象"的信息在**短语选择**一步已经
   用掉了（选出 top 身份）；定位一步（框中心）没有额外信息增益，只有额外方差。

## 教训（供后续路线）

- 检测器落位要赢 DENSE 点，必须**时序聚合**（如按身份做整行跟踪平滑）或**置信度门控**，
  单帧直通不可行——与 P0（两次顺序一致的 rerank 复核，+0.0093）形成对照：
  同为"再看一眼"，候选裁剪空间复核有效、单帧检测框直通无效。
- 若重启此路线：先离线验证"检测框中心 + 时序平滑 > DENSE 点"这一前置命题，命题不成立
  则整条路线不投入。
- AVE-PM 类事件标签、类目、伪标签不得冒充人工裁剪 GT；本路线全程未用任何分类/事件标签。

## 复现

```
PYTHONPATH=/home/supie/AIC /opt/miniconda3/envs/cv/bin/python scripts/v5_p2_ground_pilot.py --stage split
PYTHONPATH=/home/AIC/supie/AIC ... --stage qwen    # vLLM TP4, 24 视频 ≤6 关键帧点名
... --stage detect                                  # GroundingDINO-tiny fp32 cuda:4
... --stage eval                                    # 配对 bootstrap 5000
```

（详细参数见脚本；detections/qwen_subjects/eval 产物在 `/data/aic/experiments/V5_P2_GROUND/`。）

## 局限

- 24 视频 pilot，非全 confirm2；但 4 分层一致为负 + 两个变体一致为负，方向性结论可信。
- 只测了 tiny 模型、单帧直通；时序聚合变体未测（超 pilot 时间盒）。
