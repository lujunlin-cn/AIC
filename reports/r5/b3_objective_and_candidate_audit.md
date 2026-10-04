# B3 objective and candidate audit (round-5 A5)

日期：2026-10-04。目的：在训练任何新空间评分器之前，核对 B3 的训练目标是否
已经是 R5 §2.3 提议的"候选效用"目标；若是，换目标的路线就是重复实现。

## 结论

**B3 已经是候选效用头。** "训练评分器拟合 u_j（候选窗口对全部标注者的平均
IoU）、部署 argmax"——与 R5 §2.3 的建议逐字相同。R5 的警告命中："若已使用，
不把重复实现命名为新路线。"

## 证据（v8_s_train_multidata.py，仓库 commit 40ca4cd）

- 候选几何：129 个等距合法最大窗口（`--n-cand 129`，`aic/max_window_path.geometry`）。
- 目标量：`scripts/v8_s_train_multidata.py:226`
  `u = iou(win_px[:, None, :], gt[None, :, :]).mean(...)` —— 每候选对
  全部标注者平均 IoU（RV 6 人，LIVE 1 人），与 R5 §2.3 的 u_{vf,j} 定义一致。
- 损失：`v8_s_train_multidata.py:319` `loss, _ = huber_pair(x, u)` —— 直接
  huber 回归 u，softmax-IoU 损失（R5 §2.3 的 L_IoU）是它的替代形式，不是新目标。
- 评估：`head-argmax IoU` vs `center`（中心候选）vs `best`（候选 oracle），
  即 R5 要求的"旧评分器 / oracle"差距三件套，round-5 S0 审计直接消费了它。
- 训练域：rv_native + rv_rot（旋转增强）+ live_train（LIVE-YT-VC 训练划分）；
  live_dev 用于选择，live_confirmation 从未用于训练或选择（S0 锁定源）。

## 对空间路线的指向

1. 换损失/换目标 = 重复实现，关闭。
2. S0 审计（round5_spatial_oracle.json）显示锁定源上 oracle−head = 0.247
   [0.227, 0.268]，70.2% 帧差距 ≥0.05，head−center 仅 +0.071 —— 剩余差距
   大头在**评分器的判别能力**（同一 129 候选内选不对）而非候选几何覆盖
   （oracle 与 head 用同一候选集）。
3. 因此攻击面排序（R5 §2.4 框架内）：
   - **更强的候选表征**（B3 输入 2305 维 = 几何 + 图像特征；换/加更强的
     编码器特征是第一杠杆，需符合参数档位）；
   - **S4 TTA**（多尺度/平移等变平均，零参数增量，可先离线验证）；
   - **监督广度**（y 轴真值不足是已知限制；LIVE 只支持 x 轴）；
   - S2/S3（教师加权、分割辅助）按 R5 先验不乐观，排后。

## 重开条件

发现部署权重与训练目标不一致（部署包用的头不是本脚本产出的谱系），或
候选几何本身被重设计（候选集合变化会同时移动 oracle 上界，需重测差距）。
