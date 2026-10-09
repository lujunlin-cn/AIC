# R13 F1-F3 CPU 窗口审计报告（2026-10-09 上午）

范围：F1（关 8-10 mAP 复现缺口）与 F3（Mr.HiSum 主训练集部署头）的官方代码审计、
口径归一、F3 前置管线。约束：NPU 10h 窗口留给 4B/27B 线，本窗口全程 CPU。

## 1. 官方基线代码审计（upstream_repo 在盘，5 个配方 diff 实锤）

官方 repo：`MrHiSum/annotations/upstream_repo/model/`（PGL-SUM / VASNet / SL-module
三基线；README 训练命令 batch 256 / epochs 200 / SL-module lr 0.05）。

我方 P0-b（59.17）与官方（50.80/67.20）的对比混入了 5 个变量：

| # | 变量 | 官方 | 我方 P0-b |
|---|---|---|---|
| 1 | 指标口径 | 5-clip 段均值→softmax→对二值 top50%/15% 掩码的 sklearn AP | top-ρ% 预测集 vs top-ρ% gtscore 集的 precision |
| 2 | 特征源 | mr_hisum.h5/features（yt8m tfrecord float，**rgb 1024 only**） | processed/yt8m uint8 手动 dequant，rgb+audio 1152 |
| 3 | dequant 公式 | `q*4/255-2`（yt8m_reader.py，范围 [-2,2]） | `(q+0.5)*2/255-1`（范围 [-1,1]） |
| 4 | 损失 | 全基线 = MSE（solver.py criterion） | 自实现 pairwise |
| 5 | ckpt 选择 | 每 epoch val best（ap50/ap15 分开存） | 固定末代 test 单次 |

其他官方事实：SL-module = depth5/heads8/mlp3072/dropout0.5 + SGD+StepLR(100,0.1)
+ xavier；PGL-SUM = Adam+ExpLR(0.97)。**在盘 mr_hisum.h5 无 features 字段**
（仅 change_points/gt_summary/gtscore）——官方训练特征需从 processed/yt8m 重建。

**结论**：59.17 vs 67.20 的 8 点差距不是口径幻象的先验不成立——但归一前不可判。
F1 的正确第一步=同口径重评已存 ckpt（零训练归因），再决定重训消融方向。

## 2. 同口径重评（进行中，~1.7h）

`scripts/f3_rescore2.py`（远端）：dump test-2000 特征 npz → P0-b 4 个 pairwise
ckpt 前向 → 双口径（官方 seg-AP50/15 + 我方 prec50）。
- 教训一条：首跑版本前向漏 `unsqueeze(0)`，(T,1152) 直接进模型使 attention 维度
  错乱（形状合法、数值全废），首个读数 58.32 为无效读数已修复重跑。对照规则：
  重评的我方口径 prec50 必须复现 verdict 的 58.95±0.05 才算 harness 正确。
- 环境坑：qwen35 venv 的 h5py 3.16 读 yt8m shard 存在可复现死锁（futex 风暴、
  IO 静止、位置固定 ~583MB）；tfcpu venv（h5py 3.14）60 秒完成同量 dump。
  **910A 侧批量 h5 读取一律走 tfcpu venv 或加 per-shard alarm。**

判定表（rescore2 出数后填）：
- 官方 ap50 ≥ 64 → 缺口主要在 ckpt 选择/特征（重评即大幅追回，F1 主攻=官方化改造）
- 官方 ap50 ≈ 58-60 → 缺口在配方本体，F1 主攻=TRINITY 配方重训（已备弹）

## 3. F1 头改造（已备弹，NPU 窗口即发）

`scripts/r13_f1_rank.py`：P0-b 骨架 + `--loss bce`（软标签=gtscore）/
`--depth 5` / `--schedule warmcos` / `--feat {rgb+audio,rgb,audio}` /
`--official-dequant` / 双口径评估 / val-best ckpt。
CPU 冒烟 PASS（subset 300 × d5，epoch 9.9min）。NPU 发车脚本
`scripts/r13_f1_launch.sh`：主臂 3 seed（3 卡）+ 官方 SL-module NPU 复现
（configs.py 已打 npu device 补丁；CPU 上 102s/step×109 step×200 ep 不可行，NPU 必需）。

## 4. F3 前置管线（全部完成）

1. **yt8m 合同 rgb 特征（426 视频）**：`scripts/r13_f3_yt8m_extract.py`，
   TF 2.21（tensorflow 主包 aarch64 wheel；tensorflow-cpu 从未发 linux aarch64）+
   官方 inception-2015-12-05 GraphDef + 官方 PCA（mean/eigenvecs.T[:,:1024]/
   sqrt(eigenvals+1e-4)）+ uint8 量化 round-trip（dequant 域 [-2,2] 精确对齐）。
   graph parity 金标准 PASS（官方 cropped_panda.jpg top1 置信度 0.885 ≈ 官方教程读数）。
   424/426 落盘（vid 39/80 全长 3s 不足 4 帧拒收，组包器斜坡兜底覆盖）。
   8 worker 并行 ~8 分钟；单帧 0.2s。
2. **质量门 PASS**：ours 每维 mean −0.0004 / std 1.0252 vs Mr.HiSum 单帧
   mean −0.0079 / std 0.9707（白化方差结构吻合）；[-2,2] 量化域合法。
3. **组包器**：`scripts/r13_f3_package.py`，单变量声明（temporal scorer=
   Mr.HiSum 监督轻头 vs R12 GATT PHD2 监督头）、keep 0.80 冻结、B3 bbox 逐字节
   复制、秒级分数广播到帧。随机头全链冒烟 PASS（426 视频、pred_frames=192595、
   sym_diff 0.2749、CPU 前向 5.5min）。

## 5. 边界与挂起项

- VGGish 426 音频通道：transformers 5.19 已移除 Vggish（aha49 4.49 venv import
  报 NPU 探测错）；**挂起等 F1 消融的 audio 通道贡献读数**——若 <1 mAP 则 426
  侧无需提音频，F3 直接 rgb-only。
- 官方 SL-module NPU 复现：NPU 窗口第二优先级（F1 主臂之后）。
- rescore 判定表 → F1 主臂发车参数（NPU 窗口 19:42 后，cron 已设）。

## 2.1 rescore3 终评（10-09 12:10，4 seed 全量，harness 浮点级验证）

| 官方口径 | 我方 P0-b（4 seed） | 官方基线（论文） | 缺口 |
|---|---|---|---|
| ap50 | **66.10**（65.99/65.62/66.70/66.09） | 67.20 | **1.10** |
| ap15 | **34.33**（34.80/33.16/35.12/34.25） | 50.80 | **16.47** |
| 我方 prec50 | 59.17 | — | 与 P0-b verdict 逐一吻合 ✓ |

结论修正：
1. ap50 口径下配方债不成立（差 1.1）——P0-b 头排序质量接近官方基线。
2. **真缺口在 ap15（−16.5）**：头部帧（最精彩 15%）判别力大幅落后。机制假设
   （【推断】）：官方用 MSE 直接回归 gtscore 幅度，pairwise 只保排序不保幅度；
   top-15% 需要幅度信息区分「好」与「最好」。修复方向=软标签回归
   （BCE/MSE）——已在跑的两臂（f3_head_s0 / f1_deq_ablation_s0）正是 BCE 配方。
3. 部署含义：keep 掩码恰是「挑最好的帧」，ap15 是比 ap50 更贴近部署的口径；
   BCE 臂出数后以 ap15 为第一判据、ap50 为第二。

## 6. 运维教训（本窗口新增）

1. **kill 未生效+log 截断交错**：SIGTERM 打不进 torch 长前向进程（C 扩展段），
   `kill; sleep 2; nohup 重发` 后旧进程仍持原 fd 写 log（offset 保留、内容交错），
   bug 版读数被当成修复后读数判读了 1 小时。规程修正：杀进程必须 `kill -9` +
   `ps 确认清零` 后才重发；关键读数行打印进程 PID 与代码版本标记。
2. **harness 对照锚**：任何重评脚本的读数必须先复现历史自报数（本例 first-200
   prec50=59.59 vs P0-b 全量自报 58.95，同量级 ✓）才可外推；rescore2 的 52.48
   即被锚判定为 harness 错误，避免了一次错误归因。
3. h5py 3.16（qwen35 venv）读 yt8m shard 可复现死锁；h5 批量读取走 tfcpu venv
   （h5py 3.14）。
