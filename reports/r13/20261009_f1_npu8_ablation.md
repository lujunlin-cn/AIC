# R13 F1 NPU 8 臂消融记录（2026-10-09 晚）

背景：F1 口径重评（rescore3）确认 ap50 缺口仅 1.1（66.10 vs 67.20）、ap15 缺口
16.5（34.33 vs 50.80）后，用户 20:00 授权 NPU。8 卡 8 臂配置级并行（无 DDP），
20:44:42 齐发，单臂 3 epochs + 每 epoch 全量 val（1998）+ 单次 test 双口径。
本文件为中期记录（ep1-ep2 读数，21:42），FINAL 终判后补 §4。

## 1. 臂矩阵（每卡一臂）

| 卡 | tag | 配方 | 检验因子 |
|---|---|---|---|
| 0-2 | f3_head_s0/1/2 | BCE/d5/warmcos/rgb/官方dequant | 主臂 + seed |
| 3 | f1_deq_p0b_s0 | 同上但 P0-b dequant | 特征尺度合同 |
| 4 | f1_mse_s0 | MSE 替换 BCE | 官方原版损失 |
| 5 | f1_d3_s0 | d3 替换 d5 | 容量 |
| 6 | f1_audio_s0 | rgb+audio 1152 | 音频通道 |
| 7 | f1_pair_s0 | pairwise 替换 BCE | 损失族对照 |

环境配方：jit_compile=False、CPU init 权重、set_device 显式落卡（--npu-id）。
吞吐 ~0.72s/step（CPU 2.65s/step 的 3.7×；8 臂并行等效 29×）。

## 2. 中期读数（ep1 中段 val，21:42）

| 臂 | ap50 | ap15 | 备注 |
|---|---|---|---|
| **f1_pair** | **65.16** | **34.75** | ep0 val 63.50/32.48，仍在涨 |
| f1_audio | 61.55 | 27.49 | vs s1（无音频同配方）+2.6/+3.2 |
| f1_mse | 59.74 | 24.68 | ep0 val 59.55/24.66 |
| f1_d3 | 59.52 | 25.68 | ep0 val 60.91/25.90（ep1 回落，val-best 保住） |
| f3_head_s1 | 58.95 | 24.32 | |
| f3_head_s0 | 58.56 | 22.90 | |
| f1_deq_p0b | 58.51 | 24.12 | |
| f3_head_s2 | 57.15 | 23.43 | |

参照：P0-b 头（pairwise/d3/P0b-dequant，3 epochs）官方口径 66.10/34.33；
官方基线 67.20/50.80。

## 3. 两个中期结论

1. **ap15 机制假设证伪**：原假设「pairwise 只保排序缺幅度 → 拖累 top15%」
   与数据相反——pairwise 全面碾压 BCE/MSE（34.75 vs 22-25）。**pairwise 才是
   Mr.HiSum 域正确损失**；等权回归（BCE/MSE）稀释头部判别力。F1 消融矩阵
   一次买回该答案。
2. **音频通道真实有效**：+2.6 ap50 / +3.2 ap15（audio vs 无音频同配方）。
   426 侧 VGGish 128d 提取升级为值得排期（此前挂起等此读数）。

## 4. FINAL 终判（22:32 全出，单次 test 双口径）

| 臂 | test ap50 | test ap15 | prec50 | best_ep |
|---|---|---|---|---|
| **f1_pair（pairwise/d5/官方deq）** | **65.78** | **34.01** | **58.71** | 2 |
| f1_d3（BCE/d3） | 61.68 | 26.83 | 55.60 | 0 |
| f3_head_s1（BCE/d5） | 61.52 | 27.50 | 55.14 | 2 |
| f1_deq_p0b（BCE/d5/P0b deq） | 61.52 | 26.95 | 55.10 | 0 |
| f1_audio（BCE/d5/audio） | 60.94 | 26.99 | 55.19 | 1 |
| f1_mse | 60.89 | 25.32 | 55.36 | 2 |
| f3_head_s2 | 60.51 | 24.70 | 55.14 | 0 |
| f3_head_s0 | 60.30 | 24.82 | 54.74 | 0 |

参照：P0-b 头（pairwise/d3/P0b-deq）66.10/34.33；官方基线 67.20/50.80。

判定：
1. **pairwise 唯一胜者**：65.78/34.01，ap15 领先第二名 6.5 点、ap50 领先 4.1。
   ≈P0-b 复现（差 0.3 ap50，seed 噪声内），配方债闭环。
2. BCE/MSE 假设证伪记录在案（§3）；dequant 在 BCE 下无差异（61.52 双平）。
3. **audio 增益不稳**：ep1 的 +2.6/+3.2 到 FINAL 收窄为 seed 噪声级（60.94 vs
   s0/s2 的 60.30/60.51、低于 s1 的 61.52）。VGGish 426 提取维持暂缓，待
   pairwise+audio 组合臂定夺。
4. d3 vs d5（BCE 下）：61.68 vs 60.3-61.5，d3 略优或平——容量不是 BCE 的问题。

## 5. F3 部署头冻结决策

- **冻结配方：pairwise / d5 / warmcos / rgb / 官方 dequant，Mr.HiSum 27,892
  全量监督，keep 0.80**（ckpt f1_f1_pair_s0_valbest.pt，best_ep=2）
- 单 seed 局限标注：pairwise 族 4-seed 方差已知小（P0-b ±0.4），可接受。
- **提交包已出**：R13_F3_MRHISUM_PAIR.zip（sha256 c2859eecc8e8ca45…，
  pred_frames=192642，feat_missing=2 兜底，sym_diff 0.2877 vs 35.17 包，
  空间框逐字节复制 B3；登记 file id 59091）
- 单变量声明：temporal scorer = Mr.HiSum 监督 pairwise 头（vs R12 GATT 的
  PHD2 监督头）；平台读数直接回答「外部行为监督 × 正确损失」能否突破 35.17。
