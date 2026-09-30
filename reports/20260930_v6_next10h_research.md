# AIC V6 后续 10 小时研究报告（2026-09-29 23:14 → 09-30）

执行窗口 T0=09-29 23:14 CST，覆盖 E1-AUTO / E3 / E4 / E5 四线。遵守：GPU 白名单
（V100 1/2/4/5/6/7；910A NPU 2/3/4/5）、单次训练 <12h、测试数据只做冻结算法自动推理、
不扩大已暴露 dev、不改旧阈值。无平台新回分，正式最佳保持 **INTERP 48.67**；
本报告所有 dev/confirm/AP/Spearman 均为代理指标。

## 0. 一句话结论
本轮 0 个新候选包：T1（全分辨率 TCN）通过全部预注册弱域门并拿到 indicative 迁移证据，
但按"不能仅凭 MrHiSum Spearman 出包"的 prereg 规则不出包；E1-AUTO 教师伪标签管线
打通并验证可扩展；E5 450M 零样本空间定位判负；E3 官方特征链全通。

## 1. E4 时序候选（MrHiSum 弱域 + 一次性 confirm）
预注册：configs/V6_NEXT10H_PREREG.json。dev=2,808 弱标签；confirm=4,008 fresh reserve
一次性评估；配对门=confirm 逐视频 delta vs T0 + source 聚类 bootstrap 95% CI 下界>0。

| 配置 | 参数量 | dev（种子均值） | confirm | 配对门 |
| --- | --- | --- | --- | --- |
| T0 U-Net（基线） | 1,712,897 | 0.2811 | 0.28805 | — |
| **T1 全分辨率 TCN** | **740,609** | **0.3188±0.0058** | 0.32196/0.32496/0.32890 | **3/3 过**（delta +0.0339/+0.0369/+0.0408，CI 下界 +0.0267/+0.0288/+0.0332） |
| T2 U-Net+rank | 1,712,897 | 0.2991 | 0.30248/0.30427/- | 1/3 过（s1 CI 下界 -4.0e-05）→ 否决 |

T1 另有目标域迁移证据（下节）。出包判断：**不出**。理由：证据链=弱域门过 +
迁移 indicative-mixed，仍缺学生原生域验证；把 MrHiSum 权重换进现有 bundle 不是
单一因素干净包。正确路径是把 TCN 骨架放到学生原生特征域重训（NEXT_ACTIONS #1）。

## 2. E3 原始 MP4→YT8M 兼容特征→冻结头（部署判断：DEPLOYABLE）
- 契约全部落码并 selftest 通过（scripts/v6n_e3_yt8m_extract.py）：
  1fps→TF-CPU InceptionV3 GraphDef→pool_3→官方 PCA→1024→官方量化 uint8；
  VGGish（官方转换权重 288,567,937B，sha256 10086976…561f4b）0.96s 窗→128→同族量化；
  喂头走冻结解量化 v1（与官方网格仿射等价，适配器吸收）。
- 排障三条已固化：npz PCA→上游同构 torch pth；0.96s 窗只有 94 mel 帧<96，改喂 1.0s；
  torch.hub file:// 陈旧截断缓存假 EOF。
- 官方默认行为已记录：零音频填充、300 帧上限。
- **目标域迁移（T1_s3 冻结头，YouTube-Highlights 公共 match_label，n=32，六域）**：
  macro AP **0.6162** vs 基率 0.4722；macro Spearman 0.1296。分域均值 AP：
  parkour 0.914(n=2) / gymnastics 0.688 / dog 0.674 / surfing 0.596 / skating 0.547 /
  skiing 0.526——全部域高于 0.5。证据等级 **TRANSFER_INDICATIVE**（零训练、零选择，
  仅诊断；详见 reports/v6_next10h/temporal_transfer.csv）。

## 3. E5 LFM2.5-VL-450M 零样本空间定位（负结果）
- smoke 4 帧×2 提示词（P1 box 格式 / P2 描述+box）：0/8 可解析；
  公平变体 P3（纯数字直出）：0/4。
- 关键证据：回复语义正确（钓鱼视频回 "person fishing" 等）→ 视觉通路正常；
  坐标生成退化为 `<box>Man</box>`、`-1 1 1 1` 重复循环 → 450M 级零样本
  不具备坐标输出能力。按 prereg smoke 门（≥99% 可解析）**FAIL**，不进 sweep。
- 结论：S1（模型引导裁剪）在 450M 上不可用；空间臂仍只有 S0 居中裁剪。
  空间蒸馏应改以教师点（E1-AUTO 输出）为监督。原始输出存
  reports/v6_next10h/lfm_zero_shot.csv。环境：transformers 5.1.0（/data/aic/tools/hf51），
  fp16 sdpa，~0.2s/帧。

## 4. E1-AUTO 教师伪标签 pilot（无人工标注，按 09-29 用户指示）
- 数据：PM400 pilot train split，32 条（全带 clip_context），1fps 均匀 ≤20 帧 → 640 帧。
- 教师：Qwen3-VL-32B，910A NPU 2/3/4/5，批量 eager fp16 B=4，
  prompt sha 918f0730… 断言校验。**640/640 完成，wall 56.0 分钟 ≈0.19qps（稳态=生产 0.184）**。
- 工程要点：7 种帧分辨率会各自触发 NPU 算子编译（每形状分钟级）→ 统一 720×1280
  后一次编译跑全程；批 padding 防短批形状。
- QC（仅筛选用途）：parse **100%**；保留 **29/32=90.6%**（轨迹抖动/uncertain 率淘汰 3 条）。
- 伪标签格式（pseudo_labels.jsonl）：逐帧 subject_point + 3:1/16:9/1:3 合法窗 +
  point>center>anti 偏好序 + keep_weight∈{0,0.5,1} + event_context 弱通道
  （context_only_not_crop_gt=true 如实保留）。
- 训练对照（TemporalTCN 从头训，hash 24/8 切分，3 种子，目标=公开 event 窗）：
  Arm A 仅公开标签 mean AP **0.8360**；Arm B 公开+筛选权重 **0.8429**；
  delta **+0.0069**（种子间摆动 ±0.04，**噪声内，不宣称收益**）。
  证据等级 PILOT_SCALE_FEASIBILITY：管线打通、达到用户扩展标准
  （parse 100%、保留 90.6%、~0.19qps），但伪标签权重的增益未证实。
- 达到用户设定的扩展条件 → 扩展到 108 条列入 NEXT_ACTIONS（预算教师 ~3.2h）。

## 5. 数据与曝光
台账见 reports/v6_next10h/exposure_ledger.jsonl（五条目）。标签来源清单见
labels_source_manifest.json（四来源，人工标注=false）。测试数据本轮未接触；
未使用教师测试预测/母本点/时间掩码。

## 6. 基础设施记录
- AVE-PM：按用户指示切换到用户提供的 mihomo 订阅（179.253.227.52:2096，2026-09-30 版），
  V100 临时实例 mixed 7893/controller 9093 已换新配置重启，代理连通验证通过
  （gdrive Range 探测 200/1.04s）。fetch 从 53,687,091,200/68,067,427,328 B 断点续传，
  当前处于 per-file quota 退避重试（自动恢复）；watchdog 在岗。
- aic-batch 容器补装 opencv-python-headless 5.0.0（容器重建需重装）。

## 7. 复现命令（关键）
```
# E3 官方链 selftest（V100）
PYTHONPATH=/data/aic/tools/tf_cpu:/home/supie/AIC python scripts/v6n_e3_yt8m_extract.py --stage selftest --out /tmp/e3st
# E4 训练/评估/门（V100 GPU7）
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/home/supie/AIC python scripts/v6n_e4_temporal.py --stage train --config t1 --seed 20260930 --tag s2
CUDA_VISIBLE_DEVICES=7 PYTHONPATH=/home/supie/AIC python scripts/v6n_e4_temporal.py --stage eval --weights /data/aic/experiments/V6N_E4/weights_t1_s3.pt --split confirm --out /data/aic/experiments/V6N_E4/confirm_weights_t1_s3.csv
python scripts/v6n_e4_paired_ci.py
# YTH 迁移（V100）
PYTHONPATH=/data/aic/tools/tf_cpu python scripts/v6n_e3_yth_transfer.py --stage list --n 32 && --stage extract && CUDA_VISIBLE_DEVICES=2 ... --stage eval
# E1-AUTO（910A aic-batch 容器）
python scripts/v6n_e1auto_teacher_pilot.py --stage select --n 32 && --stage frames
CUDA_VISIBLE_DEVICES=2,3,4,5 python scripts/v6n_e1auto_teacher_pilot.py --stage teacher
python scripts/v6n_e1auto_teacher_pilot.py --stage qc
# E1-AUTO 对照（V100 GPU2）
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=/home/supie/AIC python scripts/v6n_e1auto_pilot_contrast.py
```

## 8. 继续/停止判断
- T1 路线：**继续**（唯一干净候选 + indicative 迁移；下一步=学生原生域重训，见 NEXT_ACTIONS #1）。
- E1-AUTO：**继续**（达标扩展）。
- E5 450M 空间定位：**停止**（负结果终局），空间蒸馏目标改教师点。
- 出包：**本轮无**；09:30 冻结、10:00 材料就绪时间表不受影响（现有 INTERP 48.67
  与复赛包 SUB_SF_K700_SEMIFINAL_V1 均已就位未动）。

## 附：交付物索引
reports/v6_next10h/{PROGRESS_LOG.md, model_params.json, lfm_zero_shot.csv,
temporal_transfer.csv, exposure_ledger.jsonl, labels_source_manifest.json,
e2e_manifest.json, NEXT_ACTIONS.md}
远端实验目录：V100 /data/aic/experiments/{V6N_E4, V6N_E3_YTH, V6N_E1A, V6N_E5}；
910A /data/aic/experiments/V6N_E1A
