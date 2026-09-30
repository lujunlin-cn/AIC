# V7 LFM 24h 执行报告（T0 = 2026-09-30 09:15 CST，本报告写于 T+5.5h）

 prereg 契约：`configs/V7_LFM_PREREG.json`（冻结于 T0 前）。本报告覆盖：E0 门、S 线（空间主攻）、T 线（时序诊断）、
融合诊断、复赛就绪。所有训练均在 910A（aic-batch 容器）完成，910A 耗时为实测墙钟。

## 一句话结论

**450M 纯学生第一次拿到"训练后显著优于自身零样本与 B0 观测"的空间证据链**：LFM_V7_S_FINAL 已打包
（zip eb67f04d…，450,283,377 参数 ≤ 500M），官方成绩待评（按约定不预报）；KD 与解冻两个增强臂均为负结果；
时序 TCN 两 seed 过 C3（诊断记录，无提交通道）；复赛评测集的全链就绪已在 910A 验证（用户已提示不再跑当前官方集）。

## 候选矩阵落点（对照 prereg）

| 候选 | 状态 | 说明 |
| --- | --- | --- |
| LFM35_BASE_REPRO | 已有 | LFM450_RAW_GROUND_V1 官方 35.00（zip 4e4f8b91…） |
| LFM_V7_S_FINAL | **已打包待评** | 训练候选效用头替换零样本文本读出；单一因素；其余与 B0/QWEN_POINT 链全同 |
| LFM_V7_T_DIAG | 诊断（C3 过） | TCN 两 seed 超 RANDOM/UNIFORM，CI>0；无提交通道（时间轴全选不变），只记录 |
| LFM_V7_ST_FINAL | 不出包 | 时序头没有合法的提交增量通道（mask 全选是最优历史结论）；不为了组合而组合 |
| LFM_V7_SFT_CHALLENGER | 未做 | 空间主证据链 + 负结果 + 复赛就绪已占满高价值时间；SFT 无对应短板 |

## S 线（主攻）：训练候选效用头

**方法**：LFM2.5-VL-450M 单 tile NaFlex 视觉塔网格 (22,40,768)（fp16 前向，无文本提示、无 generate），
129 个等距合法窗候选（自由轴，含中心），win/out token 均值池化 + 差 + 归一位置 = 2305 维 →
MLP 2305-512-128 + 2 层 eager self-attention → 效用；Huber(δ.25)+0.3 pairwise（max gap）；
GT = RetargetVid 6 标注员平均 IoU（纯公开裁剪 GT，无教师输出、无 B0 观测、无官方测试信息）。
1,511,425 参数；训练 1200 步 batch16 lr 3e-4，~350 s（NPU）。

**选择（预注册 S_DEV 601-620）**：seed0 0.6541 / **seed1 0.6650**（vs center +0.059 / +0.072）→ 主候选 seed1
（checkpoint sha256 81dcdbac…53a011）。

**DIAG 031-100（70 源，从未训练从未选择；源视频配对 10k bootstrap）**：

| 对照 | seed0 | seed1 |
| --- | --- | --- |
| head − B0 同帧重算 | **+0.0474 [+0.0219,+0.0734]**（46/24） | **+0.0575 [+0.0332,+0.0827]**（49/21） |
| head − CENTER | +0.0604 | +0.0705 [+0.0458,+0.0973] |
| head − LFM_GROUND（35.00 零样本读出） | +0.0724 [+0.0459,+0.0994] | +0.0692 [+0.0263,+0.1126] |
| head − QWEN_T（32B 教师点） | +0.0124 [−0.0068,+0.0311] | +0.0281 [−0.0026,+0.0557] |

**C1 全过**：vs 零样本 CI 下界 > 0；全部分层均值非负、无分组 >0.01 退化；y 轴不显著负且实际显著正。
分层（seed0，head−B0）：y 3-1 关键帧级 +0.0121 [+0.0059,+0.0184]；x 1-3 +0.0845 [+0.0694,+0.0996]；
fast_motion +0.027 [−0.024,+0.077]；shot_cuts +0.064 [+0.005,+0.132]；multi_person +0.051 [+0.014,+0.090]；
single_person +0.045 [+0.011,+0.080]；regular +0.030 [−0.013,+0.074]；nonperson +0.064 [−0.008,+0.146]。
oracle best−center +0.224 → 头吃到约 1/4 的剩余上行空间。
**450M 学生头在 DIAG 上统计追平 32B 教师点**（两 seed 的 CI 均含 0、均值 +0.01~+0.03 为正）。

## 负结果（同等交付）

| 臂 | 配置 | 结果 | 判定 |
| --- | --- | --- | --- |
| GT+KD（C4） | 教师点软目标 σ=span/16，w=0.2，200 源缓存零新调用 | best S-dev 0.6465 < GT-only 0.6541（同 seed0） | KD 臂拒绝；公开 GT 足够，教师点在本数据量下无增益 |
| 解冻（C2） | 视觉塔末 2 层 + projector + head 解冻（24,078,849 参数 FP32），在线提特征（旧缓存失效） | best 0.6570 在 step 100，随后震荡 0.630→0.598→0.637→0.623→0.646 < frozen 0.6650 | 解冻臂拒绝；"先冻结"方向正确；若继续需更低 backbone lr 与更长预算 |
| blend（诊断） | 推理时教师先验 u+λ·exp(−.5((o−ot)/σ)²) | λ=0.2：blend−head **+0.0105 [+0.0018,+0.0198]**（3,074 帧/70 源）；λ≥0.4 无增益 | 显著但小；融合把 32B 拉进部署总量破 ≤500M 档 → 不打包，只记录。教师单独 −0.0068 [−0.024,+0.011] |

## T 线（时序诊断）：LFM 原生特征从头重训 TCN

TVSum 50 源 1fps pooled 特征；T1 式 6 膨胀残差块 TCN（768→128 proj，**394,241 参数**，不加载任何 Mr.HiSum 权重）；
监督 = 20 人逐帧均值（Huber δ0.1 手写 NPU 版 + 0.5 pairwise |Δy|>0.05）。dev16 段中心秒评分，200 次 seeded RANDOM 配对：

| seed | macro nDCG@15% | vs RANDOM（配对） | vs UNIFORM | 胜/负 |
| --- | --- | --- | --- | --- |
| 0 | 0.8725 | +0.1008 [+0.0592,+0.1439] | +0.1001 [+0.0565,+0.1454] | 12/16 |
| 1 | 0.8850 | +0.1133 [+0.0697,+0.1553] | — | 14/16 |

**C3 过**（预注册门槛 RANDOM 0.623 / UNIFORM 0.629，实测基线更高也过）。定位：诊断记录——LFM pooled 特征
携带可用时序信号（对照上轮"多图时间不可用"的零样本结论，从头训小头即可用）；但时间轴全选是最优历史结论，
无合法提交增量通道，不构成候选。

## E0 门（复现与验证）

- **协议冲突复现与解决（C0）**：同 4 帧、同栈、单变量 schema——官方 JSON array [0,1] 4/4 可解析；E5 `<box>`0-1000 0/4
  （与 E5 0/12 一致）；JSON+1000 措辞仍输出 [0,1]。**E5 是协议失败，不是 450M 无定位能力。**
- **910A 真实训练验证**：forward→loss→backward（4 参数组梯度全非零）→clip→step（40/40 参数实际变化）→save→reload
  （前向 diff 1.1e-4）→re-forward；0.092 s/iter、峰值 1.62 GiB。**降级：GradScaler 不可用（ascend910 无 AmpUpdateScale）→ 纯 FP32；SDPA 后端不可用 → 自写 eager attention。**

## 910A 实测耗时（aic-batch）

| 步骤 | 规模 | 墙钟 |
| --- | --- | --- |
| 特征缓存（4 卡 shard） | 200 源 4,659 帧 | ~2 分钟（0.02 s/帧） |
| S-frozen 头训练 | 1200 步 | 349 s（seed0）/ 266 s（seed1） |
| 解冻训练 | 600 步 accum4 | 632 s |
| KD 训练 | 1200 步 | ~350 s |
| 官方推理（3 shard） | 3,106 帧 | 138-187 s（0.16 s/帧） |
| TCN | 1200 步 + eval | 292 s（CPU；NPU 小算子反而 20× 慢） |
| obs cache CPU 版（复赛链） | 2 视频 | 52 s（≈26 s/视频，174 视频约 20-25 分钟 @4 parts） |
| DIAG 聚合 / 融合诊断 | CPU | ~4 分钟 / ~7 分钟 |

## 打包与 SHA

- **LFM_V7_S_FINAL.zip** sha256 `eb67f04d26a84211c5b287e8a4f5fdde9c4a2bf5cb61ff03814b9ae8a7fb5a3c`（953,468 B）
- predictions.jsonl sha256 `7ee9b202babd785519512e3871dbe7612d8cf9e5a64c12052af369d23910c9a1`（174 视频 / 87,781 帧）
- 与 B0 关系：帧掩码 diff 0、最大宽度 diff 0（唯一变量 = 自由轴位置）；独立检查器 + 独立解压回读均 valid；parse_fail 0
- 部署总量 450,283,377 参数（LFM 448,718,848 + 头 1,511,425 + YuNet 53,104）≤ 500M 硬顶 ✓
- manifest：`/data/aic/official_test_20260926/submissions/LFM_V7_S_FINAL/manifest.json`
- 提交顺序建议：**LFM_V7_S_FINAL（纯学生空间）**；教师档最佳仍是 INTERP 48.67（已评）。无第二学生包（避免非单因素堆料）。

## 复现命令（服务器 221.213.81.199:44000，容器 aic-batch）

```bash
# 0) 环境
cd /root/AIC && set -a && . /data/aic/tools/ascend_teacher.env && set +a
PY=/data/aic/tools/lfm_venv/bin/python
# 1) E0 协议对照
$PY scripts/v7_e0_protocol_check.py
# 2) 特征缓存
bash scripts/launch_v7_extract.sh   # 4 shard NPU 2-5
# 3) S 头两 seed
$PY scripts/v7_s_train_head.py --device npu --seed 0 --output-dir /data/aic/experiments_910a/LFM_V7/s_head_s0
$PY scripts/v7_s_train_head.py --device npu --seed 1 --output-dir /data/aic/experiments_910a/LFM_V7/s_head_s1
# 4) DIAG 配对聚合（seed1 同理换路径）
$PY scripts/v7_s_diag_aggregate.py --per .../s_head_s0/per_diag_031_100.jsonl \
  --prev-rows /data/aic/experiments_910a/LFM450_EVAL_V1/spatial_v2/rows_s0.jsonl \
              /data/aic/experiments_910a/LFM450_EVAL_V1/spatial_v2/rows_s1.jsonl \
  --output .../s_head_s0/diag_table.json
# 5) 负结果臂
$PY scripts/v7_s_train_head_kd.py --device npu --seed 0 --kd-weight 0.2 --output-dir .../s_kd_s0
$PY scripts/v7_s_train_unfrozen.py --steps 600 --eval-every 100 --output-dir .../s_unfrozen_s0
# 6) 官方推理 + 打包
$PY scripts/v7_s_official_points.py --head .../s_head_s1/head_s.pt --cards 2 --shard i --nshards 3 \
  --output-dir .../official_points_s1
python3 scripts/max_window_release.py --frozen configs/LFM_V7_S_FINAL.json \
  --index /data/aic/official_test_20260926/intake/index.enriched.jsonl \
  --output /data/aic/official_test_20260926/submissions/LFM_V7_S_FINAL
# 7) T 线
$PY scripts/v7_t_extract_tvsum.py --device npu
$PY scripts/v7_t_train_tcn.py --device cpu --seed 0 --output-dir .../t_tcn
```

## 复赛就绪

见 `reports/v7_lfm_24h/SEMIFINAL_READY.md`：7 步全链已在 910A 验证（解码 174/174、CPU obs cache、
B0-from-cache、提帧器 360×640、头推理 0.16 s/帧、打包校验器）；已知 PyAV 17 vs 15 像素微差（链内自洽，不影响复赛）。

## 边界与诚实声明

- 官方成绩待评，本报告不预报分数；DIAG/S_DEV 的 IoU 增益到官方 F 的传导系数未知（教师点历史上 IoU +0.04 量级对应官方空间链显著上行，但 B0 34.42→QWEN_POINT 的官方差尚未回分）。
- 训练只用 RetargetVid 公开裁剪 GT 与 TVSum 公开标注；未使用官方测试任何帧/预测参与训练；未使用 32B 测试期 mask 伪装纯学生。
- 历史分组审计：TVSum 训练 = 50 − dev16 = 34 源；V7 新划分（S_TRAIN/S_DEV/DIAG）已冻结于 prereg，历史 dev/confirm 降级为 train 已记录在 splits ledger。
