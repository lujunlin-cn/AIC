# V7 LFM 24h 进度日志（T0 = 2026-09-30 09:15 CST）

冻结契约：configs/V7_LFM_PREREG.json。成绩台账：LFM 35.00（zip 4e4f8b91…）、教师 INTERP 48.67（501cebce…）、XRERANK 48.49（027fd12a…）。

## 检查点 1 · T+1.6h（10:50）

### E0 全部通过
- **协议冲突解决（C0）**：同 4 帧（031-034/0，E5 同批）、同 910A FP16/eager/jit-off 栈、只改回复 schema：
  JSON array [0,1]（上轮 3210/3210 协议）4/4 解析；E5 P1 `<box>x1 y1 x2 y2</box>` 0-1000 = **0/4**（与 E5 0/12 一致）；
  JSON schema + 1000 措辞仍输出 [0,1] JSON 4/4（schema 主导，坐标单位措辞无效）。
  **结论：E5 是协议失败，不是 450M 无定位能力。** 证据 LFM_V7/e0/protocol_check.json。
- **910A 真实训练通过**：forward→loss→backward→clip→step→save→reload→re-forward。
  GRAD_CHECK group0-3 = 4+4+16+16 全部非零梯度（head/projector/视觉末2层）；60 步 loss 0.196→0.010；
  40/40 训练参数实际变化；重载前向 diff 1.1e-4；0.092 s/iter；峰值 1.62 GiB。
  **降级说明：GradScaler 不可用**（ascend910 无 AmpUpdateScale 算子）→ 纯 FP32 训练路径（vision 94M 参数 FP32，HBM 无压力）。
- 特征缓存 v7_feats_v1 完成：200 源全部关键帧 4,659 帧，网格 (22,40,768) fp16 + pooled，**0.02 s/帧**（4 卡 shard）。

### 运行中
- S 头训练（NPU 2）：129 等距合法窗候选 + win/out token 池化 + 2 层 set transformer，GT-only（6 标注员 IoU 均值），
  S_TRAIN 110 源 / S_DEV 20 源（601-620）/ DIAG 031-100（不训练不选择）。
- TVSum 1fps LFM pooled 提取（NPU 4,5）：50 源，T 线输入。

### 环境/踩坑记录
- docker exec 内联 source env 不生效（变量未 export）→ 必须 launch 脚本内 `set -a; . env; set +a`（同 launch_spatial.sh 模式）。
- aic-batch 内 NPU 6/7 不可见（复现既有结论）→ 分卡只用 2-5。
- Lfm2VlForConditionalGeneration 的 vision_tower 在 `.model` 下。
- 910A NPU 1 有他人 python 进程（不动），gemm-lab 容器（NPU 0-1）不碰。

## 检查点 2 · T+3.7h（12:55）—— 空间主候选已打包

### S 线（空间，主攻）全部完成
- **S-frozen 头（GT-only）**：S_TRAIN 110 源 4,498 样本（129 等距合法窗候选，win/out token 池化 + 位置 → MLP 2305-512-128 + 2 层 eager self-attn）。
  两 seed 方向一致：seed0 S-dev **0.6541** / seed1 **0.6650**（vs center +0.059 / +0.072）；训练 1200 步 ~350s。
  **主候选 = seed1**（预注册选择集 S_DEV 601-620 择优）。checkpoint 81dcdbac…53a011，1,511,425 参数。
- **DIAG 031-100（70 源，不训练不选择）源级配对 bootstrap 10k**：
  - seed1：head−B0 **+0.0575 CI[+0.0332,+0.0827]**（49胜21负）；head−CENTER +0.0705 [+0.0458,+0.0973]；
    head−LFM_GROUND（35.00 零样本读出）+0.0692 [+0.0263,+0.1126]；head−QWEN_T（32B 教师点）+0.0281 [−0.0026,+0.0557]
  - seed0 复现：+0.0474 [+0.0219,+0.0734]（46/24）
  - **C1 全过**：CI 下界>0、全部分组均值非负、无分组 >0.01 退化；
    y 轴 3-1 关键帧级 +0.0121 [+0.0059,+0.0184]（显著为正，非负信号）；
    x 1-3 +0.0845 [+0.0694,+0.0996]；fast_motion +0.027 [−0.024,+0.077]；shot_cuts +0.064 [+0.005,+0.132]；
    multi_person +0.051 [+0.014,+0.090]；single_person +0.045 [+0.011,+0.080]。
    oracle best−center +0.224（头吃到约 1/4 上行空间）。
- **解冻臂（C2）负结果**：vision 末 2 层 + projector + head 解冻（24,078,849 参数 FP32，600 步 632s），
  best S-dev 0.6570 出现在 step 100，随后震荡下行（0.630→0.598→0.637→0.623→0.646）< frozen 0.6650 → 解冻臂终止，frozen 为主。
  旧特征缓存 v7_feats_v1 对解冻臂失效（在线提特征），结论"先冻结"方向正确。
- **KD 臂（C4）负结果**：GT+KD（σ=span/16，权重 0.2，教师点 200 源缓存零新调用）best S-dev 0.6465 < GT-only 0.6541（同 seed0）→ KD 臂拒绝。
- **官方推理完成**：v7_s_official_points.py 3 shard（NPU 2/3/5）3,106/3,106 帧、174/174 视频，0.16 s/帧（vision tower fp16 前向 + 头，无文本提示）。
- **LFM_V7_S_FINAL 打包完成**：policy QWEN_POINT 全链路（EMA .25 / hold / YuNet fallback / 全选 mask / 最大宽度不变），
  width diff 0、mask diff 0，独立检查器 + 独立解压回读均 valid，parse_fail 0。
  **zip sha256 = eb67f04d26a84211c5b287e8a4f5fdde9c4a2bf5cb61ff03814b9ae8a7fb5a3c**（953,468 B；predictions 7ee9b202…c9a1）。
  部署参数 450,283,377（LFM 448,718,848 + 头 1,511,425 + YuNet 53,104）≤ 500M ✓。官方成绩待评，不预报。

### T 线（时序）
- TVSum 50 源 1fps LFM pooled 特征完成；TCN（T1 式 6 膨胀块，LFM 域从头训）运行中。
  踩坑：F.huber_loss 在 NPU 回退 CPU（~20× 慢）→ 手写 Huber 基础算子版；
  首次启动 --cards 参数不存在、旧进程残留互踩日志——已清理，单实例运行中。C3 门槛（超 RANDOM 0.623/UNIFORM 0.629）待其完成。

### 新增踩坑
- `model.config.text_config.hidden_size` ≠ 视觉特征宽：头输入宽必须 = vision hidden×3+1 = 2305（EZ1001 k-axis 报错）。
- best-checkpoint 保存对 ModuleList 切片直接 .state_dict() 会 AttributeError → 逐 blk 更新。
- points 文件 status 值需为字面 'ok'（打包器 parse_fail 判定用列表精确计数），'ok_head' 会导致 174 视频全部误标 parse_fail。
- docker exec -d 内 nohup 启动后，grep 进程数会把 bash 包装计入；pkill 后必须重新核对 python 实例数。
