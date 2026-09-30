# V6-NEXT10H 进度日志（T0 = 2026-09-29 23:14 CST）

## 检查点 2 · T+3.5h（09-30 02:45 CST）

### E1-AUTO 教师 pilot 完成（核心里程碑）
- 640/640 查询完成，wall 56.0 分钟 ≈ 0.19 qps（含 ~9 分钟首形状编译；稳态 ≈ 生产 0.184 qps）。
- QC（仅筛选用，非真值）：parse 640/640=100%；保留 29/32=90.6%（3 条因轨迹抖动/uncertain 率淘汰）。
- 产物（910A /data/aic/experiments/V6N_E1A/）：teacher_replies.jsonl（逐帧原始回复）、
  pseudo_labels.jsonl（3:1/16:9/1:3 合法窗 + point>center>anti 偏好 + keep_weight 时间权重
  + event_context 弱通道）、qc_report.json、pilot_list.json。
- 已同步 V100，pilot 训练对照（A=仅公开 clip_context 标签 / B=公开+keep_weight 筛选权重）
  在 GPU2 运行中（TemporalTCN 从头训，3 种子×2 臂，PILOT_SCALE 可行性口径）。

### E4 全部收口
- 配对门终表：T1 三种子全过（唯一干净候选）；T2 s1 CI 下界 -4.0e-05 → 1/3 未过 → 否决。
- YTH 目标域迁移（T1_s3 冻结头，8 视频 5 域）：macro AP 0.5934 vs 基率 0.4449，
  macro Spearman 0.1885；5/8 域为正（dog/skiing/surfing），gymnastics 失败（AP 0.14），
  skating 基率 0.93 退化。证据等级 TRANSFER_INDICATIVE，不足以单独出包，
  足以支持"T1 继续作为主开发候选"。

### 其他
- AVE-PM：.part 停在 53,687,091,200 B（09-29 19:18 起），fetch 进程在岗，按既定等待代理；
  watchdog 未动。
- 交付物已落 reports/v6_next10h/：model_params.json、lfm_zero_shot.csv、
  temporal_transfer.csv、exposure_ledger.jsonl、PROGRESS_LOG.md。

## 检查点 1 · T+1.7h（09-30 00:55 CST）

### E4（时序候选，MrHiSum 弱域 + confirm）— 已完成主体
- dev（MrHiSum dev 2,808）/ confirm（4,008 fresh reserve，一次性）：
  - T0 U-Net s1: dev 0.28110 / confirm 0.28805（基线，1,712,897 参数）
  - T1 全分辨率 TCN 三种子: dev 0.31545 / 0.31391 / 0.32695（均值 0.3188±0.0058），
    confirm 0.32196 / 0.32496 / 0.32890（均值 0.32527）；740,609 参数
  - T2 U-Net+rank λ=0.5 三种子: dev 0.29562 / 0.29935 / 0.30227，
    confirm s2 0.30248 / s3 0.30427（s1 confirm 评估进行中）；1,712,897 参数
- 预注册晋级门（confirm 配对 delta vs T0 + source 聚类 bootstrap 95% CI 下界>0）：
  T1 三种子全过（delta +0.0339/+0.0369/+0.0408，CI 下界 +0.0267/+0.0288/+0.0332，
  win_rate 0.551/0.563/0.572，gate_pass=true）。T2 待 s1 eval 后汇总（预计过门但弱于 T1）。
- 出包判断不变：弱域证据，目标域迁移证据未落地前不出包（预注册规则）。

### E3（原始 MP4→YT8M 兼容特征→冻结头）— selftest 通过
- 契约链全部打通：TF-CPU InceptionV3 GraphDef(classify_image_graph_def.pb)→pool_3
  2048→官方 PCA→1024→官方量化 uint8；VGGish(torchvggish 官方权重 288,567,937B，
  sha256 10086976…561f4b)→128→同族量化。selftest OK（rgb_q∈[0,255]，audio absmean 134.3）。
- 排障记录：官方默认零音频+300 帧上限已落码；npz PCA 需转上游同构 torch pth；
  0.96s 窗仅 94 mel 帧 <96，改喂 1.0s（等价官方补尾行为）；
  torch.hub file:// 陈旧截断缓存曾致假 EOF，已清。
- 正在跑：YouTube-Highlights 8 条（dog/gymnastics/skating/skiing/surfing 五域，
  match_label=1 公共标注）特征提取 → T1_s3 头逐秒打分 → 时间迁移指标
  （Spearman / AP / top20 精度 vs 基率）。证据等级将标 TRANSFER_INDICATIVE。

### E5（LFM2.5-VL-450M 零样本空间定位）— 负结果，已定判定
- smoke（4 帧×2 提示）0/8 可解析；公平变体 P3（纯数字直出）再测 0/4。
- 关键观察：回复语义正确（如钓鱼视频回 "person fishing"、舞蹈回 "dancers"）
  → 视觉通路正常；但坐标生成退化为 `<box>Man</box>`、`-1 1 1 1` 重复循环
  → 450M 级模型零样本不具备坐标定位输出能力。
- 按预注册 smoke 门（≥99% 可解析）FAIL → 不进 sweep，S1（模型引导裁剪）不可用，
  S0 居中裁剪仍是唯一可用空间臂。负结果入报告，不烧 60-90 分钟网格。
- 环境备注：transformers 5.1.0 隔离于 /data/aic/tools/hf51，fp16 峰值显存低、~0.2s/帧。

### E1-AUTO（教师伪标签 pilot，无人工标注）— 进行中
- PM400 pilot manifest（320 源，train 215 全带 clip_context，context_only_not_crop_gt=true
  仅作弱时间通道）。已抽帧：32 条（全部 train、全带 context），640 帧，1fps 均匀 ≤20 帧/条。
- 教师推理（Qwen3-VL-32B，910A 4 卡 eager fp16 B=4，prompt sha 918f0730… 断言通过）：
  首次启动因 7 种帧分辨率各自触发 NPU 算子编译而停滞，已止损改为统一 720×1280
  （归一化点坐标不受均匀缩放影响），重启后进入编译期，预计 ~60 分钟完成 640 查询。
- 产出计划：teacher_replies.jsonl → QC（parse 率/轨迹平滑度/uncertain 率，
  仅筛选用）→ pseudo_labels.jsonl（3:1/16:9/1:3 合法窗 + point>center>anti 偏好
  + keep_weight 时间权重 + event_context 弱通道）→ qc_report.json 吞吐/保留率。

### 资源与其他
- V100：GPU7 跑完 T2 训练+confirm eval；GPU4 LFM smoke 已结束；GPU0/3 上的外部进程未触碰。
- 910A：NPU 2/3/4/5 教师推理；aic-batch 容器内补装 opencv-python-headless 5.0.0（容器生命周期内有效，已记入复现依赖）。
- AVE-PM 下载与 PHD² 哨兵按既定状态维持，未动。
- 无平台新回分；正式最佳保持 INTERP 48.67，不做任何"代理指标=官方分"表述。
