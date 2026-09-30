# V6-NEXT10H 下一步最小实验清单（2026-09-30 冻结版）

## 结论先行
- 本轮不出新候选包：正式最佳保持 INTERP 48.67。T1 是唯一预注册干净候选
  （confirm 配对门 3/3 过），但只有弱域证据 + 迁移 indicative-mixed，按 prereg 不出包。
- E1-AUTO 管线已验证可扩展；E5 450M 零样本定位判负。

## 最小实验（按优先级）
1. **T1 → 学生管线的正规接入评估**（1 个工作日）
   T1(TCN 740,609 参数) 是 MrHiSum 弱域头。要在提交里用，必须在学生原生特征域
   重训同骨架并过官方 E2E，而不是把 MrHiSum 权重换进现有 bundle。
   最小版本：学生原生特征上 train TemporalTCN×3 种子 → confirm 门 → 若过，冻结做
   官方提交（单一因素=时序头骨架）。
2. **E1-AUTO 扩展到 108 条**（教师 ~3.2h @0.19qps + QC 0.5h）
   pilot 已达用户扩展标准（parse 100%、保留 90.6%）。扩展后重跑 A/B 对照
   （3 种子，报告置信区间而非点估计；本轮 +0.0069 在种子噪声内）。
3. **YTH 迁移 32→96 条**（提取+打分 ~1.5h）
   32 条 macro AP 0.593 vs 基率 0.445；扩到 ~100 条可给迁移证据一个可引用的
   置信区间，并按域分层定位失败模式（gymnastics 类）。
4. **空间臂改造**：450M 定位不可用 → 空间蒸馏目标改为 Qwen32B 教师点
   （E1-AUTO 的 point→合法窗即监督信号），学生侧用现有检测/回归头吸收。

## 明确不做
- 不扩大已暴露 dev / 不改旧阈值让 H3 翻正。
- 不把 MrHiSum Spearman、YTH AP、教师一致率当作官方分数。
- 测试数据仍只用于冻结算法自动推理与格式核验。

## 环境备忘
- vggish-10086976.pth（288,567,937B，sha256 10086976…561f4b）已就位；
  PCA 经 vggish_pca_torch/vggish_pca_params-97013be8.pth（npz 转制）。
  官方链 selftest OK；YTH/PM400 特征已验证。
- V100 临时 mihomo：/tmp/mihomo -d /home/supie/.config/mihomo_tmp，mixed 7893，
  controller 9093，config=用户订阅（2026-09-30 版，备份 /tmp/mihomo_old_config_0930.yaml）。
  AVE-PM fetch 从 53,687,091,200B 断点续传中（per-file quota 冷却，自动重试）。
- aic-batch 容器补装 opencv-python-headless 5.0.0（容器重建后需重装）。
