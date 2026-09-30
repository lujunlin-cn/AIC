# 复赛评测集就绪清单（910A 全链，2026-09-30 验证）

V7 学生空间链在 910A（aic-batch 容器）已完成全组件验证。复赛集到达后按序执行：

## 前提
- 服务器 `ssh -p 44000 root@221.213.81.199`，全部命令在 `docker exec aic-batch bash -c "cd /root/AIC; set -a; . /data/aic/tools/ascend_teacher.env; set +a; ..."` 内跑。
- 头 checkpoint：`/data/aic/experiments_910a/LFM_V7/s_head_s1/head_s.pt`（sha256 81dcdbac…53a011，1,511,425 参数）。

## 步骤（假设复赛 intake 与初赛同 schema：`video_path/width/height/frame_count/fps/targetRatioWH`）

1. **解码自检** `scripts/v7_semifinal_avcheck.py`
   - 改脚本内 index 路径为复赛版；lfm_venv PyAV；输出 ok/bad。初赛集实测 174/174（15.6s）。
2. **观测缓存（CPU）** `scripts/v7_semifinal_build_obs_cache.py --index <复赛index> --face /data/aic/pretrained/yunet/face_detection_yunet_2023mar.onnx --output <cache_dir> --part i --parts 4`（i=0..3 并行）
   - YuNet+显著性路径，跳过 FasterRCNN（QWEN_POINT 链不读）；2 视频实测 52s，174 视频量级约 20-25 分钟。
3. **B0 保底包** `scripts/v7_semifinal_b0_from_cache.py --index <复赛index> --cache <cache_dir> --output <b0_dir>`
   - 直接从缓存 b0 列生成，缓存/预测构造性一致（max_window_release 的等式检查要求）；实测 VALID True True。
4. **关键帧提取** `scripts/v7_semifinal_extract_kfs.py --index <复赛index> --cache <cache_dir> --output <kf_dir>`
   - 1s 网格 + B0 shot starts；PNG 长边 640（与官方集一致，实测 360×640 竖屏）；同时写 points/{vid}.json 骨架。
5. **学生头推理** `scripts/v7_s_official_points.py --head <head.pt> --index <复赛index> --keyframe-src <kf_dir> --cards <2..5> --shard i --nshards 4 --output-dir <pts_dir>`
   - 本轮官方集实测 0.16 s/帧（3,106 帧 3 shard ~3 分钟）。
6. **打包** 复制 `configs/LFM_V7_S_FINAL.json`，仅改 `index_sha256`（新集 sha256_file）、`b0_predictions`（步骤3）、`obs_cache`（步骤2）、`qwen_points`（步骤5 的 points 目录）→ `scripts/max_window_release.py --frozen <新config> --index <复赛index> --output <submissions/...>`
   - 自动含：B0 等式检查、宽度/掩码 diff、独立检查器、独立解压回读、SHA。
7. 登记 registry.jsonl + `pios result add`。

## 已知事项
- **解码栈差异**：910A lfm_venv PyAV(17.x) 与初赛 V100 cv env PyAV(15.1) 有像素级色彩转换差异；实测同视频 `reset` 完全一致、`raw/b0` 有小偏（状态机级联）。复赛集全新建缓存，链内完全自洽，不影响提交；但不要把复赛 B0 与初赛 B0 做逐字节对比。
- 提帧长边 640；`points` json 的 status 必须写 `'ok'`（打包器列表精确计数）。
- 复赛 index 若字段名不同，需先改 `v7_semifinal_avcheck.py`（video_path）与 `_record_path` 的适配。
- 更大帧量时先按 26s/视频估算 obs cache 墙钟，4 parts 并行（CPU 48 核内）。
