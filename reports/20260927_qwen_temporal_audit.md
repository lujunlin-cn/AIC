# Qwen 时间选择审计（P2）

状态：阻塞，未运行。Qwen 没有进入蒸馏，E3 不开。

## 审计设计（已就绪）

- 脚本：`scripts/qwen_qvh_dev_audit.py`。
- 数据：native QVH dev 24 个视频，只用 dev 标签，不接触官方测试内容。
- 教师设置与官方 SUB_Q 相同：1 fps 锚点，448 px，block 32，vLLM greedy，TP=4。
- 对照：全选，以及帧数相同的均匀抽帧。后者用来检验收益是否只来自输出量。
- 指标与 `reports/20260927_all_select_diagnosis.md` 相同，是 2 s 片段集合 F1（hi/rel 两种代理），另报重要性提升倍数。
- 准入条件：Qwen 相对全选、相对同帧数均匀抽帧的 hi 口径配对 CI 下界都 > 0，才允许进入蒸馏（`admit_distillation`）。

## 阻塞原因

- Qwen3-VL-32B FP16 权重 66.7 GB，需要 4 张 V100 32 GB 做 TP=4。
- 2026-09-28 01:20 起，GPU 4–7 被一个 root 用户的 vLLM 服务（qwen3.8-27b，非本项目）各占约 29.7 GB。
- 当前空闲的授权卡只有 GPU 2，GPU 1 有 3.5 GB 的非本项目占用。GPU 0/3 不可用。按约束不结束他人任务。
- 旧的 TVSum dev 审计（`qwen_dev_tvsum`）用 HF 后端 OOM 失败，没有可用输出。

GPU 4–7 空出后可直接运行：

```bash
PYTHONNOUSERSITE=1 PYTHONPATH=/data/aic/envs/vllm_av_overlay CUDA_VISIBLE_DEVICES=4,5,6,7 VLLM_WORKER_MULTIPROC_METHOD=spawn \
timeout --signal=TERM --kill-after=3m 710m /opt/miniconda3/envs/vllm/bin/python -m scripts.qwen_qvh_dev_audit \
  --records /data/aic/datasets/QVH_NATIVE_CANDIDATES/records.jsonl \
  --model /data/aic/pretrained/qwen3_vl_32b_instruct \
  --output /data/aic/experiments/QWEN_QVH_DEV_AUDIT_V1
```

## 已有平台证据

SUB_Q（Qwen 时间选择 + B0 同款裁剪）得分 31.20，低于全选 V0 的 34.42。在当前裁剪下，Qwen 的时间选择在平台上没有胜过全选。

## Q2 内部教师探针包

Q2 = SUB_Q 的帧集合 + 与 S2 逐帧相同的裁剪。已核验 174/174 个视频的帧集合等于 SUB_Q，裁剪等于 S2。

- 模型规模按真实部署记录：Qwen3-VL-32B 33,357,390,064 参数、66,714,912,872 B，加上检测器和 YuNet，合计 33,376,853,922 参数、66,792,990,268 B。
- 按参数口径超过 9B，按字节口径超过 9216 MB，两种口径都超限。
- 不再继承 350.5 这个占位值。JSONL 不写 model_size_mb 字段（初赛格式允许缺省，但平台是否接受缺省未核实）。
- 身份：内部教师探针，不作为合规终版，不在 S1/S2 之前上传。
- ZIP SHA-256：`9c9460fb760c88ff626e07c71d6f3b375149f41bdd639fde44ec53b96a12d235`，55,214 条预测，10 个空视频（与 SUB_Q 相同）。

## 未决问题

- 初赛是否允许测试推理时加载超过 9B 的教师：规则只在复赛/半决赛写明超限无效，初赛条款没有明确写，需要向组委会确认。
- 初赛平台能否接受不含 model_size_mb 的 JSONL，未实测。
