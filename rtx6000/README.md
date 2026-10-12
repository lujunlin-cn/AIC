# R15 RTX PRO 6000 RL 迁移包

910A → RTX PRO 6000（Blackwell sm_120）的 GRPO/OraRL 栈整体迁移。单张卡、
共卡双进程（rollout 与 trainer 常驻同卡，经文件协议严格同步交替——即
「先 rollout 后 train」的同步共卡形态；此形态没有异步收益，买的是省钱与
架构简单，见与 910A 版差异一节）。

## 文件清单

| 文件 | 运行环境 | 作用 |
| --- | --- | --- |
| `r15_common.py` | 两者 | 910A 公共函数逐字移植（prompt/预处理/loss/advantage），NPU 无关 |
| `r15_grpo_reward.py` | 两者 | reward v4 逐字复制（W1 项），零改动 |
| `r15_rollout_vllm.py` | vLLM env | rollout：HF generate → vLLM，`SamplingParams(n=3)` 组内共享视频 prefill |
| `r15_train_cuda.py` | venv_train | trainer：bf16 + sdpa + fla，输出 PEFT 目录 + .pt 双格式 |
| `r15_mine_vllm.py` | vLLM env | 难例挖掘（r14_rl_mining 的 vLLM 版，输出 schema 不变） |
| `r15_parity.py` | 两者 | Day-1 四道门：GDN kernel / 数值链 / vLLM 冒烟 / LoRA 活性 |
| `r15_safe_chunk.py` | venv_train | exp-then-mask GDN chunk（fla 缺席时的保底，CUDA 上无害） |
| `r15_merge_adapter.py` | venv_train | LoRA 合并（--lora-mode=reload 的 fallback 用） |
| `setup_rtx6000.sh` | 容器 | 一键环境：torch 校验、双解释器、模型下载 |
| `r15_pack_from_910a.sh` | **910A** | 打包模型/数据/LoRA，解包后路径同构 `/data/aic` |
| `r15_smoke.sh` | 容器 | 四道门 + 2-step 共卡端到端 |

## 租机规格

- GPU：1× RTX PRO 6000（96GB）或 6000D（84GB，fp16/bf16 主路径慢 ~15%，够用）
- CPU ≥16 核（视频解码是暗瓶颈）、RAM ≥128GB
- 磁盘：AutoDL 数据盘 1.2T 已租定（wave1 搬迁 ~235G + RL 运行账目 ~130G ≈
  400G，富余充足；wave2/PHD2 已取消不再占用）
- 宿主机驱动 ≥ r570（Blackwell 硬要求）
- 容器镜像：`vllm/vllm-openai` 2025-03 之后的 tag（自带 torch≥2.7+cu128
  的 sm_120 kernel；旧 tag 无 sm_120 会直接报 not compatible）

## 五步启动

```bash
# [910A 上] 1) 打包（LoRA 目录用户指定，默认 ckpt_060）
LORA_DIRS="/data/aic/experiments_910a/LFM_V13/q35_sft_out/ckpt_060" \
  bash rtx6000/r15_pack_from_910a.sh
# 2) 传输 + 解包（RTX 节点，必须解到 / 使路径同构）
tar xzf rtx6000_bundle_<date>.tar.gz -C /
cd /data && md5sum -c rtx6000_bundle_<date>.md5 | grep -v ': OK' || true

# [RTX 容器内] 3) 环境
bash rtx6000/setup_rtx6000.sh          # 记下打印的 ROLL_PY / TRAIN_PY

# 4) 冒烟（四道门 + 2-step 端到端；LoRA 活性门不过会自动切 reload 模式）
export ROLL_PY=<上一步打印> TRAIN_PY=/opt/venv_train/bin/python
export ADAPTER=$L/q35_sft_v2_out/ckpt_030   # 2026-10-12 定案
bash rtx6000/r15_smoke.sh

# 5) 正式跑（共卡，两进程都要挂着；step 节奏由 trainer 侧决定）
$ROLL_PY rtx6000/r15_rollout_vllm.py --pool $L/grpo_pool_v2.jsonl \
  --adapter-init $ADAPTER --out $L/rtx_grpo_out --steps 200 \
  --gpu-mem-util 0.45 &
$TRAIN_PY rtx6000/r15_train_cuda.py --pool $L/grpo_pool_v2.jsonl \
  --adapter-init $ADAPTER --out $L/rtx_grpo_out --steps 200 &
# 难例挖掘（正式 RL 前置，独立跑，可用满卡）：
$ROLL_PY rtx6000/r15_mine_vllm.py --pool $L/grpo_pool_v2.jsonl \
  --adapter-init $ADAPTER --out $L/grpo_mining_rtx
```

## 数据盘账目（数据盘 200GB 的依据；含搬迁数据时按 1TB 规划）

| 项 | 大小 | 依据 |
| --- | --- | --- |
| **910A 搬迁 wave1**（LFM 全系 + 旧 experiments + QVH_V10 + datasets/提交包 + 4B/TimeLens2-8B/Qwen3.8-27B + repo tar） | ~235G | 19 路并行实测在传；**wave2（PHD2 399G）2026-10-12 用户取消**，外部原始视频不再搬迁 |

| 项 | 大小 | 依据 |
| --- | --- | --- |
| 基座 Qwen3.5-4B（bf16） | ~9GB | 4B × 2B |
| LoRA 产物（每步 PEFT 目录 + .pt 双格式） | ~30GB | r=16 ≈ 65MB/份 × 2 × 200 步；磁盘紧张可只留最近几步，回滚窗口 3 步足够 |
| 帧图（pool 823 段 + 全 FRAG + 4K pool 预留） | ~30GB | 823 段 ~2GB；5886 段全量 ~14GB；4K 池帧更大 |
| merged_step（仅 reload 降级模式） | ~20GB | ~8GB/步，自动只留最近 2 步 |
| 传输峰值（tar 与解包共存，解完可删） | ~25GB | 基座 + 帧 |
| venv / pip 缓存（若与数据同盘） | ~10GB | |
| traj / holdout / 杂项 | ~5GB | |
| **合计** | **~130GB** | 主路径（lora 模式）稳态 ~85GB，传输期峰值 ~110GB |

## LoRA init（已定案，2026-10-12 用户指定）

`--adapter-init` = **`/data/aic/experiments_910a/LFM_V13/q35_sft_v2_out/ckpt_030`**
（SFT v2 重做中唯一真信号 ckpt：全 parse 但模板化，ckpt_030 pred-pred
pearson 0.539 为唯一非模板信号）。PEFT 目录格式，r=16 勿改。该目录已随
wave1 优先落位 AutoDL。

## 与 910A 版的差异（全部有意为之）

1. **rollout 引擎**：HF generate(batch 6) → vLLM（组内 n=3 共享 prefill、
   continuous batching、prefix cache）。采样参数不变（T=1.2 / top_p 0.85 /
   256 tokens / eos 截断），reward v4 与 OraRL advantage 链逐字未动。
   跨引擎贪心输出不要求一致（视觉 token 化路径不同，RL 采样训练不受影响）。
2. **训练精度**：fp16+eager → bf16+sdpa；GDN 走 fla triton kernel（fla
   缺席时 safe torch chunk 保底，`--safe-chunk` 可强制 A/B kernel）。
   `logits_to_keep=280` 手动 gather 保留（loss 数学逐字未动）。
3. **拓扑**：4 rollout worker + 双卡 trainer → 单卡单 rollout 进程 +
   单 trainer 进程，`traj_step{s}_0.jsonl`、adapter 双格式（PEFT 目录 +
   .pt），tmp+rename 原子协议不变。
4. **NPU 补丁全删**：safe_chunk/multinomial CPU 路由/warper where-fix/
   jit_compile=False/容器锁规避。CUDA 上标准路径即正确路径。
5. **grad checkpointing**：CUDA 上可用（`--grad-ckpt`，use_reentrant=
   False），默认关 = 与 910A 数值行为同构；910A 的线程泄漏死锁是
   torch_npu 特有，此处无此约束。

## Day-1 风险表（按处置顺序）

| 风险 | 门 | 处置 |
| --- | --- | --- |
| vLLM 不认 Qwen3.5-4B（dense GDN hybrid） | `--gate vllm` 报 no arch | 换更新 `vllm/vllm-openai` tag → SGLang → HF generate 保底（提速封顶 2–3×） |
| vLLM LoRA 注入不了 GDN 层 | `--gate lora` 无差异 | `--lora-mode=reload`（每步 merge+重建引擎，+1–2 min/step，smoke 会自动切换） |
| fla kernel 与 torch chunk 数值偏差 | `--gate gdn` > 阈值 | `--safe-chunk` 强制 910A 同构路径（慢但已验证） |
| 共卡显存互相挤爆 | TSTEP peak_gpu 打点 | rollout `--gpu-mem-util` 降 0.05 一档；96GB 卡可升 |
| 视频解码 CPU 瓶颈 | RSTEP wall 偏高 | 看 `top`，帧预解码缓存，或换更多核机型 |

## 第 2 张卡的判据（提前写死，防止拍脑袋）

跑满 10 个 step 后取 RSTEP/TSTEP wall 均值：rollout 占比 <40% → 单卡到底；
≥40% 且要冲 deadline → 加第 2 张，用途优先级：两条独立消融线（各自共卡，
实验吞吐 ×2）> 单实验异步拆分（×~1.7，且引入 staleness）。84/96GB 单卡
显存（双进程合计 ~40–50GB）不会成为加卡理由。

## 数据流（与 910A 完全同构）

```
/data/aic/pretrained/Qwen3.5-4B                     基座（打包项 1）
LFM_V13/grpo_pool_v2.jsonl                          正式 pool（823 段口径见 mining）
LFM_V13/frag_labels_posfirst.jsonl                  FRAG 全量标签（重建 pool 用）
LFM_V13/q35_sft_out/<ckpt>                          LoRA init（用户指定）
PHD2_FRAG_V1/frames/<vid>/*.jpg                     帧图（pool 过滤后打包）
每 step：rollout 写 traj_step{s}_0.jsonl → trainer 消费 →
         adapter_step{s}.pt + adapter_step{s}/ → rollout 热切换
```
