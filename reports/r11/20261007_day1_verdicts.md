# R11 Day-1 执行判定（2026-10-07）

执行地：kmsp05（910B 双卡）。R10 收窄后 CPU/NPU 全空；本地容器→910A 通道经
`ssh -p 44000 root@221.213.81.199` 打通（bundle 传输绕过其无 GitHub 出网）。
资产：r8-fullinfo@f0b72e8+。910A 侧 118 个 untracked 脚本备份于 /root/aic_untracked_backup。

## 判定一：S1 分数场 Viterbi 审计 —— WEAK_POSITIVE_BELOW_GATE（灰区）

数据：`reports/r11/s1_audit.json`（B3_s0 分数表 31.1MB，四池 8,972 帧级行；
λ 网格 dev(live_dev,124 源) 选 λ=0.25，eval 单次判定）。

| eval 池 | delta IoU | CI95 | n_src |
|---|---|---|---|
| live_confirmation（locked-out） | **+0.00243** | [+0.00043, +0.00450] | 227 |
| rv_dev | +0.00391 | [−0.00212, +0.00974] | 20 |
| rv_diag | +0.00194 | [−0.00174, +0.00550] | 70 |

- 晋级门（双池 ≥+0.005 且 CI>0）：**未过**（confirmation 量级不足、rv_dev 跨 0）。
- 关闭线（< +0.002）：未触发。换算 ≈ +0.09 官方分。
- ⚠️ 置换对照警示：λ=0.25 源内打乱帧序后 +0.00355 ≥ 真实 +0.00228——增益不依赖
  帧序的部分占大头，更像 IoU 估计的平滑正则；大 λ（依赖真实时序）eval 更差。
- 结论：解码层有微小真实正收益（零参数，可作部署端免费后处理记录）；
  「跨帧裁剪轨迹」机制在 B3 分数场上无撬动级证据。与 P1A（观测层持平）合并，
  GPT-6-PRO 第四层的 trajectory 支路降权，**空间线主杠杆压到 S2（检测器候选源）**。

## 判定二：S2c 口播占比 —— 阈值口径反转，Light-ASD 臂恢复

r10_audio 全池 AST-527（1,963 源全覆盖）。⚠️ 口径修正：npz 存 logits，
按概率域（sigmoid）判：

| 档位 | 源数 | 占比 |
|---|---|---|
| speech 主导（mean prob>0.3） | 819 | **41.7%** |
| 过渡（0.1–0.3） | 429 | 21.9% |
| BGM 主导（≤0.1） | 715 | 36.4% |

mean speech prob 0.27 与 R10 3.1 的 σ=0.30 互证。首次 logits 域误读（11.1%）
已修正。**判定：≥30% 门槛过 → S2c Light-ASD（face-track×audio 对位）按预注册
继续**；S2a 检测器 smoke 同为 S2 前置。

## 判定三：V0 dense-view 前提 —— NOT_RUN（合法记录）

三个候选 dense 观察缓存路径均不存在 → 前提 UNTESTED，dense-view KD 臂保持关门；
若要开测需先建 16-native-frame 特征缓存（新 NPU 成本，另议）。

## 判定四：O1/V1 E3 缓存不可复用且 7B 角色判别塌缩 —— V1-7B 挂起

- V4_E3_EVENT 仅 5 个 YTH vid，PHD2 3,917 frag 零覆盖；E3 原链是 vLLM+YTH 锚点（本
  host 不可用）。
- 新链 `scripts/r11_v1_event_roles.py`（Qwen2.5-VL-7B 生成式、PHD2 媒体+R10 槽时间轴）
  修通三个管线 bug 后（chat template 正常、段号键照抄占位符 K 已修、CONTEXT 图 OOM 降
  级已加），行为仍塌缩：**输出单调角色（全 SETUP / 全 RESULT）**——孤立 16s 窗+单帧代
  表 2s 段的设计下 7B 无判别力。按预设止损 **V1-7B 挂起**；负面记录入账。
- V1 后续形态（需用户裁定，不自行点火）：32B teacher+全源上下文扫描（kmsp05 无 32B
  transformers 批量链，需新写；32B×984 源为十小时级 NPU）；或放弃角色结构改用运动/
  帧差类廉价 scout。第二层其余形态（V2 counterfactual、R8 VLM_SFT）不受本判定约束。

## 判定六（追加）：V1-32B 在 kmsp05 当前栈不可用 —— vllm-ascend 线关闭 + transformers generate hang

用户指示尝试 vllm-ascend 提速后，实测与文档证据如下（2026-10-07 晚）：

1. **vllm-ascend 官方从未支持 910A**：v0.7.3 与 v0.11.0 安装文档硬件表均为「Atlas 800
   A2 系列（910B 类）」，CANN 配平 8.1.RC1 / 8.3.RC2，kernels 包 `_910b`；910A 不在
   任何版本支持表。与 10-01 实测死路（910A 缺 FusedInferAttentionScore）一致。
2. **0.18.0 引擎初始化卡死（实测）**：venv ABI 修复后（torch_npu 2.8.0.post5 配平
   torch 2.8；vllm-ascend 0.18 实配 vllm 0.13.0+torch 2.9.0——pypi 元数据三向互斥，
   --no-deps 手工配平），import 与平台插件激活通过，但 V1/V0 双引擎在权重加载前
   hang（HBM 零占用，四轮探针）。
3. **transformers 32B generate 在 10-04 重建栈上 hang（实测）**：Qwen3-VL-32B 加载
   71–78s 正常（6 卡 device_map），但 generate 全形态卡死——多图长 prompt、单图
   contact-sheet、batch=1 三试均零产物，HBM 静止 5235MB。关键背景：**历史 32B 推理
   （0.70 qps，9-28/29）跑在 10-04 栈重建之前；重建后 32B generate 从未被验证过**，
   今日判定其已退化。`[Check][offset] storage_offset result is untrustworthy` 警告
   反复出现，疑似 CANN 9.0.0 算子行为变化所致（未深究，属驱动/CANN 层）。
4. 后果与选项（用户决策）：① 910A 装 CANN 8.x 双栈专供 32B 推理（动驱动层，风险
   高）；② 有 910B 机器则 vllm-ascend 0.11/0.18 按官方文档直接可用；③ V1 事件结构
   线挂起等环境。**S2 空间主杠杆不受影响**（S2a 走 CPU，见判定五）。

vllm18 venv 最终状态：torch 2.7.1+torch_npu 2.7.1+transformers 4.57.1（0.11 线主体，
vllm 0.10.0 源码编译被终止）；0.13/0.18 组合的配平知识记录在案。

## 判定五：S2a 检测器 —— NPU 算子坑实锤后转 CPU 32 分片，覆盖率门过

- RT-DETR r50vd 在 910B→实为 **910A（用户纠正；npu-smi 显示 910B 不可信，lspci
  19e5:d801=910A）** 上加载成功（hf-mirror 5.9s），但 NPU 前向崩于 **AsStrided 算子**
  （error_detail 落盘）——「910A 算子覆盖待实测」的 RT-DETR 实锤，不再赌 NPU 检测。
- **转 CPU 全并行**：RT-DETR r50 CPU 3.5 s/帧（OMP 6），3,917 frag×32 分片 ≈1h 墙钟，
  755GB host 内存充裕。覆盖率门（conf≥0.3 person 帧 ≥0.6）**已过（gate_ge_060=true）**。
  全量提取 32 分片后台运行中（r11_s2a/p0..p31）。
- 教训（三项，均已修入脚本/流程）：
  1. `HF_ENDPOINT` 的 setdefault 必须在 `import transformers` 之前——huggingface_hub
     在 import 时固化 endpoint 常量，之后设置静默失效，from_pretrained 在
     huggingface.co 上无超时挂死（faulthandler 栈定位）；
  2. NPU 作业必须 `source /data/aic/tools/ascend_teacher.env`；**当前栈严禁
     jit_compile=False**（2026-10-04 rebuild note：kernel-parse bug；R9 旧脚本里的
     该行不得再抄）；
  3. RT-DETR logits 无 background 列，标签偏移需按类宽判别（首版 coverage=0 的根因）。

## 资源与过程

- 并行形态：CPU 32 分片（S2a 全量）+ NPU 空闲（V1 挂起后无 NPU 作业；AsStrided 坑未
  解前不在 NPU 上跑检测）。
- 910A git：GitHub 不可达（出网），bundle 直传；ext_data 本地改动 stash 于 910A
  （pre-r11 ext_data WIP），118 个 untracked 脚本备份 /root/aic_untracked_backup。
