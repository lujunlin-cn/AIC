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

## 判定六修订（2026-10-07 晚，用户纠正后）

**撤回两处错误结论**：
1. 「CANN 9.0.0 裸缺 910 算子」——错。CANN 9.0.0 改了算子包组织：8.x 的
   `Ascend-cann-kernels-910_<ver>` 在 9.0.0 改名 `Ascend-cann-910-ops_<ver>`（含算子
   库/TBE/aclnn/kernel 二进制），oepkgs 镜像 kernels 名止于 8.x 不能推出 9.0.0 缺包。
2. 「CANN 8.3.RC1+910 kernels 配 vllm-ascend 0.11.0」——作废。0.11.0 官方矩阵为
   CANN==8.3.RC2、torch==2.7.1、torch-npu==2.7.1.post1；且 vllm-ascend FAQ（v0.11.0
   至 v0.23.0rc1）明文将 Ascend 910 / 910 Pro B（Ascend-cann-kernels-910）列为不支持。

**已执行（按用户诊断树）**：
- 第 1 步：本机原无 910-ops（opp/Ascend910 仅 3 个 aicpu 包，无 AI Core kernel）。
- 第 2 步：补装 `Ascend-cann-910-ops_9.0.0_linux-aarch64.run`（官方 OBS 直链，
  --install --quiet）——success（legacy/transformer/math/ras 四子包，cann-9.0.0 内
  share/info/ops_* 注册 + lib64 出现 910 系 HCCL 算子对象）。
- 复验（含 env 顺序修正，ASCEND_OPP_PATH 正确指向 cann-9.0.0/opp）：
  **32B generate 仍 hang**（transformers contact-sheet，batch 2/1、HBM 静止 5236MB、
  零产物）。ops 缺失不是 hang 根因（或非充分修复）。
- 第 4/5 步（vllm-ascend 非官方 910A 路径、aclnn fallback、AsStrided 是否落入 910A
  未适配执行路径）为深度排障，未完成——需逐算子探针/strace/torch_npu 源码级分析。

当前状态：910-ops 9.0.0 已补全（对其他工作负载无害且可能有益）；32B generate hang
根因未定，进入深度排障范畴；vllm-ascend 官方不支持 910 系的结论维持（FAQ 明文）。

## 判定七（深度排障，进行中）：TEFUSION 编译死锁定位 + 栈状态偏移警示

**确定级发现（py-spy Python+native 双栈）**：32B generate 挂点=
`generate → _sample → 首次 forward`，native 栈停在 `Tensor.to()` 跨设备拷贝的
`eventfd_read`（等一个永不完成的 stream event）。ASCEND DEBUG 日志（555MB）尾部
为 **TEFUSION 死循环轮询 `GetFinishedCompilationTask`（graph 编译任务永返 size=0）**
——即某算子触发了动态图编译，编译任务投给 TBE 后台 worker 后 worker 不消费，
主线程死等。gitee ascend/pytorch #IC6DXR 同款问题的已知两类成因：kernel 包缺对应
算子二进制 / 环境变量配置问题。

**对照实验**：裸 P2P copy 与 MATMUL 探针正常 → 基础通道健康，挂点在「特定算子的
JIT 编译」层。7B 同日对照：下午可跑形态（8 图）在晚间重测也卡死——与 910-ops
装卸循环的时间线吻合（见下）。

**栈状态偏移警示（重要）**：910-ops 的 uninstall 脚本连带删除了 opp/built-in
（5.5G→45M，主链算子库），已用同一包重装恢复（5.5G，cann-opbase 子包新增注册）；
但 910-ops 装卸循环后，**下午可跑的 7B 原版当前不可跑**，清 TBE 缓存无效果——
当前栈状态与 10-04 重建基线存在未表征的偏差（opbase/lib64 文件集可能与重建时
不同）。**未再继续盲试**。

**建议（需用户/熟悉该机人员决策）**：
- 选项 A：按 10-04 note 配方做一次干净重建（toolkit runfile + 910-ops + 主链
  venv 重验），消除装卸循环残留——半小时级，风险=重建期间主链不可用；
- 选项 B：保持现状，32B 排障转「离线分析」（torch_npu 源码/GitHub issue/
  华为工单），不再动栈；
- 无论 A/B：7B 下午可跑状态与 32B hang 的对比是根因的最强线索（同日装改前能跑、
  装改后不能），重建后可用 7B 原版作金标准探针快速验证栈健康。

## 判定八（选项 A 干净重建后）：JIT 编译子系统状态损坏实锤 —— 主链多模态前向不可用，升级上报



用户裁定选项 A 后执行与探针矩阵（2026-10-07 深夜）：

**重建执行（四步链式，全过零事故）**：卸 910-ops → 卸 toolkit → 装
`Ascend-cann-toolkit_9.0.0`（OBS 直链 1.22GB，--install --force --quiet）→ 装
`Ascend-cann-910-ops_9.0.0`（2.36GB）。重建后 `opp/built-in` 5.5G、
set_env.sh 恢复、driver 24.1.0 未动（装卸不触 driver）、NPU 温度 30°C。
顺手修复：python3.9 补装 psutil 7.2.2（10-04 配方坑 1 的 TBE 双侧依赖缺一项；
修复后无行为变化，排除单因）。

**重建后探针矩阵**：

| 探针 | 基线（装卸循环前） | 重建后 | 判定 |
|---|---|---|---|
| 纯文本前向 L=512（7B） | ~0.1s | 0.1s/iter | ✓ 健康 |
| 7B generate（金标准，同脚本同命令） | **17.5s/源**（r11_e3_roles_pilot 6 源 87s，mtime 铁证） | 探针×3，10–15min 零产物 | ✗ 慢 40 倍+ |
| 主链多模态前向（1 图+短文） | 1.4s/样本（0.70 qps） | **>600s 未完成一次** | ✗ 慢 400 倍+ |
| 32B generate | 继承判定六/七 | —（更重，不测） | ✗ 死锁态 |

**根因收窄（排除法）**：
- ✗ toolkit/ops 文件层：重建换新仍病；
- ✗ 驱动版本：24.1.0 全程未动；
- ✗ 硬件：30°C、HBM/DDR 读数正常、无降频迹象；
- ✗ python3.9 TBE 依赖：psutil 补齐无变化；
- ✓ **锁定 TBE/JIT 编译子系统持久状态损坏**：TBE 缓存残留仅 3.7MB（几乎空），
  每个 shape 触发 JIT 重编译且**编译不收敛/不落盘**（10 分钟连一次多模态前向
  编不完；纯文本算子集小已被会话内反复触发编完故 0.1s）。日志裸点输出 =
  编译进度残留（脚本本体无 print('.')）。
- 分水岭 = 19:46 的 910-ops 装卸循环（卸载连带删 built-in 5.5G→45M 事故已知，
  判定七）。**同版本干净重建不恢复**——损坏在缓存数据库/注册标记/共享态层，
  不在 runfile 覆盖的文件层。

**后果**：R8–R10 主链（多模态特征提取/训练）当前在 NPU 上不可用；
V1-32B/7B generate 线继续挂起。**CPU 通道与不依赖 NPU 的 R11 工作（S2 读出、
Light-ASD、S2a 已完成的 3781/3917 产物的消费）不受影响。**

**剩余选项（需用户/平台裁定）**：
1. **整机重启或 NPU 驱动级复位**——JIT 子系统 host 侧状态（daemon/共享内存/
   缓存库）最彻底清零；本机为租用机，重启需平台侧操作，`npu-smi` 单卡复位
   可试但风险自担；
2. 华为工单 / 离线分析（原选项 B 路径）；
3. 接受现状：R11 推进全部走 CPU/已完成产物，NPU 线挂起等窗口。

## 判定九（深夜收官）：根因闭环 —— python fork×编译线程锁死（官方 FAQ 同款），32B/7B 全线复活

用户三步授权全部执行后收束（2026-10-07 23 时）。**判定八的「JIT 子系统持久
损坏」结论被推翻**——栈从未损坏，真凶是应用层并发形态。

**执行序列**：① 整机重启（up 1min 验证，NPU 健康）——复测仍冻结；② 清编译
缓存（kernel_cache 5.2M→8K）+ 预热跑——预热跑揭示 **NPU OOM**（7B 权重
15.5GB 后 HBM 仅剩 20MB）；③ 互联网检索命中官方 FAQ。

**根因链（三层）**：
1. **fork×编译线程锁死（官方 ptacase_0037 同款）**：python 3.8–3.11 在算子
   编译场景下 fork 多进程会锁死。落点 = transformers 5.18 **fast image
   processor 多图时起 forkserver 进程池** × TBE 编译线程持锁。决定性实验：
   单图（无进程池）+jit_compile=False → 健康；10 图同配置仍冻结；
   **10 图 + `use_fast=False`（慢速 processor，无子进程池）→ 全绿**
   （10img fwd 6.54s、generate50 10.0s）。
2. **「装卸循环」的真实角色**：910-ops 装卸清空算子编译缓存 → 每次首任务
   触发编译线程 → forkserver 锁死条件成立。**下午健康 = 缓存全热无编译
   线程**；晚间全部 hang = 冷缓存+编译线程常驻。CANN/ops 文件层全程清白。
3. **连锁伤害（「慢 400 倍」的真相）**：历次 kill 未清 forkserver 孤儿组
   → 僵尸进程霸占 HBM（实测卡 6 残留 16.5GB）→ 后续每个探针在 OOM 边缘
   挣扎（OOM-重试循环表现为「极慢+零产物」）→ 再留新僵尸的恶性循环。
   OOM 日志（`Tried to allocate 88MiB, 20MiB free`）为铁证。

**修复配方（生产化，已入脚本）**：
- 推理脚本：processor `use_fast=False`（必改）；
- 推理可 `torch.npu.set_compile_mode(jit_compile=False)`（10-04 note 的
  禁令限**训练**——InplaceNormal kernel-parse bug；推理无该算子，实测安全
  且必要：保持进程内无编译线程）；
- 进程清理必须杀全组（主进程+forkserver+worker），跑前 `npu-smi info -t
  proc-mem` 确认 HBM 干净。

**互联网对照**：hiascend 官方案例库 ptacase_0037「在模型运行时脚本突然
卡死」精确命中（py 3.8-3.11+非二进制编译+fork→子进程锁死）；官方解法
「spawn 代替 fork」落地为 use_fast=False，「二进制场景」落地为
jit_compile=False；TE_PARALLEL_COMPILER=1 无效（锁死源不在 TBE 并行编译
fork 数，在 processor forkserver）。

**复活验证**：
| 探针 | 结果 |
|---|---|
| 7B 金标准（同脚本同命令，基线 17.5s/源） | **ok=3 err=0 72s（24s/源）** |
| 32B contact-sheet（判定六/七的原始故障） | **loaded 135s + generate 40s/2 源，hang 解除** |
| 32B 解析修复后（seg-N 键归一化） | **ok=2 bad=0**，叙事弧（SETUP→MAIN→RESULT）质量良好 |

**V1-32B 全源扫描已启动**（1961 源，~11h 后台，卡 2–7，
/tmp/v1_32b_fullscan.log，产出 r11_e3_roles_32b/<src>.json）。

## 判定五：S2a 检测器 —— NPU 算子坑实锤后转 CPU 32 分片，覆盖率门过

- RT-DETR r50vd 在 910B→实为 **910A（用户纠正；npu-smi 显示 910B 不可信，lspci
  19e5:d801=910A）** 上加载成功（hf-mirror 5.9s），但 NPU 前向崩于 **AsStrided 算子**
  （error_detail 落盘）——「910A 算子覆盖待实测」的 RT-DETR 实锤，不再赌 NPU 检测。
- **转 CPU 全并行**：RT-DETR r50 CPU 3.5 s/帧（OMP 6），3,917 frag×32 分片 ≈1h 墙钟，
  755GB host 内存充裕。覆盖率门（conf≥0.3 person 帧 ≥0.6）**已过（gate_ge_060=true）**。
  全量提取 32 分片后台运行中（r11_s2a/p0..p31）。
  **【收口更新 10-07 晚】** PHD2 32 分片完成：3,917/3,917 全量、coverage 0.766
  [0.717–0.830]、0 错误。⚠️ 本条提取跑在 PHD2（keep-mask 协议池、无空间候选），
  readout 数据合同要求 B3 分数场同池 → 补 live 池提取后完成读出 a/b/c：
  **a 门 FAIL（+0.0000045 vs +0.02，低 3 数量级）、c 压线未过（+0.00454 <+0.005，
  CI>0）→ 检测器源收口关闭**。判定与机制见
  `reports/r11/20261007_s2a_readout_verdict.md`。
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

## 判定十（2026-10-08 凌晨验收）：V1-32B 全源扫描 4 小时验收 + O1 scout 门 PASS

背景：判定九破解 generate hang（fast processor forkserver × TBE 编译线程 fork 锁死，
栈从未损坏；推理配方 `use_fast=False` + 推理 `jit_compile=False`）后，32B V1 线复活，
双实例（--cards 0-3/shard 0 与 --cards 4-7/shard 1，batch 6）全源扫描 PHD2/R10 池。

### 扫描验收（判定：完成、全齐）

- 产物 `r11_e3_roles_32b/`：**1,963/1,963 json 全齐**（目标 1,963）。
- shardA：965/965，ok=889 bad=76，9,900s（0.10 src/s）；shardB：964/964，ok=874 bad=90，
  9,572s（0.10 src/s）。合计 ok=1,763 / bad=168，双实例干净退出、0 残留进程。
- **bad 168 个全部 status=`mostly_unsure`**（多数段输出 UNSURE；raw 内仍有部分角色判断）。

### O1 scout 门（`r11_o1_v1_features.py --e3-dir … --out r11_o1_e3_feats_32b`）

```
manifest frags=3917, role_coverage_frac=1.0, missing_e3=0
mean_keep_gt_overlap = 0.9132   →  gate_o1_ge_06 = TRUE（门限 0.6）
feature frags written=3917
```

- **按预注册：门 PASS，O1-verifier 保持排队**。
- 角色分布（14,955 段）：MAIN 37.2% / SETUP 27.2% / RESULT 20.4%（keep 族 84.8%）；
  UNSURE 8.8% / JUNK 4.1% / IDLE 2.3%。
- ⚠️ **机制警示（同报，影响下游读数设计）**：keep 族占 84.8%、63.6% 的视频全段 keep、
  overlap median=1.0 / p10=1.0——`KEEP={MAIN,SETUP,RESULT}` 三类并集过宽，keep 并集几乎
  盖满时间轴，overlap 门对「角色定位 GT」的**区分度有限**（0.9132 主要反映并集覆盖）。
  与判定四 7B 的单调塌缩不同，32B 有真实三类分化（37/27/20，UNSURE 仅 8.8%）——
  **V1-7B 的挂起结论不外推到 32B**；但下游读数不应再用 keep 并集口径，改用**角色序列
  形态**（MAIN 集中度、SETUP→MAIN→RESULT 时序、非 keep 段位置）才有判别力。

### 过程与成本

- 判定九配方验证链：32B contact-sheet hang 解除（ok=2、40s/2 源）→ 全源双实例 2.75h
  跑完 1,963 源（含 8 卡满占期间我方探针避让，见 R12 记录）。
- 产物：`r11_o1_e3_feats_32b/`（3,917 npz：e3_role (8,5) + slot_cov + keep_prior + t）+
  `scout_readout.json`；统计脚本 /tmp/o1_stat.py（一次性）。

## 判定十（10-08 晨）：e3_role 判别力预检 FAIL —— 角色结构与 keep_prior 冗余，O1-verifier 不再花

V1-32B 产物（判定九）经 R10 同协议双门读出（`r10_audio_head.py --feature
e3_role`，3,917 frag / 1,963 源 / 5 折 CV / 2,000 次 bootstrap）：

| 臂 | f1 | vs champion | vs folded prior | 置换对照 |
|---|---|---|---|---|
| R1_prob_resid | 0.6612 | +0.00401 CI[0.00123,+0.00689]✓ | **+0.0004 CI[−0.00018,+0.00102] 跨 0 ✗** | +0.00039（≈真值，削弱） |
| R2_util_resid | 0.6609 | +0.00364 CI>0 | +0.00008 null | POLICY_ONLY |

**判定：FAIL（vs prior 门）**——角色 one-hot 对折叠先验（keep_prior+slot_cov）
零增量。机制：角色分布退化，MAIN+SETUP+RESULT=84.8%、IDLE+JUNK 仅 6.4%
（14,955 段统计）→ 与 keep_prior 共线；scout overlap 0.913（门 0.6）是
「几乎全 keep」分布的天然产物，非精细判别证据。

**结论**：32B contact-sheet 形态的事件角色结构作为 slot 残差特征块收口
关闭；**O1-verifier（4 NPU-h）前提塌掉，不再投入**。判定四的 V1-7B 塌缩
属形态问题（判定九修复了形态），但信息量本身与廉价先验冗余（本判定）——
两层各自成立。与 R9（编码器 8 读出全负）、R10（音频三线全负）、S2a
（dense 候选关闭）同向：独立廉价信息源在 PHD2 池接连证伪，残余杠杆集中
在 S1 灰区、S2c/S2b 与官方域差（S4）。

## 判定十一（10-08 晨续）：S2b null 关闭 + S4 预审计基线固化 + S2c 权重缺口

**S2b（presence/box 稳定性，slot 粒度跟踪连续性代理）**：从 S2a 检测框提取
3,917/3,917 frag 的 (8,5) 特征（presence/conf/面积/中心抖动/检测数），R10 同
协议读出 → **R1 delta_vs_prior = 0.0（严格 null，λ 全零 = 模型弃用特征）**、
R2 +0.00007 null。**判定：关闭**。检测框 schema 备忘：per-slot 仅 1 框
（best-conf），n_det 为原始检测数；工程教训：批量 np.load 必须 close（fd 耗尽
被裸 except 吞成「malformed」假象）。

**S4 预审计（`scripts/r11_s4_preaudit.py` → r11_s4_preaudit.json）**：官方域
（live_dev keep 密度 0.248 / live_confirmation 0.232）vs 诊断域（rv_dev 0.404 /
rv_diag 0.425），**gap −0.17、p90 gap −0.415**（rv 有大量 0.78 密度源，live 尾部
截于 0.36）。**结论：rv 上校准的 keep 密度倾向搬到 live 会系统性 over-predict**；
任何空间候选的 S4 门基线 = live 自身分布（≈0.24）。此读数与「B3 官方 raw 38.76
初赛仅 +0.33 转化」的稀疏 GT 机制一致。

**S2c Light-ASD**：前提门（speech 41.7%）已过，但 **Light-ASD 权重不在本地**
（find 无资产），需下载+face-crop 管线搭设——列为待续项，本轮未跑。

**R11 累计判定面**：关闭=S2a dense、V1/O1 角色结构、S2b；灰区=S1（+0.0024）；
待续=S2c（权重缺口）、S4 正式门（待候选）、S3（条件挂起）；环境正资产=32B
推理复活（判定九）。本地廉价信息源五连 null 后，剩余方向=官方域差（S4 预审计
已给出第一手密度结构）与 R12 平台差分归因。
