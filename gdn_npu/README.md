# qwen35_gdn_npu — Qwen3.5 GDN / causal_conv1d 的 910A NPU 加速

2026-10-10 准备阶段产物（调研报告：`../GDN_NPU算子调研报告_20261010.md`）。
目标：替换 transformers 5.19 `modeling_qwen3_5` 的 4 个 PyTorch 回退函数，
服务 RL（GRPO）rollout 与 on-policy distillation。

## 文件

| 文件 | 用途 | 状态 |
| --- | --- | --- |
| `qwen35_gdn_npu.py` | 主补丁：4 个 NPU 实现 + apply()/unpatch() | CPU 对齐 50/50 PASS；NPU 44/45 PASS |
| `test_gdn_correctness.py` | 自包含数值对齐测试（内嵌生产参考实现） | 双环境全绿 |
| `bench_gdn.py` | M1 基准：chunk/decode/conv 计时 + 显存峰值 | 已跑定版 |
| `probe_graph.py` | M1 探针：npu graph capture 可行性 | 已判死留档 |
| `qwen35_attn_npu.py` | full-attention 分块+checkpoint（激活显存解） | CPU 13/13 PASS；NPU 待空闲 |
| `test_attn_correctness.py` | attention 分块对齐测试 | 沙箱 CPU 全绿；NPU 等空闲 |
| `bench_attn_npu.py` | attention 显存/速度基准（4 任务） | 备好待空卡 |

## ✅ full-attention 分块方案（2026-10-12，RL 量产前激活显存解）

### 原生融合算子判死（文件级证据，零 NPU 消耗）
- CANN 9.0.0 `opp/built-in` 全树无 FlashAttention 类 kernel（仅 detect_* 检测域变体）；
  ascend910 家族预编译融合算子仅 4 个 conv2d 融合；tbe impl 仅 9 文件无 attention；
  libopapi 有 FlashAttentionScore host 符号（203 hits）但无 kernel 文件 = 空壳。
- → `npu_fusion_attention` / `npu_prompt_flash_attention` / `npu_incre_flash_attention`
  本机全部不可用。同法验证 RecurrentGatedDeltaRule host 符号 18 hits 也无 kernel
  → GDN 手写路线仍是唯一路。
- 顺带发现：torch_npu 2.9 public API 有 `npu_recurrent_gated_delta_rule` /
  `npu_fused_causal_conv1d` 封装（910B 专用，910A 无 kernel）。

### 方案：纯 torch 分块 attention + 块级 checkpoint（qwen35_attn_npu.py）
- 前向 score 峰值 O(S²) → O(block×S)；训练时 score 不进 autograd 保存区（重算）。
- 数学逐块等价：causal 下每行 query 的 key 范围完整落在块内，softmax 行内独立
  精确，无 online-softmax 近似。CPU 对齐 **13/13 PASS**：前向 0~2.3e-8、
  梯度 ≤8.2e-7、ckpt/非 ckpt、4D mask/pad/cross/decode 回退全过。
- 显存账（4B，S=4900，B=1，fp16）：eager 每层保留 score 0.77 + repeat_kv 0.77 +
  softmax 输出 0.77 ≈ 2.3GB × 8 full 层 ≈ **18.6GB** → 分块+ckpt ≈ **1GB 级**。
- 契约：attention_mask=None 且 Sq==Skv 时按 causal 前缀截断（生产 is_causal 层
  语义）；Sq≠Skv（chunked prefill 增量）自动回退 eager；S_q==1（decode）回退。
- 锚定 transformers 5.19 `Qwen3_5Attention.forward`（类级 patch，仅文本 full
  attention，视觉塔与 GDN 层不受影响）。

### 用法（与 GDN 补丁独立，可叠加）
```python
import sys; sys.path.insert(0, "/root/AIC/gdn_npu")
import qwen35_attn_npu as attn
attn.apply()                    # 默认 block=1024, ckpt 开
# 回滚: attn.unpatch() 或环境变量 QWEN35_ATTN_NPU_DISABLE=1
```

### 待 NPU 空闲指令后执行（bench_attn_npu.py，4 任务）
```bash
cd /root/AIC && set -a && source /data/aic/tools/ascend_teacher.env && set +a
export ASCEND_RT_VISIBLE_DEVICES=<空闲卡>
for t in align infer layer stack8; do
  timeout 900 python3 bench_attn_npu.py --task $t > /tmp/bt_$t.log 2>&1
done
```
- align：NPU fp16 数值门（fwd/grad <3e-3）
- infer：prefill 峰值+速度（预期 chunked 峰值 ≪ eager）
- layer：训练单层 fwd+bwd 峰值+速度（ckpt 重算开销量化）
- stack8：8 层串联训练图峰值（4B 真实账：eager 预期 ~18GB vs chunked ~1GB 级）
- 风险提示：新 shape 首编译可能触发 TBE hang（三大坑①），timeout 包裹+重试；
  `use_reentrant=False` checkpoint 若触发 910A per-layer ckpt 线程坑，
  退路=apply(use_ckpt=False)（分块仍省推理峰值与 autograd repeat 物化）。



## 用法（模型构建前）

```python
import sys; sys.path.insert(0, "/root/AIC/gdn_npu")
import qwen35_gdn_npu as gdn
gdn.apply()                     # 默认(数值对齐优先: triangular+seq)
gdn.is_active()                 # True = 已接管
# 可选: gdn.apply(solver="nsquaring", scan="affine", chunk_batch=8, conv_impl="unfold")
# 回滚: gdn.unpatch() 或环境变量 GDN_NPU_DISABLE=1
```

注意：
- `apply()` 后**不要再**调用 `r13_qwen35_safe_chunk.apply()`（会覆盖回慢速版）。
- kernel 契约：调用点已把 q/k repeat_interleave 到 Hv 头，函数内 q/k/v 全 Hv 头。
- 已知疑点：transformers 5.19 chunk 回退的 solve 分支解 (I+L)X=V，dynamo 分支解
  (I−L)，符号相反；生产走 solve 分支，本实现全部对齐 solve 分支。

## 配置项（apply 参数）

| 参数 | 取值 | 说明 |
| --- | --- | --- |
| `solver` | `triangular`(默认) / `nsquaring` | 三角求解；后者纯 bmm 可微，CPU 对齐 rel 3e-8 |
| `scan` | `seq`(默认) / `affine` | chunk 间递推；affine=仿射前缀合成去串行化，CPU 对齐 rel 4e-6，T=8192 时 kernel 数 NC×6→O(log NC) |
| `chunk_batch` | 0(默认)/8/16 | intra 量分批，压 pairwise_decay 峰值显存（fp32 全量 4B@T=8k 约 2.1GB、27B 约 3.2GB） |
| `conv_impl` | `conv1d`(默认) / `unfold` | prefill 深度卷积；unfold 用 bmm 形式（避开大 numel fp16 广播乘的 CPU 内核缺陷，NPU 上也更省） |

## M1 结果（2026-10-11，910A 卡5 实测）

### 数值验证
- **NPU 44/45 PASS**（quick 全矩阵，isolate 模式）。唯一 FAIL=`chunk_batch4_seq`
  （特定 shape 的 TBE 编译 hang，非数值问题；同构 batch4_affine/real4b_seq 均 PASS）。
- 门全过：out(fp16) rel ≤3e-3（实测 ≤5.1e-4），state(fp32) ≤1e-3（实测 ≤4.4e-4）。

### 性能（NPU 实测）
| 项 | 结果 |
| --- | --- |
| decode mine | **988 µs/层/步 @B=1**；B=1→8 不敏感（988→946µs）= launch 开销主导 → 手写融合 kernel 是正道 |
| chunk 最优 | **`nsquaring+affine` 6.8ms/call @T=512，peak 0.33GiB**（cb 无关） |
| solve_triangular | **隐性热点**：同 scan 下 nsquaring(纯 bmm) 快 1.3-10× |
| affine vs seq | tri 下 38.7k→18.1k µs；nsq 下 11.3k→7.0k µs |
| chunk_batch 分批 | 负收益（tri+seq+cb8 病理 2.2s：逐批 solve 触发重复 JIT）→ 弃用（cb=0） |
| conv | **conv1d 胜 unfold**（173µs vs 320µs @T=8192，与 T 无关）→ 默认 conv1d |

### 推荐配置
```python
gdn.apply(solver="nsquaring", scan="affine")   # chunk 最优组合（数值门已过：nsq rel 3e-8 / affine rel 4.8e-6）
# decode/conv 无参数；conv_impl 保持默认 conv1d
```

### 路线判定
- **A+（npu graph capture）关闭**：`torch.npu.NPUGraph` capture 失败，
  驱动 24.1.0/固件组合不支持（`Feature is not supported`）——驱动特性级结论。
- **路线 B（手写 decode 融合算子）权重上调**：decode 988µs/层/步、launch 主导、
  graph capture 不可用 → AICPU/TBE kernel 是 decode 的唯一根本解。

### 本机 NPU 三大坑（M1 新发现，影响所有算子开发）
1. **4 维 matmul + transpose 视图 → TBE BatchMatMul 在线编译失败/hang**
   （ge BuildSingleOpModel 500002）。生产参考实现 scan 里就有此调用 → 参考系
   在 NPU 上结构性跑不了（bench 的 ref 基线用 `.contiguous()` 兼容版）。
   对策：循环内一律 3 维 bmm（mine 已全改）。
2. **进程退出阶段 TBE daemon 死锁**（atexit join 卡死）→ 子进程 `os._exit()`
   强退 + 输出落文件（不得用 pipe 捕获，孤儿持有 pipe 写端会吃满超时）。
3. **同进程连续新 shape 触发多次 TBE 编译 → 累积死锁** → case 级进程隔离
   （isolate 模式），hang case 重试一轮（编译缓存热后可过）。

### 手写算子编译链状态（M3 前置探针）
- msopgen AICPU 工程生成 ✓；build 卡在 op_proto 链接（旧模板 CMake 与
  CANN 9.0.0 布局不匹配：ASCEND_INC 指 ascendc/include 缺 graph 头、
  libgraph 需显式链 libgraph_base、交叉 crt 污染 host 库链接）。
  已修三个（include 路径/libgraph_base/工具链软链），剩 op_proto+crt 问题待解，
  或改走 `-lan cpp`(ascendc) 模板重试。cmake 已装入 qwen35 venv。

## ✅ M3 里程碑：手写 AICPU 算子全链路打通（2026-10-11）

**GdnDecodeStep（decode 单步 GDN 递推融合 kernel）在 910A NPU 上执行成功，数值双 MATCH**
（out diff 6.1e-5 = fp16 底噪；state diff 7.5e-9 = fp32 精确；对照 CPU 逐式参考）。

### 工程（/root/msop_aicpu，沙箱源码 gdn_npu/aicpu_gdn + aicpu_add）
- 算子：`GdnDecodeStep`（q/k/v fp16 + g/beta fp32 + state fp32 → out fp16 + new_state fp32）
- 验证：msopst 生成 ACL 调用程序 + 手造输入 bin + numpy 逐式参考对比

### CANN 9.0.0 + Ascend910A 自定义 AICPU 算子全配方（踩坑沉淀）
1. **build 环境变量**（四件套）：
   ```bash
   export ASCEND_OPP_PATH=/usr/local/Ascend/cann-9.0.0/opp
   export ASCEND_AICPU_PATH=/usr/local/Ascend/cann-9.0.0
   export ASCEND_TENSOR_COMPILER_INCLUDE=/usr/local/Ascend/cann-9.0.0/aarch64-linux/include
   export AICPU_SOC_VERSION=Ascend910
   export TOOLCHAIN_DIR=/usr/local/Ascend/cann-9.0.0/tools/hcc   # 目录形式
   ```
2. **`cmake/config.cmake` 必须去掉 SHARED_LINKER_FLAGS 的 `-pie`**——模板把 exe
   链接标志混进 so，GNU ld 拉 Scrt1.o → `undefined reference to main`（四小时级坑）。
3. `ASCEND_TENSOR_COMPILER_INCLUDE` 指向 `aarch64-linux/include`（不是 ascendc/include）；
   op_proto 需显式补链 `libgraph_base.so`（新版 ld 严格）。
4. `REGISTER_CPU_KERNEL(NAME_IDENT, Class)` 宏：第一个参数是**标识符**
   （`const char *NAME_IDENT = "OpName"`），不是字符串字面量。
5. REG_OP 结束用 `.OP_END_FACTORY_REG(Name)`（无 OP_END）。
6. atc/msopst 的 soc_version = **`Ascend910A`**（不是 Ascend910；platform_config 目录名为准）。
7. 执行链：安装 `.run --install-path=$CANN/opp` + `ASCEND_CUSTOM_OPP_PATH=$CANN/opp/customize`
   + customize 目录要有 `version.info`；msopst 生成的 main 需要
   `test_data/data/<case>_input_i.bin` 数据文件；生成的 ACL 工程 cmake 需手工补
   `-I $A/include -L $A/lib64 -lascendcl -lruntime`。
8. cmake 不在系统 → pip 装 qwen35 venv + symlink /usr/local/bin。

### 待办（M3 后半）
- torch 侧接入：`torch.library` 自定义 op + acl 单算子调用封装（aclopExecuteV2 已验证可用），
  替换 `qwen35_gdn_npu.npu_recurrent_gated_delta_rule` 的循环体。
- V2 融合：conv_update + g/beta + l2norm 进 kernel（V1 只融合 GDN 递推）。
- 性能阶梯：AICPU V1（预估 2-4× decode GDN 段）→ 观察 B=8 是否吃满单核 → 决定是否上 TBE。

## ⚠️ M3 后半判定（2026-10-11）：aclop host 开销构成结构性墙

**torch 侧 ctypes 直调已全链路打通、数值 MATCH**（`aicpu_call.py`：
`aclopSetModelDir` 注册离线 om 静态库 → `aclopExecuteV2` 按 shape 匹配执行；
aclop 动态模型生成不可用，静态 map 通道可用；desc 缓存 + dataset 构造；
注意 SetModelDir 用绝对路径。B=1/B=8 out diff 9.8e-4、state ~1e-7）。

**性能判定（B=1 实测）**：
| 路径 | µs/次 |
| --- | --- |
| om-once（desc 缓存+dataset 临时构造） | 3900 |
| om-cached（dataset 全缓存，仅 ExecuteV2） | 3878 |
| torch 向量化（现实现） | **783** |

cached≈once → 开销在 aclopExecuteV2 内部（模型匹配/GE 调度，结构性 ~3.9ms），
非构造开销，不可优化。**结论：任何经 aclop 通道的自定义算子（AICPU 或 TBE om）
在 decode 场景都必然输给纯 torch 向量化**。若要打赢 783µs，只有两条重路：
1. TBE kernel + 自定义 aclnn 通道（新栈特性，910A 支持未知——graph capture 已死是前科；估 2-4 天，建议先 1 小时探可行性再投入）
2. op_plugin 源码集成进 torch_npu（host 开销 ~30µs 级，但需源码编译 torch_npu，数天+版本耦合）

**当前工程判断（待用户裁定）**：decode GDN 段以 torch 向量化（783µs/层/步）收官；
本 M3 的全部配方（编译链/om 通道/验证链）已沉淀——若未来升级 CANN 或迁移 910B，
手写 kernel 可直接复用并自动受益于更快的调度通道。

## 部署位置（910A）

代码：`/root/AIC/gdn_npu/`（本目录同步）。跑前：
`cd /root/AIC && set -a && source /data/aic/tools/ascend_teacher.env && set +a`
