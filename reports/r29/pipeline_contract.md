# R29：现役信息路径与梯度合同

2026-10-16；基于本地父版本 `d3e6a40e417321be67229f2add54aca34f80d858` 的有限源码核查。本文未访问远端、GPU、protected 源或赛事输入；空间缓存状态沿用 R28 已记录的盘点，不声称重新完成原视频部署验证。

**当前 AHA 是完全冻结的原生三头推理，R29 只拟合已存两标量的微型读出；没有梯度进入视觉塔或语言模型。B3 早已是候选 IoU utility head，本轮不能把这个思想当作新方案。旧 Qwen ckpt030 的精确 LoRA/视觉池化合同仍缺原训练文件，不能由模型名称补齐。**

## 1. AHA：观测、因果状态与输出

```text
视频原始帧索引
  → 约 1fps 选帧（R25：metadata FPS；R26 公共：实际 PTS）
  → ffmpeg 原帧解码、禁止自动旋转、等比缩放与黑边填充至 384×384 RGB
  → AHA 原生 image_processor → visual_embed → 每帧 49 个视觉 token
  → 首块前缀为通用任务文本，后续只追加视觉 token
  → 因果 transformer + 跨块 past_key_values
  → 各帧最后一个 token 的 hidden state
  → informative / relevance / uncertainty 原生头
  → 缓存三个标量；时间排序只用 informative + relevance
  → 最近采样点所属区间广播 → 预算选帧 → 取已有 B3 框 → JSONL 原帧 ID
```

- **R25 采样不是实际 PTS 最近邻。** `r25_aha_submit.py:105–106` 使用 `round(arange(ceil(duration))*metadata_fps)`，裁至合法帧并去重。`r26_public_score.py:103–108` 则验证媒体/轴 SHA，使用 `arange(0, pts[-1], 1s)` 对实际解码 PTS 最近邻并去重，保存原帧索引和实际 PTS。这两种合同不得互换解释；R29 公共试验继承后者。
- 解码与原生 processor：`r25_aha_submit.py:52–65,89–94`。`visual_embed` 导入远端 AHA 类，49 token/frame 是本地运行断言；其内部视觉 token 压缩、projector/pooling 细节不在这次有限本地核查中，**UNKNOWN**。不能由 49 推断原始图像只经过某一种池化。
- 因果路径：`r25_aha_submit.py:61–85`。每块默认 8 帧，调用方传递上一块返回的 KV、不主动重置或裁剪；第 f 帧读取自身最后一 token，能使用允许的过去上下文和自身 token，不能看未来帧。实际历史可注意范围还依赖已加载 config/cache 的隐式窗口，本次未实例化核实，不能只据调用方代码宣称全部历史始终可见。离线整片处理并没有使它成为双向整片模型。首块加入固定通用 query：`Identify the most informative and important highlights of this video.`；没有逐视频人工任务文本或 GT query。
- 输出：`informative=softmax(head)[1]`，`relevance=sigmoid(head)`，uncertainty 保存但不参与排序；头在全 token 上计算，再取帧末位置（`r25_aha_submit.py:53–58,74–83`）。首块与 native forward 对照要求最大绝对差 `<1e-5`。词表 LM projection 不反馈评分头或 KV，因此在这个 forward 中省略；不能由此把仍加载且参与参数清单的权重从申报中随意删除。
- `model.eval()`、`currently_training=False`、`requires_grad_(False)` 与 `torch.inference_mode()` 明确关闭训练（`r25_aha_submit.py:33–38,53–59`；R26 同结构）。loader 检查 missing/unexpected/mismatched keys 为空，三头权重与 adapter 对应张量精确一致；移除并以非注册别名恢复 `vision_encoder` 的装载处理有明确代码，不能把历史序列化张量和直接当实际推理参数。

## 2. 广播和预算：三个版本必须区分

| 路径 | 广播距离 | 输出预算与 cell 内决策 |
|---|---|---|
| R25 历史提交 | 原始帧索引最近邻，平距先早采样点 | 按采样分数稳定降序，整 cell 加入至达到预算，可超 `round(.85N)` |
| R26 official exact builder | 原始帧索引最近邻 | 最后 cell 按距采样原帧由近到远截断，平距先早帧；精确 K |
| R26/R29 公共评估 | 实际 PTS 最近邻 | 最后 cell 按距采样 PTS 由近到远截断，平距先早帧；精确 K |

证据：`r25_aha_submit.py:124–134`；`r26_build_diagnostics.py:19–32`；`r26_public_evaluate.py:22–46`；`r29_public_decision_probe.py:44–57,157–161`。这些广播会把单点判断扩成区间判断；精确 K 修复数量差，不自动恢复边界信息。R25 的空间参数输入是已经覆盖全部原始帧的框 JSONL，脚本仅选择这些框；**AHA 未重算或更新 B3**。

当前安全包是 deterministic center85 + B3，时间分支本身不调用 AHA。R27 中 75%/95% 的用户回分低于 center85；本轮未再扫官方 keep。上述公共实际 PTS 合同通过不等于官方素材全部 VFR/PTS 合同已获独立证明。

## 3. 本轮与旧头训练，梯度实际到哪里

| 路径 | 固定输入与可学习部分 | 监督、压缩与结论边界 |
|---|---|---|
| **R29 已执行微型读出** | 已存 AHA 两标量，经 cell 广播；可加三次位置基；只优化 logistic 系数 | TRAIN 每源等权 BCE、固定 ridge、TRAIN 标准化；SciPy 优化，不运行 encoder，不回传 AHA。插件预算用 `sum(p)`，并非完整 M 后验。`r29_public_decision_probe.py:60–93` |
| **R24 已完成对照** | 冻结 VPB/CLIP → 每采样点 768d → 768→512 GELU→1；仅头有梯度 | 12 点实际 PTS；VPB 空间 token 均值，CLIP native processor + pooler_output；十人 point 或 block-density 标签，BCE；不同来源 TRAIN/DEV。不是新待跑方案。`r24_clean_compare.py:44–108` |
| **旧 VPB/GIF 头** | 缓存 temporal/spatial features → TRAIN norm → 768→512 GELU→1；仅头有梯度 | `gif_hard`/`_target`，MSE + .5 pairwise hinge；旧 row-level hold125 的泄漏/标签问题已知，不作泛化证据。原部署确实读取 checkpoint norm，采用 GELU；VPB 输出 `.mean(2)` 得到时间特征再与 Qwen 已存分数融合。`reports/r22/head_identity/train_vp_head.py:53–67,103–121,156–179`；`score_426_vp.py:103–110,166` |
| **旧 Qwen3.5 ckpt030 SFT/LoRA** | **精确路径 UNKNOWN** | 旧文档指向的 `scripts/r13_sft_lora.py` 在当前 checkout 缺失；本次有界搜索未找到该原训练入口。LoRA target_modules、视觉塔/projector 是否冻结、pooling、loss 到哪些 token 均不作猜测。需原 trainer、adapter_config、trainable tensor 清单才能恢复。其他 Qwen2.5 空间 trainer 不能替代此证据 |

R29 的负结果仅约束“两个冻结标量 + 当前读出 + 当前公共目标”的迁移，不约束完整 hidden state 中是否还存在可读出内容，也不证明语言端或视觉塔适配无用。若区别已在 `visual_embed` 或池化中丢掉，后端梯度不能逆向创造输入；但目前没有证据证明是哪一层已经丢掉这种区别。

## 4. B3：已有的 utility 模型与当前缓存部署

历史可核实训练合同：native processor → LFM2.5-VL **仅视觉塔**、单 tile NaFlex → valid token 2D grid → 129 个最大合法裁剪窗（只沿自由轴移动，固定尺度）。每候选输入为窗内均值、窗外均值、二者差及归一化位置，共 2305d；头为 MLP 2305→512→128 + 两层四头自注意力 → 候选 scalar。标签已经是候选对人工框的 annotator-mean IoU（RV 六人、LIVE 一人），loss 为 δ=.25 Huber + .3 best/worst softplus 排序。训练只读缓存特征，仅头接收梯度。

源码证据：`scripts/v9_phd2_lfm_feats.py:42–65`（同 V7/V8 冻结视觉抽取实现）；`scripts/v8_build_samples.py:61–101`；`scripts/v8_s_train_multidata.py:203–227,236–290`；配置 `configs/LFM_V8_B3_SEMIFINAL.json`。这些说明 **129 候选 utility/预期 IoU 排序不是 R29 新算法**。视觉 grid 的候选均值压缩可能丢位置关系，但是否导致当前误差仍需固定候选的 GT 诊断，不能从结构直接判罪。

当前可重放路径是缓存 keyframe 中心 → hold → 镜头 reset → EMA=.25 → 最大合法比例框 `[x,y,w]`。`reports/r28/summary.md:11–21` 记录对 426 视频/234159 帧的坐标回放最大差 0，以及在记录路径的有限搜索中 B3_s0 和 LFM 基座缺失、仅找到 V7。**缓存沿袭和几何一致已获证据；原视频到 B3 输出的当前独立部署未获证据。** 不能用 V7 冒充恢复 B3；旧 `qwen_points/qwen_centres` 字段名也不证明推理时在用 Qwen。

若下一轮恢复原权重，应绑定实际 head/base SHA、native processor、抽取 token mask、候选 grid 与训练 norm，再执行公共真实帧对照。当前误差层次仍须分开：候选覆盖、空间特征、utility 读出、几何映射、hold/EMA；不能以缓存回放成功排除其中任何 GT 质量问题。

## 5. 对本轮资源决策的约束

1. R29 scalar 校准没有对现有模型做适配；失败后盲目扩大同类标量读出不能诊断视觉能力。若研究 hidden features，应使用可追溯监督、相同观测和来源隔离的位置对照。
2. 当前 AHA 已是约 1fps 的因果输入；12 个整片稀疏点的问题不能直接解释其失败。需要区分静态点命中标注 run 与真正看到局部状态变化，必要时改变局部观测或未来上下文。
3. 空间的新主张必须越过“已有 utility head”这一事实，并解决原权重身份/真实输入合同；新检测/grounding/分割可生成候选，但教师框不能取代人工 native GT。
4. 时间公共 F 与独立裁剪池 IoU 仍无同源同帧联合桥梁。本文信息路径不提供官方涨分证据，也不改变总推理组件 ≤9B 的合规口径。
