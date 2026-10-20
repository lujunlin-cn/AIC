# AIC 视频高光：Valen 多模态决策路线技术交接与执行指南

> 文档版本：v1.0  
> 编写日期：2026-10-20  
> 目标读者：接手 `lujunlin-cn/AIC` 的 Coding Agent  
> 状态：**指导性技术方案；Valen 与 AIC 的任务对接尚未运行，所有增益均待验证。**  
> 本次参考仓库基线：`lujunlin-cn/AIC` 的 `main` 提交 `6ac4cb648924c18143db0745a72a2adfc940c66f`（2026-10-20 03:18:29 UTC）。实施前必须重新检查当前 HEAD。  
> 设计原则：复用现有 `aic/`、`scripts/`、`configs/`、`reports/` 和历史预测缓存；**不推倒重建，不直接替换现役提交链路**。

## 0. 给 Coding Agent 的执行摘要

本研究准备引入 [Valen](https://github.com/Valen-Team/Valen)——受 Jev 启发的多模态候选决策模型——解决 AIC 的两个决策：

1. **时间轴**：在已经筛出的高召回时间窗口中，判断窗口保持、前移、后移；条件满足后，再研究边界附近的精确原始帧选择。
2. **空间轴**：在现有 **129 个合法构图候选框**中，先用轻量排序头选出 Top-K，再让 Valen 比较候选裁剪图像，选择一个 `[x,y,w]` 框。

**首选研究模型**：`Valen-Team/Valen-2B`。速度对照：`Valen-Team/Valen-0.8B`。优先使用官方 `Choice` 决策接口，保留 `Noul`、`Score` 的可扩展性。模型提供的是**选项集合内概率**，不等于已校准的真实正确率。

**首选工程路线**：先验证“冻结已有时间集合，仅替换空间决策”的 Valen 分支；再验证“冻结已有空间框，仅替换时间窗口策略”的分支。Valen-2B + 轻量召回模型是一条**拟议新管线**，不是已验证提交系统。

**重要更新**：AIC 在 2026-10-20 新完成 DINOv3 matched probes，时间与空间主实验均**未通过**预注册门。不得按之前的未执行计划重复启动相同实验。DINOv3 只剩一个具有新机制的 pairwise/listwise 空间排序诊断可考虑执行；相关细节见第 3 节。

**硬限制**：比赛全部实际推理参数量 `<= 9,000,000,000`。R54/R70 的 TimeLens2+V9 推理链约 `8.8545B` 参数；与额外 2B Valen 相加会超限。正式方案必须**替换**该大模型链，而不是叠加 Valen。历史包仅用作不可训练的性能参考或冻结父输出的可控比较。

### 0.1 首次接手时必须输出的结果

在修改代码之前，Agent 应先生成 `reports/valen_route/asset_audit.md`，至少确认：

- 当前 Git HEAD、Python/PyTorch/Transformers 版本、GPU 型号与显存、CUDA 是否可用。
- `aic/`、`scripts/` 里解码、原始帧映射、时序头、129 候选框、裁剪回映射、打包与独立校验的实际入口。
- 当前可访问的 V9 vision-only 模型、TCN checkpoint、R70 候选特征缓存、R54/R71 冻结父预测及 DINOv3 缓存；对**不存在**的文件登记缺失，不自行假造。
- Valen-2B/0.8B 下载、依赖、许可证、模型 revision 和完整参数量；下载前先做运行环境兼容性检查。
- 当前官方提交 schema 是否仍要求 `model_params_b`，以及本地历史 checker 是否仍只接受 `model_size_mb`。

不得把“Git 中出现模型名/配置文件”当成“远端 checkpoint 当前存在”。

---

## 1. 赛题的真实任务与硬约束

### 1.1 输入与输出

输入：原始视频、`video_id`、目标画幅 `targetRatioWH`（如 `[16,9]` 或 `[9,16]`）。输出为每个视频的 `predictions` 列表；每个元素具有：

```json
{
  "frame": 120,
  "bboxes": [120.0, 80.0, 640.0]
}
```

`frame` 是**原始视频从 0 开始的整数帧编号**，不是抽帧序号、时间戳或视频 clip 编号；`bboxes` 是单个扁平的 `[x,y,w]`，不是多个框数组。对应框高不提交，而由 `h = w * target_h / target_w` 计算。所有坐标使用原始视频像素空间且必须合法。一个原始帧最多提交一个框；允许 `predictions: []`。

新包记录所有实际推理模型总参数量：`model_params_b = parameter_count / 1e9`。旧包中的 `model_size_mb` 属于历史版本字段，不能不经核实复制进新包。样例 JSON 的元字段以最新赛事 schema 和独立校验为准。

### 1.2 官方目标函数

对视频 `v`，令 `N_p` 为预测帧数，`N_g` 为 GT 帧数，`S_v` 为**相同 video_id 且 frame 精确相同**的配对帧上连续空间 IoU 之和，则

\[
F_v=\frac{2S_v}{N_p+N_g},\qquad
\mathrm{Score}=100\times\frac1V\sum_{v=1}^{V}F_v.
\]

特殊规则：`N_p=N_g=0` 时 `F_v=1`；仅一侧为空时 `F_v=0`。没有时域匹配容忍带。视频间是**宏平均**，不能用全部帧的 micro-F1 替代。当前公式**不含**历史 `k_size` 系数。

必须理解：高召回候选集本身不等于最终高分。误选帧增加分母却没有 IoU 贡献；正帧上的构图错误也会降低得分。现有很多历史系统以 `keep=0.85` 提交大量连续帧，**这只是历史策略，不是比赛强制预算**。

### 1.3 合规与截止

- 赛方**不提供训练集**，不公开正式测试集 GT。训练只能使用许可允许的公开数据、合规自建标注和公开预训练权重。
- 正式测试集只可进行冻结推理、输出检查和合规提交。严禁对测试素材/模型输出进行新训练、测试集伪标注，或利用平台回分反推标签并做逐样本调参。
- 所有推理组件（冻结与训练、辅助视觉、检测器、裁剪头）都计入 9B 总参数量；共享权重只计一次，量化**不减少参数个数**。
- 仓库规则基线记载：复赛结果截至 `2026-10-23 20:00`、作品材料截至 `2026-10-25 23:59`、每日最多 5 次提交。官方通知未明确标注时区；Agent 应复核平台，不自行推断时区。

来源：[赛事规则与评分](https://github.com/lujunlin-cn/AIC/blob/main/01_%E8%B5%9B%E4%BA%8B%E8%A7%84%E5%88%99%E4%B8%8E%E8%AF%84%E5%88%86%E6%A0%87%E5%87%86.md)（链接若变更，以仓库根目录实际文件 `01_赛事规则与评分标准.md` 为准）。

> 注意：某些历史 `aic/evaluation.py` 路径仍引用旧 `size_coefficient`；`scripts/independent_submission_check.py` 在本次审计时仍校验旧 `model_size_mb`。Agent 必须**新增或兼容升级现行规则校验**，保留旧包校验入口，不能直接用旧 checker 的“通过”声称现行官方格式通过。

---

## 2. 当前仓库技术栈：已实现、可复用与不能假设的内容

### 2.1 高层链路

```text
原始视频 + 官方索引
        |
        v
现有解码/PTS/原始帧号映射
        |
        +--> [时间路径] 位置先验 / 已有 TimeLens2 冻结参考 / PHD2 TCN / 轻量视觉特征
        |          |
        |          +--> 高召回候选窗口 / 原始帧集合
        |
        +--> [空间路径] 现有 V9/B3 相关资产、合法候选框与候选评分
                   |
                   +--> 129 合法窗口 -> 候选框与特征
        |
        v
合法性约束 + 逐帧 [x,y,w] + JSONL/ZIP + 独立 validator
```

**这张图描述既有能力与候选研究模块，不代表所有分支已同时存在可部署 checkpoint。**

### 2.2 仓库中的关键资产

| 模块 | 现有入口/证据 | 目前判断 |
| --- | --- | --- |
| 赛题规则 | `01_赛事规则与评分标准.md` | 以其现行参数、逐帧与公式契约为准 |
| 状态与实验史 | `STATUS.md`、`EXPERIMENTS.md`、`DECISIONS.md`、`registry.jsonl` | 先查再做，避免重名与重复试验 |
| 视频原始帧契约 | `aic/temporal_contract.py` 及既有解码/索引脚本 | 已纠正 anchor 与 clip center 约定；新模块必须复用并核对 PTS |
| 时间模型/训练基础 | `aic/models.py`、`aic/train.py`、现有 `R92/R97/R102` 脚本 | Temporal U-Net/TCN、弱监督和缓存流程已有历史实现 |
| 空间候选 | R70 相关报告、`scripts/dinov3_spatial_patch_probe.py`、历史 B3/V9 路径 | 129 个合法候选窗口及原生 crop 代理评测已实现 |
| DINOv3 实现 | `scripts/dinov3_runtime.py`、`scripts/dinov3_temporal_ablation.py`、`scripts/dinov3_spatial_ranking_probe.py` | 已有源码封装和 matched probes；不应重新从零搭建 |
| 提交验证 | `aic/contract.py`、`aic/evaluation.py`、`scripts/independent_submission_check.py` | 保留旧兼容路径；添加现行 `model_params_b` 检查与独立回归测试 |
| V9 vision-only | `configs/v9_temporal_components.json`、R54/R70 打包报告 | R54 报告中的空间视觉模块约 87.4M 参数；需确认**实际可加载权重与前处理** |
| 历史 B3 checkpoint | `reports/r72_b3_asset_recovery_20261018/README.md` | 原始 V8 B3 head 与 base 在审计环境**未恢复**；不能用 V7 文件冒充 |

新代码只允许作为 `aic/` 的薄适配层和 `scripts/` 的实验入口；不要新建一套与仓库平行的视频解码器、时间映射或官方输出协议。

### 2.3 Python、推理框架与依赖的实际边界

以下版本来自本次读取的两份 `pyproject.toml`，**是仓库依赖声明，不是当前机器已安装版本**：

| 依赖 | AIC 主仓库 | Valen 官方仓库 | 处理方式 |
| --- | --- | --- | --- |
| Python | `>=3.10` | `>=3.10` | 独立 Python 3.10+ 环境 |
| PyTorch | `torch>=2.2,<3.0` | `torch==2.6.0` | 以 Valen CUDA 兼容性为准，和现有 NPU 环境隔离 |
| TorchVision | `>=0.17,<0.25` | `==0.21.0` | 不在现有 AIC 环境直接升级/降级 |
| PyAV | `av>=12,<17` | `av==16.1.0` | 复用视频原始 PTS 映射，但保留媒体解码一致性测试 |
| 图像/数值 | `Pillow>=10`、`numpy>=1.24,<3` | `Pillow>=10`、`numpy>=1.26` | 固定实际安装版本和测试图片 SHA |
| Transformers / PEFT | AIC 主依赖未固定 | `transformers==5.4.0`、`peft==0.18.1` | 由独立 Valen 环境安装，**不能**直接宣称与既有推理环境兼容 |
| 空间/视觉可选包 | `opencv-python-headless`、`timm` | Valen 依赖自有 processor | 复用 AIC 几何和候选逻辑，不以 Valen processor 替代坐标回映射 |

链接：[AIC `pyproject.toml`](https://github.com/lujunlin-cn/AIC/blob/main/pyproject.toml)、[Valen `pyproject.toml`](https://github.com/Valen-Team/Valen/blob/main/pyproject.toml)。Valen 官方脚本的 CUDA 12.4 PyTorch 安装示例需要与实际 GPU 驱动兼容；不得在未经检查的 Ascend `torch_npu` 环境内直接执行。

### 2.4 经记录的近期官方平台分数

| 历史包 | 用户报告的平台分数 | 可得结论 |
| --- | ---: | --- |
| R71 | 36.43 | 当前较稳的历史高分锚 |
| R107 | 36.43 | 事件生命周期并未超过 R71 |
| R70 | 36.42 | 空间候选排序出现过正向官方回分 |
| R93 | 36.32 | 4 FPS 端点残差较 R54 小幅上升 |
| R54 | 36.22 | TimeLens2 中心锚定残差是可比对照 |
| R108 | 35.62 | 采样率分歧直接当高光价值未奏效 |
| R91 | 33.87 | ResNet18 + TVSum 的当前完整轻量方案失利 |

上述数字是 AIC 仓库登记的用户平台反馈；平台上传文件与本地 ZIP 的逐字节映射并未全部独立核验，**不能写成已经做过本次 Valen 实验**。初赛 48.67 与当前复赛的 36.xx 使用不同条件，不可直接比较。

关键报告：

- [R107/R91/R108/R93 回分复盘](https://github.com/lujunlin-cn/AIC/blob/main/reports/r107_r91_r108_r93_official_postmortem_20261020/summary.md)
- [R54/R46 归因分析](https://github.com/lujunlin-cn/AIC/blob/main/reports/r54_r46_official_postmortem_20261018/summary.md)
- [R70 相关空间结果](https://github.com/lujunlin-cn/AIC/tree/main/reports/r70_topsoft_objective_20261018)

### 2.5 不得忽略的 DINOv3 最新负结果

[2026-10-20 matched probes](https://github.com/lujunlin-cn/AIC/blob/main/reports/dinov3_v1/20261020_dinov3_probe_results.md) 已完成：

- DINOv3 ViT-B/16：`85,669,632` 参数，768D CLS/patch，已完成 5,886 个 PHD2 片段的 CLS 缓存。
- 时间位置消融（同一个公开代理划分）：position-only `0.457027`；DINO CLS `0.438451`；DINO+position `0.433453`；DINO+delta `0.435161`。**当前 DINO+TCN 不优于位置基线，不能作为已验证召回升级。**
- 空间候选：中心宏平均 IoU `0.497588`；DINO patch `0.484223`；候选 oracle `0.803418`；CLS 广播 `0.101462`；patch 置换 `0.261900`。patch 存在候选敏感信息，但还不够达到中心基线；**不晋级**。
- 下一项只保留结构不同的 `scripts/dinov3_spatial_ranking_probe.py`（pairwise/listwise 排序目标）可选诊断。未超过中心且未达域外预注册门槛则关闭该部署路线。

**本指导文档不要求 Coding Agent 再跑 DINOv3 union-BCE、原 SmoothL1 空间回归或重做 E0–E6。**

### 2.6 轻量时间召回模型的现实排序

1. **首个工程对照**：确定性的中心连续窗口，或冻结的既有窗口生成代码。它不需要新参数，且在正式回分上有充分历史支撑。
2. **首个学习型候选**：使用确实存在的 V9 vision-only 视觉表示与现有 TCN 训练/解码协议；此组合仍属**待训练/待验证设计**，先核实 V9 权重和特征访问，不能凭 `configs/v9_temporal_components.json` 视为已就绪。
3. **退化/负对照**：ResNet18 + TVSum（R91），已经得到 33.87；可用于速度和迁移失败分析，不应原样作为主路线。
4. **条件研究**：DINOv3 的新排序目标仅在独立证据出现后再考虑；时间 CLS+TCN 的当前配方已经失败。

召回层的目标是 **high recall at budget K**，不是单独追求排序准确率；但比赛没有公开原始帧 GT，因此真正的 strict `Recall@K` **只能在具备原始帧真值的合法内部数据**上计算。在 PHD2/YouTube Highlights 等区间/投票标签上只能报告其各自的代理召回，不能伪称官方精确帧召回。

---

## 3. Valen：外部开源依赖及技术边界

### 3.1 已核对的开源入口

- 源码：[Valen-Team/Valen](https://github.com/Valen-Team/Valen)；Apache-2.0 **代码**许可；基础模型、数据集各自遵守独立许可。
- 权重：[Valen-2B](https://huggingface.co/Valen-Team/Valen-2B)、[Valen-0.8B](https://huggingface.co/Valen-Team/Valen-0.8B)。当前均声称支持 text/image/video 及 `Choice`、`Noul`、`Score`。
- 核心文档：[模型 README](https://github.com/Valen-Team/Valen/blob/main/README.md)、[中文数据格式](https://github.com/Valen-Team/Valen/blob/main/docs/data-format_zh.md)、[训练指导](https://github.com/Valen-Team/Valen/blob/main/valen/training/README.md)、[快速本地推理](https://github.com/Valen-Team/Valen/blob/main/cookbook/local_inference_zh.ipynb)。
- 代码依赖 Python 3.10+；官方 GPU 安装示例使用 PyTorch CUDA、`transformers`、`av`、可选 Flash Attention。当前**没有核实**华为 Ascend 910A/NPU 上可用。先在 CUDA 环境 smoke，不把 NPU 支持当作事实。
- 官方作者在 VisualDecisionBench-Video 上报告 Valen-2B 约 `82.85%`，Valen-0.8B 约 `79.03%`；**该基准不是 AIC 高光帧精确匹配或裁剪 IoU。**
- 读取权重使用 `trust_remote_code=True`；必须固定 revision、核查源码与许可证并记 SHA。模型卡说明 `dtype="auto"` 保持模型保存的 FP32 参数形式，在 BF16 autocast 中运行 backbone，决策 Mixer 保持 FP32；不要未经一致性测试强行全部 `.half()`。

### 3.2 官方接口，不要自行编造生成式 JSON API

以下是 Valen README 的**文本 smoke 入口**；模型尚未在本项目环境执行：

```python
import torch
from transformers import AutoModel

model = AutoModel.from_pretrained(
    "Valen-Team/Valen-2B",
    trust_remote_code=True,
    dtype="auto",
    attn_implementation="sdpa",
).to("cuda").eval()
torch.set_float32_matmul_precision("highest")
torch.backends.cudnn.allow_tf32 = False

result = model.predict({
    "state": "A cat is on the sofa.",
    "questions": {
        "animal": {
            "type": "choice",
            "instructions": "Which animal is present?",
            "criteria": {"cat": "A cat", "dog": "A dog"},
        },
    },
}, execution="shared_state")
print(result)
```

图像/视频输入需要使用 Valen **官方 schema**：`request.state` 设为 `{"messages": [{"role":"user", "content": [...] }]}`，媒体条目为 `image_url` 或 `video_url`，媒体路径本地存在。`request.questions` 使用小写 `type: choice` 和 `criteria: {id: description}`。训练 JSONL 每行一条，`targets` 是 `question_id -> {probabilities: {candidate_id: p}}`。目标概率必须非负、有限、并求和为 1。

**不要**直接假设 `model.predict` 返回标准 OpenAI chat-completion JSON、绝对正确概率或可以输入 URL 自动拉视频。Agent 应先打印官方 smoke 的真实输出结构，再实现本地 `ValenDecisionAdapter` 将其转成项目内格式。

### 3.3 建议项目内部接口（新接口，不冒称 Valen 官方 API）

```python
# 这是给 AIC 的建议适配层契约，不是 Valen 官方函数签名。
from dataclasses import dataclass
from typing import Protocol

@dataclass
class DecisionResult:
    question_id: str
    candidate_probs: dict[str, float]  # 顺序和候选身份必须保留
    selected_id: str
    raw_response: object            # 复现/调试，不写入正式提交
    model_revision: str

class DecisionBackend(Protocol):
    def decide(self, state: object, question: dict) -> DecisionResult: ...
```

正式输出必须从**经过验证的 candidate ID**映射到窗口或 `[x,y,w]`；禁止根据模型自由生成文本解析 `frame` 或浮点框。可使用 `Choice` 硬决策，或在公共验证集经过校准后使用 `candidate_probs` 做混合排序。必须记录候选顺序、随机种子、输入媒体摘要、模型 revision、概率、推理耗时。

---

## 4. 新目标架构：两次决策，但共享一套合法输入输出协议

```text
原视频 + 原始 frame/PTS 映射 + targetRatioWH
  |
  +-- A. 时间候选生成（复用冻结位置窗口/可用的轻量 TCN）
  |       -> 候选窗口 + 逐帧索引映射 + 预算 K
  |       -> Valen Choice：left / keep / right [后续可拓展]
  |       -> 确定性窗口解码和原始帧集合
  |              [第一阶段不允许任意改 K]
  |
  +-- B. 裁剪候选生成（复用现有 129 个合法框）
          -> 轻量候选排序 -> Top-K 图片/框编号
          -> Valen Choice：选择最佳 candidate_id
          -> candidate_id -> 原图像素 [x,y,w]
  |
  +-- 校验 frame 唯一、有序；坐标合法；参数计数合规
          -> predictions JSONL -> ZIP -> 独立格式检查
```

### 4.1 A：时间决策——先判断窗口位置，不要强制逐帧 Jev

AIC 现役高分父链更接近“中心锚定连续窗口，利用事件信号微调位置”，而不是仅保留 5 个精彩帧。R54 对 R26 的复盘显示，其时间选择几乎都是连续窗口。因此第一阶段的动作空间为**同预算的窗口策略**：

- `left`：保留 `K` 个原始帧，窗口向前平移 `delta`；
- `keep`：保持冻结父窗口；
- `right`：保留 `K` 个原始帧，窗口向后平移 `delta`。

`delta` 是显式配置的**待验证超参数**，不是来自模型预测的自由坐标；首先固定一种预注册值与候选数量，不以正式平台回分搜索。窗口越界时，执行明确的确定性回退。对首轮所有候选保持相同 `K` 和同一空间父框，以便识别时间决策本身的贡献。

视频输入应明确给出每个候选窗口的起止原始帧、预算 K 和位移 delta，并提供能看见候选差异的左右边界镜头/图像；不能仅给视频和“前移/后移”几个字。Valen 默认视频抽样 16 帧，这**不能**自动保证定位精确帧。边界精排仅在出现合法原始帧级监督和高召回候选后启动；不凭 1 FPS 或 4 FPS 代理声称精确原帧定位成功。

**第二阶段（条件）**：只有明确存在“无高光”真值，才增加 `NONE` 或 `Noul`。只有窗口位置任务达到门槛且存在可靠预算监督，才研究 `shrink/expand` 或自适应 K；不能一次改变窗口、预算、采样率和裁剪框。

### 4.2 B：空间决策——129 候选 -> Top-K -> Valen

保留当前原生裁剪候选生产、targetRatioWH、像素几何和原图坐标回映射。流程为：

1. 在同一选中帧上构造 129 个合法候选框。
2. 用现有候选头或独立训练的**轻量排序头**得到前 K 个候选，研究阶段先以 `K=4` 和 `K=8` 做预算对照；验证两种 K 下的 Oracle 覆盖损失。
3. 对每个候选生成裁剪图片/带框原图，并保持一份 sidecar `candidate_id -> [x,y,w]`。媒体/选项必须一一对应。
4. 让 Valen 对这 4–8 个候选执行 `Choice`。禁止由 Valen 直接生成 `[x,y,w]`。
5. 对同一个源的合法原生 crop GT 计算所选框 IoU，与中心框、R70/已有空间头、Top-K oracle 对比。

推荐第一阶段**仅评测样本级候选排序**，不直接对正式 426 个视频约 199,037 个历史输出框逐一运行 Valen。成本必须实测，并计算整套系统的总吞吐量、视频级延迟和显存。若 Valen 只改善难例，应考虑训练小排序头蒸馏其选择，但**蒸馏也必须仅用合规公开训练素材**。

### 4.3 关于视觉编码器共享

“V9 vision-only + TCN”是优先尝试的**轻量召回工程组合**，不是已经验证的完整时间高召回模型：

- 历史 V9 vision-only 空间模块约 87.4M 参数，现有配置同时记录 V9 PHD2 TCN 头约 246k 参数；是否拥有完全一致可加载的视觉 checkpoint / 预处理 / feature cache，需先审计。
- `Valen-2B` 具有自己的多模态视觉路径。**不要假设 Valen 和 V9 共享骨干权重**；不共享时，全部参数各计一次。
- 如果 V9 权重/前处理缺失，先使用确定性中心窗口作为候选生成对照。不要用 DINOv3 v1 的负结果“补成已验证 V9 checkpoint”。
- 任何部署优化（共享视觉、TensorRT、ONNX、量化、NPU 移植）均需保持输出一致性并单独测试；模型卡的自动通用 vLLM/SGLang 示例不能代替 Valen `Choice` 接口可用性的测试。

---

## 5. 数据、标签与训练协议

### 5.1 真实可用的数据源及用途

| 数据或素材 | 可用监督语义 | 用法 | 不能假设 |
| --- | --- | --- | --- |
| PHD2 / PHD-GIFs | 人类选择片段、user/vote 信息；已有 5,886 干净片段 | 时间窗口偏好、候选召回/排序代理 | 未选片段并非可靠负样本，不是原始帧精确 GT |
| YouTube Highlights | MTurk 选择/投票；现存可解码媒体需逐一核对 | 第二时间行为监督；源/域外检验 | 自动 match 标签不能不经审计就当高光 GT |
| RetargetVid / LIVE native crop | 公开 native crop 框/几何标签 | 空间候选 IoU、软目标、Oracle 分析 | 不是与 PHD2 配对的官方时间空间联合 GT |
| R42 内部公开素材人工标注 | 540 份结构化标注、12 公开源，存在重复与一致性问题 | 定性错误归因、辅助诊断 | 非独立高可靠多人真值；不能作为官方 GT |
| AIKU | 1024D 预提取序列、摘要评分/二值标签 | 条件式时序 trunk 迁移研究 | 无原始视频及公开特征提取器，不能直接部署 |
| TVSum | 连续摘要重要性分数 | 负对照和跨任务研究 | R91 证明当前直接迁移方案表现差 |
| 官方测试视频/提交记录 | **无公开 GT** | 合规冻结推理、格式检查与有限平台反馈 | 禁止用于训练、伪标注、逐样本调参 |

相关审计：[`reports/dinov3_v1/supervision_source_audit_20261020.md`](https://github.com/lujunlin-cn/AIC/blob/main/reports/dinov3_v1/supervision_source_audit_20261020.md)。所有 split 必须在**原始来源视频层级**隔离，不能把相邻片段或同视频不同画幅分进 train/val；检查 URL/source/media SHA 泄漏。

### 5.2 构造 Valen 时间 Choice 训练样本

对一条拥有可用行为标签的公开视频：

1. 固定原始视频和候选窗口集合，在 `left/keep/right` 候选上计算有标签证据的覆盖或加权得分。
2. 根据已有投票强度或偏好构造 `targets.question_id.probabilities`；未观察区间用 mask，不直接写成 0。
3. 以 `source_video_id` 为 `group_id` 隔离 train/val/test；`request.state` 的多媒体仅包含推理时真实可取得的可观测内容。
4. 记录原始窗口 `[start_frame,end_frame)`、`delta`、预算 `K`、label source、video hash、frames/PTS 映射到**独立元数据/manifest**；绝不把最优候选或 GT IoU 放进输入提示。

以下是**无标签推理记录**的字段示意，不代表已生成真实训练样本：

```json
{
  "group_id": "PUBLIC_SOURCE_ID",
  "request": {
    "state": {
      "messages": [{
        "role": "user",
        "content": [
          {"type": "text", "text": "示例视频 N=300 帧，K=255，delta=10 帧。left=[12,267)，keep=[22,277)，right=[32,287)。请比较相同预算的三个窗口；真实部署还需提供窗口边界的镜头证据。"},
          {"type": "video_url", "video_url": {"url": "assets/clip.mp4"}}
        ]
      }]
    },
    "questions": {
      "window_choice": {
        "type": "choice",
        "instructions": "哪一个候选窗口更适合本视频的高光选择？",
        "criteria": {
          "left": "向前平移相同帧数的连续窗口",
          "keep": "保留当前连续窗口",
          "right": "向后平移相同帧数的连续窗口"
        }
      }
    }
  },
  "meta": {"domain": "public_highlight", "modality": "video"}
}
```

上例的 N、K、delta 和三个窗口区间仅展示合法候选的编码方式，不对应真实视频标签。训练时新增 `targets.window_choice.probabilities`，其值**必须根据真实可用的公开标签计算**，不能使用这个示意文本编造监督。`assets/clip.mp4` 是相对 JSONL 所在目录的真实文件路径，必须由 Agent 在数据预处理阶段生成；不能给 Valen 传任意 HTTP URL。

### 5.3 构造 Valen 空间 Choice 训练样本

为同一真实帧构造编号 `A/B/C/D...` 的裁剪图。`state.messages[].content` 按编号放文本说明和对应本地 `image_url`；`criteria` 的 key 与编号严格一致。每个候选具有 sidecar `candidate_id -> original_[x,y,w]`。公开原生裁剪 GT `b_gt` 存于 label 数据中，**不进入 state 文本或媒体**。

对候选框 `b_i`，建议使用以下**研究性软标签**：

\[
y_i = \frac{\exp(\operatorname{IoU}(b_i,b_{gt})/\tau)}
{\sum_{j=1}^{K}\exp(\operatorname{IoU}(b_j,b_{gt})/\tau)}.
\]

`tau>0` 是待预注册/仅在训练验证源上设置的温度。对同一来源存在多个合法真值框时，必须先遵从该数据集既有标注/匹配约定，不能未经说明直接平均。首轮训练优先使用 Valen 自带的 SFT/LoRA 分类损失；只有 SFT 与候选目标确实成立后，再考虑论文中提出的 RLCD。不得在没有独立证据时声称 RL 必然优于 SFT。

### 5.4 训练只在通过零样本门后启动

- 必须先对 2B/0.8B 的零样本时间与空间测试做速度、候选偏差、格式、OOM 检查。
- 如需要 LoRA，从已发布 **Valen-2B 完整 checkpoint** 继续训练；不要误把“准备原始 Qwen3.5 base 并预训练 Valen”当成最低成本微调路径。
- 参考官方 [`configs/train/qwen/sft_finetune_lora.json`](https://github.com/Valen-Team/Valen/blob/main/configs/train/qwen/sft_finetune_lora.json)，复制到**单独实验配置**，修改 `model.model_path`、`data.path`、`training.output`、数据长度和 batch/token 预算。
- 官方样例默认 `data/smoke/text.jsonl` 且 `max_steps=100`，仅是 smoke；换成多模态正式数据前不得宣称已经训练图像/视频任务。
- 用公开 train/val 选择温度、LoRA 超参数与 stopping；独立 source-held-out 测试用于最终一次验收。保存所有对比 seed 和模型哈希。

---

## 6. 实验矩阵、评价与验收门

**原则：空间先于时间；零样本先于微调；公开代理先于官方单变量包。**

| ID | 只改动的变量 | 冻结内容 | 首要指标 | 条件/停止 |
| --- | --- | --- | --- | --- |
| Z0 | 验证 Valen 原始 Choice + 真实视频/多图输入 | 无正式 AIC 改动 | 成功率、logits/概率、延迟、显存、候选顺序敏感性 | 失败即修兼容，不训练 |
| S0 | `Valen-2B` 零样本选 Top-K crop | 同帧、同 129 候选、同 GT | source-macro IoU、oracle gap、耗时 | 若低于中心/已有空间头，先作错误分析 |
| S0-small | `Valen-0.8B` 同样本对照 | 同 S0 | 效果/吞吐比 | 只有 2B 超过才考虑付更高成本 |
| S1 | 空间 Valen-2B LoRA SFT | 同一候选池、切分和几何 | source-macro IoU，分域/分比例 | 只允许合规 native crop 监督 |
| T0 | 冻结确定性窗口候选 + Valen 的 left/keep/right | 空间框、K、输入预算、源划分 | 行为标签代理 source-macro F、相对位置基线差 | 未超过位置基线不进入进一步训练 |
| T1 | 学习型候选召回：可用 V9 特征 + TCN | Valen、候选动作、所有评价条件 | 固定 K 的公开代理召回/F、延迟 | 若权重/数据不完整，不伪造可运行路径 |
| T2 | Valen 时间 SFT（条件） | 候选召回与空间模块 | source-held-out 行为代理、选项一致性 | 必须先取得可靠窗口偏好标注 |
| J0 | 两个已经独立晋级的模块合用 | 固定模型 checkpoint、预算、校验链 | 有合法联合 GT 时算联合 F | **没有联合 GT 不填“官方预测分”** |

### 6.1 时间指标要有正确边界

- 若存在**原始帧级** GT：在相同 `K` 比较 strict `Recall@K = |GT_frames ∩ candidate_frames| / |GT_frames|`、帧级 Precision/Recall、window coverage、每视频 macro F1。明确零 GT 视频的单独统计规则。
- 若只有 PHD2/YouTube Highlights 的片段/投票监督：报告所用时间网格、投票规则、mask、跨域 source-macro 代理 F/AP，以及与 position-only 的差异；**不能称为 AIC 官方时间 F1**。
- 重点检查 Valen 是否只是学习“中心位置”，需要 `position-only`、源内时间顺序置换、候选顺序循环置换 3 个控制。相邻镜头不能跨不同原视频泄漏。

### 6.2 空间指标必须包含候选上限

- 对每帧统计：center IoU、已有空间头 IoU、Valen top-1 IoU、Top-K oracle IoU、129-window oracle IoU。
- 对每个 source 先平均，再 macro 平均；RetargetVid 与 LIVE 分域统计。报告 **source-level paired bootstrap 95% CI** 和失败视频清单。
- 如果 Top-K oracle 相比 129 oracle 大幅下降，则先修召回/Top-K，不是微调 Valen；如果 oracle 足够高但 Valen IoU 差，才进入 SFT 排序。
- 研究性**晋级门**（非历史实验结果）：同一 source split 下，相对强控制 source-macro IoU `>= +0.005`，配对 bootstrap 95% CI 下界 `>0`，RetargetVid/LIVE 两域不出现方向相反的显著下降，并经过候选顺序置换稳定性检查。未通过则停在研究分支，不制作新官方包。

### 6.3 概率、算力与总参数

- 对 `candidate_probs` 做求和、有限性、候选 ID 完整性断言；仅在独立验证集上做温度校准，并报告 Brier/ECE。**Softmax 候选概率不是“80% 的客观正确率”。**
- 实测每个决策 state 的 GPU 耗时（含 decode/processor）、峰值显存、候选数 K、输入视觉 token/视频帧数、批处理吞吐量。根据**实际要处理的帧数**外推正式 426 视频成本，外推量标为“估计”。
- 参数清单对每个**实际加载的唯一模型**使用 `sum(p.numel() for p in model.parameters())`；注意 LoRA 合并、共享权重、附属视觉模型/检测器。输出单一 manifest 统计；总量必须 `<= 9B`（实现判断表达式使用 `<= 9_000_000_000`）。不能把多次加载的共享模型双重计数，也不能漏掉独立视觉编码器。
- 比较时同一父时间集/空间框必须固定，严禁模型大小、分辨率、输入 fps、预算 K、点源、框源一并变动后将分数归因给 Valen。

### 6.4 现行评分回归测试（与旧版本隔离）

添加 current-rule evaluator 的**单元测试**，覆盖：

1. `video_id + frame` 精确匹配、错 1 帧不产生 IoU。
2. 连续 IoU 的极值与非完全相交情形；不是 IoU 大于阈值就按 1 计。
3. 预测空、GT 空、两者都空的三种条件。
4. 同一视频重复帧/无序帧、越界框、非数值和 NaN/Inf。
5. `targetRatioWH=[16,9]`、`[9,16]` 的合法回映射与高的计算。
6. video-macro 与 frame-micro 不一致时，只允许 video-macro 符合官方定义。
7. `model_params_b` 与 9B 实际参数限制；保留历史 `model_size_mb` checker 的旧模式以便重现过去包。
8. JSONL video_id 全覆盖、ZIP 解压往返 SHA、一致的原始视频帧索引。

说明：这是依据公开公式的本地重建，**未取得已验证的同版本官方可执行 evaluator**，不得把通过本地测试等同于平台验收。

---

## 7. Agent 按顺序执行：不要同时启动多个未知分支

### Phase 0：只读审计（必做）

1. 拉取/打开当前 AIC 仓库；记录 HEAD 和 `git status`。不要覆盖现有用户改动。
2. 阅读 `STATUS.md`、`DECISIONS.md`、`EXPERIMENTS.md`、规则文件和最新 `reports/dinov3_v1/20261020_dinov3_probe_results.md`。
3. 检查 V9、R70、R54、DINOv3 相关缓存/权重/运行脚本实际路径，并校验 SHA；缺失则标记 BLOCKED，不编造。
4. 检查 GPU、依赖、可用显存和 Valen 官方安装要求。隔离环境，避免破坏现有 AIC Python/NPU 环境。
5. 生成 `reports/valen_route/asset_audit.md`，指出可复用入口、缺项和执行路径。

### Phase 1：Valen 零样本可运行性

1. 在隔离 CUDA 环境安装 Valen 依赖，使用官方文本 smoke 验证 `model.predict`。
2. 固定 Valen-2B/0.8B 的模型 revision 和文件 SHA；不允许隐式升级最新版本。
3. 使用**合法公开视频**执行 1 个 `video_url` `Choice`；使用同一合法公开帧的 4 个真实裁剪图片执行 1 个多图 `Choice`。
4. 打印并归档 Valen 原始响应 schema，封装 `DecisionBackend`；证明候选 ID 顺序映射无误。
5. 记录模型真实参数量、CPU/GPU/RAM/显存、视频采样和单样本耗时；OOM 则调整图片/视频预算，而不是隐瞒失败。

### Phase 2：空间 S0（优先）

1. 对齐原生 crop 数据、129 候选和 source-disjoint train/dev/test；检查无 GT 泄漏。
2. 比较 center、既有候选头、Top-K oracle、Valen-2B、Valen-0.8B。
3. 分别统计 RetargetVid/LIVE、横竖屏、候选顺序置换敏感性和推理耗时。
4. 如果 S0 失败，分析失败原因；仅在有充分数据监督且候选 oracle 足够高时再启动 S1 LoRA。
5. S0/S1 未过晋级门，不制作官方替换包，不假称提升。

### Phase 3：时间 T0/T1（条件）

1. 先冻结中心/父连续窗口，确认 `K`、`delta`、左右位移合法且原始帧映射正确。
2. 使用 PHD2/YouTube Highlights 的可观察行为监督构建窗口偏好，不把未标区间写成负例。
3. 先比较位置规则与 Valen 零样本；只有可辨识收益才进一步训练。
4. V9+TCN 分支先核验真正可加载的模型和一致的特征提取协议。不存在时使用 position-only，不从零重新造视觉引擎。
5. 仅当 T0/T1 有可靠 source-held-out 结果时启动 T2 和精确边界实验。

### Phase 4：打包（只有独立过门才执行）

1. 固定父提交包和对应 manifest。空间包只换 `bboxes`，所有 `frame` 和每视频预算一致；时间包只换 `frame` 集合，公共帧的原有 `bboxes` 保持一致。
2. 对所有新增帧，必须通过已冻结空间模块实际推理取得合法框；不能从相邻帧直接复制未验证的框。
3. 核算唯一模型总参数量 `<=9B`，正式新包使用最新 `model_params_b` 契约，留存独立文件与 SHA。
4. 通过 current-rule contract、独立 checker、ZIP round-trip 与视频覆盖检查后，才产生候选提交包。
5. 禁止用平台回分逐样本反推出 GT。外部平台分数只能作为固定方法的一次整体反馈，不用来寻找隐藏测试标签。

### Phase 5：研究结论与回退

每一个实验独立产出：`config.json`、模型与代码 SHA、数据清单/来源许可、source split、metrics、bootstrap CI、时间/显存、预测分布、错误案例、是否晋级、回退理由。没有跑通的实验标记 `NOT_RUN` 或 `BLOCKED`，没有官方回分时标记 `OFFICIAL_SCORE_UNAVAILABLE`。

---

## 8. 目录建议与最低交付物

仅在 Phase 0 审计确认现有实现没有同功能入口时才新增以下文件。路径为**建议**，不是声称仓库已有：

```text
aic/
  valen_adapter.py                 # Valen 原生输出 -> DecisionResult
  highlight_decisions.py           # 候选窗口/候选框 ID 映射，纯策略
  current_score.py                 # 现行公式；与历史 evaluator 分离
scripts/
  valen_smoke.py                   # text/video/multi-image
  valen_spatial_probe.py           # S0 + S0-small
  valen_temporal_probe.py          # T0（有合法时间代理后）
  valen_param_audit.py             # 实际 unique-param manifest
configs/
  VALEN_HIGHLIGHT_S0.json          # 预注册候选预算与输入尺寸
reports/
  valen_route/
    asset_audit.md
    zero_shot_metrics.json
    spatial_source_metrics.json
    temporal_proxy_metrics.json
    parameter_manifest.json
    final_verdict.md
```

**第一轮的最低交付标准**：

- 可运行 Valen `Choice` 的本地 smoke 脚本与离线样例（仅使用可授权素材）；
- AIC 候选 ID、帧索引和裁剪坐标协议测试通过；
- 至少 1 组公开原生 crop 的 Valen 2B vs 0.8B vs center/当前空间头结果，包含样本量与 source 分组；
- 实际参数总量、GPU 峰值显存、单 state 耗时；
- 一份不夸大结论的 `final_verdict.md`。若 Phase 1 因权重、环境或 GPU 阻塞，交付完整的 blocker 证据，不伪造运行结果。

**明确不要求**：一开始重训练 DINOv3、重新执行 E0–E6、复刻 TimeLens2 全链、盲目训练 Valen RLCD、直接提交没有公开域外证据的完整替换包、承诺涨到 40 分。

---

## 9. 决策原则与研究优先级

**P0（优先且便于归因）**：Valen-2B 在 129 个合法框中的 Top-K 再排序；对比 Valen-0.8B；测真实 IoU 与耗时。空间方向同时有公开原生 GT、显著候选 oracle gap 和 R70 历史正向线索，数据/评价对齐程度更高。

**P1（条件启动）**：Valen 在冻结窗口集合上做 left/keep/right 时间选择。当前官方时间强基线是中心锚定连续窗口，R91 轻量 ResNet18/TVSum 直接替换失败，DINOv3 CLS+TCN 首轮低于位置控制；不要假设另一轻量编码器自然突破。

**P2（仅在训练监督成立时）**：Valen-2B LoRA/SFT，随后研究用 Valen 作为教师蒸馏到可批量推理的轻量时间/空间头。无可用监督时不启动 RLCD 或对正式测试媒体做伪标签训练。

**P3（探索性）**：DINOv3 pairwise/listwise 空间候选排序；与 Valen 的结果独立评估，不通过就停止。

### 最重要的三个问题

1. **候选覆盖**：轻量时间/空间候选是否已经把目标答案漏掉？先测 oracle 上限再优化决策器。
2. **真实收益**：Valen 是否超越位置/中心/现有排序头，而非只输出一组看起来合理的高置信概率？
3. **可部署性**：Valen 的实际参数、媒体 token 和逐帧调用成本能否在 AIC 的 9B 参数上限及推理预算内运行？

只有这三个问题都有实证答案，才考虑把 Valen 接入正式提交链路。

---

## 10. 主要原始入口（按阅读顺序）

1. [AIC 当前项目状态](https://github.com/lujunlin-cn/AIC/blob/main/STATUS.md)
2. [AIC 赛事规则根目录文件](https://github.com/lujunlin-cn/AIC)——打开 `01_赛事规则与评分标准.md`
3. [DINOv3 2026-10-20 实测结论](https://github.com/lujunlin-cn/AIC/blob/main/reports/dinov3_v1/20261020_dinov3_probe_results.md)
4. [R107/R91/R108/R93 官方回分复盘](https://github.com/lujunlin-cn/AIC/blob/main/reports/r107_r91_r108_r93_official_postmortem_20261020/summary.md)
5. [R72 B3 权重恢复审计](https://github.com/lujunlin-cn/AIC/blob/main/reports/r72_b3_asset_recovery_20261018/README.md)
6. [DINOv3 监督源审计](https://github.com/lujunlin-cn/AIC/blob/main/reports/dinov3_v1/supervision_source_audit_20261020.md)
7. [Valen 官方 README](https://github.com/Valen-Team/Valen/blob/main/README.md)
8. [Valen 数据格式与多媒体字段](https://github.com/Valen-Team/Valen/blob/main/docs/data-format_zh.md)
9. [Valen-2B 模型卡](https://huggingface.co/Valen-Team/Valen-2B)
10. [Valen 训练/LoRA 配置](https://github.com/Valen-Team/Valen/blob/main/configs/train/qwen/sft_finetune_lora.json)

> 交接结束条件：Coding Agent 能区分**已执行实验**、**仅有代码/配置的能力**、**新提出但尚未验证的 Valen 路线**；先建立可复现实验，再决定是否投入正式提交。不得混淆公开代理、历史平台分数与新实验预测。
