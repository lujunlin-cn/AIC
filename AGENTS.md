# AGENTS.md — 2026 AIC 通用视频高光剪辑长期实验协议

## 0. 项目使命

本仓库用于参加：

**2026 AIC「基于视频大模型的通用视频高光剪辑」**

你的角色不是单纯的软件开发 Agent，而是：

> **Research Engineer + Video Understanding Researcher + Competition Optimization Agent**

最终目标是在比赛规则允许范围内，充分利用现有计算资源，持续进行：

- 数据研究与清洗
- baseline 构建
- 模型训练
- 消融实验
- 参数搜索
- 特征工程
- 蒸馏
- 视频时序建模
- 空间重构图
- 推理优化
- 模型压缩
- 官方指标验证
- 错误分析

目标是：

> **最大化最终比赛 Score，而不是追求模型规模、论文新颖性或者单项离线指标。**

允许积极探索不同实现，不需要因为方案“不够优雅”“不是最新架构”而放弃。

但是所有实验必须：

1. 可复现；
2. 有明确假设；
3. 有完整日志；
4. 能与已有 baseline 比较；
5. 符合比赛规则；
6. 单次训练严格小于 12 小时。

---

# 1. 永久 Source of Truth

每次开始新的研究、实现或训练任务前，首先阅读：

```text
01_赛事规则与评分标准.md
02_数据集_模型路线_特征工程研究.md
AGENTS.md
```

优先级：

```text
官方最新规则
    >
01_赛事规则与评分标准.md
    >
AGENTS.md
    >
02_数据集_模型路线_特征工程研究.md
    >
历史实验记录
    >
模型个人判断
```

其中：

## `01_赛事规则与评分标准.md`

负责定义：

- 官方评分方法
- 模型大小限制
- 提交格式
- 数据使用限制
- 推理限制
- 比赛阶段
- 其他硬性规则

如果任何代码、实验方案或本文与 `01` 冲突：

> **以 `01_赛事规则与评分标准.md` 为准。**

---

## `02_数据集_模型路线_特征工程研究.md`

负责定义：

- 数据集候选
- A / B / C 三条主要研究路线
- Feature Bank
- Spatial Reframing
- Teacher Distillation
- 当前研究假设

该文件是技术研究基础，但不是不可修改的真理。

如果实验数据证明其中某项判断错误：

> 允许修改技术路线，但必须留下实验依据。

---

# 2. 本地环境

本机主要负责：

- 编写代码
- Git 管理
- 实验设计
- 分析结果
- 小规模测试
- 保存最终代码和需要保留的模型权重

工作目录：

```bash
/home/hajimi2025/AIC
```

所有本项目代码应位于：

```bash
/home/hajimi2025/AIC
```

不要把项目散落在其他目录。

---

# 3. 远程训练服务器

远程计算服务器：

```text
Host: 119.62.14.24
Port: 22022
User: supie
Authentication: SSH key
```

连接形式：

```bash
ssh -p 22022 supie@119.62.14.24
```

使用系统现有 SSH key / SSH agent。

**禁止：**

- 将 private key 内容写入仓库；
- 将 private key 复制进代码；
- 将认证信息提交 Git；
- 为方便运行而关闭必要的 SSH 安全机制。

---

# 4. 远程目录规范

远程代码：

```bash
/home/supie/AIC
```

远程大容量数据盘：

```bash
/data/aic
```

其中所有大型内容必须优先存入 `/data/aic`。

推荐结构：

```text
/data/aic/
├── datasets/
├── weights/
├── pretrained/
├── features/
├── teacher_labels/
├── checkpoints/
├── experiments/
├── predictions/
├── cache/
└── tmp/
```

例如：

```text
/data/aic/datasets/TimeLens-100K
/data/aic/datasets/QVHighlights
/data/aic/datasets/TVSum
/data/aic/datasets/RetargetVid

/data/aic/pretrained/resnet18
/data/aic/pretrained/vit_small

/data/aic/checkpoints/A1_resnet18_tsm_tunet
```

不要将大型 dataset 和大量 checkpoint 存入：

```bash
/home/supie/AIC
```

`/home/supie/AIC` 应主要保存：

- source code
- configs
- scripts
- small metadata
- experiment manifests
- documentation

---

# 5. 本地与远程代码同步

本地目录：

```bash
/home/hajimi2025/AIC
```

远程目录：

```bash
/home/supie/AIC
```

保持二者代码一致。

推荐使用：

```bash
rsync
```

而不是重复手工复制文件。

同步时禁止意外覆盖：

```text
/data/aic
```

或其他远程数据目录。

大型：

- dataset
- checkpoint
- extracted frames
- cached features

不应该通过普通代码同步流程反复传输。

---

# 6. GPU 资源

服务器有多张 GPU，但本项目允许使用的物理 GPU 为：

```text
GPU 1
GPU 2
GPU 4
GPU 5
GPU 6
GPU 7
```

共：

```text
6 × NVIDIA V100 32GB
```

禁止默认使用：

```text
GPU 0
GPU 3
```

除非用户之后明确授权。

---

# 7. 极其重要：CUDA_VISIBLE_DEVICES 映射

V100 的物理编号不是连续：

```text
1,2,4,5,6,7
```

因此必须非常小心 CUDA logical device mapping。

如果：

```bash
export CUDA_VISIBLE_DEVICES=1,2,4,5,6,7
```

则 PyTorch 内部看到的是：

```text
cuda:0 -> physical GPU 1
cuda:1 -> physical GPU 2
cuda:2 -> physical GPU 4
cuda:3 -> physical GPU 5
cuda:4 -> physical GPU 6
cuda:5 -> physical GPU 7
```

**不要把 physical GPU ID 与 local rank 混用。**

例如正确：

```bash
CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 \
torchrun \
  --standalone \
  --nproc_per_node=6 \
  train.py ...
```

PyTorch：

```text
LOCAL_RANK=0...5
```

不是：

```text
1,2,4,5,6,7
```

---

## 使用两张物理 GPU 4、5

正确：

```bash
CUDA_VISIBLE_DEVICES=4,5 \
torchrun --standalone --nproc_per_node=2 train.py
```

程序内部：

```text
cuda:0
cuda:1
```

---

# 8. 每次训练前必须检查 GPU

禁止盲目启动训练。

首先运行：

```bash
nvidia-smi
```

检查：

- GPU 型号
- GPU 编号
- GPU memory
- 已有进程
- 当前利用率
- 是否有其他任务占用

需要时进一步检查：

```bash
nvidia-smi \
  --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu \
  --format=csv
```

如果目标 GPU 已存在明显训练进程：

> 不要直接杀掉其他人的进程。

重新选择空闲 GPU 或等待适当资源。

---

# 9. GPU 使用原则

调试阶段不要无脑使用 6 卡。

优先：

```text
1 GPU
↓
确认 pipeline 正确
↓
2 GPU
↓
确认 DDP 正确
↓
再扩展到多 GPU
```

推荐并行实验：

```text
GPU 1       -> Experiment A
GPU 2       -> Experiment A ablation

GPU 4,5     -> Experiment B

GPU 6,7     -> Teacher / Feature extraction
```

只有正式大规模训练确实受益时才使用：

```text
6 GPU DDP
```

目标是最大化：

> **experiment throughput**

而不是最大化单个任务 GPU 数量。

---

# 10. 单次训练 12 小时硬限制

这是不可违反的硬约束。

任何单个模型的单次训练：

```text
wall-clock training time < 12 hours
```

必须主动设计训练计划，使正常训练预期：

```text
<= 10.5~11 hours
```

不要设计“理论上刚好 12 小时”的训练。

---

## 所有正式训练必须带超时保护

Linux 环境优先使用类似：

```bash
timeout \
  --signal=TERM \
  --kill-after=3m \
  11h50m \
  <training command>
```

DDP 示例：

```bash
CUDA_VISIBLE_DEVICES=1,2,4,5,6,7 \
timeout \
  --signal=TERM \
  --kill-after=3m \
  11h50m \
  torchrun \
    --standalone \
    --nproc_per_node=6 \
    train.py \
    --config configs/A1.yaml
```

同时必须定期保存 checkpoint。

建议：

```text
每 15~30 min
或
每个 epoch
```

保存一次 recoverable checkpoint。

---

# 11. 12 小时不是必须跑满

如果模型：

- validation 不再提高；
- 明显过拟合；
- loss 异常；
- 指标远低于 baseline；
- 出现严重实现错误；

应尽早停止。

禁止为了“把 12 小时用完”继续无意义训练。

---

# 12. 比赛优化目标

永远优化：

\[
Score_{competition}
\]

而不是只优化：

\[
Loss
\]

或：

\[
Raw\ F1
\]

如果当前比赛规则采用：

\[
Score = F_{video}\times k_{size}
\]

则所有模型选择必须考虑：

```text
raw metric
+
model-size coefficient
```

---

# 13. 模型大小是一级指标

每次正式实验都必须记录：

```text
Parameters
Checkpoint Size
Inference Weight Size
FP32 Size
FP16 Size
INT8 Size（如适用）
Competition Size Tier
Competition Size Coefficient
```

模型必须符合比赛最大权重限制。

当前具体阈值始终以：

```text
01_赛事规则与评分标准.md
```

为准。

如果当前规则仍为：

```text
<= 100 MB       -> coefficient 1.00
100~500 MB      -> coefficient 0.95
500 MB~9 GB     -> coefficient 0.90
> 9 GB          -> invalid
```

则优先探索：

```text
<=100 MB
```

其次：

```text
100~500 MB
```

进入：

```text
>500 MB
```

必须有实验证据证明原始指标提升足以覆盖 size penalty。

---

# 14. 不允许只看参数量估算模型大小

必须实际导出比赛推理使用的权重文件并检查。

例如：

```bash
du -h model.pt
ls -lh model.pt
```

如果最终系统同时加载：

```text
backbone
detector
tracker
OCR model
adapter
LoRA
spatial network
auxiliary network
```

则要按比赛规则计算总权重。

不能只报告主 backbone。

---

# 15. 核心任务定义

比赛核心任务分为两个部分。

## Task A — Temporal Highlight Detection

目标：

```text
Video
↓
Frame / Clip Importance
↓
Highlight Frames / Segments
```

---

## Task B — Spatial Reframing

目标：

```text
Highlight Frame
↓
Subject / Saliency
↓
Crop Window
↓
[x,y,w]
```

整体不是普通视频分类：

```text
Temporal Selection
+
Spatial Reframing
```

---

# 16. 三条主实验路线

除非实验给出非常明确证据，否则保持以下研究主线。

---

# Experiment A

## CNN + Temporal U-Net

核心：

```text
Video
↓
Frame Sampling
↓
ResNet18 / ResNet34
↓
TSM / lightweight temporal module
↓
1D Temporal U-Net
↓
Highlight Probability
```

Spatial：

```text
Visual Features
↓
Subject / Saliency
↓
Tracking
↓
Crop Optimization
↓
Path Smoothing
```

优先目标：

```text
<=100 MB
```

A 是整个项目的：

> Safe Baseline。

任何高级方案都必须与 A 比较。

---

# Experiment B

## Small ViT + Temporal U-Net

核心：

```text
Video
↓
Small ViT
↓
Frame Embeddings
↓
Temporal U-Net / TCN
↓
Highlight Probability
```

重点考察：

> 更强视觉语义是否值得额外参数量和计算开销。

优先：

```text
ViT Tiny / Small
TinyViT
MobileViT
EfficientFormer
其他小型视觉 backbone
```

不要因为“大 ViT 看起来先进”就默认使用大模型。

---

# Experiment C

## Large Teacher → Small Student Distillation

大型模型主要用于：

```text
offline teacher
```

而不是最终比赛推理。

流程：

```text
Video
↓
Frozen VLM / Video Teacher
↓
Structured Soft Labels
↓
Student Training
↓
Small Competition Model
```

Teacher 可产生：

```text
highlight probability
semantic importance
action importance
subject clarity
composition
quality
emotion
event completeness
audio-visual importance
```

最终比赛推理应尽可能只加载：

```text
Student
```

---

# 17. 推荐实验顺序

默认实验顺序：

```text
A0
↓
A1
↓
A2
↓
A3
↓
B0
↓
B1
↓
A vs B
↓
选择最优 Student
↓
C0
↓
C1
↓
C2
```

含义：

```text
A0 = ResNet18 + Temporal U-Net

A1 = A0 + TSM

A2 = A1 + Feature Bank

A3 = A2 + Spatial Tracking / Smoothing

B0 = Small ViT + Temporal U-Net

B1 = B0 + Feature Bank

C0 = Best A + Teacher soft labels

C1 = Best B + Teacher soft labels

C2 = Multi-task Distillation
```

可以插入额外消融，但不要丢失清晰的实验谱系。

---

# 18. Feature Engineering 是核心研究方向

本比赛不应只依赖 neural network。

优先研究：

## Motion

```text
optical flow
motion magnitude
motion acceleration
camera motion
scene change
shot boundary
```

## Subject

```text
face
person
object
subject persistence
subject area
subject position
trajectory
occlusion
```

## Quality

```text
blur
sharpness
exposure
noise
compression
camera shake
```

## Audio

```text
energy
onset
beat
speech
applause
cheering
music climax
silence
```

## Semantic

```text
visual embedding
action
event
scene
OCR
human-object interaction
```

## Composition

```text
saliency center
face center
rule of thirds
headroom
lead room
edge clipping
subtitle preservation
logo preservation
object completeness
```

---

# 19. 优先使用 Zero / Low Parameter Enhancement

因为比赛具有模型大小惩罚，应高度重视不增加 neural weights 的算法：

```text
OpenCV
optical flow
histogram
Laplacian
FFT
shot detection
Kalman Filter
EMA
Dynamic Programming
Temporal NMS
hysteresis
gap filling
crop path optimization
motion smoothing
```

如果：

```text
+0 MB
```

的方法能够提升：

```text
+0.5% ~ +2%
```

官方指标，

其价值可能大于显著扩大模型。

---

# 20. 视频数据 IO 是一级工程问题

6×V100 不应该长时间等待视频解码。

监控：

```text
GPU utilization
CPU utilization
I/O throughput
DataLoader wait
```

如果 GPU 利用率异常低，优先检查：

```text
video decode
disk I/O
augmentation
DataLoader workers
frame extraction
```

---

# 21. 允许离线缓存

对于稳定 backbone，可以预计算：

```text
video
↓
frames
↓
backbone
↓
[T,D] feature
```

保存：

```bash
/data/aic/features/
```

之后 Temporal U-Net 实验直接加载 feature。

优点：

- 极大提高 ablation 速度；
- 避免重复视频解码；
- 避免重复 backbone forward；
- 可以快速搜索 temporal architecture。

---

# 22. 数据集策略

不要简单把所有数据 concat。

每个 dataset 必须回答：

```text
它在教模型什么能力？
```

例如：

```text
TimeLens          -> temporal grounding
QVHighlights      -> highlight / saliency
TVSum / SumMe     -> generic importance
YouTube Highlights-> action highlight
LIVE-YT Crop      -> reframing
RetargetVid       -> human cropping
LaSOT             -> tracking
LSVQ              -> video quality
```

如果 dataset 无法明确贡献某种能力：

> 不要因为“可能有用”就下载和训练。

---

# 23. 防止 Data Leakage

所有训练、验证和测试数据必须追踪来源。

建立：

```text
dataset manifest
video id
source
split
hash if practical
```

特别注意：

- 相同 YouTube 视频可能出现在多个公开 dataset；
- 同一原视频可能被不同 benchmark 二次剪辑；
- teacher 数据不应污染 validation/test。

---

# 24. 实验必须有明确 Hypothesis

禁止：

```text
试试看这个会不会更好
```

每个实验开始前必须写：

```text
Hypothesis:
Change:
Expected effect:
Risk:
Control:
```

例如：

```text
Hypothesis:
TSM 可以改善短时动作建模，同时几乎不增加模型参数。

Change:
A0 -> A1 only adds TSM.

Expected:
Temporal F1 +1~3%.

Risk:
Long-range semantic event unchanged.

Control:
Same dataset / seed / epochs / backbone.
```

---

# 25. 一次实验尽量只改变一个主要变量

优先：

```text
A0
vs
A1
```

而不是：

```text
ResNet18 + BCE + 16 frames
```

直接对：

```text
ViT + focal loss + 64 frames + different augmentation
```

否则实验结果无法解释。

---

# 26. 每个实验必须记录

建议维护：

```text
experiments/experiment_registry.csv
```

或：

```text
experiments/experiment_registry.jsonl
```

至少记录：

```text
experiment_id
date
git_commit
host
gpu_ids
config
dataset
dataset_version
seed
architecture
parameter_count
weight_size
precision
batch_size
learning_rate
optimizer
scheduler
training_time
best_checkpoint
temporal metrics
spatial metrics
competition metric
size coefficient
final estimated score
notes
status
```

---

# 27. Experiment ID

建议：

```text
A0_001
A0_002
A1_001
A2_001
B0_001
C0_001
```

禁止：

```text
test1
new
final
final2
final_new
```

---

# 28. 每次运行必须保存完整 config

所有训练参数进入：

```text
configs/
```

例如：

```text
configs/
├── A0_resnet18_tunet.yaml
├── A1_resnet18_tsm_tunet.yaml
├── B0_vits_tunet.yaml
└── C0_distill.yaml
```

禁止关键参数只存在 shell history。

---

# 29. 日志目录

所有远程实验输出：

```bash
/data/aic/experiments/<experiment_id>/
```

例如：

```text
/data/aic/experiments/A1_004/
├── config.yaml
├── command.txt
├── git_commit.txt
├── environment.txt
├── train.log
├── metrics.json
├── best.pt
├── last.pt
├── predictions/
└── summary.md
```

---

# 30. 每个正式实验必须记录启动命令

例如：

```text
command.txt
```

内容必须允许未来直接复现。

---

# 31. 保存环境

正式结果至少记录：

```bash
python --version
pip freeze
nvidia-smi
git rev-parse HEAD
```

必要时记录：

```bash
nvcc --version
```

---

# 32. Validation 不能只看总分

至少分析：

## Temporal

```text
Precision
Recall
F1
segment IoU
boundary error
mAP if applicable
```

## Spatial

```text
bbox IoU
center error
crop stability
crop velocity
crop acceleration
crop jerk
```

## System

```text
weight size
VRAM
latency
FPS
training time
```

---

# 33. 按视频类别做 Error Analysis

比赛测试集包含不同 domain。

验证时尽量分：

```text
sports
outdoor
performance
travel
vlog
classroom
meeting
media
screen recording
fixed camera
first-person
```

避免：

```text
overall F1 提升
```

掩盖某个重要场景完全失效。

---

# 34. 保留 Failure Cases

每个重要模型至少保存：

```text
Top false positive
Top false negative
Worst temporal boundary
Worst crop IoU
Worst crop jitter
```

最好生成可视化视频或 contact sheet。

错误样本通常比继续随机调参更有价值。

---

# 35. Checkpoint 策略

不需要保存每个 epoch 的全部模型。

至少保留：

```text
best
last
```

重要 milestone 可以额外保存。

旧实验大量 checkpoint 应及时清理，避免 `/data/aic` 被无意义占满。

但删除前必须确认：

- best checkpoint 已保留；
- experiment metadata 已保存；
- 不再需要复现。

---

# 36. 不允许破坏性清理

未经确认不得：

```bash
rm -rf /data/aic
rm -rf /home/supie/AIC
```

禁止广泛 wildcard 删除。

清理时必须针对明确目录。

---

# 37. Git 工作原则

重要改变必须可追踪。

在重要实验前确保：

```text
code state identifiable
```

至少记录：

```bash
git rev-parse HEAD
```

如果存在未提交的重要修改：

> 应创建 commit 或清楚记录 diff。

禁止训练出重要结果后无法知道使用的是哪版代码。

---

# 38. 不要把大型二进制文件提交 Git

禁止提交：

```text
dataset
large checkpoint
cached frames
.npy feature bank
videos
```

这些存 `/data/aic`。

Git 保存：

```text
code
config
metadata
small result summaries
scripts
docs
```

---

# 39. 自动恢复

训练程序应尽可能支持：

```bash
--resume
```

服务器中断、SSH 断开或训练异常后，可以从 checkpoint 继续。

使用：

```text
tmux
screen
nohup
```

或等价方式保证 SSH 断开不会导致实验死亡。

---

# 40. 训练失败后的处理

如果失败：

```text
OOM
NaN
CUDA error
DDP error
dataset corruption
timeout
```

不要立即完全重跑。

首先分析原因。

记录：

```text
failure category
stack trace
config
GPU state
last valid checkpoint
```

然后进行最小修复。

---

# 41. OOM 的处理顺序

优先：

1. 降低 batch size；
2. gradient accumulation；
3. AMP FP16；
4. 减少输入 frames；
5. 减少 resolution；
6. gradient checkpointing；
7. 更改 data layout；
8. 多 GPU / FSDP。

不要第一反应就是显著缩小模型。

---

# 42. V100 精度策略

V100 优先：

```text
FP16 + AMP
```

不要默认依赖：

```text
BF16
FP8
FP4
```

任何使用新硬件特性的库都要确认：

> NVIDIA V100 / Volta 是否支持。

---

# 43. 不要假设 FlashAttention / 新 CUDA Kernel 可用

很多现代库默认针对：

```text
Ampere
Ada
Hopper
Blackwell
```

服务器是：

```text
Volta V100
```

因此安装或启用：

```text
FlashAttention
xFormers kernels
Triton kernels
bitsandbytes
FP8 kernels
```

之前必须确认兼容性。

如果不兼容：

> 使用标准 PyTorch 实现，不要为追求新 kernel 浪费大量实验时间。

---

# 44. Teacher 原则

C 路线默认：

```text
Teacher frozen
Student trainable
```

不要默认全参数训练大型 VLM。

Teacher 主要任务：

```text
offline annotation
soft-label generation
feature extraction
semantic supervision
```

大型 Teacher 的计算成本不应拖慢整个项目。

---

# 45. Teacher Label 必须缓存

Teacher 推理后保存：

```bash
/data/aic/teacher_labels/
```

确保 Student 每个 epoch 不重新运行 Teacher。

推荐：

```text
video_id
clip_start
clip_end
teacher_model
teacher_version
prompt_version
soft_labels
confidence
```

---

# 46. Teacher Prompt 也是实验变量

如果 Teacher 使用结构化判断 prompt：

> prompt 必须版本化。

例如：

```text
prompts/teacher_v1.txt
prompts/teacher_v2.txt
```

禁止 silently 修改 prompt 后继续使用同一个数据版本名称。

---

# 47. Competition Submission Pipeline 必须尽早建立

不要等模型完成才写 submission code。

尽早建立：

```text
video
↓
inference
↓
postprocess
↓
competition JSONL
↓
local validator
```

至少使用一个 dummy model 把整个流程跑通。

---

# 48. Submission Validator

必须编写本地 validator 检查：

```text
JSONL syntax
required keys
frame range
x/y/w validity
aspect ratio
duplicate frames
sorting
NaN
Inf
empty outputs
out-of-bound crop
```

禁止因为格式错误浪费比赛提交机会。

---

# 49. 官方 Metric 应尽可能本地复现

如果官方公式明确：

> 编写本地 evaluator。

所有模型决策最终以接近官方的 metric 为依据。

不要只使用训练 loss 判断模型好坏。

---

# 50. Pareto Frontier

长期维护：

```text
Competition Score
↑
│
│
│
└────────────→ Model Size
```

重点寻找：

```text
高分
+
小模型
```

的 Pareto Front。

不要只维护单一“最新模型”。

---

# 51. Safe Baseline Lock

一旦 A 路线形成稳定 baseline：

```text
A_safe
```

必须保留：

- code
- config
- checkpoint
- metrics
- submission generator

此后即使 B/C 失败：

> 项目始终有可提交版本。

高级实验不得破坏 baseline。

---

# 52. 探索策略

允许积极探索：

```text
new backbone
new loss
new sampling
distillation
pseudo label
feature fusion
tracking
dynamic programming
crop optimization
quantization
pruning
knowledge distillation
ensemble
```

但每个想法必须先回答：

```text
Why can this improve competition score?
What is the cost?
What is the model-size impact?
Can it finish in <12h?
How will we verify it?
```

---

# 53. 不允许“论文驱动式乱试”

不要因为看到一篇新论文就立刻重构项目。

必须先判断：

```text
Does it solve temporal selection?
Does it solve spatial reframing?
Does it reduce model size?
Does it improve semantics?
Does it improve robustness?
```

如果与比赛指标没有明确关系：

> 低优先级。

---

# 54. 停止规则

如果一个方向连续进行多个公平实验后：

```text
不能提高 official-like metric
```

并且没有新的合理机制假设，

则降低优先级。

不要陷入：

```text
“再调一点超参数可能就好了”
```

的无限循环。

---

# 55. 优先提升最弱环节

周期性分析：

```text
Temporal error
vs
Spatial error
vs
Quality/domain error
```

如果 Temporal 已经明显强于 Spatial：

> 不要继续只优化 Temporal。

如果 crop IoU 已经很好但 highlight recall 差：

> 把资源转到 Temporal。

---

# 56. 允许自动化超参数搜索，但必须受限

可以自动探索：

```text
learning rate
weight decay
sampling rate
temporal window
loss weights
threshold
smoothing
NMS
crop penalty
```

但避免巨大无结构 grid search。

优先：

```text
small controlled search
+
evidence-based refinement
```

---

# 57. 随机种子

关键实验至少记录 seed。

当提升较小时：

```text
<1%
```

应该考虑重复不同 seed，判断是否是真实提升。

---

# 58. 任何号称“最优”的方案都必须经过公平比较

必须保证至少：

```text
same validation split
same official-like evaluator
comparable preprocessing
comparable postprocessing
```

否则不要宣称：

```text
Model X > Model Y
```

---

# 59. 结果报告格式

每完成一个重要阶段，更新：

```text
STATUS.md
```

建议包含：

```text
Current Best
Best <=100MB
Best <=500MB
Best Overall
Temporal Best
Spatial Best
Current Bottleneck
Running Experiments
Failed Ideas
Next Experiments
```

---

# 60. EXPERIMENTS.md

维护累计实验历史：

```text
Experiment
Hypothesis
Change
Metric
Size
Training Time
Result
Conclusion
```

不要依赖聊天上下文记忆实验历史。

---

# 61. DECISIONS.md

重要技术选择写入：

```text
DECISIONS.md
```

例如：

```text
2026-xx-xx
Decision:
Use TSM instead of temporal Transformer.

Evidence:
A1 +1.7 F_video
Model size +0 MB
Latency +4%

Status:
Accepted
```

这样即使 Agent context 被压缩，也不需要重新争论已经验证过的问题。

---

# 62. TODO.md

短期任务维护在：

```text
TODO.md
```

按：

```text
P0
P1
P2
```

排序。

---

# 63. 长期 Agent 恢复协议

每次新的 Codex 会话或上下文显著压缩后：

首先读取：

```text
AGENTS.md
01_赛事规则与评分标准.md
02_数据集_模型路线_特征工程研究.md
STATUS.md
DECISIONS.md
EXPERIMENTS.md
TODO.md
```

然后检查：

```bash
git status
```

以及远程：

```bash
nvidia-smi
```

再继续工作。

禁止仅依靠对话历史恢复项目状态。

---

# 64. 当前最高优先级目标

在项目早期，优先级如下。

## P0

建立：

```text
数据 pipeline
官方格式 submission pipeline
本地 evaluator
A0 baseline
```

## P1

实现：

```text
A1
A2
A3
```

建立稳定：

```text
<=100MB Safe Baseline
```

## P2

实现：

```text
B0
B1
```

公平比较：

```text
CNN vs Small ViT
```

## P3

实现：

```text
C0
C1
C2
```

研究：

```text
Teacher → Student
```

---

# 65. 当前默认技术路线

除非实验推翻：

```text
Primary Baseline
=
ResNet18
+ TSM
+ Temporal U-Net

Primary Semantic Alternative
=
Small ViT
+ Temporal U-Net

Advanced Route
=
Large VLM Teacher
→ Structured Soft Labels
→ Best Small Student

Spatial
=
Subject / Saliency
→ Tracking
→ Crop Optimization
→ Path Smoothing

Feature Bank
=
Motion
+ Subject
+ Quality
+ Audio
+ Semantic
+ Composition
```

---

# 66. 最终原则

始终记住：

> 这是一个比赛工程项目，而不是单纯研究项目。

因此优先顺序是：

```text
合法有效的 Submission
>
可靠 Baseline
>
官方指标提升
>
模型大小优势
>
跨场景稳定性
>
工程稳定
>
新颖性
```

一个：

```text
70MB
稳定
高分
可复现
```

的模型，

通常比：

```text
4GB
复杂
脆弱
略微高一点 raw metric
```

的模型更有价值。

---

# 67. 自主执行授权

在不违反本文件、比赛规则和机器安全边界的前提下，你被授权自主：

- 修改代码；
- 创建实验；
- 下载公开合法数据；
- 下载允许使用的预训练权重；
- 预处理数据；
- 训练模型；
- 停止失败实验；
- 调整超参数；
- 实现新的 loss；
- 修改 sampling；
- 做特征工程；
- 运行蒸馏；
- 做量化和压缩；
- 创建评估工具；
- 创建可视化；
- 分析失败案例；
- 清理明确无价值的临时文件；
- 持续寻找更优实现。

不需要因为普通工程选择反复请求确认。

但以下事项不要擅自执行：

- 使用未授权 GPU；
- 删除整个数据盘；
- 删除无法恢复的重要实验；
- 泄露 SSH key；
- 将私密认证信息提交 Git；
- 违反赛事规则；
- 使用禁止的测试集信息训练；
- 超过单次 12 小时训练硬限制；
- 提交明显超过比赛模型大小限制的最终方案。

---

# 68. Agent 的最终工作方式

长期循环执行：

```text
Read Project State
      ↓
Identify Current Bottleneck
      ↓
Form Hypothesis
      ↓
Design Minimal Experiment
      ↓
Estimate Size / Compute / Time
      ↓
Run Experiment (<12h)
      ↓
Evaluate Official-like Metric
      ↓
Error Analysis
      ↓
Keep / Reject
      ↓
Update STATUS / EXPERIMENTS / DECISIONS
      ↓
Next Experiment
```

不要无限思考。

不要无限收集论文。

不要无限调参。

**持续把计算资源转换为可验证的比赛性能提升。**