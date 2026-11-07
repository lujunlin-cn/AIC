# 路线 07：主体定位、显著性、构图读出与路径优化

整理日期：2026-11-07。总入口：[index.md](index.md)。本页讨论早中期独立空间机制；LFM训练主线见路线04，Video-ORA后期完整框见路线12。

## 主假设

同一高光时间集合上，框更接近人工观看构图即可提高联合F。项目试过中心/脸/主体/显著性/teacher点/候选窗口/直接坐标/轨迹平滑。物体检测框、视觉注意力中心和观看构图框并非同一种监督。

## 子方向与结果

| 家族／实验 | 实际内容 | 结果与限制 |
|---|---|---|
| center/saliency/subject | 参数为零的中心、传统显著性，YuNet人脸+EMA | 原视频到JSONL链通过；早期初赛DeiT+YuNet比同时间集合中心高+.56 |
| SPATIAL_GT_001/NATIVE_002 | RetargetVid20源、6标注员、两目标比例、固定方法 | 得到真实crop IoU；补全作者negative clamp后重算。不能再把全部空间GT写成null |
| SPATIAL_GROUP_003 | 面积/置信度加权top3脸中心与EMA | 相对单脸EMA −.00264，CI[−.00726,0]；多脸加权未改善 |
| subject_scale / semantic subject | 人脸/COCO/语义主体的尺度、置信和窗口策略 | 主体更完整不保证IoU更高；历史尺度包正式负回分。teacher点的成功见路线03 |
| MAX_WINDOW_PATH | EMA、镜头内L1 DP、静态路径、shot reset | 公开指出EMA滞后/错误跟随，部分DP/静态更好；不能用路径更平滑替代绝对框质量 |
| R28/R29 Qwen3.5-4B pilot | 四个已暴露RV源×3帧×2比例，生成自由坐标 | Qwen=.599645、中心=.579117；20/24输出中心、y轴12/12中心；运行成功但选择能力弱，没有同帧B3对照 |
| R29 oracle129 | 同输入枚举129合法偏移 | oracle=.868315，远高于当前输出；只证明几何候选够用，不保证模型可学或官方可涨 |
| R46/R53 asset/readout audit | 恢复B3实际权重、V9 runtime、观察与候选读出拆分 | 精确旧B3一度缺失，V7/V9近似头不能冒充；后续空间研究须绑定真实head/base |
| R69 direct grounding | TimeLens直接预测整个视频或分段crop，保留R54时间 | 分段门控候选官方24.95，比36.22−11.27；轨迹总变差降低46.5%也没挽救绝对位置错误 |
| R84 coordinate consistency | 镜像/坐标一致性空间读出 | 已做公共诊断；无已确认官方收益，不把等变性检查当crop能力 |
| R86 T5 H1 | R54时间固定，T5映射用于V9_A1raw点 | 官方34.88；同时改变点源与映射，是混杂实验 |
| R88 corrected H1 | 相同时间、系数，仅改用正确B3点源 | 完整426候选已生成，未发现回分；因此“B3上T5已失败”仍不成立 |
| R89 low-level composition | 梯度、局部对比、清晰度、相邻帧差、合法几何，native crop监督 | 272源confirm IoU .512363→.516466，+.004104；去差分+.004108，说明当前差分没有额外价值。候选构建过，未确认官方分 |
| R94 U²-Netp | 通用前景主体作为观察增量 | 公开CI跨零，见路线01 |
| R98 UNISAL | 视觉注意力质心/50%mass中心替换Qwen点 | LIVE148源分别−.03723/−.03364，CI均负；关闭直接替代配方 |
| R100 pooled context | 邻候选池均值 × 当前候选低秩交互 | Δ+.000747，CI跨零；置换后仍+.000374，未显示新观察价值 |
| R101 ResNet context | 当前高分辨率候选×邻帧ResNet18语义 | Δ+.000197，CI跨零；置换反而+.001546，未识别真实上下文增量 |
| R103 spatial tokens | 两邻帧各4×4 V9 token交叉注意力 | Δ−.0006575；一帧源全mask NaN修复后完整重跑，结果仍负 |

## 当前可支持的判断

固定R25时间mask，B3框换中心后35.18→30.23，说明空间本身是高杠杆组件。这个差值不能推导“空间占全部误差多少”，也不表示任意新空间方法都能再加4.95分。

最大合法窗口是大量历史方案的工程选择，当前官方详细契约允许合法局部`[x,y,w]`，并非强制最大窗。后期完整框实验已经实际尝试，不能再称“项目从未探索尺度”。

通用显著性直接替代失败，与将显著性作为受native crop GT监督的辅助特征是不同假设；后者R105存在弱正信号，见路线09。

## 与 GPT 讨论的问题

1. 当前损失主要是主体观察、候选内选择、自由轴位置、尺度还是部署轨迹？需要哪组完整链评价分开它们？
2. 怎样建立横屏转竖屏、竖屏转横屏、native比例的真实留出，避免转置验证冒充原生域？
3. 若重新考虑tracking，应怎样证明改善逐帧IoU，而不只减少速度/抖动？

## 证据入口

- `reports/20260925_spatial_group_and_ood.md`、`reports/20260928_max_window_path.md`。
- `reports/r29/spatial_review.md`、`reports/r53_v9_spatial_audit_20261018/summary.md`。
- `reports/r69_segmented_stability_gate_20261018/README.md`、`reports/r83_official_local_validity_20261019/summary.md`。
- `reports/r86_r87_observation_and_transfer_20261019/summary.md`、`reports/r89_lowlevel_composition_20261019/summary.md`。
- `reports/r98_unisal_human_saliency_native_crop_20261019/summary.md`、`reports/r100_spatial_context_utility_20261019/summary.md`、`reports/r101_spatial_resnet_context_20261019/summary.md`。
