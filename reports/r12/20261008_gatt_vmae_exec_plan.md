# R12-GATT / R12-VMAE 执行方案（2026-10-08 凌晨批准批）

状态：GATT 待 P0-b 损失判定后冻结配方开训；VMAE 特征已确认在盘（免提取）；
AHA 权重下载中（951MB，本地→910A 传输待下轮）。

## 1. R12-VMAE 臂降本裁定

- `LFM_V11/phd2_vmae_feats/`：**6,001 npz 全量在盘**（V11 时代已提取），每 frag
  `ord [8,768]`（16 帧 linspace 窗）+ `rep [8,768]`（静态控制）。
- 原计划的「8 卡 1-2h 提取」**取消**。臂形态改为两档：
  1. **VMAE-F（冻结特征轻头，CPU，先行）**：ord/rep [8,768] → 轻排序头，与 GATT 同框训练；
     rep 条件天然构成「运动贡献」对照（同 R11 三输入控制设计）。
  2. VMAE-E2E（端到端微调，NPU，M01 桥已验 pass 7.2e-05）——视 F 档读数再定。

## 2. R12-GATT 头：数据合同盘点结论

- B3 时间头（TCN keep 掩码）的训练脚本不在 v8_* 命名集（`v8_s_train_multidata.py`
  经核实为**空间头**：129 格点 win/out 特征 + IoU 标签）；血缘在 `r8_temporal_dualbase.py`
  一线，精确合同下轮考古。
- 已确认可用的输入表示：`LFM_V8/{live_feats,rv_feats_rot,diag_feats_dense}` per-frag
  SigLIP `pooled (768,) f16`（B3 同款，同表示换头=干净单变量）；PHD2 侧时间监督
  （GIF-union in-interval，keep-mask 协议）的构建代码待从 r8 链确认。
- **配方冻结条件（预注册）**：P0-b 双臂终评（test mAP@ρ=15/50，明日午后）出来后，
  损失变量按胜者定（pairwise 胜 → GATT 直接上 multiple pairwise ranking；MSE 胜或平 →
  BCE/MSE 口径）。架构固定 = TRINITY 式（768→512 投影 + 3 层 Transformer 8 头 +
  1D RoPE + 逐帧标量头，与 `p0b_mrhisum_rank.py` 同构）。
- 评估：本地只做预检（能否训出排序），判定走平台差分（每包 ×2 变体）。

## 3. 执行序（下一轮）

1. P0-b 终评读数 → 冻结 GATT 损失配方；
2. 考古 `r8_temporal_dualbase.py` 时间标签构建 → 复刻数据合同；
3. GATT（SigLIP pooled 768d）与 VMAE-F（ord/rep）同框训练，live_dev 选参 /
   live_confirmation 单次，过预检后按 R12 纪律打包上平台（×2 变体）；
4. AHA 权重落 910A 后：Mr.HiSum 协议复现验证（权重有效性门），再议 LoRA。
