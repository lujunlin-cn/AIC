# V6 E0 证据与接口审计（2026-09-29）

指南：`AIC_V6_48.67分停滞诊断与60分突破执行指南_20260929.md`。起点 commit `1c1cedf`（teacher-v5-20260929）。
本报告只更正记录与重判规则，不覆盖历史实验事实；未改任何划分、未放宽任何门槛。

## 1. 成绩账本更正

- `configs/QWEN32B_INTERP_XRERANK_V5_FINAL.json` 的 `official_platform_score` 48.67 → **48.49**
  （复制母本遗留），`official_score_source` 更正为本包成绩的用户回报；母本字段保持 INTERP=48.67。
  配置内新增 `correction_log` 记录两处更正的字段名/原值/原因，历史记录（STATUS/EXPERIMENTS/DECISIONS）不动。
- 数据方向描述更正：LIVE-YT VC 是**横屏源→竖屏区域裁剪**，与目标任务（竖屏→16:9）方向相反；
  V5 简报中"与竖屏→16:9 直接同构"的说法已在 `reports/20260929_v5_package_and_data_brief.md` 修正。
  H3 的 LIVE 表全部为 axis0 与此一致；LIVE 只可作 x 域训练/回退诊断。

## 2. source_group 级 H3 CI 重算（rv 双比例合并）

脚本 `scripts/v6_e0_source_group_metrics.py`（头/归一化来自 V5 `h3_heads.pt` 快照，无重训）。
**自校验**：4 集 × 3 头的单元级口径（同展平顺序、同 bootstrap seed 20260929）与 V5 保存的
`metrics.json` CI **全部逐位一致**（12/12 pass），重算管道可信。

源组定义：RV 同一源视频的 axis0+axis1 两单元合并（源内**等权**平均——V5 预注册未约定权重，
等权为本 E0 冻结的约定并在此声明）；LIVE 每源 1 单元，源级=单元级。

| 集 | 头 | 单元级 n / CI95（=V5 原值） | 源组级 n / CI95 | Δ |
|---|---|---|---|---|
| rv_dev | G | 40 [+0.0056,+0.0175] | 20 [+0.0066,+0.0165] | 略窄 |
| rv_dev | V | 40 [−0.0118,+0.0370] | 20 [−0.0133,+0.0384] | 略宽 |
| rv_dev | VQ | 40 [−0.0112,+0.0361] | 20 **[−0.0114,+0.0356]** | 略宽 |
| rv_confirm2 | G | 200 [+0.0038,+0.0116] | 100 [+0.0036,+0.0119] | ≈ |
| rv_confirm2 | V | 200 [−0.0003,+0.0307] | 100 [−0.0010,+0.0314] | 略宽 |
| rv_confirm2 | VQ | 200 [+0.0071,+0.0313] | 100 **[+0.0064,+0.0326]** | ≈ |
| live_dev / live_val | G/V/VQ | 同单元级 | 同单元级（每源 1 单元） | — |

逐单元明细：`reports/v6_source_group_metrics.csv`（1,680 行）；汇总 `reports/h3/v6_source_group_summary.json`。

## 3. 按完整预注册重判 Gate1 / Gate2

预注册原文（`configs/V5_H3_VISUAL_PREREG.json`）gate1 要求两个条件同时成立，V5 代码 `ok1` 只执行了第一个：

| 条件 | 数值 | 判定 |
|---|---|---|
| 均衡 dev：best(V,VQ) − G > 0（关键帧 IoU） | **+0.0110**（.630361 − .619354） | ✅ |
| rv_dev 单独：best(V,VQ) − G > 0 | **−0.0031**（V −0.0052 / VQ −0.0031 相对 G +0.0185 的 sel−mother 差） | ❌ |

**gate1 = FAIL。** 这是点估计失败（V/VQ 在 rv_dev 上的关键帧 IoU 低于 G），不是 CI 宽度问题；
"扩大 rv_dev 就能通过"的假设不成立，扩样本未必翻负为正。

gate2（源组级口径）：rv_dev VQ mean +0.0110 ✓ 但 CI_low −0.0114 < −0.002 ✗（V 同败）；
confirm2 VQ +0.0190 [+0.0064,+0.0326] 重复通过。**gate2 = FAIL（rv_dev 关）**。

**最终判定：H3 V5 按 V5 预注册两道门均未通过，不出包——结论与 V5 一致，但依据更早更强**
（V5 报告把它归因于"gate2 CI 未过"，掩盖了 gate1 的 rv_dev 条件失败）。下轮重启必须：
先解决"视觉头在 rv 域点估计不胜 G"的问题（而非只扩样本），并按 E1 用原生 y 域验证。

## 4. holdout 暴露记录核对（ledger v1，2026-09-29 09:21 冻结，不改原件）

`/data/aic/external_datasets/_registry/holdout_exposure_v1.json` 实测：

| 数据集 | 组数 | 暴露记录（experiment=T5_CROPHEAD_V3 等历史） |
|---|---|---|
| LIVE_YT_VC | 1,800 | fit(live_train) 1,466 + selection(live_dev) 156 + **confirmation(live_confirm3) 178** |
| RetargetVid | 200 | SPATIAL_SUBJECT_002 / MAX_WINDOW_* / T0-T2 / T5（fit 80 + selection 20 + confirmation 100）+ V4_E1_OBS025 in_progress 170 |
| QVHighlights(in-house) | 160 | QVH_NATIVE_BOUNDED_V1：fit 96 + selection 24 + confirmation 40 |
| YouTubeHighlights | 124 | T3/T4 selection 124 |

三点裁定：

1. **ledger 中不存在任何标记 `holdout_confirm_reserved` 的组，也没有 "227" 组的记录**
   （split_name 含 reserve 的行数为 0）。V5 预注册"release confirmation reserve (227) 已有
   T5 live_train fit exposure"的说法在 ledger v1 中**无对应条目**，无法核实其身份；
   该说法可能是 ledger 冻结前的手工记录或笔误。ledger 的类别定义（fit-only 暴露可保留在
   reserve 中并逐组记录）是规则，不是"已登记的视频集合"。
2. **"live_val 未见过/全新"的表述错误**：live_val = ledger 中的 `live_confirm3`（T5 confirmation
   角色，178 组 → 建表 172 单元）。V5 仅将其用作 report-only 是合规的，但它对 T5 的模型选择
   已暴露，不是新鲜集。更正后的规范表述："live_val（T5 confirm3，对 T5 选择已暴露；V5 report-only）"。
3. 各名称映射：rv 001–080 = T5 fit；rv 081–100 = T5 selection；rv 601–700 = T5 confirmation；
   live 三段见上表。跨数据集同组拷贝（如 dhf1k:NNN ↔ rv:NNN）继承暴露身份——混训按 group 去重时以 ledger 为准。

## 5. H3 部署接口修复（V5 快照的 5 项缺陷）

| # | V5 缺陷 | 修复 |
|---|---|---|
| 1 | `build_units` 不生成 X，`unit_tensors` 无条件读 `u['X']` → KeyError | `unit_tensors` 仅 kind≠V 读 X（V5 数值路径不变）；`build_units` 现用 `kf_features` 生成 26 特征表 |
| 2 | CLI 只收 kind=V | `--kind G/V/VQ` |
| 3 | 拼接器仅支持 x 行 | 新 `scripts/v6_e0_splice_spatial.py`：`--axes 0/1/0,1`，G1（DENSE 重算=母包）/G2（仅目标轴自由分量可变）/G3（非目标行逐字节不变）硬守卫 + 往返一致；`--only` 仅限 E2E 诊断切片（manifest 标 diagnostic_only） |
| 4 | 母本候选被吸附到网格（addendum 承认） | `--exact-mother`：33 网格 + 精确母本偏移（34 槽位，去重），默认 grid33 保持 V5 可复现 |
| 5 | 官方 ds='off' → (ds_rv,ds_live)=(0,0) 为训练未见组合 | score 阶段输出域标识审计（`rerank/_domain_audit.json` + 已知 OOV 警告）；根治（去 source one-hot）属 E1 |

## 6. 小样本端到端复现（E2E，已完成 2026-09-29）

真实数据全链路：tables → features → score(VQ 与 V) → v6_e0_splice_spatial --axes 0 / 1，
远端 `/data/aic/experiments/V6_E0/off_e2e/`。

### 6.1 E2E 过程中新发现并修复的缺陷（第 5 节表格之外）

| # | 缺陷 | 修复 |
|---|---|---|
| 6 | `build_units` 计算了 26 特征表但未存入 unit dict → score 阶段 KeyError('X') | `'X':X` 入 dict（与缺陷 1 同根） |
| 7 | `--exact-mother` 时 gap≤1e-6 的行走 33 宽赋值到 34 宽数组 → broadcast 崩溃 | 统一拼接精确列；added 标志改循环内记录（exact_mother_added_frac 口径更准） |
| 8 | **官方关键帧缓存 1 fps（stride 60），dense 点位 0.5s（stride 30）**；V5 训练域 rv 为每 dense 关键帧一张图（rv_dev 57/57 实测）。官方 png 只覆盖一半 dense 键 | `v6_e0_render_missing_kf.py` 从 intake index 的源 mp4 补渲染 2,882 帧（半分辨率、noautorotate、coded_pixels 口径，现有一律不改），provenance manifest 落盘 |

### 6.2 各阶段结果

| 阶段 | 结果 |
|---|---|
| tables | 174 units / 5,586 关键帧 / 34 候选列；`exact_mother_added_frac` 均值 **0.879**（min 0.158 / max 1.000）——量化缺陷 4：官方 ~87.9% 的关键帧上精确母本位置被网格吸附丢失 |
| features | 5,586 关键帧 DINOv2 ViT-B/14 letterbox 池化，全部缓存 |
| score VQ | 174/174 出 rerank JSON；`_domain_audit.json`：ds_rv=ds_live=**0.0**（训练未见组合，已知 OOV 警告在档，见缺陷 5） |
| score V | 同上（同一特征缓存） |
| splice axes 0（vid=12，VQ） | G1（DENSE 重算=母包逐字节）✓ G2（仅 x 自由分量可变）✓ G3（非目标行不变）✓；148/148 帧与母包不同；独立校验 + ZIP 往返 valid（zip `4f0b86e6…`，`diagnostic_only=true`） |
| splice axes 1（vid=0，V） | G1/G2/G3 ✓；578/630 帧不同；独立校验 + 往返 valid（zip `9d5d7056…`） |

**判定：H3 部署接口五项缺陷的修复全部在真实官方数据上端到端跑通；三轴守卫与身份链（index sha256 → 母包 sha256 → predictions sha256 → ZIP 往返）成立。**
两个诊断包均为 INTERMEDIATE_SPATIAL_ALL_FRAMES_NOT_FOR_UPLOAD，不得上传（单视频切片）。

E2E 之后 score/table 产物另见：`off_e2e/off_units.pkl`、`off_e2e/features/`、`off_e2e/rerank_vq/`、`off_e2e/rerank_v/`、`render_missing_manifest.json`。

## 7. model_params_b 契约（aic/contract.py）

- 新增 `size_coefficient_params(P)`：≤0.1B→1.00、≤0.5B→0.95、≤9B→0.90，**P>9B 直接 ContractError**
  （复赛上限；量化/LoRA 不减免统计）。行级 `model_params_b` 校验：类型/范围/全行一致/与实测匹配。
- 旧 `size_coefficient`（MB 计档）保留为显式 legacy 入口（历史报告可读）；final 行只有 MB 无 params_b
  → warning（legacy 行可读），两者皆缺 → error。旧行为不变（无 model_params_b 时路径与之前逐位一致）。
- 函数级测试通过（6 组边界值 + 6 组非法值 + legacy 3 组）。
- 注意：教师系统 Qwen3-VL-32B(+YuNet) ≈ 33.357B **不可作为复赛系统**；合规候选另行统计（E1/E3 产物）。

## 8. 结论与对下一步的约束

- H3 V5 最终判定：gate1 FAIL（rv_dev 点估计）+ gate2 FAIL（rv_dev 源组级 CI）→ 不出包，维持。
- E1 必须以"候选含精确母本（34 槽位）+ 显式几何输入 + 去教师依赖 + 去域 one-hot"的修正头重做，
  主指标用原生 y 域人工标注的完整管线表现；RV/LIVE 只作方向诊断。
- 本 E0 未改动：任何划分、任何门槛、任何已登记分数。
