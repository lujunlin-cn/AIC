# Spatial benchmark — 2026-09-25

## 当前可信程度

已从“合法输出诊断”进入**真实人类crop GT评测**，但仍没有证明一个稳定替代center的空间模块，更没有AIC联合分数。

RetargetVid annotations/evaluator/SmartVidCrop输出来自作者仓库。恢复DHF1K官方video.rar中非solid RAR成员001–020，所有原视频完整解码、CRC通过，总12,365帧，与6标注者和2ratio逐帧数量完全相同。数据manifest和文件SHA见 `representation_generalization/spatial_manifest.json`。

首版 `SPATIAL_GT_001` 固定参数后运行；随后发现完整upstream evaluator含负坐标clamp，新增只重计分的 `SPATIAL_GT_NATIVE_002`，保留旧结果。自有预测与GT没有负坐标，分数不变；上游SmartVidCrop含6421个负坐标，必须clamp后才能作为参照。**先max(coord,0)，再inclusive +1 IoU**，与未修改upstream函数全量890,280次比较误差0。

所有结果固定同一原frame集合；不使用temporal预测挑容易帧。两ratio各自按视频均等宏平均，6human等权。

## 固定方法对照

| Method | IoU 1:3 | IoU 3:1 | 新权重 |
|---|---:|---:|---:|
| center | .481898 | .739514 | 0 |
| gradient saliency | .476395 | .721160 | 0 |
| subject_proxy | .483416 | .748387 | 0 |
| subject_proxy + EMA | .483567 | .748409 | 0 |
| true_face | .485475 | .751261 | 232,589 bytes |
| true_face + EMA | .486342 | .752144 | 232,589 bytes |
| upstream SmartVidCrop released outputs | .430868 | .704610 | 未在本轮运行，不作本地size/runtime比较 |

SmartVidCrop的原结果仅作为相同20条子集参照，不代表全部200条benchmark排名。

Subject proxy仍是置信度回拉中心的梯度代理。True face使用OpenCV Zoo YuNet 2023mar：SHA256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`，confidence0.8，面积/置信度与前一观察位置用于关联；无face回退proxy。它不是通用object/person detector。平均视频face detection rate .11828。

EMA alpha固定.25；64×36缩略图绝对差均值>.25进行shot reset。所有阈值在首次GT评测前固定，无GT搜索。Crop宽度仍是目标比例下最大合法宽度，尚未实现learned大小选择。

## 不确定性与轨迹

双ratio每视频先平均，以20个source做20000次paired bootstrap。

- Proxy+EMA−center delta **+.00528 [−.00381,.01651]**。
- Face+EMA−center delta **+.00854 [−.01054,.02855]**。
- Pure saliency−center delta **−.01193 [−.05992,.04198]**。

均跨0，不提升为默认crop。所有方法合法率1.0。轨迹以宽高归一化、按fps换算秒，并排除跨shot差分；当前输出坐标经整数round，量化会影响高阶导数。

| Method | Center error/diagonal | Velocity | Acceleration | Jerk |
|---|---:|---:|---:|---:|
| center | .06325 | 0 | 0 | 0 |
| saliency | .06274 | .05178 | 2.20769 | 118.2063 |
| subject_proxy | .06157 | .00880 | .47574 | 26.5708 |
| subject_proxy+EMA | .06156 | .00397 | .20506 | 12.0198 |
| true_face | .06268 | .04589 | 2.49021 | 140.4030 |
| true_face+EMA | .06229 | .02611 | .63931 | 32.1633 |

平滑确实降低轨迹抖动，但它不自动改善人类crop匹配；IoU只微增。

## 失败案例

`representation_generalization/figures/*_human_crops.jpg` 叠加一个真实annotator（绿色）、center白、saliency青、proxy+EMA橙、face+EMA红；不是用saliency自己评判自己。

- 003：单人讲解，face观察帮助跟随，双ratio IoU相对center +.1389。
- 016：双ratio +.0881。
- 020：多人场景，face倾向右侧显著人脸，而人工窗口关注中央多人互动，delta −.1031。
- 002：delta −.0488。

这些表明“能检测人脸”不等于“知道应保留哪个主体”。下一空间假设应围绕多人内容覆盖/主体选择，而不是继续调EMA使jerk更漂亮。

## Raw inference一致性和实际权重

`aic.inference` 增加显式 `spatial_protocol=dense_v1`；保留旧 `legacy_sampled_v1`。新模式在每个原frame更新observer/association/shot-reset/EMA，包括temporal不选中的帧；最后只序列化被选中的frame。

`DENSE_E2E_001`：A0 center/saliency/proxy/proxy_smooth/true_face/true_face_smooth，以及DeiT proxy_smooth、ViT-B face_smooth，共8条真实视频链路，3600条预测全部通过validator；round到benchmark坐标后的最大误差0。此次threshold=0只是为了覆盖全部帧的集成检查，不作为模型阈值/效果选择。

| Bundle | 实际总bytes |
|---|---:|
| A0 + center/proxy | 25,685,169 |
| A0 + YuNet | 25,917,758 |
| DeiT-S + proxy | 46,618,447 |
| DeiT-S + YuNet（组成字节和，尚未单独E2E） | 46,851,036 |
| ViT-B + YuNet | 175,219,036 |

DHF1K001上A0 center2.95s、A0 face+EMA4.33s、DeiT proxy+EMA3.40s、ViT-B face+EMA6.41s，是一次小视频raw路径时延，不能推断大视频P95。无额外OCR/audio/tracker neural weights。

无联合temporal+crop GT，**official_f_video和competition_score保持null**。以上RetargetVid IoU不是AIC官方IoU或联合得分；比例迁移9:16/16:9与可变尺寸仍需验证。

## 固定发行入口实际阈值复核

`ENGINEERING_RELEASE_002`在DHF1K001/003/020用真实阈值：A0 .40三条均空，DeiT .35仅003选48帧。仍合法，但不能据此声称时间识别有效或无效，因为没有temporal GT。

`ENGINEERING_RELEASE_003`使用历史TVSum DEV的AwmHb44_ouw/37rzWOQsNIw，目标9:16。A0center与A0faceEMA选同1263原帧，DeiTcenter2297帧，三者均非空且valid。A0center33.01s、faceEMA157.78s，说明dense face path真实CPU开销不可忽略；权重小不代表运行免费。此处只验证部署，无9:16人工crop GT，不推断该比例IoU。

发行manifest、SHA、离线入口在 `releases/20260925_v1/`；FP16文件运行时转FP32标准kernel，不宣称FP16算子加速。
