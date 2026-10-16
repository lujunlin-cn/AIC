# AIC_MANUAL_ANNOTATION_V1 操作说明

这是一批用于路线判断的小规模人工先验数据，不是官方测试集，也不是可直接训练成 AIC GT 的数据。

## 已取数据

素材来自 AutoDL 已有的 `QVH_NATIVE_CANDIDATES`，没有下载整套 1.3GB 数据。固定抽取了 12 个公开源视频的 30 秒窗口，按 1 fps 导出 360 张 JPEG，同时保留带单声道 AAC 音频（16 kHz、64 kbps）的低码率 MP4 供标注者先看过程。重新编码后本机批次约 25MB，本机目录为：

```text
artifacts/annotation_batch_v1/
```

其中 6 个视频做双人重叠标注，另外 6 个先做单人标注。选择覆盖动作结果、危险动作、讲话信息、多人舞台、景观、普通活动等对当前路线有区分力的类型。QVHighlights 的 query 只留在 manifest 中，第一遍不展示给标注者。

完整选择、时间窗和原因见 [`configs/ANNOTATION_BATCH_V1.json`](../configs/ANNOTATION_BATCH_V1.json)；逐帧判断流程见 [`reports/annotation_protocol_v1.md`](annotation_protocol_v1.md)。当前协议版本为 v1.1：先完整播放并建立时间线，再逐帧判断；0/1/2/3 按同一视频内部相对保留价值区分，不确定不能改写成 0。

这批任务没有逐视频 `targetRatioWH`。因此 `crop_box` 只表示自由比例的构图可行性诊断，不是官方 native crop GT，也不用于直接训练空间框回归；必须先判断高光等级为 2/3，才决定是否存在稳定合理窗口。

这轮只下载这个小批次，不再把 TVSum、SumMe、VideoXum 或 LIVE-YT 整库搬到电脑。它们的摘要/构图标注政策与 AIC 不同，先用同一批原视频同时做时间、上下文和构图标注，才能判断路线是否需要改变。若这轮结果仍无法区分机制，再从同一 `QVH_NATIVE_CANDIDATES` 清单补 6 个源视频；在结果出来前不扩大数据集。

## Label Studio 常驻服务与访问

Label Studio 已配置为用户级 systemd 常驻服务，当前监听本机
`127.0.0.1:18081`，并通过 FRP 中转到公网：

```text
http://47.108.220.174:3002
```

服务配置文件为 `~/.config/systemd/user/label-studio.service`，本地文件服务根目录为仓库根目录。服务状态可检查：

```bash
systemctl --user is-enabled label-studio.service
systemctl --user is-active label-studio.service
curl http://127.0.0.1:18081/health
curl http://47.108.220.174:3002/health
```

FRP 客户端由系统级 `frpc.service` 常驻，代理名为 `label-studio`，映射
`127.0.0.1:18081 -> 47.108.220.174:3002`。不要再用临时的 `label-studio start` 进程占用端口。

登录账号使用项目管理员账号；密码不写入仓库或本说明。

### 导入任务

项目 `AIC Manual Highlight Smoke v1`（ID 1）已经创建，interface 使用 [`configs/label_studio_annotation_v1.xml`](../configs/label_studio_annotation_v1.xml)，当前已导入 60 个 Smoke 任务：

```text
artifacts/annotation_batch_v1/label_studio_tasks_smoke_local_files.json
```

Smoke 验收通过后，完整文件 `artifacts/annotation_batch_v1/label_studio_tasks_local_files.json` 共 360 个任务，包含 Smoke 的 60 个任务。为避免重复导入，建议新建一个正式项目并导入完整文件，或先在 Smoke 项目中删除已有任务再导入完整文件。图像路径形如 `/data/local-files/?d=artifacts/...`。当前任务是盲标版本：只含视频 ID 和时间，不含类别、抽样原因或 QVH query；这些信息只在独立 manifest 中，供标注完成后比较。如果 Label Studio 环境允许浏览器直接读本机文件，也可使用同目录的 `label_studio_tasks.json`（`file://` 路径版本）。

先只标以下两个视频验收界面和导出：

```text
B6AEPicK9Bk_210.0_360.0
rAHQY2FHcOo_210.0_360.0
```

确认图片可见、RectangleLabels 能画框、导出 JSON 中保留 `video_id`、`source_time_sec` 和框坐标后，再继续其余 10 个视频。6 个重叠视频的第二位标注者必须在看不到第一份结果时完成。

如果只有一位标注者，先完成全部 12 个视频；不要把单人结果伪装成一致性验证。若有第二位标注者，复制一个 Label Studio 项目，双方分别导入同一盲标任务，导出文件分开保存。

## 这批标注要回答什么

1. 高光核心是否集中在完成/结果，还是需要看到动作变化过程；
2. 静态 RGB 是否足够，还是声音、语言或前后上下文占主导；
3. QVHighlights query 相关区间与通用高光是否错位；
4. 高光帧是否同时具有稳定的目标比例构图窗口。

完成后先做双人一致性、源级留出和上述四项诊断，再决定是否把某种先验写入部署候选。不能把这 12 条视频的标签当作官方 GT，也不能用它们直接声称官方分数提升。
