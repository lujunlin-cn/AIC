# 双人标注操作说明（V1.3，2026-10-17）

完整批次已开放，无需再从 AutoDL 下载素材或手工导入任务。

## 两个人分别打开哪里

| 人员 | 项目 | 当前已完成 | 新增待做 | 内容 |
| --- | --- | --- | --- | --- |
| 甲（已做Smoke的人） | [标注员甲](http://47.108.220.174:3002/projects/1/data) | 60 | 240 | 共10视频，已有2个，新增8个 |
| 乙 | [标注员乙](http://47.108.220.174:3002/projects/2/data) | 0 | 240 | 共8视频，全部独立完成 |

甲还有6道旧题需要删除多余框，题号34、35、38、57、59、60；见[修订质检队列](r42_smoke_revised_20261017/repair_queue.csv)。只保留一个完整窗口，不必改等级，不机械取所有框的并集。原60份结果和2份草稿已保留，不自动合并草稿。

两项目合计540份任务，覆盖360张独立图片、12个公开源；6视频/180图片由两人分别标。分工在 `configs/ANNOTATION_ASSIGNMENTS_V13.json` 固定，按照原有单人列表顺序拆分，未按标注结果挑选。

当前为项目分开保存，未配置不同账号的访问权限隔离。可使用现有登录方式，但两人必须只进入自己的项目，不查看对方结果；不自动发送邀请邮件。乙不要看甲的Smoke逐题质检与修订报告。项目名、任务项目和说明决定A/B身份，导出时不能只按账号ID判断是谁。

## 打标人员先读什么

[在线V1.3完整标准](http://47.108.220.174:3002/data/local-files/?d=artifacts/annotation_batch_v1/guides/annotation_protocol_v1_3.html)（需登录，可浏览器打印为PDF）。任务顶部也有入口，右上角Instructions可查看说明。仓库版为[标注标准V1.3](annotation_protocol_v1_3.md)。旧V1.2文档保留在 `annotation_protocol_v1.md`，以V1.3开展新增标注。

1. 换视频后，先完整看上方30秒片段并开声音；同一视频只需完整看一次，可回放。
2. 逐秒图填写等级、阶段、类型、把握。
3. 0/1跳过依赖和画框；2/3填写信息依赖，并画一个完整窗口或说明无合理窗口。
4. 点击Submit提交，再做下一题；页面旧缓存用Ctrl+F5刷新。

没有比例、数量或去重配额；精彩持续几秒，可以连续2/3；整段普通可以全0/1。框是一帧一个完整保留窗口，涉及多人、物体、字幕时一起保留，不分别画对象框。两人只讨论通用规则，不讨论本批具体答案。

每人新增240题；按每题20–40秒，加上看视频与休息，排班约1.5–3小时。修订Smoke存储的lead_time合计1811.987秒/60题，约30.2秒/题，仅作耗时参考，不是实际在岗计时。先做30题后按实际速度调整，不复制标签赶进度。

## 素材与服务

这批只使用既有 `QVH_NATIVE_CANDIDATES` 中12个30秒窗口、360张1fps JPEG和12套MP4/WebM视频。第一遍不展示query、类别、抽样理由、原标签或模型预测。只提供30秒上下文，不是原始长视频全片。

素材与生成的任务、私有导出、截图位于 `artifacts/annotation_batch_v1/`，不提交Git。完整源manifest含盲标人员不应看的原query/理由，负责人保管，不发给乙作为说明。无逐视频目标比例，所以框只用于自由比例构图可行性诊断，不是native crop GT。

常驻服务：用户级 `label-studio.service`（127.0.0.1:18081），系统级 `frpc.service`（公网47.108.220.174:3002）。服务配置和凭据放本机私有目录，仓库不保存密码。

```bash
systemctl --user is-active label-studio.service
systemctl is-active frpc.service
curl http://127.0.0.1:18081/health
curl http://47.108.220.174:3002/health
```

存储连接只用于读取frames/videos/guides，不点击Sync，避免产生意外任务。现有任务已去重导入，不要再次手工导入360张全量文件。

## 负责人复现与导出

部署命令、环境、数量、hash与验收见 `reports/r42_annotation_v13_20261017/deployment.json`。部署脚本按video_id+time_sec去重，第二次运行必须新增0题；它不写标注结果，不复制甲的答案给乙。完整SQLite备份在Label Studio私有backups目录，含认证数据，禁止提交Git或发给标注者。

```bash
/home/hajimi2025/.venvs/label-studio/bin/python scripts/build_annotation_v13_interface.py \
  --protocol reports/annotation_protocol_v1_3.md \
  --base-config configs/label_studio_annotation_v12.xml \
  --output-config configs/label_studio_annotation_v1.xml \
  --output-html artifacts/annotation_batch_v1/guides/annotation_protocol_v1_3.html
```

完成后，每项目分别导出正式结果与草稿（不合并）。为每次导出选择新的目录：

```bash
python scripts/snapshot_label_studio.py \
  --database /home/hajimi2025/.local/share/label-studio/label_studio.sqlite3 \
  --project 1 --output artifacts/annotation_batch_v1/exports/A_COMPLETION_TIMESTAMP
python scripts/snapshot_label_studio.py \
  --database /home/hajimi2025/.local/share/label-studio/label_studio.sqlite3 \
  --project 2 --output artifacts/annotation_batch_v1/exports/B_COMPLETION_TIMESTAMP
```

用 `scripts/audit_annotation_smoke.py` 和 `configs/R42_ANNOTATION_QC_V13.json` 检查新增V1.3结果；甲旧60按原协议单独保留来源。数据完成前不分析未完成任务的缺字段率。

先检查字段、媒体映射、单框、缺失与不确定，再做每源一致性和事件/声音依赖诊断。主一致性统计使用双方同V1.3的4个新增重叠源；两Smoke源含甲V1.2历史，作为版本敏感性结果单列。每人框不平均成唯一真值，自报依赖不当作模型能力上限；12源用于机制诊断，不能充当官方GT、独立泛化确认或官方涨分证据。
