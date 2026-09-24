# 技术决定

## 2026-09-25：从 Phase 0 / 1 启动

Status：Accepted（工程选择，非模型效果结论）。

Evidence：本地只有三份文档，远程项目与数据目录为空，无历史 baseline 可继承。

Decision：先实现比赛契约和 A0；TVSum 作者原视频为首个数据源；用显式指标名称区分时间代理与联合比赛指标；无同版本官方脚本时称 official-like，不称官方一致性验证。

## 2026-09-25：TVSum 许可证闸门

Status：Blocked。

Evidence：包内 `WebscopeReadMe.txt` 要求签署 Yahoo Data Sharing Agreement、获批非商业学术研究并禁止再分发/网络存储；README 的 CC-BY 表述不能自动覆盖该冲突。文件 SHA-256=`407d340bcd06fdc6d17374ebe6760b4a96816bcace228559c8283d9fb2520dea`，50/50 视频字节与解码元数据已核验。

Decision：用户明确授权本项目将可下载的数据集和模型视为可用于实验。保留 `license_gate`、来源和条款哈希用于追踪，但不再因该字段暂停训练；恢复 A0_001 资产，后续实验按用户授权继续。A0 的 TVSum 标签仍是 summary proxy，不是官方联合 GT。

## 2026-09-25：资源与时间边界

Status：Accepted（用户硬约束）。

Decision：物理 GPU 0/3 永不默认使用；GPU1 已有任务，首轮优先2并启动前重查。所有训练使用内部预算、定期 best/last 与外层 timeout，断线不终止。大文件仅在 /data/aic；代码 rsync 不删除远端数据。
