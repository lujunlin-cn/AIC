# R27：围绕36.04最佳中心时间窗的四项对照

## 假设与边界

中心85+B3已取得用户报告复赛36.04。保留全部B3逐帧框，仅改变保留预算或窗口位置，区分“两端低价值”与“预算选择”机制。不是新模型训练，不使用测试GT、外部服务或预留14公共源；不把公共时间F1映射成官方分。配置在运行前冻结，见 configs/R27_CENTER_WINDOW_20261016.json。

## 已执行

426/426视频；重新构造中心85，199,037个frame+bbox与R26最佳包逐项完全相等。四包均通过独立checker与项目contract，零issues；每ZIP根目录只含predictions.jsonl，解包内容核对通过。全部450,283,377参数，权重reported903,762,857B，仅存储记录。CPU缓存重组7.93秒，0训练、0新增GPU推理；AutoDL已在线核对GPU空闲。

| 建议顺序 | 包 | 帧数 | 相对最佳新增 / 删除 | 官方分 |
|---|---|---:|---:|---|
| 1 | R27_CENTER095_B3 | 222454 | 23417 / 0 | 待提交 |
| 2 | R27_CENTER075_B3 | 175623 | 0 / 23414 | 待提交 |
| 3 | R27_EARLY085_B3 | 199037 | 11705 / 11705 | 待提交 |
| 4 | R27_LATE085_B3 | 199037 | 11705 / 11705 | 待提交 |

75%和95%都是居中连续窗口；前移/后移臂保留精确85%，相对居中起点分别移动 round(−.05*N)/round(+.05*N)。按原帧编号计算，与R26中心选择定义一致，不改用PTS持续时间。

## 可下载文件

- [R27_CENTER095_B3.zip](../../submissions/R27_CENTER/R27_CENTER095_B3/R27_CENTER095_B3.zip)；SHA256 `7e87584d8c54ff619d9469a53045b0b5107795c72e628a3cfcfa3d6ca2015a1d`。
- [R27_CENTER075_B3.zip](../../submissions/R27_CENTER/R27_CENTER075_B3/R27_CENTER075_B3.zip)；SHA256 `32debd144b9bdd67f6d5573f7a395f5ccf55754d6ffbdd902d169bb99fb9ed94`。
- [R27_EARLY085_B3.zip](../../submissions/R27_CENTER/R27_EARLY085_B3/R27_EARLY085_B3.zip)；SHA256 `2e4ae4395eaf114f7e59900e72543f9d5f7cbe9c43ca666eb52f9f96c09c9251`。
- [R27_LATE085_B3.zip](../../submissions/R27_CENTER/R27_LATE085_B3/R27_LATE085_B3.zip)；SHA256 `2552877aca333f726d52153ed4177de53a1fb8edbe6190b4478ccfa73717f5af`。

## 复现

```bash
PYTHONPATH=. python scripts/r27_center_windows.py --config configs/R27_CENTER_WINDOW_20261016.json
```

需要ignored输入：/tmp/r25_index.jsonl、artifacts/r26/b3_allframes.jsonl、submissions/R26_DIAG/R26_POS_CENTER_EXACT085_B3/。全部SHA冻结于config；路径可以按配置迁移但不能换内容。run.json记录环境、代码/配置SHA与父提交；build.log为完整输出。现有B3缓存不构成完整部署交付，权重、原始视频推理入口仍需另行打包核验。

## 判定规则

95、75先判断预算方向，再用85前/后移判断位置非对称。四臂回分全部登记；不承诺提升、不因包已通过校验而宣布效果成功。若全部≤36.04，停止此预算/位置方向，保留当前最佳，转向固定中心mask的空间监督；若有提升先冻结新最佳，不自动扩网格。缺逐视频官方分，不声称显著性。

后续公开训练实验：源级交叉拟合去时间趋势/双头校准、有限上下文/query单变量控制，目标为超越冻结位置先验的增量；独立公开crop GT上的主体关联/尺度对照。两者均需新预注册与公平基线；本轮尚未执行这些训练。

## AutoDL备份

四个ZIP、manifest、脚本、配置与报告已备份到`/root/autodl-tmp/experiments/R27_CENTER_WINDOW_20261016/`（保留仓库相对目录），远程逐包SHA与本地一致。此备份不含全部ignored输入或B3权重，不宣称是完整部署包。
