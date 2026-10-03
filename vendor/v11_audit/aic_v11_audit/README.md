# AIC V11 审计附件

本目录包含期望 F 参考函数、合成测试和待执行验证计划。

## 验证边界

`numerical_checks.json` 只记录动态规划的合成值与梯度验证。
没有在此环境中训练 AIC 模型，也没有复算平台分数。
两个计划检查文件只说明计划结构检查通过。
所有真实项目实验均为 `NOT_RUN`。

## 运行合成测试

前置条件：安装 PyTorch 与 NumPy。
本次验证使用 Python 3.13.5 和 PyTorch 2.10.0+cpu。

风险：运行命令会重写 `numerical_checks.json`。
需要保留原始记录时，先复制本目录。

```bash
python test_expected_f.py
```

## 使用参考函数

`expected_f(logits, costs, gains, gt_count, empty_value=1.0)` 接收单个视频的块动作参数。
`costs` 必须为每块原帧数。
`gains` 必须为固定裁剪框的匹配 IoU 总和。
`gt_count` 必须为该视频的全部 GT 帧数。
该函数只用于合法训练和诊断，不读取推理阶段的 GT。

不同块动作条件独立。裁剪框必须独立于保留掩码。
若不满足这些条件，不得沿用当前递推并声称精确。

## 证据文件

`source_manifest.json` 固定项目提交与用户附件哈希。
`validation_plan.json` 定义独立验证门。
`statistical_plan.json` 定义拟合范围与统计单位。
`document_checks.json` 只检查主文档的标题与代码块结构。