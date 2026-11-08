# RetargetVid 原始媒体接入

本轮是CPU媒体与多人标注审计，不运行模型，不创建训练划分。200个源全部已在`spatial_crop_v1`曝光；恢复原视频不能清除曝光。

## 当前路径

- 远端代码：`/root/autodl-tmp/videoora42_clone/retargetvid_media_audit_newday/`
- Python：`/root/autodl-tmp/videoora42_clone/.venv/bin/python`
- 原始AVI：`/root/autodl-tmp/videoora42_clone/datasets/DHF1K/video/{001..100,601..700}.AVI`
- 原始RAR：`/root/autodl-tmp/videoora42_clone/datasets/DHF1K/video.rar`（保留，只解所需200源）
- 原始标注：`/root/autodl-tmp/videoora42_clone/datasets/RetargetVid/annotations/annotator_{1..6}.zip`；原字节解到`annotations_all/annotator_N/`。
- 工作registry：上述`datasets`根下的`DHF1K/processed/media.jsonl`与`RetargetVid/processed/{media,annotations}.jsonl`。
- 原始显示时间轴：`DHF1K/processed/timelines/dhf1k_NNN.npz`。
- 交付：`artifacts/retargetvid_full_media_audit/`。`runs/initial/`保留首次200源完整解码证据，顶层是最终代码的200源缓存复验；两次只做一次全量解码。

200源均为640×360、30FPS；源实解122684帧，六人/两画幅共1472208框。不是DHF1K全部1000源，也没有接入DHF1K显著性/眼动标注。

## 复现

已有200 AVI时不依赖RAR工具，不复制、不会重新解包。首次需要解RAR时，本机仅单独安装`libarchive-c==5.3`在`retargetvid_media_tools/libarchive_c_5_3`，未更改模型venv；安装身份保存在审计证据中。编码/解码使用已有PyAV，无ffmpeg/ffprobe/OpenCV命令依赖。

```bash
cd /root/autodl-tmp/videoora42_clone/retargetvid_media_audit_newday
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=ext_data:/root/autodl-tmp/videoora42_clone/retargetvid_media_tools/libarchive_c_5_3 \
/root/autodl-tmp/videoora42_clone/.venv/bin/python -u \
  ext_data/scripts/ingest_dhf1k_retargetvid.py \
  --existing-media-root /root/autodl-tmp/videoora42_clone/datasets/DHF1K \
  --registry-root /root/autodl-tmp/videoora42_clone/datasets \
  --retargetvid-root /root/autodl-tmp/videoora42_clone/datasets/RetargetVid \
  --evidence-root ext_data \
  --audit-output artifacts/retargetvid_full_media_audit
```

等价入口：`python scripts/audit_retargetvid.py --full-media --root <RetargetVid> --media-root <DHF1K> --registry-root <datasets> --evidence-root ext_data --output <audit-dir>`。

首次每源只调用一次真实`DenseCropAdapter(decode=True,size=None)`：返回frame 0 RGB，同时观察同一个解码循环到EOF，保存PTS/时间轴及首中末RGB哈希。后续复验核对视频SHA、解码合同SHA、时间轴SHA，全部匹配才复用，变化/不确定状态会报告失败而非盲重跑。两种画幅/六人不重复全解码。不抽取或持久化全量图像。

注意：重复运行更新的是本轮工作registry与顶层审计。首次执行证据已经保留到`runs/initial/`；原始ledger与frozen release永不改写。更换视频或解码实现时应保留旧审计，使用新的显式版本目录，不能覆盖不确定缓存。

## 后续实际读取

```python
from pathlib import Path
from aicext.adapters.spatial import DenseCropAdapter

root = Path('/root/autodl-tmp/videoora42_clone/datasets')
ds = DenseCropAdapter(root=root,
                     split=['holdout_dev_exposed', 'holdout_confirm_exposed'],
                     ratios=['1-3', '3-1'], fps=None, decode=False)
assert len(ds) == 400  # 每源两画幅，六个标注者保留在同一单元内
sample = ds[0]
assert sample['boxes_xywh'].shape[0] == 6
assert sample['frame_idx'][0] == 0
assert sample['valid'].all()
assert len(DenseCropAdapter(root=root, split='train')) == 0
```

运行时设`PYTHONPATH=ext_data`；或显式设`AIC_EXT_ROOT`在导入前指向上述registry根目录。默认EXT_ROOT的历史`/data/aic/external_datasets`不是本次远端数据路径；不修改全局默认、不将旧frozen release重绑定到新路径。

## 标注、统计及使用边界

原始`left,top,right,bottom`按`x=l,y=t,w=r-l,h=b-t`转为`xywh_halfopen_v1`，不加1、不裁边。每文件第i+1行对应PyAV显示顺序frame i，0-based、无排序或丢帧；保存真实PTS而非使用`round(t*fps)`重建。报告核对的是源身份、解码顺序/时间轴、完整长度与坐标合同，不是重新人工标注所有帧内容。

六人保留独立框。Source-Macro在每视频内汇总帧/15对标注者，再对200个源等权平均；2000次bootstrap以视频为单位。两画幅分开报告。多峰为预先定义的探索性图连通簇代理，含阈值敏感性，不是由六人确认的总体统计多峰。系统偏好是相对另外五人同帧中位位置/尺度的描述，不是准确率。

全部标注都是各画幅的最大窗（发布舍入误差≤1px），同帧Scale分歧为0；可研究Position、多解构图和部署回归，不能伪装成可变Scale人工偏好。画幅1:3/3:1也不是原生AIC 9:16/16:9视频域。

全部200源继续`exposed_eval`。DHF1K官方Train/Val只作为原发布元数据，工作project_split采用历史曝光，六人/两画幅及跨DHF1K副本均属同一个`dhf1k:NNN` group。原ledger和release/quality的SHA审计前后相同。本轮不建训练版本，不提供fresh validation，不能解除P2独立视频域门。未来训练必须新建明确标记已暴露的实验版本，这些源不再用于独立泛化确认。

旧release逐媒体manifest未在当前克隆中恢复，故与历史AVI逐文件SHA一致性为`null`；本次实际AVI与原始标注已有完整SHA与实际读回证据，不能夸大为历史逐字节认证。

## CPU验证

```bash
python -m compileall -q aic scripts tests ext_data
PYTHONPATH=ext_data:. pytest -q ext_data/tests/test_retargetvid_full_media.py
PYTHONPATH=ext_data:. pytest -q ext_data/tests
PYTHONPATH=. pytest -q
PYTHONPATH=ext_data:. pytest -q tests ext_data/tests
```

合并468通过（项目435 + external-data33，其中本轮17个）。远端本轮针对性测试与实际数据验证见`validation_cpu.json`；GPU使用0。

远端模型venv缺少pytest，首次测试因`No module named pytest`退出，原失败保留在`runs/runtime_missing_pytest.json`。测试依赖的pure-Python wheels已与官方PyPI SHA256核对后解到独立`/root/autodl-tmp/videoora42_clone/retargetvid_media_tools/pytest_9_0_3`，未安装进模型venv；身份清单见`runtime_tools_install_identity.json`。复验命令：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=ext_data:.:/root/autodl-tmp/videoora42_clone/retargetvid_media_tools/pytest_9_0_3 \
/root/autodl-tmp/videoora42_clone/.venv/bin/python \
  artifacts/retargetvid_full_media_audit/capture_runtime.py
```
