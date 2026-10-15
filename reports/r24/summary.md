# R24：干净源级 point/block、VPB/CLIP 与 AHA 尝试

日期：2026-10-15。结果：**不晋级，不评估预留源，不提交新比赛包。**
R23 官方复赛用户回分 P250=18 / P500=26 已另行登记；初赛分数不参与本轮判断。

## 协议与证据

公共 VideoXum 26 train / 14 dev，源 ID 交集为零；14 protected 未打开。
每源重新完整解码，检查实际帧数与 R22 PTS 轴一致、media/axis SHA一致。
全视频均匀12帧，按实际 PTS 最近邻映射原始帧。point 为采样点10人标签均值；
block 为该采样点 Voronoi 块内原始帧、10人标签均值。

VPB与CLIP本轮重新抽特征，禁止复用历史缓存。CLIP使用 vision pooler_output 768维，
无512维投影或额外适配器。两种表示都用768→512→GELU→1头，同seed相同初始化、
train-only逐维标准化、soft BCE、AdamW、lr=.001、wd=.01、固定300 epoch最终模型。
各backbone遵守自身预训练输入合同，因此比较包括该预处理差异。
未按dev选epoch。主指标 keep=.85；.6/.25仅预注册辅助读数，不据其换门槛。
逐人时间F1→视频macro，按原始帧精确保留round(N×keep)，块内同分按原帧顺序。
不是官方F_video：没有空间GT，也没有比赛测试监督。

## 主指标

| 表示 / 监督 | seed13 F@.85 | seed17 F@.85 | 均值 |
|---|---:|---:|---:|
| VPB point | .296544 | .298774 | .297659 |
| VPB block | .294371 | .296831 | .295601 |
| CLIP point | .292495 | .286712 | .289603 |
| CLIP block | .279953 | .284288 | .282120 |

控制：position_only=.297467；equal_score_fixed_block=.297467；
select_all=.292570；fixed32_random=.285002。
position_only为只拟合train块标签的三次多项式；random为固定0–31种子的原帧随机排序。
位置控制与同分前缀在.85上相同是本次排序结果，不是手工指定相等。

VPB block−point：seed13 −.002173，source bootstrap95%CI[−.010575,.005216]；
seed17 −.001943，CI[−.015237,.009990]；均值−.002058。
预定门要求均值≥.005、两seed非负且block胜全部控制：**失败**。
CLIP block−VPB block：−.014419 / −.012543；置信区间均跨零。
因此不能宣称CLIP普遍更差，但本轮不支持换CLIP。
VPB point也未两seed都超过位置控制，不能称为稳定学到了有效高光排序。

辅助：F@.25，VPB point=.225054/.243905、block=.240831/.246058；
同分前缀控制=.264970、全选=.292570。block在低keep优于point的趋势也没有胜过这些控制。
不放松主指标，不换seed补跑，不将上述读数乘100当作复赛分数。

## AHA

按20min timeout、offline、不安装/下载的有界协议启动，2.455秒首次失败：
`models/live_llava/video_head_live_llava_qwen.py` 导入 `wandb` 时 ModuleNotFoundError。
**模型未加载、帧未解码、无视觉前向、无三头结果，无质量判断。**
此外SigLIP cache存在.incomplete文件；尚未验证是否可用，不能把它当作完整资产。
本轮保存失败并停止，不把启动脚本当作真实视觉smoke通过。

## 复核、资源与可复现性

`verification.json`：独立排序实现重建8臂×3keep原始帧F，全部精确相等；
保存checkpoint在CPU重放，概率最大误差≤1e−5；train/dev源交集0。
原始权重SHA、实际采样PTS、media/axis SHA、环境、命令、逐源预测均随本报告入库。
脚本状态以`git_commit.txt`和`code_sha256.txt`绑定；检查脚本另在源码目录。
单卡RTX4080 32GB，训练外层45min timeout；每个小头300epoch实测.55–.70秒。
抽特征VPB3.14秒/CLIP3.96秒，不含前面的CPU解码与校验。
PyTorch峰值allocated459601408字节（不是nvidia-smi整机峰值）。
每头394241参数；实际checkpoint字节见metrics。无最终比赛推理组合导出，故不认证大小档。
VPB预训练文件228815765字节；CLIP源文件598530372字节，包含未使用的CLIP文本等权重，
不能把源文件大小当作已导出vision-only推理大小。
远端features/checkpoints保留于`/root/autodl-tmp/experiments/R24_EXEC_20261015`，大文件不入Git。

限制：14源dev很小且已用于本轮研究；无全局fresh保证；无可信域标签、native crop GT或专门无高光池。
12帧全视频输入与公开摘要标签均可能失配真实高光任务，不能由本轮失败推断模型容量或内容信号不存在。

## 后续建议

优先检验采样分辨率：固定本轮源划分和表示，先用原始GT计算12/48采样的排序oracle差距，
再决定是否值得做48采样的可学对照；oracle不是模型增益。暂停当前12帧block配方。
AHA后续若继续，应单独做依赖与视觉塔完整性修复，再重开有界视觉实验；本轮没有蒸馏依据。
空间方向仍需native crop GT，不能从这些时间指标排除空间瓶颈。
