# R26 复现与数据边界

基准代码父提交 `b56a9e7`。实际运行脚本和配置均另存SHA256；完整环境与模型文件SHA在AutoDL隔离目录 `/root/autodl-tmp/experiments/R26_DIAG_20261016`。未训练新模型，使用授权GPU0，公共集评分带42300秒超时。原来写成`11h45m`的timeout在启动前报错，已改成合法秒数后启动；失败未占用GPU或生成评分。

公共数据根目录是R22已绑定轴的26TRAIN/14DEV，按media/axis SHA校验，使用实际PTS最近帧1fps采样；每帧先等比例384黑边padding，再沿用R25 image processor、generic query、BF16、49视觉token、chunk8、完整因果缓存。DEV是已在R24观察过的公共源，不是新holdout。预留14源未打开。

```bash
# AutoDL；command.txt中同一命令，完整日志public_score.log
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=3m 42300s \
 /root/autodl-tmp/aha_env/bin/python r26_public_score.py \
 --root /root/autodl-tmp/experiments/R26_DIAG_20261016 \
 --model /root/autodl-tmp/experiments/R25_AHA_SUBMIT_20261015/merged_llava_bf16_v2 \
 --index /dev/null --spatial /dev/null --config R26_DIAG_20261016.json
/root/autodl-tmp/aha_env/bin/python r26_public_evaluate.py --config R26_DIAG_20261016.json

# 本地；426原始评分、index、B3全帧框属于私有/ignored输入
PYTHONPATH=. python scripts/r26_build_diagnostics.py \
 --index /tmp/r25_index.jsonl --scores artifacts/r26/r25_scores \
 --spatial artifacts/r26/b3_allframes.jsonl --r25 submissions/R25_AHA_BF16_K085 \
 --output submissions/R26_DIAG --report reports/r26
python scripts/r26_plot_diagnostics.py
```

`--index /dev/null --spatial /dev/null`是公共评分脚本继承R25加载器的未使用必填参数；公共数据来源只取config.data_root及其已冻结train/dev列表。

公共对照：info、rel、sum三臂；纯位置prefix/suffix/center、仅TRAIN拟合三次位置回归、32固定随机种子、全帧。主要预算.85，预注册次要预算.60/.25；所有非wholecell/non-all控制均逐视频精确round(N*keep)。公共所有者/最后cell截断按PTS最近距离，官方固定R25评分使用原帧距离。最后cell截断以距采样点最近优先、距离同分早帧优先，排序cell同分早采样点优先。

公共指标：逐原帧10份人工摘要mask，分别算F1后按标注者均值、再按源宏平均；CI为14源配对bootstrap10000次、seed20261016。不是官方联合时空Score。不依据DEV选择head、预算、阈值或seed。随机对照每源每seed固定重置，不必与R24随机数流相同。

官方对照ZIP使用固定时间戳/压缩参数，输入和Python/zlib版本相同时可重建字节；每包根目录仅predictions.jsonl。独立validator与项目contract均校验完整426视频。R26三包现已用户回报分数，见official_score_events.json；未独立核对平台上传SHA。重跑构建脚本会生成原始“待上传”manifest，不得覆盖后续分数事件作为当前状态。

参数组成：AHA8,030,363,168；B3=LFM2.5-VL448,718,848 + spatial head1,511,425 + YuNet53,104 =450,283,377。文件bytes只用于存储与复现，model_size_mb兼容字段沿用参数百万单位并显式记录语义。无AHA的position控制450,283,377参数，无B3的center空间控制8,030,363,168参数；不虚报同参数。当前项目记录复赛回分不乘k_size；参数仍如实声明。

新包同时写入规范`model_params_b`字段，项目contract按实际参数量交叉核验；保留legacy字段用于兼容。
