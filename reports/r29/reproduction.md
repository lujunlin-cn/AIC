# R29 复现与证据索引

2026-10-16；工作目录 `/home/hajimi2025/AIC`；执行时父代码 `d3e6a40e417321be67229f2add54aca34f80d858`。R29 新脚本在执行时尚未提交，因此以脚本、协议和输入 SHA256 绑定实际实现，不能仅凭父 commit 复现。主协议为 [R29_BREAKTHROUGH_20261016.json](../../configs/R29_BREAKTHROUGH_20261016.json)，结论见 [summary.md](summary.md)，源码中的信息、梯度和时轴合同见 [pipeline_contract.md](pipeline_contract.md)。

本页记录已经发生的计算，并另列**尚未执行的安全重跑方法**。数学穷举是合成后验验证；公共时间是 VideoXum 摘要机制诊断；空间是四个已暴露 RV 源的工程 pilot。均不是独立新源确认、可上传新候选或官方涨分证据。没有训练 encoder、访问 protected14、使用赛事测试标签或自动上传。下一阶段监督协议尚未执行。

## 1. 输入恢复、位置与来源

公开证据导出器为 [r29_collect_public_evidence.py](../../scripts/r29_collect_public_evidence.py)。它在远端只读取 `/root/autodl-tmp/experiments/R24_EXEC_20261015/bindings.json` 的 40 个唯一 TRAIN/DEV 绑定：26 TRAIN、14 已暴露 DEV，无 protected。每源导出三项：

- `R22_INSTRUMENT_20261015/axis_arrays/<id>.npz`：原始 PTS、十人原帧摘要成员标签等轴数据。
- `R26_DIAG_20261016/scores_native_shape/<id>.json`：已缓存的 AHA 原生约 1fps 分数、采样原帧和实际 PTS。
- `R24_EXEC_20261015/<id>.npz`：原有公共特征证据；R29 时间读出不重新运行 encoder。

空间仅按 R28 冻结配置读取 RV `031/032/033/034` 的六人注释及缓存 PNG；两个比例、每源三个帧位置。12 个 PNG 产生 24 个 frame×ratio 项。导出器另保存 `sampled_rv_gt.json`、`bindings.json`、`source_hashes.json`。后者绑定 132 个 payload（120 个时间文件＋12 个 PNG），本页整理时已逐一重新校验匹配。原 RV 媒体路径曾不可访问，当前证据是缓存名称/像素 SHA 与注释绑定，**不是重新完成的原媒体解码/帧号 QC**。

远端归档：`/root/autodl-tmp/experiments/R29_BREAKTHROUGH/exposed_public_evidence.tar.gz`。本地归档：`artifacts/r29/exposed_public_evidence.tar.gz`；已解包：`artifacts/r29/public_evidence/`。`artifacts/` 和二进制特征不进入 Git；完整报告 checkout 仍须恢复这些公开输入才能重跑。精确的历史导出/传输 shell 调用未单独保存，下述恢复命令是等价恢复方法，不冒充历史终端转录。

| 文件 | bytes | SHA256 |
|---|---:|---|
| `exposed_public_evidence.tar.gz` | 6265219 | `21629035232bf5434309c09f3230f892063bac972d84fbf31a159686edabf7f9` |
| `public_evidence/bindings.json` | 26466 | `bf2a877af1c766acaca81dd927729bef96999a31d95dd7e87e71d41042f4531e` |
| `public_evidence/source_hashes.json` | 15626 | `67ff50b9c6f6fbfb1e85972f69594a49ac2248d5a84b2a51fbd1711bac6e719e` |
| `public_evidence/sampled_rv_gt.json` | 15345 | `178cb21ebb9b568cc80917c6444545462fbf8e2bd2e243843c804b70e6889717` |
| `reports/r22/r22_instrument_report.json` | 104361 | `abf3f30d66004dd31dc940a9cac4446f0ec032bf888afc08ea685f18c7422756` |
| `reports/r26/public_metrics.json` | 115269 | `24ab360abc5d2a20e46620446acb34038103bf17c03f5d97428e048b201336a0` |

以下示例将归档恢复到**新目录**。它没有在编写本页时运行。SSH 保留严格 host-key 验证；`HostKeyAlias` 是本轮已保存 SSH 检查使用的历史可信主机别名，见 [closeout_resource.json](closeout_resource.json)，并非关闭校验。若本机没有对应可信 known_hosts 条目，应先正常核实服务器身份。

```bash
task_recovery=artifacts/r29/recovery_20261016T120000Z
mkdir "$task_recovery"
scp -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=yes \
  -o 'HostKeyAlias=[connect.westc.seetacloud.com]:40206' -P 40188 \
  root@connect.westd.seetacloud.com:/root/autodl-tmp/experiments/R29_BREAKTHROUGH/exposed_public_evidence.tar.gz \
  "$task_recovery/exposed_public_evidence.tar.gz"
python3 - "$task_recovery" <<'PY'
import hashlib, json, sys, tarfile
from pathlib import Path
root = Path(sys.argv[1])
archive = root / 'exposed_public_evidence.tar.gz'
assert hashlib.sha256(archive.read_bytes()).hexdigest() == '21629035232bf5434309c09f3230f892063bac972d84fbf31a159686edabf7f9'
dest = root / 'public_evidence'
dest.mkdir(exist_ok=False)
with tarfile.open(archive, 'r:gz') as tar:
    for member in tar.getmembers():
        path = Path(member.name)
        assert member.isfile() and not path.is_absolute() and '..' not in path.parts
    tar.extractall(dest, filter='data')
expected_metadata = {
    'bindings.json': 'bf2a877af1c766acaca81dd927729bef96999a31d95dd7e87e71d41042f4531e',
    'source_hashes.json': '67ff50b9c6f6fbfb1e85972f69594a49ac2248d5a84b2a51fbd1711bac6e719e',
    'sampled_rv_gt.json': '178cb21ebb9b568cc80917c6444545462fbf8e2bd2e243843c804b70e6889717',
}
for name, digest in expected_metadata.items():
    assert hashlib.sha256((dest / name).read_bytes()).hexdigest() == digest, name
hashes = json.loads((dest / 'source_hashes.json').read_text())
assert len(hashes) == 132
for name, digest in hashes.items():
    path = Path(name)
    assert not path.is_absolute() and '..' not in path.parts
    assert hashlib.sha256((dest / path).read_bytes()).hexdigest() == digest, name
bindings = json.loads((dest / 'bindings.json').read_text())
assert len(bindings) == len({r['video_id'] for r in bindings}) == 40
assert {r['split'] for r in bindings} == {'train', 'dev'}
print('Verified public input payloads:', len(hashes))
PY
```

不要直接重跑导出器覆盖远端原归档：gzip/tar 元数据也可能改变归档 bytes，且现有归档已经绑定。缺包可在单独环境安装对应版本；不要为重跑修改原 evidence 环境。

## 2. 已执行的 CPU 计算

CPU 环境为 `spark-e602`、Linux `6.17.0-1026-nvidia-aarch64-with-glibc2.39`，Python `3.13.12`（conda-forge），NumPy `2.4.4`、SciPy `1.17.1`，空间图像使用 Pillow `12.2.0`。时间运行设置 `OPENBLAS_NUM_THREADS=1`、`OMP_NUM_THREADS=1`。以下相对路径均以仓库根目录为工作目录。

### 2.1 数学目标与 decoder 穷举

实际命令：

```bash
timeout 5m python3 scripts/r29_decision_theory.py --output reports/r29/decision_theory.json
```

协议 [decision_theory_protocol.json](decision_theory_protocol.json) 的 SHA256 为 `5cfe51540500068527cfbcb1bb722342f02b894fb230b52a391c4c2600f72fc2`；脚本 SHA 为 `c103281b24bf2bdca7251edb84759bf5fbd1d3bccbecc7f9683313cd9b843fbe`。结果和实测环境在 [decision_theory.json](decision_theory.json)：CPU 3.7933 秒、exit 0。240 次 decoder 对全动作穷举、51200 个已知 M 等式、40640 个增删等式、81 个独立 Bernoulli 排序检验、24 个动态规划检验均 PASS，最大精确差 0。48 个概率的约 0.0078 秒 profile 使用**合成概率**，不是 48 次视频观测或实测视频推理成本。

### 2.2 TRAIN 校准与公共集合决策

实际命令：

```bash
/usr/bin/time -v timeout --signal=TERM --kill-after=30s 30m \
  /home/hajimi2025/miniforge3/bin/python scripts/r29_public_decision_probe.py \
  --config reports/r29/temporal_probe_config.json
```

冻结配置 [temporal_probe_config.json](temporal_probe_config.json) SHA 为 `57cff3487b9fbe6e6ebdbab1cfb491b0ea1630c6e2a39ddc13f4f9c0ead9b5c4`；脚本 SHA 为 `efe9a8ea0da5d4af1231eb4cd8f008ef30f44563e3234b438670899d7436fe3a`。配置绑定上述两个历史报告及 `artifacts/r29/public_evidence/`。它拒绝使用已经存在的输出目录，不能对原目录直接重跑。

输出 [public_decision_probe/](public_decision_probe/) 包括原配置、输入绑定、拟合系数、逐源结果、源级 bootstrap 和观察/广播诊断。`metrics.json` SHA 为 `9c581ad2a8bd36ec00420ac7da190e265e3d961a9fa0ab17ec0c1597c903ca6d`。环境 [temporal_probe_environment.json](temporal_probe_environment.json) 记录 wall 1.0249 秒、exit 0；[temporal_probe_resource.log](temporal_probe_resource.log) 记录 elapsed 1.01 秒、最大 RSS 113048 KiB、exit 0；stdout 在 [temporal_probe_stdout.log](temporal_probe_stdout.log)。没有 GPU 使用或 encoder 梯度。

### 2.3 独立数值复核与原生观察附录

实际命令：

```bash
/home/hajimi2025/miniforge3/bin/python scripts/r29_temporal_independent_review.py
OPENBLAS_NUM_THREADS=1 python reports/r29/public_decision_probe/native_observation_recompute.py
```

独立复核只读取已拟合参数、不重新拟合。脚本 SHA `f627dc88e3f8dfbcfd2bad64bdd2641affb0dba1e97dff41e67b3a2251a09164`；[temporal_independent_review.json](temporal_independent_review.json) 保存 `PASS_WITH_SCIENTIFIC_LIMITS`、0.7059 秒、exit 0、520 个源×arm 的 F 最大差 0。数值独立复核不是独立数据确认。

原生观察附录脚本 SHA `60a319306c0981b979707e0f09ff755d5240ecf6c672c3eabd8122a8f9485827`；[native_observation_appendix.json](public_decision_probe/native_observation_appendix.json) 保存 0.0903 秒、父 metrics SHA 和完整源表。该 JSON 未单列退出码，不能将补写的退出码冒充原始记录。它是结果后的描述性附录，不拟合、不据此选择 DEV 超参；只检验现有原生 1fps 对摘要正 run 的命中，不证明动作/语言可观测，也不证明 R25 metadata-FPS 轴与公共 actual-PTS 轴等价。

这两个脚本没有 output CLI，读写路径固定。尤其独立复核会覆盖 `reports/r29/temporal_independent_review.json`，附录会覆盖原附录；请使用第 5 节的隔离工作目录。

## 3. 已执行的空间续跑：保留旧 13 条，只生成 11 条

真实执行快照是 [spatial_completion/r29_run_spatial.sh](spatial_completion/r29_run_spatial.sh)。远端工作目录 `/root/autodl-tmp/experiments/R29_BREAKTHROUGH`；Python 为 `/root/autodl-tmp/experiments/R28_SPATIAL_AUDIT/qwen_env/bin/python`；冻结配置 `/root/autodl-tmp/experiments/R28_SPATIAL_AUDIT/R28_SPATIAL_PUBLIC_PILOT.json`；旧 13 条输入 `/root/autodl-tmp/experiments/R28_SPATIAL_AUDIT/public_pilot_cached/predictions.jsonl`。

实际 preflight 和 GPU 命令（变量定义见启动器快照）：

```bash
"$task_python" r29_spatial_continue.py --config "$task_config" \
  --previous "$task_previous" --output spatial_continuation --validate-only > spatial_preflight.json
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 20m \
  "$task_python" r29_spatial_continue.py --config "$task_config" \
  --previous "$task_previous" --output spatial_continuation > spatial_continuation.log 2>&1
```

输入校验为 expected 24、previous 13、remaining 11，加载模型前检查旧行与同 PNG/几何/注释相符。原始 R28 未完成运行材料保存在 [recovered_r28/](recovered_r28/)，没有用 R29 续跑伪装旧运行完整。

| 绑定 | SHA256 |
|---|---|
| 续跑脚本 | `8fbf88dec2a2363d9dbd0c9ca169fd30561cbf821af2fb02acf9b77aa7c777b5` |
| R28 冻结配置 | `52ee988dd66cb9e69b78b2eae034600766209234416917f6ba81c916b1912530` |
| 原 13 条 predictions | `4ff041352250a0d9b70bf2f8ca62c583d44648ffa1749961ed6982178f0759df` |
| 完整 24 条 predictions | `6ec82c74f7bfe9466c239dbe59be62dd8b0ede2dd3f56c0a191164eae36c8113` |
| 续跑 sampled_gt.json | `a79a996c22a8d73a729034d40bf9eb00343a8f930c8c91c2d6cdbd73709f7776` |

### 启动器缺陷与修复身份

已执行旧启动器用了以下进程检查：

```bash
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | rg '[0-9]'; then
  exit 73
fi
```

远端没有 `rg`，[spatial_launcher.log](spatial_completion/spatial_launcher.log) 明确记录 `line 12: rg: command not found`。该管道在 `if` 条件中，没有触发 `set -e` 退出，因此检查失败开放。主负责人另行事前读取 GPU 状态为空闲；本轮没有处理外来进程，但**不能声称旧启动器有效执行了自动冲突保护**。

仓库当前 [scripts/r29_run_spatial.sh](../../scripts/r29_run_spatial.sh) 已改为：

```bash
task_gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$task_gpu_pids" ]]; then
  printf '%s\n' "$task_gpu_pids"
  exit 73
fi
```

在 `set -e` 下查询失败即退出，有 PID 也退出，不再依赖 rg。**修正版没有运行**；其身份与实际执行快照不同。它仍硬编码原工作目录，不能原样启动来覆盖证据；安全的新目录方法见第 6 节。

### 实测环境、资源与结束状态

[environment_observed.json](spatial_completion/spatial_continuation/environment_observed.json) 与 `pip_freeze.txt`：Python 3.12.3、Torch 2.8.0+cu128、Transformers 5.19.0、CUDA 12.8；GPU0 RTX4080，32760 MiB。冻结生成设置：seed 20261016、do_sample=false、thinking=false、max_new_tokens=128、max_image_side=640。

[metrics.json](spatial_completion/spatial_continuation/metrics.json)：24 条完成、13 条复用、11 条新推理；加载 8.1667 秒、新生成 14.0396 秒、总 27.2867 秒；peak allocated 9358655488 bytes、reserved 9435086848 bytes；[spatial_exit_code.txt](spatial_completion/spatial_exit_code.txt) 为 0。运行时唯一 Qwen 参数 4539265536，仅是此单模型 pilot 的统计，不能替代新组合系统的完整推理参数申报。

`gpu_before.csv` 为 1 MiB、util 0%；立即 `gpu_after.csv` 为 1 MiB、util **100%**（采样滞后），不能把这个文件说成 util 0%。后续只读 [closeout_resource.json](closeout_resource.json) 在 `2026-10-16T10:41:20.873435+00:00` 记录 GPU 1 MiB、util 0%、无计算进程，exit 0。结束检查未启动或停止任务。

完整快照、preflight、stdout/stderr、输入 SHA、Python/pip 清单及预测都保存在 [spatial_completion/](spatial_completion/)。它们说明这次真实视觉推理成功，不说明空间候选通过比赛晋级。

## 4. 已执行的空间 CPU 复算、GT 辅助候选与失败图

实际命令：

```bash
python3 scripts/r29_spatial_recompute.py \
  --predictions reports/r29/spatial_completion/spatial_continuation/predictions.jsonl \
  --sampled-gt reports/r29/spatial_completion/spatial_continuation/sampled_gt.json \
  --output reports/r29/spatial_recompute.json
python3 scripts/r29_spatial_oracle.py \
  --protocol configs/R29_SPATIAL_GT_ASSISTED_DIAGNOSTIC.json \
  --predictions reports/r29/spatial_completion/spatial_continuation/predictions.jsonl \
  --sampled-gt reports/r29/spatial_completion/spatial_continuation/sampled_gt.json \
  --output reports/r29/spatial_oracle.json
python3 scripts/r29_spatial_contact_sheet.py \
  --oracle reports/r29/spatial_oracle.json \
  --sampled-gt reports/r29/spatial_completion/spatial_continuation/sampled_gt.json \
  --image-root artifacts/r29/public_evidence/public_rv_images \
  --output reports/r29/spatial_failure_cases.png
```

[spatial_cpu_evidence.json](spatial_cpu_evidence.json) 记录三项均 exit 0，以及语法检查命令/退出状态、完整脚本和结果 SHA。复算与保存 IoU 最大差 `2.22e-16`。四源 24 项连续 raw-GT IoU 为 Qwen `.5996451766`、中心 `.5791167941`；129 候选 GT 辅助最优 `.868315`；原生 RV round/clamp/inclusive 口径另列，不能混为官方连续 IoU。

oracle 协议 SHA `760c095523464c30336a9d6d438ff6e27501160b42a9dd8afd217b6d306c156d`，oracle CPU 约 `.006` 秒。它在已见 GT 后诊断候选覆盖，不是当前输入的可学习上界。没有同帧 B3 对照。[spatial_failure_cases.png](spatial_failure_cases.png) 为 127325 bytes、SHA `acc2c128f794a5e46f7e9b2687ca710142c30be23617c486adf0f1e288ddd3ba`；图与指标共同支持坐标选择失败，不能单独归罪 LFM 编码器。进一步解释见 [spatial_review.md](spatial_review.md)。

## 5. 隔离 CPU 重跑：保留本轮原件

以下是**未执行的复现配方**，在仓库根目录运行，使用当前 Python；需上述 NumPy/SciPy/Pillow。它将原配置保持逐字相同，在新的 ignored `artifacts/r29/replay_*` 下镜像相对路径，解决无 CLI 脚本固定读写路径的问题。若输入是第 1 节刚恢复的新目录，只修改 `evidence` 为已校验的恢复目录，不改冻结配置。它不访问远端或 GPU。

```bash
python3 - <<'PY'
import datetime, json, os, shutil, subprocess, sys
from pathlib import Path

repo = Path.cwd().resolve()
assert (repo / 'scripts/r29_public_decision_probe.py').is_file()
evidence = repo / 'artifacts/r29/public_evidence'
assert (evidence / 'source_hashes.json').is_file()
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
replay = repo / 'artifacts/r29' / ('replay_' + stamp)
replay.mkdir(exist_ok=False)
for folder in ['scripts', 'reports/r29', 'reports/r22', 'reports/r26', 'artifacts/r29']:
    (replay / folder).mkdir(parents=True, exist_ok=True)
for name in ['r29_decision_theory.py', 'r29_public_decision_probe.py',
             'r29_temporal_independent_review.py']:
    shutil.copy2(repo / 'scripts' / name, replay / 'scripts' / name)
for name in ['decision_theory_protocol.json', 'temporal_probe_config.json']:
    shutil.copy2(repo / 'reports/r29' / name, replay / 'reports/r29' / name)
for relative in ['reports/r22/r22_instrument_report.json', 'reports/r26/public_metrics.json']:
    (replay / relative).symlink_to(repo / relative)
(replay / 'artifacts/r29/public_evidence').symlink_to(evidence, target_is_directory=True)
env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
records = []

def run(label, command):
    with (replay / (label + '.stdout.log')).open('w') as stdout, \
         (replay / (label + '.stderr.log')).open('w') as stderr:
        result = subprocess.run(command, cwd=replay, env=env, stdout=stdout, stderr=stderr)
    records.append({'label': label, 'command': command, 'exit_code': result.returncode})
    (replay / 'replay_commands.json').write_text(json.dumps(records, indent=2) + '\n')
    result.check_returncode()

run('theory', ['timeout', '5m', sys.executable, 'scripts/r29_decision_theory.py',
               '--output', 'reports/r29/decision_theory.json'])
run('temporal', ['/usr/bin/time', '-v', 'timeout', '--signal=TERM', '--kill-after=30s', '30m',
                 sys.executable, 'scripts/r29_public_decision_probe.py',
                 '--config', 'reports/r29/temporal_probe_config.json'])
run('independent_review', [sys.executable, 'scripts/r29_temporal_independent_review.py'])
native = Path('reports/r29/public_decision_probe/native_observation_recompute.py')
shutil.copy2(repo / native, replay / native)
run('native_observations', [sys.executable, str(native)])

predictions = repo / 'reports/r29/spatial_completion/spatial_continuation/predictions.jsonl'
gt = repo / 'reports/r29/spatial_completion/spatial_continuation/sampled_gt.json'
spatial = replay / 'spatial_cpu'
spatial.mkdir()
common = ['--predictions', str(predictions), '--sampled-gt', str(gt)]
run('spatial_recompute', [sys.executable, str(repo / 'scripts/r29_spatial_recompute.py'),
                         *common, '--output', str(spatial / 'recompute.json')])
run('spatial_oracle', [sys.executable, str(repo / 'scripts/r29_spatial_oracle.py'),
                      '--protocol', str(repo / 'configs/R29_SPATIAL_GT_ASSISTED_DIAGNOSTIC.json'),
                      *common, '--output', str(spatial / 'oracle.json')])
run('contact_sheet', [sys.executable, str(repo / 'scripts/r29_spatial_contact_sheet.py'),
                     '--oracle', str(spatial / 'oracle.json'), '--sampled-gt', str(gt),
                     '--image-root', str(evidence / 'public_rv_images'),
                     '--output', str(spatial / 'failure_cases.png')])
print('Isolated replay:', replay)
PY
```

重跑主要比对输入/配置绑定、逐源 F、拟合系数、穷举 PASS 和空间 IoU；`wall_seconds`、新父 commit、结果文件中的父 metrics SHA 等会随环境/重跑改变，不能要求所有结果 JSON bytes 与旧文件相同。各脚本内嵌 `command` 是冻结的历史字符串；隔离重跑的实际命令/退出状态以新 `replay_commands.json` 和 stdout/stderr 为准。此配方本身未被当成已执行新实验。

## 6. 如需再次运行空间 GPU：新路径且先查占用

本页未启动新 GPU 运行。若另一次研究确需复现，使用本轮已存在环境、权重、冻结 config 和原 13 条输入，无需重新下载。应先上传当前续跑脚本到一个新的任务目录，并绑定其 SHA；不要直接运行硬编码原目录的启动器。以下仅为远端新目录执行示例；它不生成与本轮相同身份的历史证据。

```bash
set -euo pipefail
task_root=/root/autodl-tmp/experiments/R29_SPATIAL_REPLAY_20261016T120000Z
task_python=/root/autodl-tmp/experiments/R28_SPATIAL_AUDIT/qwen_env/bin/python
task_config=/root/autodl-tmp/experiments/R28_SPATIAL_AUDIT/R28_SPATIAL_PUBLIC_PILOT.json
task_previous=/root/autodl-tmp/experiments/R28_SPATIAL_AUDIT/public_pilot_cached/predictions.jsonl
mkdir "$task_root"
cp /root/autodl-tmp/experiments/R29_BREAKTHROUGH/r29_spatial_continue.py "$task_root/"
cd "$task_root"
sha256sum r29_spatial_continue.py "$task_config" "$task_previous" > spatial_input_sha256.txt
"$task_python" r29_spatial_continue.py --config "$task_config" --previous "$task_previous" \
  --output spatial_continuation --validate-only > spatial_preflight.json
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv > gpu_before.csv
task_gpu_pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$task_gpu_pids" ]]; then
  printf '%s\n' "$task_gpu_pids"
  exit 73
fi
"$task_python" --version > python_version.txt
"$task_python" -m pip freeze > pip_freeze.txt
set +e
CUDA_VISIBLE_DEVICES=0 timeout --signal=TERM --kill-after=30s 20m \
  "$task_python" r29_spatial_continue.py --config "$task_config" --previous "$task_previous" \
  --output spatial_continuation > spatial_continuation.log 2>&1
task_exit=$?
set -e
printf '%s\n' "$task_exit" > spatial_exit_code.txt
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv > gpu_after.csv
exit "$task_exit"
```

这里 `cp` 的源应与绑定的续跑脚本 SHA 相同；若旧远端目录已变，先恢复对应脚本。query 失败或检测到任何 compute PID 都中止，不杀进程。显存/利用率记录还须人工核对，PID 检查不是资源预约，不能保证检查之后没有任务进入。

## 7. 最终 manifest：首次生成与只读验证

在所有纳入的报告、脚本、配置和项目状态文档定稿后，由主负责人首次运行：

```bash
python3 scripts/r29_evidence_manifest.py
python3 scripts/r29_evidence_manifest.py --verify
```

[evidence_manifest.json](evidence_manifest.json) 绑定 R29 报告/代码/配置、主状态文档、登记、历史轴报告，以及 ignored 公开输入。生成器**拒绝覆盖**已存在 manifest；后续只运行 `--verify`。本页写作时未代主负责人生成 manifest。生成后不再修改纳入文件；有后续研究应使用新实验目录/新 manifest，不重写这批原件。

完整证据只能复现本轮的实现和开发判断。公共时间与 RV 空间不是同源同帧联合 GT，不能相乘推算官方 Score；同版本可运行官方 evaluator、原 B3 完整部署材料和独立新源确认仍是独立缺口。
