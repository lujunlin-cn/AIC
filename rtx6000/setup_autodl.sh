#!/bin/bash
# R15 AutoDL bring-up (RUN ON THE AUTODL BOX, in GPU mode).
# AutoDL instance is a bare image (no python3/conda confirmed 10-12 night),
# so this builds everything from zero. Pair with rtx6000/ code; data comes
# from the evac bundle already under /root/autodl-tmp/evac.
#
# Sequence after this script: bash /root/AIC/rtx6000/r15_smoke.sh
set -euo pipefail

EVAC=/root/autodl-tmp/evac
PIP=${PIP:--i https://mirrors.aliyun.com/pypi/simple/}
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=0

echo "== [0/6] GPU sanity =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

echo "== [1/6] base python (Miniconda via TUNA) =="
if [ ! -x /root/miniconda3/bin/python3 ]; then
  wget -q https://mirrors.tuna.tsinghua.edu.cn/anaconda/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
  bash /tmp/mc.sh -b -p /root/miniconda3
fi
export PATH=/root/miniconda3/bin:$PATH
python3 --version

echo "== [2/6] unpack repo + path mirroring =="
if [ ! -d /root/AIC ]; then
  tar xzf "$EVAC"/root/AIC_repo_*.tgz -C /root
  echo "repo unpacked (all branches incl. evac snapshot)"
fi
# the evac tar is the 910A-side snapshot and LACKS the latest rtx6000/ code;
# the sandbox pushes the live copy to /root/autodl-tmp/rtx6000 -- prefer it
if [ -d /root/autodl-tmp/rtx6000 ]; then
  cp -r /root/autodl-tmp/rtx6000/. /root/AIC/rtx6000/
  echo "rtx6000/ overlaid from /root/autodl-tmp (latest, incl. 5-gate parity)"
fi
# 910A-identical layout so pool frames_dir absolute paths stay valid
[ -e /data ] || ln -s /root/autodl-tmp/evac/data /data
ls /data/aic/pretrained/Qwen3.5-4B/config.json

echo "== [3/6] roll env (vLLM stack) =="
if [ ! -x /root/miniconda3/envs/roll/bin/python ]; then
  conda create -n roll -y python=3.12
  /root/miniconda3/envs/roll/bin/pip install $PIP vllm "transformers>=5.19" \
    opencv-python-headless
fi
ROLL_PY=/root/miniconda3/envs/roll/bin/python
$ROLL_PY - <<'EOF'
import torch, transformers
import transformers.models.qwen3_5.modeling_qwen3_5  # Qwen3.5 present?
print(f"roll: torch {torch.__version__} transformers {transformers.__version__}")
EOF

echo "== [4/6] train env (torch pinned to roll's torch) =="
if [ ! -x /root/miniconda3/envs/train/bin/python ]; then
  TORCH_VER=$($ROLL_PY -c "import torch;print(torch.__version__.split('+')[0])")
  conda create -n train -y python=3.12
  /root/miniconda3/envs/train/bin/pip install $PIP \
    "torch==$TORCH_VER" "transformers==5.19.0" peft accelerate \
    flash-linear-attention opencv-python-headless sentencepiece
fi
TRAIN_PY=/root/miniconda3/envs/train/bin/python
$TRAIN_PY - <<'EOF'
import torch, transformers, peft
import transformers.models.qwen3_5.modeling_qwen3_5  # noqa: F401
try:
    import fla  # noqa: F401
    print("fla available -> GDN fast path")
except Exception as e:  # noqa: BLE001
    print(f"fla unavailable ({e}) -> safe-chunk fallback")
print(f"train: torch {torch.__version__} transformers {transformers.__version__} peft {peft.__version__}")
EOF

echo "== [5/6] GPU compute check (sm_120 bf16) =="
$TRAIN_PY - <<'EOF'
import torch
cap = torch.cuda.get_device_capability(0)
print(f"{torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]}")
assert cap >= (12, 0), "need Blackwell sm_120"
x = torch.randn(1024, 1024, device="cuda:0", dtype=torch.bfloat16)
assert torch.isfinite((x @ x).float().sum()), "bf16 matmul broken"
print("sm_120 bf16 matmul OK")
EOF

echo "== [6/6] summary =="
cat <<EOF

ROLL_PY=$ROLL_PY
TRAIN_PY=$TRAIN_PY
repo:   /root/AIC (evac snapshot; rtx6000/ included)
data:   /data -> /root/autodl-tmp/evac/data (910A-identical layout)

next: cd /root/AIC && bash rtx6000/r15_smoke.sh
      (smoke 默认 ADAPTER=/data/aic/experiments_910a/LFM_V13/q35_sft_v2_out/ckpt_030)
EOF
