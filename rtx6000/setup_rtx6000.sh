#!/bin/bash
# R15 RTX PRO 6000 one-shot environment setup. Run INSIDE the container
# (recommended base image: vllm/vllm-openai:<2025-03+ tag> -- ships
# torch>=2.7+cu128 with sm_120 kernels and a matching vLLM).
#
# Layout contract (identical to 910A, so pool frames_dir paths stay valid):
#   /data/aic/pretrained/Qwen3.5-4B        base model
#   /data/aic/experiments_910a/LFM_V13     pools / labels / outputs
#   /data/aic/experiments_910a/PHD2_FRAG_V1/frames/<vid>  extracted frames
#   /root/AIC                              this repo
#
# Two interpreters, one shared torch (venvs inherit the image's torch via
# --system-site-packages, so there is exactly one sm_120 torch on disk):
#   ROLL_ENV  (image env)     vLLM rollout / mining / parity gates
#   TRAIN_ENV (/opt/venv_train) trainer / merge / gdn+numerics gates
set -euo pipefail

BASE_MODEL=/data/aic/pretrained/Qwen3.5-4B
TRAIN_ENV=/opt/venv_train
REPO=${REPO:-/root/AIC}
HF_MIRROR=https://hf-mirror.com

echo "== [1/5] GPU / torch sanity =="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
PY=${ROLL_PY:-python}
$PY - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA not visible inside container"
cap = torch.cuda.get_device_capability(0)
name = torch.cuda.get_device_name(0)
print(f"torch {torch.__version__} cuda {torch.version.cuda} dev {name} sm_{cap[0]}{cap[1]}")
assert cap >= (12, 0), "expected Blackwell sm_120; wrong image?"
x = torch.randn(512, 512, device="cuda:0", dtype=torch.bfloat16)
assert torch.isfinite((x @ x).float().sum()), "bf16 matmul broken"
print("torch sm_120 bf16 matmul OK")
EOF

echo "== [2/5] vLLM + transformers pairing (rollout env) =="
$PY - <<'EOF'
import vllm, transformers
print("vllm", vllm.__version__, "| transformers", transformers.__version__)
import transformers.models.qwen3_5.modeling_qwen3_5 as mm  # noqa: F401
print("Qwen3.5 modeling import OK")
EOF
# the vLLM image may ship an older transformers without Qwen3.5; the model
# processor (config classes) must exist for both engines
if ! $PY -c "import transformers.models.qwen3_5.modeling_qwen3_5" 2>/dev/null; then
  echo "image transformers lacks Qwen3.5 -> upgrading to 5.19.0 in rollout env"
  $PY -m pip install -q "transformers==5.19.0"
  $PY -c "import vllm; import transformers.models.qwen3_5.modeling_qwen3_5" \
    || { echo "FAIL: vLLM incompatible with transformers 5.19.0; pin a newer" \
         "vllm image or build /opt/venv_roll manually"; exit 1; }
fi
$PY -c "import cv2" 2>/dev/null || $PY -m pip install -q opencv-python-headless
$PY -c "import peft" 2>/dev/null || $PY -m pip install -q peft
echo "rollout env OK: $($PY -c 'import sys;print(sys.executable)')"

echo "== [3/5] training venv ($TRAIN_ENV) =="
if [ ! -x "$TRAIN_ENV/bin/python" ]; then
  $PY -m venv --system-site-packages "$TRAIN_ENV"
fi
TP=$TRAIN_ENV/bin/python
$TP -m pip install -q -U pip
# torch comes from the image via system-site-packages; verify before pinning
$TP -c "import torch; assert torch.cuda.is_available()" \
  || { echo "FAIL: venv does not see image torch"; exit 1; }
$TP -m pip install -q \
  "transformers==5.19.0" peft accelerate \
  flash-linear-attention opencv-python-headless sentencepiece
$TP - <<'EOF'
import torch, transformers, peft
import transformers.models.qwen3_5.modeling_qwen3_5 as mm  # noqa: F401
try:
    import fla  # noqa: F401
    print("fla available -> GDN uses triton kernels (fast path)")
except Exception as e:  # noqa: BLE001
    print(f"fla UNAVAILABLE ({e}) -> GDN falls back to safe torch chunk")
print(f"train env: torch {torch.__version__} transformers {transformers.__version__} peft {peft.__version__}")
EOF
echo "train env OK: $TP"

echo "== [4/5] base model =="
if [ -f "$BASE_MODEL/config.json" ]; then
  echo "base model present: $BASE_MODEL"
  du -sh "$BASE_MODEL"
else
  echo "downloading Qwen/Qwen3.5-4B via hf-mirror..."
  export HF_ENDPOINT=$HF_MIRROR
  export BASE_MODEL
  $PY -m pip install -q -U "huggingface_hub[cli]"
  $PY - <<'EOF'
import os
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen3.5-4B",
                  local_dir=os.environ["BASE_MODEL"], max_workers=8)
EOF
fi
test -s "$BASE_MODEL/config.json"

echo "== [5/5] summary =="
cat <<EOF

ROLL_PY=${ROLL_PY:-$(command -v python)}   # rollout/mine/vllm-parity interpreter
TRAIN_PY=$TP           # trainer/merge/gdn-parity interpreter
base model: $BASE_MODEL
repo:       $REPO

Day-1 gate order (from $REPO/rtx6000/):
  1. $TP  r15_parity.py --gate gdn      --pool <pool.jsonl> --base $BASE_MODEL --adapter-init <lora_dir>
  2. $TP  r15_parity.py --gate numerics --pool <pool.jsonl> --adapter-init <lora_dir>
  3. $PY r15_parity.py --gate vllm     --pool <pool.jsonl> --adapter-init <lora_dir>
  4. $PY r15_parity.py --gate lora     --pool <pool.jsonl> --adapter-init <lora_dir>
     (FAIL here => rollout must use --lora-mode=reload, see README)
  5. bash r15_smoke.sh   # 2-step colocated E2E
EOF
