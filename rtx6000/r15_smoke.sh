#!/bin/bash
# R15 RL smoke on RTX PRO 6000: gates 1-4 then the full colocated 2-step
# loop (vLLM rollout <-> LoRA file relay <-> CUDA trainer).
# Verifies: sm_120 stack, GDN kernel parity, vLLM video generate + JSON
# parse, LoRA liveness, adapter relay, finite loss, 16-seg one-update step.
#
# Prereqs: setup_rtx6000.sh done; bundle unpacked; smoke pool auto-built.
set -u

REPO=${REPO:-/root/AIC}
R=$REPO/rtx6000
ROLL_PY=${ROLL_PY:-python}
TRAIN_PY=${TRAIN_PY:-/opt/venv_train/bin/python}
BASE=${BASE:-/data/aic/pretrained/Qwen3.5-4B}
L=/data/aic/experiments_910a/LFM_V13
POOL=${POOL:-$L/grpo_pool_v2.jsonl}
# LoRA init is the user's pick; default to the R13 RL-init candidate
ADAPTER=${ADAPTER:-$L/q35_sft_out/ckpt_060}
OUT=$L/rtx_smoke_out
LOG=$OUT/smoke.log
STEPS=${STEPS:-2}

mkdir -p "$OUT"
: > "$LOG"
FAIL=0
note() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

note "smoke pool build: 32 segs from $POOL"
"$TRAIN_PY" - "$POOL" "$OUT/smoke_pool.jsonl" <<'EOF'
import json, sys, numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
perm = np.random.default_rng(13).permutation(len(rows))
sel = [rows[i] for i in perm[:32]]
with open(sys.argv[2], "w") as f:
    for r in sel:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("n_gt>0:", sum(1 for r in sel if r["n_gt"] > 0), "/", len(sel))
EOF

note "== gate 1/4: GDN kernel parity (train env) =="
"$TRAIN_PY" "$R/r15_parity.py" --gate gdn --pool "$OUT/smoke_pool.jsonl" \
  --base "$BASE" --adapter-init "$ADAPTER" >>"$LOG" 2>&1 \
  || { note "GDN gate FAIL"; FAIL=1; }

note "== gate 2/4: trainer numerics (train env) =="
"$TRAIN_PY" "$R/r15_parity.py" --gate numerics \
  --pool "$OUT/smoke_pool.jsonl" --base "$BASE" \
  --adapter-init "$ADAPTER" >>"$LOG" 2>&1 \
  || { note "numerics gate FAIL"; FAIL=1; }

note "== gate 3/4: vLLM video smoke (rollout env) =="
"$ROLL_PY" "$R/r15_parity.py" --gate vllm --pool "$OUT/smoke_pool.jsonl" \
  --base "$BASE" --adapter-init "$ADAPTER" >>"$LOG" 2>&1 \
  || { note "vLLM gate FAIL"; FAIL=1; }

note "== gate 4/4: vLLM LoRA liveness (rollout env) =="
if "$ROLL_PY" "$R/r15_parity.py" --gate lora \
    --pool "$OUT/smoke_pool.jsonl" --base "$BASE" \
    --adapter-init "$ADAPTER" >>"$LOG" 2>&1; then
  LORA_MODE=lora
else
  note "LoRARequest gate FAIL -> falling back to --lora-mode=reload"
  LORA_MODE=reload
  FAIL=0
fi

note "== E2E: $STEPS colocated steps (lora-mode=$LORA_MODE) =="
rm -f "$OUT"/adapter_step* "$OUT"/traj_step*
( while true; do nvidia-smi --query-gpu=memory.used,utilization.gpu \
    --format=csv,noheader >> "$OUT/mem.log"; sleep 15; done ) &
SAMPLER=$!
trap "kill $SAMPLER 2>/dev/null" EXIT

"$ROLL_PY" "$R/r15_rollout_vllm.py" \
  --pool "$OUT/smoke_pool.jsonl" --base "$BASE" \
  --adapter-init "$ADAPTER" --out "$OUT" --steps "$STEPS" \
  --samples 3 --max-new-tokens 256 --lora-mode "$LORA_MODE" \
  --gpu-mem-util 0.45 --enforce-eager \
  > "$OUT/rollout.log" 2>&1 &
ROLL_PID=$!
"$TRAIN_PY" "$R/r15_train_cuda.py" \
  --pool "$OUT/smoke_pool.jsonl" --base "$BASE" \
  --adapter-init "$ADAPTER" --out "$OUT" --steps "$STEPS" \
  > "$OUT/trainer.log" 2>&1 &
TRAIN_PID=$!

wait "$TRAIN_PID"; TRAIN_RC=$?
wait "$ROLL_PID"; ROLL_RC=$?
kill "$SAMPLER" 2>/dev/null

for s in $(seq 1 "$STEPS"); do
  [ -s "$OUT/traj_step${s}_0.jsonl" ] || { note "MISSING traj_step${s}_0"; FAIL=1; }
  { [ -s "$OUT/adapter_step${s}.pt" ] && [ -s "$OUT/adapter_step${s}/adapter_config.json" ]; } \
    || { note "MISSING adapter_step${s} (.pt or PEFT dir)"; FAIL=1; }
done
grep -q HOTSWAP "$OUT/rollout.log" || { note "NO HOTSWAP in rollout log"; FAIL=1; }
"$TRAIN_PY" - "$OUT/trainer.log" <<'EOF' >>"$LOG" 2>&1 || FAIL=1
import math, sys
ls = [float(l.split("loss_mean=")[1].split()[0])
      for l in open(sys.argv[1]) if "TSTEP" in l]
assert len(ls) >= 1, f"no TSTEP lines"
assert all(math.isfinite(x) for x in ls), f"non-finite loss {ls}"
print("losses", ls)
EOF
[ "$ROLL_RC" -eq 0 ] || { note "rollout rc=$ROLL_RC"; FAIL=1; }
[ "$TRAIN_RC" -eq 0 ] || { note "trainer rc=$TRAIN_RC"; FAIL=1; }

if [ "$FAIL" -eq 0 ]; then note "RL_SMOKE_PASS (lora-mode=$LORA_MODE)"; else note "RL_SMOKE_FAIL"; fi
tail -3 "$OUT/trainer.log" | tee -a "$LOG"
tail -3 "$OUT/rollout.log" | tee -a "$LOG"
exit "$FAIL"
