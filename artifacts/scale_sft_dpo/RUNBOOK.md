# 35092 live preflight update

The GPU smoke and original LIVE recovery are complete; see `gpu_preflight_35092.md`. No full P1 position generation or formal training is authorized/running. Do not auto-start a long cache job merely because the GPU is available. The original singleton P1 projection is the default; batch16 failed coordinate parity.

Prebuilding a cache file is optional in principle, but actual P1 centers are still required for fixed-Position labels and coverage audits. The current executable pipeline consumes offline, hash-bound centers; it does not implement online P1 inference inside training. Removing cache writes would not eliminate model generation. Limited pilot validation is complete, while full-data coverage remains unmeasured.

When a full preflight scope is explicitly agreed, use the clean public labels at `/root/autodl-tmp/videoora42_clone/scale_sft_dpo_preflight_20261108/clean_public_labels/labels.jsonl`. `scripts/generate_p1_scale_positions.py` verifies and resumes complete sources and writes an actual usage receipt; use an external timeout as well. `scripts/calibrate_scale_training.py` consumes a complete hash-bound position join on CPU; its `--pilot` option cannot lock the formal configuration. Mapping changes require a new real A/B model smoke. Downstream measured SFT/candidate/preference/DPO stage gates remain mandatory. No trained checkpoint or learned Oracle result has been produced.

Historical CPU runbook follows.

# Executable entrypoints and remaining preflight

Run from the repository root. The original remote Python is `/root/autodl-tmp/videoora42_clone/.venv/bin/python` (torch 2.8.0+cu128 / transformers 5.5.4 / peft 0.20.0). Preserve it. Local tests use CPU PyTorch; model-specific imports are lazy.

```bash
python -m compileall -q aic scripts tests
PYTHONPATH=. pytest -q tests/test_scale_*.py
PYTHONPATH=. pytest -q
python -m scripts.audit_scale_readiness --root /home/hajimi2025/AIC
python -m scripts.verify_scale_real_geometry \
  --snapshot artifacts/scale_sft_dpo/remote_snapshot \
  --bins artifacts/scale_sft_dpo/config/scale_bins.json \
  --output artifacts/scale_sft_dpo/smoke_test/geometry_validation.json
```

`remote_snapshot/` is an ignored local audit input, not a committed dataset/checkpoint. It contains the small original public manifests/code inspected on the remote; verified remote paths and hashes are in `audit/remote_assets.json`. Copy those paths with strict pinned-host SSH if repeating the audit in a fresh checkout. Do not copy official media or labels. GAICD/SACD/GNMC raw public inputs stay outside Git.

The single stage runner is the equivalent of four separate scripts:

```bash
python -m scripts.run_scale_stage --help
python -m scripts.run_scale_stage \
  --stage scale_sft \
  --run-config artifacts/scale_sft_dpo/config/run_config.json \
  --data-audit artifacts/scale_sft_dpo/audit/data_audit.json \
  --preference-audit artifacts/scale_sft_dpo/audit/preference_audit.json \
  --train artifacts/scale_sft_dpo/manifests/train.jsonl \
  --val artifacts/scale_sft_dpo/manifests/val.jsonl \
  --output artifacts/scale_sft_dpo/runs/scale_sft --device cpu
```

This command is **currently expected to exit 2 at the gate**. The missing manifests are not replaced by dummy data. After upstream requirements are genuinely passed, use `--stage candidate_generation`, `--stage preference_build`, or `--stage spatial_dpo` and provide the preceding `--sft-metrics` / `--candidate-metrics` files. These must contain measured metrics, not just a `gate_passed` flag. Candidate/DPO initialization is hash-bound to the SFT checkpoint. Use `--resume <checkpoint-directory>` to restore the optimizer, isolated Scale tensors and RNG. DPO references are cached before the first update in eval/no-grad/frozen mode and verified on resume; no second 9B inference model is introduced.

Before enabling any formal GPU worker:

1. Resolve the competition training/weight-submission permissions for every selected dataset. Keep unknown rights `null`; do not infer dataset rights from software MIT or a citation request.
2. Recover immutable **final deployed** P1 centers on the clean public LIVE/GNMC sources, with the complete parent identity and each original `[x,y,w]`. Preserve original P1 smoothing/geometry; do not use GT-response centers from old training JSON. Each cache row needs `dataset_source`, `source_id`, `image_id`, `split`, `target_ratio`, `image_sha256`, original image dimensions, `parent_identity_sha256`, `position_kind=P1_final_deployed`, and `p1_aic_bbox_xyw`.
3. Export real public labels first with `python -m scripts.export_scale_labels --help`. A local export was verified: 123,700 A/B train/val views, SHA256 in `audit/public_label_export.json`. These contain no P1 centers. The GNMC archive is verified locally; copy it to the declared execution path before using ZIP-backed input. That remote archive path is currently not verified. Join using `python -m scripts.prepare_scale_manifests --labels ... --positions ... --parent-sha256 ... --output ...`. The join rejects missing/unused rows, ratio/split/hash mismatches and leakage. Record stored media dimensions separately (`media_size`) when cached image resolution differs from the original coordinate system.
4. Run `aic.scale_audit.position_conditioned_coverage()` on Train only. Lock the mismatch/noise thresholds, source weights, minimum effective coverage/sample sizes and collapse bound using train statistics. Review the provisional grid using this coverage. Do not use Val to refit bins or training labels. Check ≥1,000 real training-center geometry cases per arm, keeping native and transpose evidence separate.
5. Run the real CPU model probe on a sufficient-memory host:

```bash
python -m scripts.smoke_scale_videoora --config artifacts/scale_sft_dpo/config/run_config.json \
  --sample public_train_sample_with_P1_center.json --output artifacts/scale_sft_dpo/smoke_test/real_cpu
```

This loads actual weights, differentiates JSON SFT and DPO, checks nonzero gradients, verifies unchanged P1 tensors, saves/restores a real adapter checkpoint, and checks geometry without taking optimizer steps. Do not mark it passed from `inspect_scale_model.py` or Tiny model tests. No GPU smoke is authorized this round.

6. Re-run the stage gate. Bind exact Train/Val paths/hashes, original runtime source hashes and all promotion evidence. Freeze the config before validation selection. Only a later user authorization may enable `formal_gpu_training_authorized`.
7. For an authorized future GPU run, use `--device cuda:0 --budget artifacts/scale_sft_dpo/config/gpu_budget.json`. The configured single ledger is mandatory, including restarts. SFT 4 h / candidates 3 h / preferences 1 h / DPO 4 h, total 12 h. A supervised child includes model loading in its conservative wall-time accounting. Unreconciled active sessions or prior overruns block subsequent jobs. Benchmarking/validation must fit their stage budget.

Final integration is `aic.scale_runtime.apply_scale_to_p1_predictions()`: it preserves the original selected frame IDs and derives each Scale crop from that immutable P1 deployed center. `python -m scripts.infer_scale --help` exposes a CPU inference entrypoint; it requires real-model smoke and exact checkpoint identity. DPO promotion compares source-macro deployment geometry to frozen SFT. If it fails, keep SFT.

## Delivered remote copy

The `scp` destination is `/root/autodl-tmp/videoora42_clone/scale_sft_dpo_newday_20261108/code`. Use this isolated checkout for the repaired code. The original runtime is imported read-only from the hash-locked path in `run_config.json`. The remote no-card allocation still cannot hold the real base model. Test-only `pytest==8.3.5` and `opencv-python-headless==4.12.0.88` are installed under the adjacent `test_deps/` directory; the original virtualenv is preserved. CPU Scale verification uses `PYTHONPATH=.:../test_deps`, `CUDA_VISIBLE_DEVICES=""` and `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`. See `audit/remote_code_sync.json` for actual results, not an assumption of model readiness. The geometry arm is part of the checkpoint contract. A/B jobs cannot silently reuse one another's Scale checkpoint.

Final submission assembly (only after real smoke, checkpoint promotion and P1 input provenance pass):

```bash
python -m scripts.infer_scale \
  --config artifacts/scale_sft_dpo/config/run_config.json \
  --input selected_public_or_official_frame_manifest.jsonl \
  --checkpoint promoted_checkpoint_directory --checkpoint-sha256 ACTUAL_TRAINER_SHA256 \
  --output scale_action_diagnostics.jsonl --p1-predictions original_p1_predictions.jsonl \
  --index decoded_metadata_index.jsonl --submission-output predictions.jsonl
```

These are argument descriptions, not claims that checkpoint/input files already exist. Official inference inputs contain no GT labels. The exporter binds every action to the original P1 frame/bbox, preserves empty-video rows and frame selection, and validates full index coverage, exact ratio-derived height, finite coordinates and measured total inference parameters. The legacy shared validator still has an MB-required `final` branch; this entry uses its coverage/geometry/parameter checks with an explicitly injected measured `model_params_b`, without inventing MB values or applying the superseded 9GB limit. This produces JSONL; competition ZIP packaging remains the existing submission packaging step.
