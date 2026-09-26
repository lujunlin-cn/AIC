# 2026-09-26 evaluation-set readiness

## Frozen intake path

`AIC_EVAL_INTAKE_V1` is now the required first step for a newly received
evaluation set. `scripts/prepare_eval_set.py` accepts the platform compact
JSON/JSONL index and resolves `video_path`, `path`, `video`, `file_name`, or
`filename` relative to the supplied video root. It fully decodes each video,
checks actual width/height/frame count and strictly increasing display PTS,
records coded-pixel coordinate convention and SHA-256, then writes an
enriched index and a hash-bound intake manifest. It refuses to overwrite an
existing intake result unless `--force` is explicitly supplied. Labels are
never read and `official_f_video`/`competition_score` remain null.

`scripts/run_eval_candidates.py` verifies that the intake manifest matches the
index and then runs each frozen release candidate in a new output directory.
`scripts/release_batch.py` remains the direct raw-video fallback for a small
batch or smoke check. All release paths still verify model hashes and actual
loaded weight bytes before writing JSONL.

## Evidence

- Local compact index → real 5-frame video → PTS/SHA intake → final dummy JSONL → validator: valid.
- Local A0 real-weight inference: `loaded_weight_bytes=25,685,169`, final JSONL validator valid, exit 0.
- Remote `/data/aic` A0 release smoke under the V100 environment: same loaded bytes and validator valid, exit 0.
- Local regression suite: `83 passed`.
- Remote source compilation and `prepare_eval_set.py --help`: passed. Remote environment does not have pytest installed, so only the local full suite is authoritative for tests.
- Remote asset inventory: `/data/aic/asset_inventory/asset_manifest_20260926.jsonl`; 11,038 records, 7,293 hashed regular files, 3,702 metadata-only dataset files, 43 valid feature symlinks, no missing/error/changed records.

## First action when the platform set arrives

Preserve the original archive and compact index. Copy videos into a fresh
`/data/aic/eval_incoming/<set_id>/videos` directory, run intake once, then run
`A0_center` first. Record the run directory, index/manifest hashes, per-video
latency and empty-output rate. Run `DeiT_center` and `A0_face_ema` only after
the A0 output passes local validation. Do not tune thresholds or use test
labels; use the platform evaluator only in a separate official-results
directory when it is provided.
