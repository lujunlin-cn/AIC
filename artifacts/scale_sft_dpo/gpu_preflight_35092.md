# 35092 GPU preflight — 2026-11-08

Formal SFT/DPO has **not** started. All optimizer-step counts are zero. The real-weight integration blocker and missing original LIVE videos are resolved. Full-data Position coverage and train-only calibration are still incomplete. A cache file is optional; obtaining real fixed P1 centers is computationally necessary for the requested labels and pre-training audits.

## Measured results

- Execution host: `ssh -p 35092 root@connect.westd.seetacloud.com`; RTX 4090 48 GiB, 96 GiB container RAM. Host CPU is Xeon Platinum 8470Q, 208 visible logical CPUs; cgroup `cpu.max=2000000 100000` limits sustained compute to 20 CPU equivalents. CPU visibility is not the allocation.
- Real Base + frozen P1 A/B + isolated Scale loaded. Complete conservative inference count is **8,999,976,258**, including buffers, below 9B. The original base, tokenizer, adapters and runtime hashes are checked by the actual backend, not inferred from submission ZIP hashes.
- Real `[9,16]` and `[16,9]` action generation, `causal_completion_sft_loss` forward/backward, nonzero Scale gradients, frozen P1 identity, checkpoint save/restore and actual geometry pass. DPO initial loss is 0.693147 for both ratios, initial policy/reference log probabilities match, reference eval logps repeat exactly, and swapped float32 logp gradients have zero sign error. The initial zero-update Scale checkpoints are differentiation evidence, **not trained SFT checkpoints**. Evidence: `smoke_test/gpu_35092_verified.json`, `model_load.json`, `sft_forward_backward.json`, `dpo_forward_backward.json`.
- An overly strict parameter-gradient swap probe failed in BF16; the failed remote `smoke_AB_final/receipt.json` is retained. The corrected probe checks mathematical swap symmetry at float32 completion-logp inputs, and independently checks actual nonzero model parameter backprop. No precision tolerance was silently relaxed.
- Pilot: **64 view/frame centers** from one original LIVE video (30 original frames × A/B) and two GNMC Train images (A/B). All 30 LIVE annotated cached-frame pixel arrays exactly match original-video decoding at their original frame IDs/PTS. **512/512** Scale projections pass, maximum center drift **0.5 px**. This is a limited real pilot, not the required full training-center distribution. Evidence: `smoke_test/gpu_35092_singleton/`, `geometry_validation.json`.
- The strict real label/center join passes all 64 pilot views and rejects a missing center and a corrupted image hash. This checks the actual preparation path without substituting GT centers. Evidence: `audit/real_position_join.json`.
- LIVE public Box originals: **1,800/1,800** recovered and hash-verified locally, then **1,800/1,800** independently verified on the GPU host, zero missing or mismatched sources. Originals: `/root/autodl-tmp/videoora42_clone/data/reproduction_sources/LIVE_YT_VC/study_videos/`. Local originals: `/home/hajimi2025/AIC/.cache_videoora_data/LIVE_YT_VC/study_videos/`. Source/download recovery code: `scripts/download_scale_live_media.py`; execution-host audit: `scripts/audit_scale_live_media.py`; evidence: `audit/live_remote_verified.json`.
- Source split: preserve LIVE clean 1,595 Train / 200 Val sources. Four GNMC Train IDs with byte-identical Val media are quarantined in both views: `001183`, `001972`, `003413`, `009560`. GNMC effective Train is 6,996 IDs / 6,983 distinct image byte hashes; official Val is preserved. Raw annotations are not modified. Clean public label inventory has **123,692 views**, SHA `a98aa78ada1f3f6b2a349c29c4bad15b65ede955dee761c586ced83f9d7d96c0`. Evidence: `audit/isolation.json`; executable: `scripts/isolate_scale_public_labels.py`.
- GAICD Journal Train input/candidate export: **2,636 images / 5,272 ratio views**, SHA `87c9fced9c21d59dc5fd38d6000ba1abccef4fac3651deb071598793f6fcfa8e`. Export opens Train annotations only. Full P1-conditioned candidate/bin retention and preference counts remain unmeasured, not zeroed to imply failure or completion. SACD has no confirmed explicit subject ranking and remains outside DPO.
- Dataset authorization ownership follows the user's explicit instruction. Original legal fields remain `null`; no permission is fabricated. An explicitly declared user-managed review removes agent license investigation from the technical checklist. Known `false` permissions still block. Formal training authorization remains false.

## Throughput and why CPU thread count is not the cache cost

| Torch threads | Source wall seconds | Model generation seconds | Other stages seconds | Average CPU equivalents |
|---|---:|---:|---:|---:|
| 8 | 84.683 | 78.922 | 5.761 | 1.964 |
| 20 | 86.780 | 80.426 | 6.354 | 3.586 |

Both runs execute the same original P1 functions on the same complete public Train video. They produce identical final boxes. The 100 model calls include four singleton parity-check calls, so steady source processing requires up to 96 calls. About 93% of this pilot's wall time is inside model generation, not frame-cache writing. PyTorch intra-op threads, decode workers and process concurrency are distinct controls; blindly assigning 20 threads to every component oversubscribes the container.

Batch=16 failed raw/coordinate parity: the B candidate changes `[500,508]` to `[500,504]`, and the subject box changes too. This path is fail-closed and is not used for formal P1 centers. The safe default calls the original singleton functions. Evidence: `audit/p1_batch16_nonparity.json`.

Two independent singleton processes fit (~35,622 MiB observed VRAM), keep the same original P1 output, and finish two duplicated source jobs in 150.620 seconds including loading. Conservative speedup relative to twice the 84.683-second warm single-source measurement is **1.124×**; the processing-only estimate is about 1.35×. This is a small throughput probe, not a full-data ETA or proof that 20 CPU equivalents are useful. Evidence: `audit/parallel_profile.json`, `audit/profile.json`.

The earlier ~50 GPU-hour estimate was a single-source extrapolation for full original singleton LIVE+GNMC+GAICD, not an approved budget or a measured full-job duration. No such job was launched. Streaming positions without saving a file does not remove those model calls. A selected source cohort is a valid smaller experiment only if declared explicitly; it cannot be reported as a full-data audit.

The current executable training pipeline expects offline hash-bound centers; online P1 inference is not implemented. The completed pilot can validate integration without building a full cache, but it cannot establish full-data label coverage or GAICD retention.

## Remaining blockers and safe continuation

1. Full real-P1 Train/Val center generation scope and its GPU allowance remain undecided following the user's questions. No full cache is running. The exact-original projection runner is resumable, verifies original source SHA, original decoded frame pixels, parent/model identity, QP floor, frame-set and path-cache hashes: `scripts/generate_p1_scale_positions.py`, `aic/scale_parent.py`.
2. Full-data effective Scale-SFT labels, source coverage, position-limited categories, GAICD candidate/pair retention, train-only bin/threshold/collapse criteria remain unknown. `aic/scale_calibration.py` and `scripts/calibrate_scale_training.py` provide the CPU execution path once complete hash-bound positions exist. Fitting/calibration are split by independent Train sources; GNMC duplicate bytes are grouped. Candidate support, 8/16-bin selection, pixel-discretization sensitivity and source-bootstrap diagnostics use no Val/Test labels. Proposed settings are not locked by this delivery; pilots cannot lock formal config.
3. The old provisional eight-bin grid remains unapproved. It must be refitted with clean Train data and actual centers. A changed mapping resets real-model-smoke readiness until it is retested. No SFT policy, model-sampled Oracle gain, validated preference count or DPO improvement exists.
4. Formal SFT/DPO remain blocked by measured stage gates and `formal_gpu_training_authorized=false`. The 4/3/1/4-hour formal stage ledger has no runs. Actual preflight GPU use is separately recorded below; it is not claimed to be zero, nor does it authorize unlimited preprocessing. Finishing all later stages within 12 hours remains unverified.

## Evidence and delivery

All real preflight task receipts, including failures and parallel overlap, are reconciled in `audit/preflight_gpu_usage.json`. Same-device elapsed intervals are merged so two workers do not double count the same physical GPU wall time. Process-hours are also retained. No historical experiment, P1/P2 weights or official output contract was removed or changed.

The actual RGB preprocessing was Pillow BICUBIC throughout, while old metadata incorrectly called it nearest. This metadata is now corrected and BICUBIC is explicit in both public-image loading paths and the DPO contract. The verified real A/B smoke uses the corrected contract.

Remote code: `/root/autodl-tmp/videoora42_clone/scale_sft_dpo_newday_20261108/code/`.
Remote preflight evidence: `/root/autodl-tmp/videoora42_clone/scale_sft_dpo_preflight_20261108/`.

Measured preflight same-device wall use: **0.196897 h**; sum of process lifetimes: **0.238310 h**. Formal optimizer updates: **0**.

Validation at delivery: `python -m compileall -q aic scripts tests` passed; `PYTHONPATH=. pytest -q tests/test_scale_*.py` **155 passed**; `PYTHONPATH=. pytest -q` **278 passed**. The final Scale-SFT gate remains closed on uncalibrated thresholds, incomplete hashed full manifests, provisional bins and the collapse bound; real model identity/smoke and user-managed authorization ownership are no longer its blockers.

The final local CPU command outputs and execution-host Scale test result are retained in `audit/preflight_validation_35092.json`. Execution-host Scale tests use the preserved isolated `../test_deps` directory; no pytest installation or dependency replacement was made in the original model environment. Code delivery hashes and its final commit marker are verified separately on the remote host.
