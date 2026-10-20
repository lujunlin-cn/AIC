# Valen route asset audit — 2026-10-20

Status: Valen-2B SFT and batched real-media confirmation completed. The frozen checkpoint passes an exposed RetargetVid 601--700 source-level transfer confirmation; no official score or submission claim is made.

User requests this route and pauses DINOv3. The supplied guide is a proposal, not an experimental result. Its DINOv3 V1 findings were superseded by `reports/dinov3_v2/README.md`; they are not used to select this route.

## Runtime and provenance

- Local HEAD before work: `fd7301f`. Existing modified submissions and untracked packages are preserved.
- AutoDL: SSH port 56200, `autodl-container-2c1e409ad1-4d8375b1`, RTX4080 GPU0, 32760 MiB. GPU idle at audit.
- Actual cgroup memory limit: 66,571,993,088 bytes; free data disk about 143 GiB at audit.
- Base: Python 3.12.3, PyTorch 2.8.0+cu128, torchvision 0.23, CUDA available. Separate `/root/autodl-tmp/valen_env` uses existing torch, Transformers 5.4.0, PEFT 0.18.1; exact installed environment will accompany the run.
- Valen source HEAD resolved through GitHub: `7b432422c18428f519ba1970f716df8a5feb5f89`. Git clone failed with TLS early EOF; archive fallback obtained. Archive bytes and model runtime files must be hashed before reporting an executed model revision.
- Valen 2B model revision: `fb9205c2541abc97ce276670e66f79b78d9068fc`; HF inventory 2,214,825,577 FP32 serialized elements. Runtime unique parameter count remains to be measured.
- Valen 0.8B revision: `c2c481fcb319bd9ae834abc7c6373271710cf24e`; HF inventory 854,305,641 FP32 serialized elements. Runtime unique parameter count remains to be measured.
- Code and model cards declare Apache-2.0. Public crop media keep their original dataset attribution/permissions; the model license does not confer media rights.
- Direct HF download failed with network unreachable. Pinned revision through hf-mirror is being tested; byte identity will be verified using HF LFS hashes.

## Reusable assets

- Geometry/candidate contract: `scripts/r70_spatial_utility_pilot.py`: `geometry`, `candidate_offsets`, `boxes`, `table_gt`; 129 maximum-size legal crop windows.
- V9 inference and batched pooling: `scripts/r70_official_points.py`; public readout checkpoint really present at `/root/autodl-tmp/experiments/R70_OBJECTIVE_VARIANTS/topsoft.pt` (6,083,337 bytes).
- Public four crop tables and media really present under `/data/aic/experiments/T5_CROPHEAD_V3`; RV source frames under `/root/autodl-tmp/evac/data/aic/experiments/QWEN_RV200_KEYFRAMES`. LIVE dev images remain in `keyframes_train`.
- Public candidate caches: `/root/autodl-tmp/experiments/R70_SPATIAL_UTILITY/full/shards`. Existing aligned row metadata can be reused; no DINO model is run for this route.
- Frozen R54/R70/R71 JSONL and manifests exist locally. R70 and R71 ZIPs exist; they retain approximately 8.855B total inference parameters including TimeLens2.
- Original V8 B3 weights have not been recovered; do not describe R70 or another head as an exact B3 reconstruction.
- `aic/temporal_contract.py`, existing index `reports/r47_timelens2_index_enriched_426.jsonl` retain original decoded frame/PTS contract. Reuse legal keyframes; do not create a new video time axis.
- Primary validator `aic/contract.py` already has `model_params_b`; independent historical checker still uses `model_size_mb`/byte-era limits. Add a separate current-rule mode without invalidating historical reproduction.
- Existing evaluator `aic/evaluation.py` already reconstructs exact-frame continuous IoU/video macro and empty cases. Verify current stage behavior before adding duplicate machinery.

## Deployment decision

Valen cannot be added to the 8.855B TimeLens2 stack. Initial deployable system uses a deterministic exact-85% center time window, V9 top-K crop recall and Valen spatial choice, all counted. Compare public crop decisions with the same recall and geometry; compare any resulting official candidate with the R26 center-time reference. Keeping R54 frame sets would still require counting the TimeLens2 model that produced them.

S0 starts on source-balanced public native-crop samples, measures K=4/8 oracle recall and candidate-order sensitivity, then expands by evidence. SFT may use public TRAIN only; checkpoint choice uses a TRAIN source holdout. Historical dev is a transfer diagnostic, not fresh independent confirmation. No official media enter training.

## 2026-10-20 confirmation update

The full `rv_confirm2` exposed pool contains 4,081 rows from 100 RetargetVid sources. Valen inference used the restored SFT checkpoint with requested and effective batch size 16: 255 full batches plus one tail batch, 614.5 seconds wall time, and no official test data. The run log records `states=16` for every full batch; the measured throughput was 6.64 rows/s.

On source-macro IoU, Valen is `0.675226` versus the existing head `0.656760`, a paired source delta of `+0.018466` with bootstrap 95% CI `[+0.010735,+0.026371]`; 71/100 sources improve and 2 decline by at least 0.05. The ratio splits are both positive: 1:3 `+0.014766` CI `[+0.001868,+0.028027]`, 3:1 `+0.022196` CI `[+0.013865,+0.030809]`.

The four-candidate oracle is `0.762726`, all-129 oracle `0.833979`, so candidate-recall loss is `0.071254` and candidate-internal selection loss is `0.087500`. The next experiment is a utility-aware candidate-choice objective on public TRAIN, still with batch 16 and source-held-out evaluation. It must improve the frozen checkpoint on a source-level public split before any deployment packaging.
