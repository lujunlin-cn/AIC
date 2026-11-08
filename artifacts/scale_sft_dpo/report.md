# Current GPU preflight update

The latest verified state is in [gpu_preflight_35092.md](gpu_preflight_35092.md). Real A/B Video-ORA smoke now passes, LIVE originals are complete, and full-data P1 coverage/calibration remain blocked. No formal SFT/DPO has started. The CPU-delivery snapshot below predates this GPU run; its no-GPU/missing-original-media statements are historical.

---

# Scale SFT-DPO repair — 2026-11-08

**Formal GPU training is not ready. No GPU task ran; all four formal stage budgets remain unused.** This delivery repairs executable code, audits real public data and parent weight identities, and preserves the unresolved gates. It does not claim SFT/DPO training success or an Oracle/official-score improvement.

## Repairs and verification

- `scale_geometry.checked_scale_action()` measures the actual AIC `[x,y,w]` center using ratio-derived height. Actions with more than 1 px drift are marked `position_shift_invalid` and excluded from SFT/preference/candidate construction. P1 centers are never moved to rescue a Scale action. Tests cover nine center locations, two ratios and four scales; a separate historical public-dev audit covers 1,200 A / 2,000 B outputs across their actual center ranges, with maximum accepted drift 0.5 px. It does **not** satisfy the missing 1,000-training-centers-per-arm requirement.
- `scale_preferences.fixed_center_sft_target()` has one `aic_bbox_xyw` key and separately stores the exact full crop and discrete action. The actual multimodal SFT backend calls **`causal_completion_sft_loss`**, with selected completion positions, prompt/padding exclusion, no new tokens and strict JSON parsing (XML wrappers, Markdown fences and explanations are rejected). Classification loss is a spare interface, not this training path.
- `scale_videoora.py` verifies the original loader/base/tokenizer/runtime identities and loads real P1 banks instead of accidentally creating fresh parent LoRA. A separate rank-1 Scale delta has an exact allowlist; frozen P1 banks retain their files and parameters. Structural PEFT stacking, tokenizer round-trip and the conservative complete count **8,999,976,258 ≤9B** passed on the remote CPU. Full weight forward/backward remains untested.
- `export_scale_labels.py` exported 123,700 real public A/B Train/Val label views without substituting P1 centers; its manifest SHA is recorded. `run_scale_stage.py` provides real SFT, model-generated candidate, CPU preference and DPO entrypoints. `scale_execution.py` implements source-balanced Dataset/DataLoader sampling, microbatch accumulation, AdamW, immutable SFT-reference log-probability caching in eval/frozen mode, checkpoint/optimizer/RNG restore, SHA/config/bin/data/geometry-arm binding, and separate frozen-SFT validation predictions for DPO comparison. DPO is compared against SFT, not only P1 scale=1. Online source-macro training preference accuracy is reported separately by dataset and quality semantics; it is never treated as held-out selection evidence. Preference pair rate records the fraction of eligible input views yielding a pair.
- Candidate diagnostics separate model-sampled, fixed-grid and combined Oracles after deployment deduplication. K=8 and the K=4 fallback are supported; seeds and failures are recorded. Fixed-grid improvement cannot pass the model-sampling promotion gate. No actual SFT candidate run has been performed.
- Preference inputs bind image bytes, prompt, preprocessing, ratio and frozen center. Pair geometry is checked against the official ratio and center. LIVE/GNMC geometry pairs explicitly identify their train fixed-grid provenance; GAICD retains MOS semantics; SACD needs explicit same-image/subject rankings. A direct human-center check prevents coordinate tolerance from hiding more than 1 px of Position change. No preferences were generated in this run.
- Source grouping uses `(dataset_source, source_id)`; candidate totals use dataset-macro of source-macros. Bootstrap uses independent sources within each dataset. Human-validation helpers separately report GAICD image-macro MOS and SACD subject-conditioned preference accuracy, without inventing MOS for unannotated actions.
- Stage gates now check measured deployment/Oracle evidence, real-model smoke, frozen Position, hashed manifests, calibrated coverage/collapse criteria and licenses. Mechanical 2000-pair/300-source defaults were removed; those minima remain unset until actual train coverage/stability evidence exists.
- A persistent, locked GPU ledger accounts for child model loading through termination, records GPU name/timestamps, enforces 4/3/1/4 h and 12 h total, and rejects inconsistent or previously overrun ledgers. No formal GPU execution is authorized in this delivery. Final inference additionally requires the checkpoint to pass its measured SFT/DPO deployment gate, and rechecks the P1 center against the original deployed bbox before loading the model. The CLI can join actions to original P1 predictions and export full submission JSONL with exact frame/index coverage, `[x,y,w]` geometry and the measured ≤9B declaration. The old shared validator’s MB eligibility branch is not applied to current parameter-based submissions; its syntax, coverage, geometry and parameter checks are retained.

An existing unmatched parenthesis in `scripts/r13_tvsum_certify.py` prevented the required repository-wide compile check; only that syntax was corrected. Historical experiment records and parent weights were not deleted or modified.

## Required answers

| Question | Evidence-backed answer |
|---|---|
| What was repaired? | The code and contracts listed above; tests and semantic checks below |
| What remains? | Actual train P1 deployed-center cache, calibrated label/quality/collapse/sample-size criteria, final train/val manifests, competition permissions, and real-weight smoke |
| Which datasets can enter formal training legally now? | **None is cleared by this delivery's gate.** Competition-use authorization remains `null`. GNMC explicitly supports noncommercial research; that alone is not marked as competition/weight-submission authorization. GAICD software MIT is not treated as an automatic license to its separately downloaded images. Author README/page evidence and SHA256 are archived under `audit/license_evidence/` |
| How many valid Scale-SFT labels? | **Unknown (`null`)**. Clean LIVE has 47,850 frames from 1,595 train sources per A/transposed-B view; GNMC has 7,000 train sources. These are available annotation counts, not P1-conditioned effective labels |
| How much Scale improvement space remains at P1 centers? | **Unmeasured** without actual train centers. `position_conditioned_scale_coverage.json` blocks this claim. The implemented auditor separates position-limited, position+grid/ratio mismatch, and Scale-optimizable samples |
| How many GAICD candidates/preferences? | Journal Train: 231,489 candidates; 227,755 finite nonsentinel rated entries; 227,750 also within MOS 1–5 and valid geometry. At 1 px ratio tolerance only **54** fit 9:16 and **2,082** fit 16:9, before P1-center/bin filters. Center/bin retention and effective preference counts remain `null`; actually constructed pairs = 0. No center condition was relaxed |
| Does SACD have usable preferences? | No confirmed explicit subject-linked ranking schema. `best_box` is not converted into pairs. Group split repaired to **2,203/286/288 images**, **2,281/309/316 subject annotations**. SACD stays outside DPO |
| Are LIVE media and source splits complete? | Historical 1,600/200 source manifest recovered; 54,000/54,000 cached frame hashes match, none missing. Five train sources quarantined for one cross-split byte-identical group: clean split **1,595/200**. Raw source videos are absent at the recorded remote path; complete PTS re-decode and full deployed P1 center generation are still pending |
| Does actual candidate Oracle@K improve? | **Not measured**; no SFT checkpoint/candidate run exists. No fixed-grid or GT-centered diagnostic is reported as a model gain |
| Is real Video-ORA smoke complete? | **No.** Exact real weight-file hashes, tensor counts, real architecture/PEFT structural injection and tokenizer were checked; the CPU-only container has **2 GiB RAM / 0.5 CPU**. `model_load.json`, `sft_forward_backward.json`, `dpo_forward_backward.json` are explicit blocked records. There is a real-weight CPU smoke entrypoint for a sufficient-memory host and public sample; meta/Tiny tests are not accepted as substitutes |
| Can formal training start? | **No**, see `audit/scale_sft_gate.json`; no bypass or lower acceptance threshold was added |
| Is the 12 h budget executable? | Timing/locking/termination enforcement is implemented and CPU-tested. Feasibility/throughput on the intended GPU is not benchmarked, so completion of all stages within 12 h is not promised |

## Scale-bin analysis

Only training labels were used. The current first gap is **0.431489** (0.29885→0.730339); five upper bins cluster at ≥0.948719. Mean video-level LIVE human scale is 0.875519 versus GNMC image scale 0.906448. The current grid's **GT-centered geometric upper-bound** mean IoU is 0.909515 on LIVE frames / 0.948680 on GNMC; separate train-quantile 16-bin diagnostic grids give 0.930338 / 0.974739. These are neither learned results nor P1-conditioned gains. Eight bins cannot yet be declared sufficient. Current bins are marked provisional; no validation/test label was used for fitting and no automatic refit was deployed.

## Validation

- `python -m compileall -q aic scripts tests`: passed.
- `PYTHONPATH=. pytest -q tests/test_scale_*.py`: **139 passed**.
- `PYTHONPATH=. pytest -q`: **262 passed**.
- Remote delivered-code compileall and **139 Scale tests**: passed with CUDA hidden. Remote full-suite environment limitations are recorded in `audit/remote_code_sync.json`; the **262-test full-suite pass is local**.
- Real-config remote metadata probe: passed; real-weight load/forward/backward: blocked.
- Historical public-dev A/B deployed geometry: 25,600 checked actions, accepted drift ≤0.5 px; training-geometry requirement remains blocked.
- Gate invocation exits 2 before backend/model/CUDA initialization, as expected. No official hidden labels or public Test crop labels were used for fitting, training, preference construction or selection.

See `RUNBOOK.md` for executable commands and the precise remaining work. Formal budget file remains at 0 h. Stop after this repair delivery; do not launch training.

## Remote code delivery

Final code is synchronized by `scp` to `/root/autodl-tmp/videoora42_clone/scale_sft_dpo_newday_20261108/code`. The directory is isolated from the original inference code, datasets and parent weights. `audit/remote_code_sync.json` records the uploaded source hashes and remote CPU validation; the final Git commit and archive receipt are also written beside the remote code. No background training or GPU worker is started.
