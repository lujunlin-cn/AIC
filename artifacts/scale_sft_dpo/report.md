# Scale SFT/DPO CPU Implementation Status

## Scope

This branch implements the spatial Scale-only path. Position and temporal selection stay frozen. Scale bins are fitted from training-only crop ratios; every quality metric uses the final AIC `[x,y,w]` projection and its exact evaluator-derived height. No GPU job, AIC hidden-test asset, or official score was used for training or preference construction.

## Semifinal baseline identity

The user-reported scores are P2 **44.65**, P1 **44.63**, and B_MIX **42.88**. The associated submission ZIP SHA256 values are recorded in `config/run_config.json`. The packages in the local semifinal output directory contain prediction JSONL files, not model weights; these hashes establish score-package identity only. P1 remains the planned frozen-position parent, with P2 as the higher-scoring comparison. Remote startup must verify the actual parent weights, adapters, tokenizer, parameter count, and center-output equivalence before training.

## Implemented

- Fixed-center crop projection, exact AIC geometry, scale-bin fitting, transpose checks, and train-only fixed-position SFT targets.
- LIVE-YT-VC sparse-label, GNMC native-ratio, GAICD journal/MOS, and explicit-only SACD ranked-pair adapters.
- Candidate K construction, post-projection deduplication, source-macro Oracle metrics, paired source bootstrap, and strict preference schema validation.
- Causal scale-action SFT loss, standard DPO objective with frozen reference, gradient/parameter allowlist checks, tokenizer round-trip check, and stage/total GPU-hour ledger.

## Current Gates

The current machine has GAICD journal metadata and labels, a full GNMC ZIP, SACD files, and LIVE labels plus 37 original videos. LIVE's complete source split and 1,800-source media are not local. SACD supplied split files share source images; current annotations contain no confirmed explicit same-image/same-subject ranked-pair records. GNMC train/validation manifests are present, but its license is PolyForm Noncommercial. The P1 model/tokenizer and source-level P1 center cache are absent from this checkout.

Accordingly, this turn completes code and CPU semantic checks only. It does not claim that Stage 0, SFT, Oracle, preference-volume, or DPO promotion gates passed. `scale_bins.json` is fitted from the 7,000 GNMC training sources only: native 16:9 annotations supply the same width/max-legal-width scale ratio for the transposed 9:16 geometry. The fitted values are `[0.29885, 0.7303391904, 0.8751093035, 0.9487190201, 0.9649496514, 0.9858763345, 0.9982873665, 1.0]`. The remote P1 centers are still required to build fixed-position SFT labels and calibrate position-mismatch thresholds.

## CPU Validation

`PYTHONPATH=. pytest -q tests/test_scale_*.py`: **37 passed**; `PYTHONPATH=. pytest -q`: **160 passed**. The Scale tests include 1,200 deterministic geometry/transpose cases, train-only split membership, LIVE `dev` normalization and held-out-split rejection, MOS missing-value filtering, fixed-center GAICD mapping, explicit SACD pair requirements, test-split Oracle rejection, candidate deduplication, source-macro metrics, tokenizer JSON actions, CPU SFT/DPO gradients, gate blocking and GPU-budget rejection. `python -m compileall -q ...` passed. `scripts/verify_scale_geometry.py` wrote `audit/geometry_contract.json`; `scripts/check_scale_gate.py` wrote `audit/scale_sft_gate.json` with CUDA initialization blocked by missing parent, LIVE split and license preflight items. GPU time used: **0 h**.
