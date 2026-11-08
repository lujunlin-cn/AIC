# Semantic review — 2026-11-08

This is a code/data contract review, not a model-quality measurement.

| Invariant | Implementation / evidence | Outcome |
|---|---|---|
| Position is immutable across Scale choices | `checked_scale_action`, train/preference filters, boundary tests; final integration derives center from the original P1 crop | Code/CPU contracts passed; real parent-parameter smoke pending |
| Raw requested center is not silently turned into a new training position | Manifest preparation requires `P1_final_deployed` with original bbox + parent identity; no GT center fallback | Implemented; full real training cache missing |
| Height follows official target ratio | `deployed_center`, exact crop geometry, pair validation; final output carries three bbox values | CPU checks passed |
| Actual SFT is autoregressive JSON, not classification | Real backend `completion_loss` calls `causal_completion_sft_loss(already_shifted=True)` after gathering causal prediction positions | Selected/full-logit loss equivalence tested; real weight backward pending |
| Input contains no GT crop/MOS | `build_prompt` includes only original fixed center, target ratio and versioned bins; image preprocessing is deterministic | Code reviewed and collator tests passed |
| No new tokenizer tokens | Actual remote processor/tokenizer round-trip for all eight actions; no token additions | Passed real tokenizer check; full model generation pending |
| DPO inputs do not change between actions | Canonical image/prompt/geometry/preprocessing SHA contract, pair bbox center check; generic trainer also compares pixel/grid tensors and prompt token prefixes | CPU mismatch and human-center tolerance tests passed |
| Frozen reference uses no dropout or updates | Cache created before policy update in model.eval, all parameters frozen, no_grad; resume validates original SFT identity/cache SHA; generic trainer forces reference.eval | Numerical/dropout/immutability tests passed; real model pending |
| DPO must improve on SFT | Initial SFT validation predictions saved before DPO; reports compare against those immutable predictions and source-bootstrap; measured gate rejects loss-only claims | Final inference rejects unpromoted DPO; online training preference accuracy is separate from validation; no measured DPO gain |
| Model candidates are actual model outputs | Runner calls backend.predict with seeded stochastic generation, records parser failures, separates model/grid Oracle | Entry implemented; not run on real SFT weights |
| No source leakage or count-weighted opaque macro | Dataset-qualified source keys; source balancing and source-stratified bootstrap; explicit dataset macro | CPU tests passed; LIVE byte duplicates quarantined |
| Public Test labels are unused | GNMC readers enumerate Test media names only; calibration/bin audits read Train labels; SACD held-out membership is metadata, not supervision | Code/data review passed |
| MOS is not IoU; SACD best_box is not a ranking | Explicit score semantics and candidate filters; no SACD preference construction without same-subject explicit pairs | Passed; zero real preferences constructed |
| 9B includes every actual inference bank | Verified base+P1 A+B headers and real-architecture Scale injection; conservative buffer inclusion gives 8,999,976,258 | Structural pass; real-weight full pipeline still pending |
| No unverified rights enter training | `competition_use_authorized=null`; gate rejects before model initialization | Blocked as required |
| Formal GPU budget cannot be reset by restart | One configured persistent ledger, lock, active-session record, elapsed-time accounting, overrun/inconsistency rejection and watchdog | CPU subprocess tests passed; 0 formal GPU h used |

The remaining blockers are substantive. They are not waived because unit tests pass. The gate and the three real-weight smoke records must remain false/blocked until new evidence exists.
