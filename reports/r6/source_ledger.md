# R6 source ledger — LIVE / RetargetVid exposure accounting

Date: 2026-10-05.  All counts measured on 910A directly (commands in git
history of this file's companion json).  Purpose: R6 Q3/Q8 requires a
source-level exposure ledger before any S-REREAD confirmation read.

## LIVE_YT_VC (1800 sources total, verified on disk)

| Set | n | Status |
|---|---:|---|
| train_index (T5_CROPHEAD_V3/live_train_index.jsonl) | 1622 | training-pool ancestors |
| val_index (live_val_index.jsonl) | 178 | see verdict below |
| B3_s0/s1 dev read (round5 per_live_dev) | 124 | DEV EXPOSED (round-5 S0/S1) |
| B3_s0/s1 confirmation read (per_live_confirm) | 227 | EXPOSED (round-5 S0/S1) |
| round-4/5 spatial audits touching both | — | covered by the 351 above |

Key verifications (all measured 2026-10-05):

1. `train 1622 ∪ val 178 = 1800`, overlap 0 (ID-level).
2. The 351 dev/confirm-exposed sources are ALL inside train_index.
3. **B3 (v8_s_train_multidata, manifest v8_manifest.jsonl): live_train
   split = 33,450 samples over 1,115 vids, ALL 1,115 inside train_index;
   intersection with val178 vids = 0.**  The champion B3 never trained on
   any val178 source.
4. T5_CROPHEAD_V3 (V5-round teacher crop head, DEPRECATED - never part of
   the shipped stack) built feature caches on all 178 val sources
   (obs_cache_live_val) and 1622 train (obs_cache_live_train).
   Per R6 Q7-C case 1 this is NOT training leakage for the current line:
   no T5 parameter, pseudo-label, or selection result flows into the B3 /
   VTREPLAY champion.  Recorded for the ancestry audit; the val178 pool
   stays eligible as a one-shot confirmation set WITH this history noted.
5. B3 manifest splits confirm the pool structure: live_dev 3,720 frames
   (=124 vid) and live_confirmation 6,810 frames (=227 vid) exist as
   EVAL-only splits; no val178 source appears in any manifest split.

## Verdict

- **LIVE_val178 = one-shot confirmation pool for R6 S-REREAD / S-ZERO.**
  Size 178 >= the R6 minimum of 120.  `BLOCKED_FRESH_SPATIAL_CONFIRM`
  does NOT fire.  Usage: ONE preregistered read per candidate group,
  every access appended to confirmation_access_log.jsonl.
- Old dev124 + confirm227: DEV ONLY from now on (already dev-exposed by
  round-5 reads; renaming them would be fraud).
- RetargetVid NEW170 residual (~170 - 70 exposed = ~100): backup pool
  only, not mixed with LIVE_val178 (different domain, cross-domain reads
  would need their own preregistration).

## Known ID-level limits (honesty items)

- Perceptual near-duplicate screening between val178 and the 1,115 B3
  training vids has NOT been run (no tooling in this cycle).  ID-level
  disjointness is verified; perceptual leakage remains possible and is
  logged as a standing limitation of the pool.
- T5-era verification passes on val178 are known to have happened but
  per-experiment access lists are not fully reconstructable; treated as
  "deprecated-line exposure, non-leaking per Q7-C case 1".

## PHD2 (temporal line)

- train.json: 110,611 sources, 201,527 intervals; interval length p50
  3.7 s, <=4 s 54%, <=6 s 17%, <=8 s 9%, >8 s 19% (measured 2026-10-05).
- Native-contract dev pool: 256 sources (round5_native_dev_sources.json),
  feature contract hash ac3b657f.  These are DEV sources for T-CONTEXT.
- Multi-position eligible events: counted by the T-CONTEXT build script
  with ineligible-ratio recorded; no deletion of ineligible events.
