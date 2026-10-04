# Round-5 report, phase A: content verdict, spatial oracle, contract rebuild

Date: 2026-10-04.  Style: ASD-STE100.  All official scores are RAW, no size
penalty (operator-confirmed after two conversion violations - see
reports/r5/score_ledger.json).  Our champion: V11_VTREPLAY 34.95.  Leader
intelligence: 45.  Real gap: 10.05 points.

## 0. Score-unit correction (done first, before any new number)

The round-4 "raw 38.83/38.60", "6.2 gap" and "63 ceiling" figures were
illegal score/k_size conversions.  Withdrawn from: round-4 report section
4c, both V11 manifest score_notes, GPT6PRO_PROMPT_V5 (erratum header),
memory.  Surviving scale-invariant results: keep-0.70 alone caps at
34.95 x 0.8/0.7 = 39.94 (cannot reach 45); the spatial marginal table
(+0.02 slide-IoU ~ +0.75 points) is baseline-independent; all local
dev/confirm deltas keep their CIs.

## 1. P0-CONTENT (the champion verdict) - DEV_ONLY, preregistered read

Frozen checkpoints, no training, two pools (dev eval 1954; old confirm 585,
demoted audit).  Source-cluster bootstrap CI95, 2000 resamples.

| arm | real−zeros | real−xperm (cross-video) | real−slotprior |
|---|---|---|---|
| champion (VTREPLAY head) | dev **+0.104** [0.095, 0.115]; confirm **+0.085** [0.070, 0.100] | dev +0.0031 [0.0002, 0.0061]; confirm +0.0041 [−0.0017, 0.0097] | dev **−0.0036** [−0.0064, −0.0008]; confirm −0.0041 [−0.0091, 0.0007] |
| saturated exactdp ×3 | 0.0 | 0.0 | 0.0 |
| healthy mse_lr1e4 ×3 | ≈0 [-0.001, 0.003] | ≈+0.002 | ≈0 |

Reading:
1. The champion is NOT fully blind (zeros drop 0.08-0.10 with clean CIs) -
   the preregistered "content-insensitive" trigger does not fire.
2. BUT its content contribution is VIDEO-level, not FRAGMENT-level: giving
   it another video's features (xperm) erases essentially the whole
   advantage (+0.003/+0.004, confirm CI includes 0).  It reads "what kind
   of video this is" (global appearance calibrating a per-slot prior), not
   "where the highlight is".
3. The PURE position prior (per-slot positive rate of the training pool)
   TIES the champion on confirm and BEATS it on dev.  The champion's
   temporal ranking is a position-prior ordering.
4. Saturated exactdp: every delta 0.0 - the purest position prior, closing
   the round-4 inversion loop.
5. Consequence (R5 §1.4): no more loss/calibration tuning on the champion.
   The push goes to re-slicing + content inputs (the native line).

Artifact: experiments/20261004_v11/round5_content_ablation{.json,
_per_video.csv}.

## 2. S0-ORACLE + A5 (spatial) - exploitable gap confirmed, loss route closed

A5 audit: B3 IS ALREADY a candidate-utility head (huber regression on
annotator-mean candidate IoU, v8_s_train_multidata.py:226,319; 129 legal
max-windows).  R5's warning hit: re-implementing the objective would be a
duplicate.

S0 readings on the FROZEN per-frame records (no new inference):

| pool | n | head IoU | oracle IoU | gap mean [CI95] | head−center |
|---|---|---:|---:|---|---:|
| live_confirmation (LOCKED-OUT) | 3,720 fr / 124 vid | 0.555 | 0.802 | **0.247 [0.227, 0.268]** | +0.071 |
| live_dev | 3,720 / 124 | 0.540 | 0.775 | 0.234 [0.213, 0.255] | +0.071 |
| rv_diag (exposed) | 9,222 / 70 | 0.681 | 0.844 | 0.163 [0.147, 0.180] | +0.056 |

70.2% of locked-out frames have gap ≥ 0.05.  Verdict: SCORER_GAP_PRESENT.
The attack is feature/backbone, candidate geometry, TTA (S4), or
supervision breadth - NOT the loss.  Caveats: the oracle is a per-frame
max (no head realizes it fully); the numbers are crop-domain (LIVE/Retat-
getVid), official-domain transfer stays unmeasured.

Artifact: experiments/20261004_v11/round5_spatial_oracle.json,
reports/r5/spatial_candidate_oracle.csv (910A), reports/r5/
b3_objective_and_candidate_audit.md.

## 3. A8 native frame contract - frozen and validated

Protocol frozen (1 s action units, centre-aligned 2 s windows, 16 native
frames, nearest-decoded-PTS, <=2-unique-frame HARD_ERROR, boundary pads
flagged, feature_contract_hash recorded).  Validation on 12 sampled
sources: 12/12 OK, 16/16 unique frames per window, nearest-match error
0.004-0.02 s (source-fps quantization, not seek defects; round-4 measured
24-1608 s).  The correct seek rule (stream time_base units) is in the
contract.  Artifact: experiments/20261004_v11/native_frame_contract.json.

## 4. A7 fresh confirm set - frozen BEFORE any model score of a member

500 NEW sources (seed 20261005, sha16 f43281bf93290cea) from a 10,337
pool: training.csv minus official family (895), minus dev-exposed (2,262),
minus every round-4 touched source (500 confirm + 8 audit).  Frozen
protocol declares: anchor + neutral slicing (v9 --neutral-starts), full
original-frame labels, all-pos/all-neg kept, keep-0.80 original-frame
macro F primary, one preregistered read per frozen candidate group, every
access logged.  Artifact: fresh_confirm_manifest.json +
fresh_confirm_access_log.jsonl.

## 5. A6 anchor/neutral stress slices - built, read pending

Same 460 old-confirm sources: anchor 920 (648 pos / 272 bg, 70.4% pos) vs
neutral 920 (149 / 556, 16.0% pos - the honest shape of label-agnostic
starts).  Frames extracted 920/920 (20 s); CPU features 16 shards running.
Read discipline frozen in reports/r5/anchor_neutral_protocol_audit.md.
Dev stress only - never a gate.

## 6. B1 parity (running) and next

CPU-fp32 vs NPU-fp16 parity (32 frozen cases, gamma > 2*eps mask rule,
|dF| <= 0.001 gate) is running on the 910A.  After it passes: the 72
NPU.h plan unlocks in R5 order - spatial S1-adjacent work (feature
upgrade for the candidate scorer) in parallel with the VideoMAEv2-B
native cache build (14 NPU.h) and the P/O/S/B/L matrix (8 NPU.h).

## 7. Budget used this phase

CPU: ~6 h wall (ablation, oracle analysis, contract validation, slicing,
parity-CPU stage).  NPU: parity stage only so far (~0.5 h of the 72 h
envelope).  No submission spent.
