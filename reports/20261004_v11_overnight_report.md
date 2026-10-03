# V11 overnight execution report (2026-10-03 night -> 10-04 morning)

Scope: the three approved axes, executed unattended while the operator
slept.  Style: ASD-STE100.  No platform scores are predicted.

## 1. State at handover

| Item | State |
|---|---|
| Candidate package 1: LFM_V11_VTREPLAY_SEMIFINAL | PACKAGED, validated, NOT uploaded |
| Candidate package 2: LFM_V11_EXACTDP_SEMIFINAL | PACKAGED, validated, NOT uploaded |
| M01 VideoMAEv2 features (426 videos) | running on 3 cards, ~minutes |
| Expected-F four-arm comparison | DONE (one arm invalid) |
| 174-to-PHD2 mapping | PAUSED, premise failed (see V10 report 7) |

## 2. The two packages, and what each tests

Both are single-variable against LFM_V9_B3_VISIONONLY_SEMIFINAL
(official 34.73): spatial boxes byte-identical on kept frames, keep 0.80,
same 178.95 MB declared size (k tier unchanged).

### 2.1 LFM_V11_VTREPLAY_SEMIFINAL - the A0 fix

Changed factor: the feature contract only.  Same probe head as the 33.85
package, now replayed on features pooled with its own training contract
(valid-token spatial-grid mean, FP32, 8,848 keyframes recomputed, 6 cards,
~150 s/card).  Manifest evidence: contract defect measured at e_z 0.289;
the replay changes 266 of 426 masks (Jaccard 0.849 vs the 33.85 mask).

Prediction this package tests: if the -0.88 came from the contract
mismatch, this package recovers to the parent's neighborhood.  If it
scores like 33.85 again, the contract defect was not the cause and the
PHD2-ranking tie-break stands as unexplained variance.

predictions sha256 b517b19a6824f7e34f107e9b953f25fa2f3a69e9bfed9e1c856f6416f38231bb
zip sha256 8433bb99c437c6a14b9d3484183f7577f46b0a30db8e55c01acd34e9791e892e

### 2.2 LFM_V11_EXACTDP_SEMIFINAL - the expected-F head

Changed factor: the training objective only.  A TCN with the same
246,401-parameter structure trained to directly MAXIMIZE THE EXACT
EXPECTED binary-temporal F through the DP recursion (vendor expected_f.py,
exhaustive-verified), on frozen valid-token features, same pool, same
sampler stream as the MSE control.  Mask replayed on the matching
valid-token contract.

PHD2 in-domain evidence (frozen pool_feats, 1,954 mixed held-out):

| Arm | AP | keep-0.80 sim F1 |
|---|---:|---:|
| MSE (incumbent) | 0.7061 | 0.6273 |
| soft-F | 0.7423 | 0.6625 |
| exact-DP | 0.7437 | 0.6625 |
| REINFORCE | (invalid - advantage saturated to zero; do not cite) |

The +0.035 sim-F1 over MSE is the first in-domain gain from changing the
OBJECTIVE rather than the features or structure.  It is a PHD2-domain
number: per rule 4 of the V10 report it cannot predict the platform
direction.  The package exists to buy that direction measurement.

predictions sha256 4f5a82db645908d533855076dc9aaa3243b64670e86221ad1c5759a0f1bc8466
zip sha256 ff52672c05949ef61df945d7711c3896f4b30192124276f1fefe01fe983b36e7

## 3. Validity notes (read before citing)

1. The four arms share init seed and sampler stream (paired design), but
   each arm ran ONE seed.  Status: PAUSED_EXPLORATORY-level evidence, not
   NO_PRACTICAL_GAIN-grade confirmation.
2. soft-F and exact-DP converged to nearly identical scores.  With 8-14
   blocks per fragment both objectives may share the same optimum once
   saturated.  The DP is preferred for deployment claims: it keeps the
   random denominator exact.
3. The REINFORCE arm needs an entropy floor or lower lr; its numbers are
   void this round.
4. Block granularity: training blocks are 1 fps keyframes (d_t = 1), the
   platform counts original frames.  The objective transfer assumes the
   keyframe grid approximates the original-frame counts; the A0 contract
   rule (train/deploy same contract) is honored for features but the
   time-grid mismatch remains a known limit.

## 4. M01 line state - first control run is DONE, and it is negative

Smoke PASS.  Conv2d bridge verified (fwd 5.7e-06, grad 0 on CPU).
Features extracted: 426 deployment videos AND 6,001 PHD2 fragments, each
fragment under TWO input conditions (ordered 16-frame window; middle
frame repeated 16x = same static content, zero motion).

Three-input control, MSE objective, paired seed/stream, 1,963 train /
1,954 held-out fragments:

| Arm | AP | keep-0.80 sim F1 |
|---|---:|---:|
| ord (ordered window, motion present) | 0.7226 | 0.6418 |
| rep (repeated frame, static only)    | 0.7193 | 0.6473 |

Delta is inside noise: NO motion-evidence gain at this sampling level.
Status: PAUSED_EXPLORATORY with a precise scope - "1 fps keyframe
resampling into a 16-frame window, 8 tubelet slots".  The scope matters:
fast motion may already be lost BEFORE the encoder (the keyframe grid),
and the 8-slot aggregation may wash out short events.  The untested
variant is native-framerate sampling (16 frames at 30 fps = a 0.53 s
window), which needs video-file decode and is the reopen condition.

## 5. Submission order recommendation (operator decides)

1. VTREPLAY first: cleanest single variable, tests the A0 mechanism.
2. EXACTDP second: tests the objective mechanism.  Submitting both in one
   batch gives two direction measurements per review cycle.
3. Parent (34.73) stays the reference; no other package changes.

