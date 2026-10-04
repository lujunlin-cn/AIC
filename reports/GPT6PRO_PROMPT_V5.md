# Prompt for GPT-6-PRO, round 5: the 6.2-raw-gap attack after both scores landed

> **ERRATUM (added 2026-10-04 after the round-5 answer returned).**  The
> score-unit premise in section 0.1 of this prompt was WRONG.  Per the
> operator: 34.95 / 34.74 / 45 are ALL raw scores with NO size penalty.
> There is no "platform vs raw" duality, no 38.83/38.60, no 6.2 gap
> (real gap = 45 - 34.95 = 10.05), and the "63 platform ceiling" was a
> double conversion.  Consume every conditional number derived from
> 38.8333 in the round-5 answer through that correction; the protocol
> and gate structure of the answer are score-unit-invariant and stand.
> This prompt body is kept verbatim for the record.

Style: ASD-STE100 Simplified Technical English.  Short sentences.  Active
voice.  Present tense.  Mark every estimate as an estimate.

Answer each question in this order: direct answer (five sentences or
fewer), mechanism, cheapest deciding experiment, failure condition.

## 0. What happened since round 4

### 0.1 Official scores (platform, k_size = 0.90 applied)

| Package | platform | raw F | parent | reading |
|---|---:|---:|---:|---|
| V11_VTREPLAY (56693) | **34.95** | 38.83 | 34.75 | NEW HIGH. +0.20 platform / +0.22 raw. A0 recovery = 122 percent: the valid-token feature-contract fix is platform-confirmed, and the repaired head BEATS the parent's all-token head |
| V11_EXACTDP (56694) | 34.74 | 38.60 | 34.75 | tie, noise level. The clean-confirm +0.031 did NOT convert to the official domain |

Operator intelligence: the current FIRST PLACE reports raw 45 (before
size penalty) - a 6.2 raw gap to us.  Reference points on our side: the
in-domain fixed-ranking prefix oracle is F = 0.6995, so a PERFECT
re-ranking of our current frames maps to a 63 platform score; our
current pooled AP is 0.74.  Our keep curve in-domain: 0.80 and 0.70 tie
at the peak, 0.50 drops 0.025 - the leader is probably not winning by
denominator cuts alone.

### 0.2 Round-4 audit results (all verified against running code)

1. The 3-seed "+0.0352" was a single-seed bootstrap; the 3-seed mean is
   +0.0271 with a seed-level t-interval crossing zero.
2. The native dp_bin arm had a reward-unit bug (slot-unit G with
   frame-unit gains; counterexample E[F] = 1.85 > 1).  Retracted.
3. The native VideoMAE caches had TWO input-contract defects, measured:
   seek offsets interpreted 33x-78x off (stream time_base vs container
   units), so selected frames sat 24-1608 s away from 8-12 s targets
   with 15/16 duplicate frames per window; and raw 0..255 pixels fed to
   the encoder with no (x/255 - MEAN)/STD.  ALL native results are
   quarantined; the +0.0795 ord-vs-rep "motion evidence" is unreliable.
4. Both packages were verified clean of these defects before their
   submissions.

### 0.3 The saturation inversion (round-4 diagnostics)

- The 3 DP seeds produce IDENTICAL top-0.80 masks on 99.97 percent of
  dev fragments; 100 percent of logits beyond |8| - full saturation.
- Content destruction: DP scores F1 = 0.6625 on real features, 0.6625
  on zeros, 0.6625 on shuffled features.  The MSE arm is similar (real
  0.6354, zeros 0.6625 - content makes it WORSE on dev held-out).
- At lr 1e-4 (no saturation) MSE and DP both converge to 0.6625 on dev
  - the objectives TIE when training is healthy.

### 0.4 The clean confirm-set verdict (500 fresh sources)

- GROUP A (the saturated package-lineage checkpoints): DP beats MSE by
  +0.031 F1, source-cluster CI95 [+0.0207, +0.0406], 3/3 seeds positive
  - passes the preregistered promote rule IN-DOMAIN.
- GROUP B (healthy lr1e-4 checkpoints): clean tie, CI95
  [-0.0030, +0.0010].
- Platform: EXACTDP ties the parent.  All three measurements agree: the
  expected-F objective adds nothing at the cut; the saturated prior
  head's in-domain robustness does not transfer to the official domain
  either.  The expected-F LINE IS CLOSED under the preregistered rule.
- Interpretation adopted: our pooled 8-slot head is a POSITION-PRIOR
  ordering with weak content; 0.6625 is the pool's prior ceiling.

### 0.5 Assets and constraints

- Clean confirm set: 500 frozen sources (sha d2695ae3), 920 fragments,
  features on disk, reusable ONLY for preregistered one-pass reads.
- 426-video drop source media 100 percent on disk; PHD2 pool ~14,550
  videos; AVE-PM 22,848; DAVSOD; ClipShots; QV/TVSum/FLMS/CPC/FCDB.
- Student size tier: <=500 M parameters for k_size 0.95; we currently
  ship 178.95 MB declared (k_size 0.90).  The shipped stack: LFM2.5-VL
  450M vision tower (spatial) + 246 K-param TCN (temporal) + B3 boxes.
- NPU is DOWN (container writable layer reset; CANN runtime missing,
  driver intact).  Recovery in progress; every NPU task queues behind
  it.  CPU: 192 cores, 755 GB - confirmed-set features already
  extracted on CPU with a verified-contract copy.
- Submission cadence so far: 2 diagnostic packages per round; platform
  return time was 3+ days this cycle.  No hard quota disclosed.

## 1. The questions

### Q1 - Decomposing the 6.2-raw gap (highest priority)

Given: our raw 38.83; leader raw 45; our prefix oracle 0.6995 (=> 63
platform ceiling with unchanged frames); our AP 0.74; our keep curve
peak at 0.70-0.80; spatial boxes byte-identical to a parent that has
never been optimized for IoU quality since the teacher rounds.  Formal
score: F_v = 2 * sum(hit-frame IoU) / (N_pred + N_gt), video-macro.
Decompose the 6.2 raw gap into plausible contributions: (a) temporal
ranking quality, (b) spatial IoU quality rho, (c) denominator structure
(keep fraction vs GT length), (d) calibration/thresholding, (e)
per-video adaptivity.  For each: what leader behavior would be
consistent with raw 45, and what is the cheapest signal we can buy (a
submission, a local experiment, or a public-dataset proxy) to test that
attribution?  State which single attribution, if confirmed, moves us
farthest toward 45.

### Q2 - Spatial IoU quality (rho): the never-optimized term

Our spatial stack (LFM2.5-VL 450M tower + B3 head) has shipped
byte-identical boxes through five submission rounds.  Design the attack
on rho: (a) direct IoU-objective training on our labeled pools (which
pools have SPATIAL ground truth of usable quality?); (b) distillation
from the Qwen3-VL-32B teacher's boxes with IoU-weighted matching instead
of the current matching; (c) a stronger modern open spatial detector
segmenter adapted to GIF-highlight frames; (d) test-time tricks (box
refinement, multi-scale, CRF-free masks).  For each: expected rho gain
(estimate), parameter budget impact, data requirement, and the cheapest
offline evaluation that predicts the official F gain WITHOUT a
submission.  What rho delta is worth a submission slot?

### Q3 - The native-framerate line, rebuilt correctly

The native line is quarantined but the ASSET is real: 426 drop source
media on disk, the Conv2d bridge for VideoMAE backward on NPU, and the
corrected extractor design (stream-time_base seek + official
normalization + hash-versioned cache keys).  Design the rebuild:
sampling protocol (sub-windows x frames, overlap, dedup rules), encoder
choice (VideoMAEv2-B vs larger, given the parameter tier), feature
level (tubelet tokens vs pooled), head, and the FAIR motion control
(ordered vs shuffled vs multi-frame static) that would finally settle
whether motion evidence exists.  Give the recipe matrix with the
smallest configuration count that still supports attribution, and the
pre-registered go/no-go gate for packaging a native-head submission.
Also: does the 1 fps grid really destroy motion evidence, or did the
round-3 measurement merely fail to find it - what is the cheapest
deciding experiment?

### Q4 - Parameter tier strategy: is a bigger encoder affordable?

k_size 0.90 applies above 500 MB declared; 0.95 presumably inside.
Candidate upgrades: VideoMAEv2-L (~305 M params), InternVideo2-1B,
V-JEPA2, or a wider LFM tower - each must coexist with the 450 M tower
we ship for spatial.  Questions: (a) does the size tier count encoder
parameters loaded at INFERENCE only (our LM is stripped but the tower
is live)?  (b) is there a tier boundary maneuver that lets us ship a
bigger temporal encoder at 0.95 (e.g., temporal-only inference path
where the spatial tower is not loaded)?  (c) expected AP gain per GB -
is VideoMAEv2-B -> L worth +0.02 AP on evidence, and where does that
show up in F?  Give the exact decision rule for "bigger encoder yes/no"
that does not depend on unverified assumptions about the k_size rule.

### Q5 - Time-head design after the expected-F closure

With expected-F closed, what is the best remaining time-head program?
Candidates: the fixed-budget conditional DP (Q3.6 of round 4 -
training shares the top-K constraint), per-video adaptive keep
(adaptively choose the cut per video from score distribution - our one
adaptive attempt failed, -0.028), calibration/threshold post-processing
on the prior head, or ensembling pooled+native ranks.  Rank them by
expected F gain per unit of compute, and define the local gate each
must pass on the EXISTING confirm set (already spent?) or on a NEW
confirm build.  Also answer: is there any principled post-hoc
transform that converts our AP advantage into F at the official
metric's structure (F_v uses N_pred + N_gt, so per-video N_gt matters -
can we estimate N_gt from our scores and set per-video keep to
min(0.8, estimated-G-length rule)?  Is that legitimate or is it GT
inversion we should not do?

### Q6 - Submission strategy with a live leaderboard

We now have a live platform and a known leader score.  Design the next
submission sequence: how many packages, which single variables, in what
order, with what preregistered readings?  Candidates: (a) keep-fraction
probe (0.70 variant - in-domain safe but is it information?), (b)
spatial-rho package (Q2 output), (c) native-head package (Q3 output),
(d) nothing until local gates pass.  Attack your own proposal: with the
leader at 45 and us at 34.95, what is the risk of burning submissions
on attribution probes instead of gap-closing candidates?  What is the
minimum submission budget to reach a defensible second place, as an
estimate?

### Q7 - Attack our evidence chain (third round)

Our confirm set: 500 sources, 920 fragments cut with the same 65
percent anchor protocol as dev.  Attack it: (a) the anchor prior
(65 percent GIF-anchored) makes position-prior heads look better than
official-domain frames where no such anchoring exists - is our
confirm-set in-domain +0.031 for GROUP A partly an artifact of the
cutting protocol?  (b) CPU-extracted features (fp32) vs dev's NPU
(fp16) - does the contract copy hold at noise level, and what check
would prove it cheaply?  (c) the confirm set was read ONCE for GROUP A
and B, but the GROUP B addition happened after the saturation finding -
is the confirm set now dev-tainted in any way that matters for future
reads?  (d) anything else in our round-4 chain that would not survive a
hostile review?

### Q8 - Resource allocation for the next cycle

NPU returns in days (CANN reinstall), CPU is ample, platform return
time ~3 days, calendar unknown.  Rank these programs by expected
official-score gain per week: spatial-rho attack (Q2), native rebuild
(Q3), bigger encoder evaluation (Q4), time-head variants (Q5), teacher
re-distillation with IoU-weighted matching, more PHD2/AVE-PM data into
temporal training.  Give the exact 72 NPU-hour plan for the first week
after CANN returns, what runs on CPU in parallel, and the single
measurement that would reorder everything.

## 2. Output format

Answer Q1 first.  Then Q2 to Q8.  For each: direct answer, mechanism,
cheapest deciding experiment, failure condition.  Mark estimates.  Do
not treat any number from the dev pool or the 65-percent-anchored
confirm set as official-domain evidence.  Keep ASD-STE100.
