# Prompt for GPT-6-PRO, round 3: audit of your V11 plan, with fresh data

Style: ASD-STE100 Simplified Technical English. Short sentences. Active voice.
Present tense. Mark every estimate as an estimate.

## 0. What changed since your V11 plan

We executed your stage-1 list. This section reports the numbers you asked
for. Section 7 lists our questions.

### 0.1 The LoRA seed-1 trajectory, with its paired frozen control

Mid-run evals use the deterministic 270-fragment prefix; step 900 is the
full 1,954. The frozen control is identical in sampler, steps, evals and
TCN updates, with NO adapters.

| Step | frozen AP | seed-1 LoRA AP | delta |
|------|-----------|----------------|-------|
| 12 (smoke, unpaired) | - | 0.7067 (full set) | - |
| 150 | 0.7067 | 0.7048 | -0.0019 |
| 300 | 0.7172 | 0.6861 | -0.0311 |
| 450 | 0.7209 | 0.7011 | -0.0198 |
| 600 | 0.7293 | 0.6943 | -0.0350 |
| 750 | 0.7269 | 0.6943 | -0.0326 |
| 900 (FULL 1,954) | **0.7462** | **0.7443** | **-0.0019** |

Seed-2 (both RNGs 20261004) prefix points: 0.7013 at step 150, 0.6943 at
step 300 - matching seed-1 within 0.008, no early slide. UPDATE: the
operator stopped the LoRA line; seed-2 has no endpoint (status
PAUSED_EXPLORATORY). The paired frozen control replaces it as the
deciding evidence.

The verdicts we draw, for you to audit - note that items 1 and 2 were
REVISED the same evening, after the platform result in section 0.6:

1. LoRA is zero gain: full-set paired delta -0.0019, inside noise. The
   prefix gap (-0.02 to -0.035) does not survive to the endpoint.
2. **REVISED: the recipe effect was an evaluation-protocol artifact.**
   Under the probe protocol, the SHIPPED head - tcn_s0.pt, the head in
   the 34.73 package - also reads AP 0.7462, identical to the probe
   frozen head. Its stored val_ap 0.6263 came from an older protocol.
   The 0.690-versus-0.746 gap between the sweep matrix and the probe is a
   protocol difference, not a training-recipe effect. The two heads tie
   on every PHD2 measure; the recipe line closes.
3. The prefix reads 0.02 to 0.03 BELOW the full set for the same model;
   it stays monitoring-only.
4. The smoke point 0.7067 is fully unmasked: any trained head sits near
   0.746 under the probe protocol; the tower was never the variable.

### 0.2 Exposure, manifests, hashes

- Sampler pool: 1,963 mixed training fragments, sampled with replacement,
  8 per step, 900 steps: 7,200 instances, mean 3.67 exposures per
  fragment, P(never sampled) = 0.025. The sweep recipe exposed each
  fragment about 81 times.
- Eval set: 1,954 mixed fragments, 984 held-out sources. Manifest
  SHA-256: 24fae58a334bd0aed3eb398929cce39f15ce5a6616be15a27db3800738e312c6
- Checkpoints: seed-1 LoRA lora_probe.pt
  c55f82786e3415e2ea26522184f55228467af84ade5d5041b6e07ad07897d748;
  frozen control lora_probe_frozen.pt
  5617b3c655b644946e3ff19e3f14cafe4c4fd53c543d0f62dc89381373c7310e
- Seed-1 RNG note: ran before the --seed flag existed - numpy 20261003,
  torch unseeded (audit defect, recorded). Seed-2 pins both.

### 0.3 The prevalence inconsistency: you were right, we measured it

You flagged that p = 0.33 and keep-all F = 0.6237 cannot share one total.
Measured on disk, mixed-only, OR-union labels at real frame timestamps:

| Pool | n fragments | keep-all binary temporal F1 | mean per-fragment p |
|------|-------------|------------------------------|---------------------|
| Eval (held-out) mixed | 1,954 | 0.6178 | 0.4722 |
| All rows incl. all-pos/all-neg | 5,886 | 0.4782 | 0.3812 |

The eval-only pair is self-consistent: g(0.4722) = 0.641 sits above the
macro 0.6178, with a plausible Jensen gap. The oracle-decomposition value
0.6237 matches the eval-side pool, not a p = 0.33 population. The p = 0.33
in our V10 report was a platform-score back-fit, not a pool measurement. It
is retired. Please redo your section 2.2.2 conditional reference (the
40.9-point extrapolation) against the corrected total, and say which V11
conclusions move.

### 0.4 Annotator census (from training.csv, which keeps user_id)

Our aggregated train.json drops the user field; the upstream training.csv
keeps it. Measured on the 2,262 feature-pool sources:

| Quantity | Value |
|----------|-------|
| Single-user sources | 83.3 percent (1,885) |
| Two-user sources | 240; three-user 74; four-plus 63 |
| (source, user) pairs with more than one interval | 60.7 percent |
| User-pair disagreement on a 1 s grid, mean / p90 | 0.127 / 0.263 |

Two consequences you should fold in. First, the interval-level vote bug you
flagged in R2 is real at scale: with 60.7 percent of users contributing
multiple intervals, our vote_fraction denominator counted intervals, not
users, for most of the mass. The three-protocol tie never tested user-level
aggregation. Second, the D2 ceiling question now has inputs: 83 percent
single-user sources, and a measured disagreement of 0.127.

### 0.5 What we did not run

No QV revival. No confirm-set construction yet. The 984 held-out sources
are dev-exposed, as you ruled.

### 0.6 The platform result that forces the revision

We packaged the probe frozen head (LFM_V10_PROBEHEAD_SEMIFINAL): spatial
boxes byte-identical to the 34.73 parent on kept frames, keep 0.80, mask
volume within 0.1 percent. Platform score: **33.85, minus 0.88 versus the
parent**. Mask analysis: 426 of 426 videos changed their kept-frame set,
Jaccard 0.773. Read together with the protocol finding above:

- Two heads that tie at AP 0.7462 on PHD2 differ by 0.88 platform points.
- PHD2-domain ranking metrics do not predict the platform delta's sign.
  Third directional failure (KD flat, QV worse, now a tied-swap negative).
- Deployment variance between ranking-equivalent heads is of order
  0.5 to 1.0 platform points - the same order as every "improvement" we
  have chased this week.

Our proposed rescue: the 174-video preliminary set carries official GT
and is disjoint from the 426-video semifinal drop. We plan to score both
heads as real F on the 174 GT and, if the ordering reproduces (old head >
new head), adopt the 174 F as the promotion gate for every future
package. Audit this plan: leakage risks (is preliminary GT legitimate for
semifinal model choice?), statistical power at 174 videos for deltas near
0.005, and the failure mode if the ordering does NOT reproduce (what does
that say about the 174-to-426 transfer?).

## 1. Framing for the questions

Your V11 plan is accepted as the working protocol. We adopt: M1/M2/M3
metrics, the delta gates, NO_PRACTICAL_GAIN closure records, the audit
table, and the stage order. The questions below ask you to sharpen
decisions the plan leaves open, now that the first data exists. Answer each
question with: the direct answer (five sentences or fewer), the mechanism,
the cheapest experiment that decides it, and the failure condition.

## 2. Q1 - label side, after the audit

2.1 Frame-level binary vs block occupancy (D1). The deployment metric is
binary at original-frame granularity: a kept frame either hits a GT frame or
it does not. For MSE ranking supervision, we believe the correct target is
block-level binary (does this second contain at least one GT frame), while
the occupancy ratio h/d belongs to the expected-F objective of your Q3,
where d_t and w_t enter the reward. Confirm or correct this split. If
occupancy is also the right ranking target, say why a soft target helps a
model evaluated under a hard metric.

2.2 Annotator sparsity cap (D2). The census is in section 0.4: 83.3
percent of feature-pool sources are single-user, 60.7 percent of users
contribute multiple intervals, and user-pair disagreement averages 0.127
(p90 0.263). Before we run the three-arm D01 test, give the function that
maps these measured quantities to an upper bound on the D2 gain, and
compute the number. With 83 percent single-user sources, is the D2 ceiling
capped near 17 percent of the source mass times the within-pair
disagreement? Give the formula and the numeric estimate as an estimate.

2.3 The 2/3 subjective-divergence cap in your 1.2.3 assumes equal-
probability divergent GT. Our measured disagreement is 0.127 mean, and the
OR union covers 47 percent of eval frames. Compute E_Y[max_a F] minus
max_a E_Y[F] for the measured disagreement level, and give the test that
decides whether the margin is learnable rather than noise.

2.4 Your D01 requires a fresh confirm set because the 984 are dev-exposed.
PHD2 ships an official train/val split. Using the official val sources as
the confirm set costs zero construction and was never touched by our
training. Is that defensible, or does the official val carry its own
selection bias that a self-built set avoids? Give the decision rule.

## 3. Q2 - what can still be trusted, and what to measure instead

3.1 The protocol confound. The sweep matrix read 0.690 under its protocol;
the probe and the shipped head both read 0.7462 under the probe protocol.
Design the one experiment that pins the confound: score the frozen probe
head AND a sweep-matrix head under BOTH protocols on the same fragments.
If protocol fully explains the gap, say which protocol component is
responsible (feature source, label timestamps, fragment filter) and give
the single canonical protocol we should freeze.

3.2 Deployment variance. Two ranking-equivalent heads moved the platform
by -0.88 (section 0.6). Is there ANY local measurable that predicts this
variance - for example, mask-set distance between heads versus platform
delta, score distribution shape, or per-video rank stability? If nothing
on disk predicts it, say so plainly, and give the cheapest design that
estimates the variance floor (how many head re-trains x platform
submissions would map it, and whether we can afford it).

3.3 NPU viability. Candidates: VideoMAE ViT-B, V-JEPA 2.1 ViT-B/16,
InternVideo2-dist-B, SigLIP2-L tower. Backend is torch_npu on Ascend 910B,
 eager attention required, no vendor plugins. For each candidate give a
10-minute smoke test that proves or kills forward+backward on this stack:
module list to instantiate, input shape, the two or three operators most
likely to lack a NPU kernel, and the pass criterion.

3.4 Your section 2.2.3 says 1 Hz global vectors may have already lost fast
motion, so differencing them cannot recover it. Our tower reads frames at
about 1 fps with 256 tokens per frame, mean-pooled to one 768-vector per
frame. The pool keeps per-frame vectors before pooling. Is there a
cheap probe on EXISTING cached frame tokens (no new encoder) that tests
whether motion evidence survives in them - for example, a temporal-shuffle
discrimination head, or a two-frame delta probe trained on GIF boundaries?
Design it. The probe must not train on eval sources.

3.5 Redo the 40.9-point conditional reference under the corrected
prevalence (0.472 eval pool), and state whether the AP jump to 0.746
changes it. All your platform-extrapolation warnings still apply; we want
the arithmetic updated, not a prediction.

## 4. Q3 - expected F and RL, on real labels

4.1 Conditional independence. Your DP recursion computes the exact
expected reward for any factored Bernoulli policy; adjacent p_t sharing TCN
context does not bias the objective, only the policy class. Confirm. Then
answer: how much of the prefix-oracle gap (+0.116) can a factored policy
recover in principle when the ranking is frozen? If a threshold policy on
calibrated p_t already extracts most of it, the DP adds nothing over a
threshold - give the diagnostic that separates these cases before we
implement.

4.2 Keyframe-to-original-frame mapping. Our labels binarize at sampled
keyframes (about 1 fps) with half-frame tolerance. The reward wants
original-frame counts d_t and per-frame IoU w_t. PHD2 ships the source
videos, so true frame timestamps are recoverable. Give the reconstruction
procedure and the validation that the reconstructed d_t matches the
official N_pred semantics (block sums to covered original frames, no
double count at segment seams).

4.3 Empty GT. GIF-domain sources all have at least one GIF. We have never
seen G = 0 in 2,262 sources. May we drop your G = 0 branch and assert
G >= 1 as a data property, or must the trainer still guard it?

4.4 The KL anchor pins the policy to p0 from the supervised init. Our
supervised head is miscalibrated (that killed the decision-layer routes A
and B). An anchor to a miscalibrated prior may drag the policy away from
the reward. Compare three stabilizers under a miscalibrated init: KL to p0,
an entropy floor, and a pairwise ranking-preservation penalty. Recommend
one, with the switch criterion.

4.5 Reward granularity. The formula gives the oracle to per-second blocks.
Our segments are 8 to 14 s, so T is 8 to 14 actions per fragment - small.
Does the REINFORCE baseline (K = 8 samples) have enough resolution at
T < 15, or does the exact-DP route dominate so early that REINFORCE is only
a cross-check? Give the sample-complexity argument, as an estimate.

## 5. Q4 - protocol calibration to our compute

5.1 Sample size. Our measured seed noise at 270 fragments is about 0.01 AP.
The paired per-source delta sigma at 1,954 fragments is unknown until
seed-2 lands; assume sigma in [0.005, 0.015]. Give the table: confirm-set
sources needed for 80 percent power at delta = 0.007, one-sided, for sigma
in that range. Say whether the official-val confirm set (about 1,100
sources, if we adopt 2.4) clears the bar at sigma = 0.01.

5.2 Two-tier closure. NO_PRACTICAL_GAIN with 3 seeds x 3 structural
variants costs about 50 NPU-hours per line. With 8 cards shared and about
6 open lines, that rule forbids exploration. Give the two-tier rule:
exploration-tier (pause, reopenable, cheap evidence) vs decision-tier (full
NO_PRACTICAL_GAIN). Specify the exact evidence each tier requires, and the
language each tier permits in a report.

5.3 Mid-run eval prefix. Mid-run AP uses a deterministic 270-fragment
prefix. Your audit table calls sample drift INVALID_EVAL_SET. We keep the
prefix for cheap monitoring and the full 1,954 for decisions. Is that
split acceptable under your rules, provided no decision ever cites the
prefix? If not, give the cheapest compliant monitoring design.

## 6. Q5 - order of execution

Given: the LoRA and recipe lines are closed; the 174-GT gate (section
0.6) is the proposed main line; seed-2 lands tonight and only confirms.
Give the exact order for the next 48 NPU-hours (8 cards), one line per
slot. Constraints: the 174 two-head scoring must run first if you approve
it; the protocol pinning test (3.1); one M01 encoder smoke on NPU (3.3).
Name the two experiments to run in parallel first, and the one result
that would reorder the plan.

## 7. Output format

Answer in this order: 0.3 redo, then Q1 to Q5. For each answer: direct
answer, mechanism, cheapest deciding experiment, failure condition. Keep
ASD-STE100. Mark estimates as estimates. Do not treat the step-12 point as
evidence in any answer.
