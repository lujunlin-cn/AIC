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
step 300 - matching seed-1 within 0.008, no early slide. Seed-2 completes
later tonight; its full-set number is the one outstanding check.

The verdicts we draw, for you to audit:

1. The recipe moved the ranking, the tower update did not. The frozen
   control reaches 0.7462 on the full held-out set. The fourteen-variant
   matrix (batch 64, 2,500 steps, about 81 exposures per fragment) capped
   at 0.696 with the SAME tower. The probe recipe (batch 8, 900 steps,
   about 3.7 exposures) lifts the same frozen tower by +0.056. The wall at
   0.677 to 0.696 was a recipe property, not a representation ceiling.
2. LoRA is zero gain: full-set paired delta -0.0019, inside noise. The
   prefix gap (-0.02 to -0.035) does not survive to the endpoint.
3. The prefix reads 0.02 to 0.03 BELOW the full set for the same model;
   it stays monitoring-only.
4. The smoke point 0.7067 is fully unmasked: the probe recipe alone
   crosses 0.70 by step 150 with a frozen tower.

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

No new platform submission. No QV revival. No confirm-set construction yet.
The 984 held-out sources are dev-exposed, as you ruled.

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

## 3. Q2 - the recipe effect and the representation

3.1 Ablate the recipe. The frozen-tower jump from 0.690 to 0.746 changed
three knobs at once: batch 64 to 8, steps 2500 to 900, exposures about 81
to about 3.7. Design the smallest ablation that attributes the +0.056:
which single knob carries it, and is it a regularization effect (early
stopping), an optimization-noise effect, or a label-coverage effect (3.7
exposures means 2.5 percent of fragments are never seen)? Give the grid,
the stop rule, and the budget in NPU-hours on our stack (one probe run is
about 2 NPU-hours).

3.2 Does the recipe transfer? Two downstream uses wait on 3.1: the M01
video-encoder runs (your section 2.4) and the decision layer, whose route
A/B were both tested against heads trained under the sweep recipe. If the
recipe effect is real, both conclusions were measured against a
handicapped baseline. Give the rule for which past closures must be
re-opened under the probe recipe, in priority order, and which may stand.

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

Given: the LoRA line is closed with data; the recipe line is open and
unexplained; seed-2 lands tonight and only confirms; your stage-2 list has
five items, two of which (decision-layer routes) were measured against
sweep-recipe heads. Give the exact order for the next 48 NPU-hours
(8 cards), one line per slot. Constraints: the recipe ablation (3.1), one
M01 encoder smoke on NPU (3.3), and a probe-recipe rerun of the decision
layer route A must all fit. Name the two experiments you would run in
parallel first, and the one result that would reorder the plan.

## 7. Output format

Answer in this order: 0.3 redo, then Q1 to Q5. For each answer: direct
answer, mechanism, cheapest deciding experiment, failure condition. Keep
ASD-STE100. Mark estimates as estimates. Do not treat the step-12 point as
evidence in any answer.
