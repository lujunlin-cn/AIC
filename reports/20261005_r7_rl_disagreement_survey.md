# R7 addendum survey: can RL recover the annotator-disagreement gap?

Date: 2026-10-05 (late evening).  Style: ASD-STE100.  All scores RAW.
Operator directive: no human labeling (micro-study dropped); weights
inventory run; re-survey the frontier for the annotator-disagreement
problem with RL-class methods.  This is the survey result and the
preregistration change it forces.

## 1. The question, stated precisely

R6 evidence: LIVE-YT-VC paper-reported adjacent-annotator IoU ~0.50 ~=
our B3 dev80 0.529; the 0.247 oracle gap may be mostly mean-annotator
vs single-annotator disagreement.  Operator hypothesis: RL-class
methods may recover part of it.

The precise decomposition:
- Part A (unrecoverable by ANY method): the variance of annotators
  around their own mean, conditional on the video.  No trainable
  function of the video can predict a specific annotator's draw the
  video does not identify.  RL adds no information; the Bayes bound
  does not move.
- Part B (potentially recoverable): the DECISION error we make on top
  of the information we have.  Our current stack fits mean IoU with a
  huber loss and takes an argmax.  Under the IoU metric this is a
  PROVABLY suboptimal decision rule - see below.

The survey says Part B is real, has named literature, and has 2026-era
RL machinery.  That is the operator's half-right intuition, made
precise.

## 2. Named evidence

### 2.1 The decision-theory core: Ahmed et al., ICCV 2015

"Optimizing Expected Intersection-Over-Union with Candidate-Constrained
Decision-Theoretic Models" (Ahmed et al., ICCV 2015, ~81 citations).
Canonical result: when ground truth is a DISTRIBUTION of plausible
boxes (annotator ambiguity), the prediction that maximizes EXPECTED
IoU is NOT the mean box.  Mean regression is optimal for squared
error, not for IoU.  Their setting - choose from a candidate set to
maximize expected IoU under box uncertainty - is structurally OUR
setting (129 legal max-windows, B3 score, argmax).  We never applied
the decision-theoretic selection layer; our head stops at "fit the
mean, then argmax".

Implication: even with B3 scores frozen, a better SELECTION RULE over
the same scores may gain.  This is testable on cached data at zero NPU
cost.

### 2.2 Distributional reward under disagreement: Diverging Preferences

"Diverging Preferences: When do Annotators Disagree and Do Models
Know?" (Zhang et al., arXiv:2410.14632).  Findings we use:
- On MultiPref and HelpSteer2, >75% of disagreements come from
  individual predilections, not noise - disagreement is structure, not
  error.  Directly supports our upgraded hypothesis.
- Standard point-estimate reward models CANNOT tell diverging from
  agreeing pairs (Diverging-ID AUROC ~0.40-0.49, chance level).
- Mean-Var distributional reward models recover it (AUROC 0.58-0.65)
  with NO loss in preference accuracy.
- Inference-time decision uses the variance: divisiveness =
  |mu_A - mu_B| - lambda(sigma_A + sigma_B).  The variance is a
  DECISION INPUT, not a nuisance.
- Not done there (explicit future work): per-annotator adaptation.
  We have no annotator IDs either - so we adopt only the
  distribution + variance-aware selection half.

### 2.3 The 2026 RL machinery for exactly this task family

- Time-R1 (NeurIPS 2025): GRPO on Qwen2.5-VL-7B, 2.5K samples,
  zero-shot Charades-STA R1@0.7 = 35.3.  Established the recipe:
  IoU reward + format reward, group-relative advantage.
- VideoTG-R1 (arXiv 2025-10-27, ACM MM 2026 track, ~9 citations):
  curriculum RL with "reflected boundary annotations" - a boundary
  reflection agent filters partially-annotated samples, then a
  difficulty curriculum.  SOTA on standard temporal grounding
  benchmarks.  Proof that "noisy boundary labels + RL" is an active
  2026 frontier, not a dead end.
- TempR1 (arXiv 2512.03963): temporal-aware MULTI-TASK GRPO (moment
  retrieval + highlight detection + more) - cited by TimeLens2 and
  VideoChat-R1.5.  Highlight detection is already an RL reward target.
- CROP (2026): DPO on expert crop preferences, FLMS IoU .621 base ->
  .871 full scheme (NOTE the R7 correction: the full-scheme delta is
  not an isolated DPO ablation).
- SpaceTools (2026-06) / AdaTooler-V (AT-GRPO): VLMs RL-trained to
  CALL a cropping tool - the tool-use direction our architecture-D
  (7B joint pipeline) would sit on.

### 2.4 The structural coincidence that matters

GRPO computes group-relative advantages within a sampled group per
prompt.  Our data IS that shape: one keyframe -> one group of 129 (or
36 shortlisted) candidates -> per-candidate reward = annotator IoU.
Candidate ranking under a within-frame reward baseline is GRPO's
native setting; a small-model version is a listwise / policy-gradient
readout on the same groups.  No other task we have touched maps this
cleanly.

## 3. What this changes in the plan

The human micro-study is dropped (operator directive).  The (i)-vs-(ii)
separator question changes shape: instead of measuring the ceiling
with humans, we PROBE Part B directly - if decision-layer fixes on
cached data can beat 0.529, Part B is nonzero and the "everything is
annotator noise" reading was overstated; if they all fail, the noise
reading strengthens.  Either way the exam is free (CPU, cached data).

### New preregistered line: DECISION_LAYER (all CPU, 2,560-row cache)

- D0 ceiling readout: the best IoU achievable by ANY monotone decision
  rule over the FROZEN B3 scores (isotonic-style bound on the cached
  score/IoU joint).  Pure diagnostic; no gate.
- D1 decision calibration: argmax over a calibrated score g(B3)
  (monotone family, dev-fit).  Gate: +0.015 vs B3, CI lower > 0.
- D2 distributional head: predict per-candidate IoU quantiles; select
  by quantile-tau expected-IoU surrogate (the Ahmed/Diverging hybrid);
  tau chosen on dev only.  Gate: same, plus vs same-capacity huber
  control.
- D3 group-relative readout retrain (small-model GRPO shape): residual
  head with listwise/policy-gradient loss on within-frame candidate
  groups, reward = annotator IoU.  Gate: same two-sided comparison.
- All read on dev80; one-shot confirm on val178/76-fresh for anything
  passing.  Risk flagged: geometric-correction families have an
  official negative precedent (scaling -4.00); these arms change
  SELECTION over legal candidates, not box geometry - recorded as a
  distinct mechanism family in transfer_pairs.

### 7B-side consequence

The VLM-CROP-EXAM gains a second role: its 2,560-row table is the
ready-made reward environment for a CROP-style preference /
GRPO-style post-training of a 7B verifier (after Probe 0 prices the
throughput).  This strengthens architecture-D's path but does not
change its gate.

### Unchanged

InternVideo2-1B temporal line (weights ALREADY local:
/data/aic/pretrained/internvideo2_stage{1,2}_1b + k710 - zero
download, the 4.3 step can start immediately).  TRANSFER-BRIDGE,
Probe 0, slot strategy all stand.

## 4. Weights inventory (910A:/data/aic/pretrained/)

| Present | Path | 9B-era usable |
|---|---|---|
| InternVideo2 stage-1 1B (+k710) | pretrained/internvideo2_stage1_1b | yes (~1.088B total) |
| InternVideo2 stage-2 1B-224p-f4 | pretrained/internvideo2_stage2_1b | yes |
| Qwen3-VL-32B-Instruct (teacher) | pretrained/qwen3_vl_32b_instruct | NO (>9B; teacher only, per frozen role) |
| Qwen2.5-VL-7B / 3B | NOT PRESENT | needs download (~16.5GB / ~7.5GB) |

Disk headroom: 2.5T free.  The 7B download stays gated on the operator
nod; the 3B fallback is cheaper and answers Probe 0's pricing question
at 2x less bandwidth.

## 5. Honest boundary

RL recovers Part B only.  If D0-D3 all fail their gates, the
annotator-noise reading survives stronger than before, and the 9B
budget concentrates on the temporal line - which is where the 28-point
gap lives anyway.  Nothing here touches official test media; every
number stays raw; no conversions.

Sources: Ahmed et al. ICCV 2015 (openaccess.thecvf.com);
Zhang et al. arXiv:2410.14632; VideoTG-R1 arXiv 2025-10-27;
TempR1 arXiv:2512.03963; Time-R1 NeurIPS 2025 / arXiv:2503.13377;
CROP arXiv 2026 (2605.12545); SpaceTools arXiv 2026-06;
AdaTooler-V (ACL); LongVTG-R1 (OpenReview); VideoChat-R1 (2025-04).
