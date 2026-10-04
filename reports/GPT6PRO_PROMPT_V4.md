# Prompt for GPT-6-PRO, round 4: strategy under two pending platform scores

Style: ASD-STE100 Simplified Technical English.  Short sentences.  Active
voice.  Present tense.  Mark every estimate as an estimate.

Answer each question in this order: direct answer (five sentences or
fewer), mechanism, cheapest deciding experiment, failure condition.

## 0. What happened since round 3

All numbers below are measured on disk or reported by the operator.  The
platform has been DOWN since the last submission round; the two new
packages are packaged, validated, registered, and NOT yet scored.

### 0.1 The two pending packages (single-variable vs the 34.73 parent)

| Package | Changed factor | Key local evidence |
|---|---|---|
| LFM_V11_VTREPLAY (id 56693) | feature contract only: the 33.85 probe head replayed on valid-token-pooled features (its own training contract) | A0 defect measured: e_z = 0.289 between all-token cache and valid-token pooling; the replay changes 266/426 masks (Jaccard 0.849 vs the 33.85 mask) |
| LFM_V11_EXACTDP (id 56694) | training objective only: TCN trained to maximize EXACT expected binary-temporal F via DP recursion | pooled-feature 3-seed result below |

Both: spatial boxes byte-identical to the parent on kept frames, keep
0.80, 178.95 MB declared, independent checker passes.

### 0.2 A0 feature-contract defect: CONFIRMED

Deployment cached ALL-token means; the probe trained on valid-token
means.  Padding is only 13.4 percent of tokens, yet e_z = 0.289 - the
padding outputs are nonzero and directional.  Head sensitivity splits:
the old shipped head (trained and deployed all-token) is fully robust
(mask Jaccard 1.000 on 16/16 probe videos); the probe head (trained
valid-token, deployed all-token in the 33.85 package) is not (min mask
Jaccard 0.667).  This is the concrete mechanism candidate for the 33.85
(-0.88) result.  Cost of the fix: recomputing all 8,848 deployment
keyframe features took 2.5 minutes on 6 cards.

### 0.3 Expected-F on pooled features: 3-seed confirmation

Paired arms (same TCN, same sampler stream), 1,963 train / 1,954
held-out mixed fragments, keep-0.80 simulated F1:

| Arm | s06 | s07 | s08 | mean +- sd |
|---|---|---|---|---|
| MSE | 0.6273 | 0.6276 | 0.6514 | 0.6354 +- 0.0138 |
| exact-DP | 0.6625 | 0.6626 | 0.6625 | 0.6625 +- 0.0001 |

Paired per-fragment bootstrap: +0.0352, CI95 [+0.0291, +0.0413].  The
DP arm's seed variance is near zero - directly maximizing F converges
to the same solution from any initialization, while the ranking
surrogate scatters.  AP moves the same direction (+0.018 to +0.041).

### 0.4 The native-framerate VideoMAE line

Setup: VideoMAEv2 ViT-B on NPU (a missing Conv3d-backward kernel is
bridged by an exactly equivalent Conv2d decomposition, CPU-verified to
5.7e-06).  Features decode the SOURCE video around each PHD2 fragment
(100 percent source coverage on disk).

Findings, in order:
1. At the 1 fps keyframe grid (8 slots): ordered-window features versus
   repeated-middle-frame features differ by AP +0.003 - no motion
   evidence survives the keyframe grid.
2. At native resolution (4 sub-windows x 16 native frames, 32-step
   input): the same paired gap is AP +0.0795 (0.6406 vs 0.5611).
   Motion evidence EXISTS and is destroyed by coarse sampling BEFORE
   the encoder.  But keep-0.80 F1 ties (0.597 vs 0.597) - the AP gain
   does not convert to the truncated decision.
3. Best native configuration: 4-window binary labels + DP at lr 1e-4 =
   AP 0.7845 +- 0.0016, paired over mse_bin 3/3 (+0.032/+0.010/+0.005).
   F1 still ties MSE.
4. Occupancy-gain labels (q_t per sub-window, the D1 direction): WORSE
   under both objectives, with larger seed spread.  Closed for this
   scope.
5. 8 sub-windows (64-step inputs): BOTH objectives collapse to
   near-random (DP 0.44-0.52, MSE 0.50+-0.01).  The TCN receptive field
   (15) covers a quarter of the sequence - a recipe-fit failure, not a
   DP failure.  Reopen condition recorded: dils (1,2,4,8,16), more
   steps, lr re-scan.

### 0.5 The lr-saturation law we found

At lr 1e-3 the DP loss on 32-step and 64-step sequences saturates into
a bad attractor that IGNORES the features (ordered and repeated inputs
converge to the identical solution).  At lr 1e-4 it recovers and
becomes the most seed-stable configuration in the study.  MSE at 1e-3
is also degraded on 64 steps.  We now scan lr for every new sequence
length / label protocol as a standing rule.

### 0.6 The 174-GT gate is dead

The 174 preliminary videos match PHD2 testing intervals statistically
(70.7 percent exact duration matches, ratio 1.000 +- 0.02 on 42 tight
mappings, 79 percent is_last users), but full-timeline alignment shows
the mp4s are NOT contiguous cuts of the downloaded source media
(first/middle/last frame alignments contradict; GIF intervals do not
land on the mp4 timeline - likely a source-version time drift).  GT is
not exportable at reasonable cost.  The platform submission is the only
official-domain measurement we have.

### 0.7 A methodological accident we caught

Float filtering (occupancy) and integer filtering (binary) produce
DIFFERENT fragment pools (1,421/1,441 versus 1,292/1,329).  Absolute
numbers compared across the two pools misled us for one report cycle
(we first attributed an F1 jump to label effects; it was mostly pool
plus lr).  Rule adopted: any filter change creates a new pool; no
cross-pool comparisons.

### 0.8 REINFORCE arm

Leave-one-out baseline collapsed to zero advantage after saturation;
the arm is void this round.  Not yet repaired.

## 1. The questions

### Q1 - Score interpretation tree (highest priority)

The two packages are independent single-variable tests against the
34.73 parent.  Give the FULL decision tree: for each combination of
{VTREPLAY >= 34.7, VTREPLAY ~ 34.3, VTREPLAY <= 33.9} x {EXACTDP >=
34.7, EXACTDP ~ 34.3, EXACTDP <= 33.9}, state what mechanism it
confirms or kills, and the immediate next action.  Also answer: should
we submit both in one batch, or VTREPLAY first and EXACTDP after
reading its score?  Where in the tree do we submit a THIRD package, and
what would it be?  Mark which tree branches are most probable as
estimates, with reasoning.

### Q2 - Theory of the DP-loss lr sensitivity

Why does the exact expected-F loss saturate into a feature-ignoring
attractor at high lr on long sequences, while MSE merely degrades?
Give the gradient-structure mechanism (sigmoid saturation times the DP
denominator structure), and, if possible, a formula predicting a safe
lr as a function of sequence length T, label prevalence, and loss type.
Propose stabilizers ranked by cost: loss normalization, gradient
scaling, logit temperature, Frank-Wolfe style step control.  Design the
cheapest experiment that discriminates "saturation" from "bad loss
landscape" on 64-step inputs.

### Q3 - Expected-F optimum versus truncation optimum

Across ALL our experiments, the DP head wins AP and seed stability but
NEVER converts to keep-0.80 F1 (pooled: +0.035 F1 yes; native: F1 ties;
AP gaps up to +0.08 do not convert).  Explain the mechanism: E[F]
optimal policies, single-draw truncation, and the calibration gap
(connect to Waegeman et al. on Bayes-optimal F-measure maximizers).
State the conditions under which an expected-F-trained head MUST beat a
MSE head at the deployment cut, and give the diagnostic that predicts,
BEFORE training, whether a dataset regime will deliver the F1
conversion.  Is there a post-hoc transformation of a DP head's
probabilities that recovers the cut performance implied by its AP?

### Q4 - Native-line deployment semantics

A native head scores 32 or 64 sub-window slots; the deployment mask
must select original frames.  Design the aggregation (slot score to
seconds to frames), the feature pipeline for the 426-video drop
(source-video decode: the intake mp4s are on disk; estimate decode
cost), and the pre-registered recipe matrix for the 8-window retry
(dils (1,2,4,8,16), steps, lr scan grid).  Also: is a hybrid sensible -
native features for RANKING inside a coarse window, pooled features
for the cut?  What is the cheapest end-to-end proof that the native
line can produce a legal submission package?

### Q5 - Official-domain signal with a dead 174 gate

With 174-GT unbuildable, the platform submission is our only
official-domain measurement.  Design the minimal-submission information
strategy: how many packages, in what order, to (a) validate the A0
mechanism, (b) validate the expected-F mechanism, (c) calibrate the
PHD2-to-official transfer coefficient with confidence intervals.  Are
there ANY other legitimate official-domain signals (the 159 no-slide
videos of the 426 set, format checker outputs, k_size verification
behavior) worth exploiting?  Attack your own proposal: what is the
risk that sequential submissions overfit the platform's test set?

### Q6 - Attack our evidence chain

We have caught ourselves twice (the pool accident; the recipe
misattribution in round 3).  Now attack this round's conclusions in
order of likely damage: (a) the 3-seed expected-F result - the 984
held-out sources are dev-exposed; how inflated is our CI likely to be,
and what clean confirm design is still available?; (b) the A0 causal
story - what else could produce the mask divergence and the -0.88?; (c)
the native motion claim - is the rep-arm a fair static control, or does
repeating a frame also destroy static-context information that ord
retains?  For each: the strongest counter-hypothesis and the cheapest
experiment that would falsify our claim.

### Q7 - Execution order for the next 72 NPU-hours

Constraints: platform return time unknown; 6 cards free; CPU abundant.
Candidates: 8-window retry with the wide-TCN recipe; deployment native
pipeline (Q4); REINFORCE repair; pooled-line confirm-set construction
(500 fresh sources, source-disjoint from dev); more DP seeds; the
signal-probe on cached tokens (motion at the token level, from round
3, never run).  Give the exact ordered list with budgets, what runs in
parallel, and the single result that would reorder everything.

### Q8 - Strategic audit of the expected-F line

Pooled DP +0.035 is local-only; native AP gains do not convert; the
decision-layer oracle (+0.116) remains unexploited.  Assess the
expected-F line as a whole: is it (a) the main line, one packaging step
from platform-validated gains; (b) an AP-improving but
F1-neutral research direction; or (c) already refuted for the platform
metric by the native F1 ties?  Define in advance: what platform or
local evidence, by when, would justify closing the line - and what
would justify making it the only line.  Consider the alternative
we have NOT tried: direct optimization against the OFFICIAL joint
metric (temporal plus spatial IoU) rather than the binary-temporal
proxy, including what spatial GT would be needed and whether the 426
drop's no-slide structure makes it feasible.

## 2. Output format

Answer Q1 first.  Then Q2 to Q8.  For each: direct answer, mechanism,
cheapest deciding experiment, failure condition.  Mark estimates.  Do
not treat any number from a dev-exposed pool as a clean confirm.  Keep
ASD-STE100.
