# Prompt for GPT-6-PRO, round 9: why every learned head converges to the position prior, and what to do about it

Style: ASD-STE100.  Short sentences.  Active voice.  Present tense.
This prompt is a DEEP-RESEARCH assignment.  Search your knowledge to the
2026-10 edge; go name-by-name; say "not sure" where your knowledge ends.
Every frontier claim needs: paper/project NAME + year + what it MEASURED.
A claim without a name is empty.  Mark every estimate.  All our scores are
RAW; no conversions of any kind.

## 0. New facts since round 8 (all verified, 2026-10-06)

### 0.1 The operator changed two rules

- **Submission quota is DAILY-5, not 5 total.**  Diagnostics and real
  candidates run in parallel; nothing crowds out anything.  The r7/r8
  one-shot slot scarcity is gone.
- **Feature redefinition may use any encoder ≤ 9B total.**  The SigLIP
  pool-variant idea is demoted to a fallback.  The main line is a
  STRONGER encoder's features on the temporal pool.

### 0.2 The central obstruction, measured four independent ways

On the champion's own evaluation pool (PHD2_FRAG_V1: 5,886 fragments /
2,262 sources; features = 8 slots x 768-d SigLIP 'mean'; labels = PHD2
selections projected to slot guard windows, MIXED-ONLY fragments; split =
eval_sources_50 source-disjoint; metric = keep-0.80 binary temporal F1 =
2·hit/(k+n_gt), fragment-macro):

| arm | keep-0.80 F1 |
|---|---|
| champion head (probe_deploy_head.pt) | 0.6588 |
| **slotprior (train-fitted per-slot positive rate, NO model)** | **0.6625** |
| champion on zero features | 0.5545 |
| fresh TCN from scratch, exact-expected-F objective, 3 seeds | 0.6625 / 0.6626 / 0.6625 (identical to slotprior to 4 decimals) |
| fresh TCN from scratch, MSE objective, 3 seeds | 0.6273 / 0.6276 / 0.6514 |

Three independent trainings (R5 content ablation, the expected-F
four-arm round, today's preregistered gate run) all converge on the
same number.  The gate threshold (+0.007 vs champion, i.e. 0.6658) has
NEVER been crossed by any learned head.

The QVH window pool (a DIFFERENT pool, centered 8-frame sliding window,
QVHighlights-derived labels): InternVideo2-1B features LOSE to the old
SigLIP-pipeline features there — dual-base TCN comparison (8,998
windows, same head from scratch, 3 seeds): old 0.180 vs IV2 0.105
pooled dev Spearman (+0.056, cluster CI [-0.001, +0.110]); P/O/L
readout: old 0.425 vs IV2 0.2333.  Two independent readouts, same
direction.  IV2 re-rooting is closed ON THAT POOL; the PHD2 slot pool
is untested for any encoder but SigLIP.

### 0.3 The spatial decision layer is equally stuck

- C/M/E objective comparison (same head, same data, 5 configs x 3
  seeds): best arm E(beta=0.03) 0.54045 vs B3 baseline 0.53108; delta
  +0.0094 [-0.002, +0.015] — CI crosses zero, gate (+0.015) failed.
  Objective choice does not move the decision layer.
- VLM_SFT (Qwen2.5-VL-7B, LoRA r8 language q/v, single-image forward +
  grad-accum 8, 4 epochs, full train240): best 0.52411 vs B3 baseline
  0.53108.  A 7B VLM reading the image and choosing among 36 candidates
  LOSES to a cached-feature head.  Permutation control 2.4x chance,
  but picks concentrate on mid-range IDs (top-5 all in 14-18 of 36).
- S-REREAD (round 7): actually observing the crop adds −0.0046 vs a
  capacity control.  Seeing is not helping.

### 0.4 Diagnostics are now a daily-5 resource (delivered, awaiting upload)

D_N (kept frames shifted +10% of n; coverage unchanged), D_X (x + 5%W),
D_Y (y + 5%H) on the frozen champion: synthetic identifiability PASS
(ΔF1 −0.10 temporal, ΔIoU −0.163 spatial).  Upload and read-back give
per-domain transfer slopes = Δofficial / Δlocal.

## Q1 - Diagnose the position-prior ceiling (the central question)

On our pool, EVERY learned head converges to the train-fitted per-slot
positive rate.  Diagnose WHY, and what has broken such ceilings
elsewhere.  Cover:

1. Is this the SLICING PROTOCOL's own structure (65% GIF-anchored
   starts in PHD2_FRAG_V1; mixed-only filtering; slot guard windows) —
   i.e., the position prior is the protocol signal, and content signal
   is thin ON THIS SLICING?  Name published analogs where a slicing
   protocol's positional structure dominated learnable content signal,
   and how authors measured it (name + year + measured number).
2. Is it the LABEL construction (PHD2 selections are creator-annotated
   GIF intervals, not dense temporal labels; mixed-only filtering
   discards 67% of fragments; mixed-only selection is itself position-
   correlated)?
3. Is it the METRIC (keep-0.80 F1 = 2·hit/(k+n_gt): a per-fragment
   top-80% keep makes within-fragment ranking nearly irrelevant — the
   top-80% threshold is lenient; is the metric's ceiling insensitive to
   ranking quality beyond the position prior?)
4. slicing vs labels vs metric — propose DISCRIMINATING EXPERIMENTS
   we can run locally (each: mechanism, cost, 48h probe, failure
   condition).  Examples we will run anyway: re-slice with
   label-agnostic (anchor-neutral) starts and re-measure slotprior vs
   learned-head gap; compute the metric's ranking-sensitivity directly
   (F1 at keep 0.80 vs 0.60 vs 0.50 curves for the same head).
   Propose others.

## Q2 - Temporal-highlight SOTA, named and measured

Give the current SOTA numbers for temporal highlight detection /
temporal action proposal / saliency-based summarization on TVSum,
SumMe, and adjacent benchmarks (name the TVSum "top-5 frames" f-tune
protocol papers that DOCUMENTED shuffle/position baselines scoring
high, with numbers).  Where does a keep-0.80 binary temporal F1 of
0.6625 on a GIF-interval pool sit relative to published
highlight-detection numbers?  Is our ceiling "low", "normal", or an
artifact of the GIF-interval label type?

## Q3 - The encoder swap: which encoder+layer on the temporal pool can beat the prior?

Feature redefinition is now the main line (operator decision).  Pool =
PHD2_FRAG_V1 (5,886 frags x 8 slots = 47,088 frames to encode; ~2.5h on
3 cards at 1.72 crops/s/card; Parity PASS for InternVideo2-1B and
Qwen2.5-VL-7B visual tower; InternVideo2-2?  no — reach 6B option:
InternVideo2-6B available via hf-mirror?  confirm).  For each of:
InternVideo2-1B stage1/stage2/mid-layers, Qwen2.5-VL-7B visual tower
(what layer?  its own paper's probing results?), InternVideo2-6B,
VideoMAE/VideoMAEv2 (weights on hand), VideoMAEv2-Giant (1.4B? fit?),
and any 2025-2026 video encoder you rank above these — rank candidates
for "features whose learned head beats a train-fitted position prior",
with named measured evidence.  Which LAYER (last / second-to-last /
multi-layer concat) has named evidence, for VIDEO temporal-localization
tasks, of carrying MORE temporal-localization content signal than the
last layer?  Which pooling of slots→feature-per-slot keeps temporal
order (mean-pool over time kills it)?
Which candidate maximizes P(beat slotprior) per NPU-hour?

## Q4 - RL directions that can plausibly break the prior

Full-information exact policy gradient on the 36-candidate table does
not move the decision layer (C/M/E, measured).  So "RL on the same
features+labels" is closed.  Which RL designs are NOT closed?

1. Frame-level / interval-level action spaces (not 36-candidate
   selection): GRPO on temporal interval proposals (Time-R1, GRPO,
   NeurIPS 2025; VideoTG-R1, ACM MM 2026) — can their action/reward
   design work when labels are GIF-interval creator annotations with
   heavy position structure?  What would the reward be (official
   arithmetic F1 directly, exact-DP differentiable already measured as
   converging to the prior)?
2. RetargetVid-style preference/crop RL on the spatial line?  only if
   it can beat 0.53108 (B3) + 0.02 gate — what does the literature say
   about RL helping when content signal is thin (RL amplifies whatever
   signal exists — if content signal is thin, RL amplifies position)?
3. Process-reward models (PRM) on temporal segmentation: named cases
   where PRM beat outcome-reward RL on dense temporal tasks?
4. Propose RL experiments that are CLOSED under our constraints (we
   will NOT run them) so we do not relitigate.

## Q5 - Distillation: the 32B teacher route

Qwen3-VL-32B predictions on the official 174-video set exist on disk
(qwen_official_174/predictions.jsonl; never platform-scored as a
package).  Officially scoring the teacher is now FREE in quota terms
(daily-5).  Design the teacher route:

1. Submit the teacher package (≤9B rule: 32B is >9B — can we submit
   the teacher at all?  The rule says total inference components ≤9B;
   a 32B teacher used ONLY for training-time supervision and never at
   inference: legal?  we believe yes (training-time only); argue the
   compliance reading, cite the rule text.
2. If the teacher is platform-scored on 174: how do we use a
   teacher-official vs student-official PAIR to calibrate distillation
   value BEFORE distilling (named prior art: teacher-official-scored
   first, then distill — who did this and what did they measure?)
3. Distillation target: slot-level soft targets from the teacher on
   OUR pool (teacher scores 5,886 fragments x 8 slots?  cost: 32B on
   910A batch transformers — measure?  the teacher has never run on
   NPU), vs trajectory/distillosses at other granularities.  Which
   granularity has named evidence of beating hard labels under
   position-heavy labels (name + measured numbers)?
4. Where does distillation help when content signal is thin (distill
   "what to look at" vs "what to score" — attention/feature distill vs
   logit distill: named comparisons with numbers)?

## Q6 - What breaks a position prior elsewhere

Find named, measured cases where a learned model beat a strong
position/content-independent prior on a temporal task with
protocol-structured slicing, and the technique that did it.  Also the
negative cases (papers where position priors REMAIN unbroken — e.g.,
TVSum shuffle baselines, GIF-interval style creator annotations).
Give both lists with numbers.  breaking a position prior with a
STRONGER ENCODER: named cases or counterexamples.

## Q7 - The metric is a suspect: keep-0.80 F1 sensitivity

Our gate metric 2·hit/(k+n_gt) at keep 0.80: with k = 0.8n and hit ≤
min(k, G): what is the metric's theoretical sensitivity to within-top-k
ORDERING vs pure coverage?  Derive the ceiling structure analytically.
If the metric's ranking-sensitivity is low, a learned head CANNOT show
its content signal THROUGH this metric even if features improve — the
metric itself may be the ceiling, not the features.  Derive: for a head
whose ranking differs from slotprior only inside the top-k, how much F1
can move?  Propose a cheap local experiment (metric sensitivity audit)
that discriminates "features cannot" from "metric cannot".

Q7 might kill the whole position-prior diagnosis: if the metric is
insensitive, Q1's answer is "the metric, not the features".  We will
run the audit regardless of your answer; your derivation tells us what
to expect.

## Q8 - Allocate the next 3 days

Operator constraints: 6x910A (48GB, fp16, eager), daily-5 quota,
teacher-only 32B available training-time, champion 34.95, leader 45,
gap 10.05.  Everything above must survive the standing rules (official
media never in training/selection; parity gate for new stacks; every
proposal: mechanism + cost + 48h probe + failure condition).

Given Q1-Q7, allocate: which encoder features get extracted first
(Q3), which diagnostic packages upload tomorrow (D_N/D_X/D_Y), whether
the teacher package goes up tomorrow (Q5), which RL/other probes run
(Q4) — a day-by-day plan with expected information value per day.

## Output contract

For every question: named evidence (name + year + measured number) for
each claim; mark estimates; "not sure" where knowledge ends.  End with:
(a) ranked next-3-day allocation with expected information value; (b)
the single most likely root cause of the position-prior ceiling (one
paragraph, committed); (c) the single experiment with the highest
information-per-cost ratio, committed.  All our scores are RAW; no
conversions.  Say "not sure" freely; we verify everything locally
before spending.
