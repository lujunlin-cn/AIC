# V10 exploration report: the ranking wall and one unconfirmed signal

Date: 2026-10-03
Scope: all V10 experiments after the GPT-6-PRO review.
Style: ASD-STE100 Simplified Technical English.

## 1. Purpose

This report records the V10 exploration results. It gives the numbers, the
test method, and the state of each line. It marks one unconfirmed signal. It
does not predict scores.

## 2. Score mechanism (measured facts)

The platform computes:

    Score = 100 x mean over videos (F_v) x k_size
    F_v   = 2 x sum of IoU on frames that hit GT / (N_pred + N_gt)

Measured facts:

- GT marks only highlight frames. CORRECTION (2026-10-03, disk audit): the
  earlier value p = 0.33 was a platform-score back-fit, not a pool
  measurement, and is retired. The V11 review flagged that it contradicted
  keep-all 0.6237. Measured pools: eval-side mixed fragments (1,954) have
  mean per-fragment positive rate 0.4722 and keep-all binary temporal F1
  0.6178 (self-consistent, g(0.4722) = 0.641 above the macro value by a
  plausible Jensen gap); all rows including all-positive and all-negative
  fragments (5,886) give 0.3812 and 0.4782. The oracle-decomposition pool
  (all index rows, per-source aggregation) is a third total. These pools
  are not interchangeable, and none of them is the official 426-video set.
- Annotator census (upstream training.csv keeps the user field that our
  aggregated train.json drops): 83.3 percent of the 2,262 feature-pool
  sources are single-user; 60.7 percent of (source, user) pairs contribute
  more than one interval; user-pair disagreement on a 1 s grid averages
  0.127 (p90 0.263). Consequence: the interval-level vote label counted
  intervals, not users, for most of the mass - the three-protocol tie in
  section 5 never tested user-level aggregation.
- 159 of 426 official videos have no slide. The predicted box is the full
  frame there. The spatial IoU is 1 by construction. The mean spatial IoU is
  about 0.76, not 0.62.
- A kept frame that is not in GT makes the denominator larger and gives no
  numerator. This is the reason a keep mask can raise F.
- The platform did not change the score when the declared model size changed
  from 862 MB to 178 MB. The scores were 34.75 and 34.73. The k_size lever
  gave no measured gain. The vision-only line is closed.

## 3. Oracle decomposition

Method: freeze the ranking. For each video, give the oracle the right to pick
the best keep count k. Compute the binary temporal F1 with the official
arithmetic. Source-level split. 1,963 sources.

| Policy                          | Binary temporal F1 |
|---------------------------------|--------------------|
| Keep all frames                 | 0.6237             |
| Fixed keep 0.80 (present)       | 0.6648             |
| Oracle A: best k per video      | 0.7804             |
| Perfect ranking (upper bound)   | 1.0                |

Reading: the decision layer holds +0.116. The ranking layer holds +0.220.
Both layers hold real space.

## 4. Decision layer tests (both closed)

### 4.1 Route A: calibration, then expected-F rule

Method: Platt and isotonic calibration on a source-disjoint fold. The fold
includes the all-positive and all-negative fragments. The decision rule picks
k* = argmax of the predicted F. Result on 94 fragments, 37 sources:

| Variant   | Temporal F1 | Prevalence MAE | Brier |
|-----------|-------------|----------------|-------|
| Raw       | 0.5406      | 0.314          | 0.280 |
| Platt     | 0.5683      | 0.246          | 0.227 |
| Isotonic  | 0.5677      | 0.247          | 0.227 |
| Fixed 0.8 | 0.5689      | -              | -     |
| Oracle A  | 0.8605      | -              | -     |

Verdict: calibration makes the probabilities better. The rule still does not
beat a fixed keep. The root cause is a representation-level selection bias.
The TCN never saw all-negative videos during ranking training. Verdict:
closed.

### 4.2 Route B: direct budget utility

Method: cross-fitted ridge. 18 score-distribution statistics per video. One
model per keep rate. 5 folds. Result on 1,963 sources:

| Policy                  | Temporal F1 |
|-------------------------|-------------|
| Best fixed keep (0.7)   | 0.6687      |
| Budget utility          | 0.6694      |
| Oracle A                | 0.7804      |

Delta +0.0007. CI95 [-0.0009, +0.0023]. The CI contains zero. The chosen
rates vary across videos, but the gain is zero. Verdict: closed.

Note: an earlier adaptive-keep test gave +0.075. That test read the true
positive count. It was a leak. The leak-free version failed. Do not quote the
+0.075.

## 5. Ranking layer tests (ten variants, one band)

Method: one matrix. Source-level 50/50 split. 984 held-out sources. Fixed
2500 steps. No checkpoint selection. Labels are the GIF-interval union.

| Variant                       | Held-out AP |
|-------------------------------|-------------|
| mean pooling, multi-scale     | 0.6959      |
| mean pooling, single          | 0.6903      |
| topk pooling, single          | 0.6885      |
| mean+delta, multi-scale       | 0.6830      |
| attn pooling, multi-scale     | 0.6859      |
| mean+attn, multi-scale        | 0.6830      |
| mean+delta, single            | 0.6826      |
| attn pooling, single          | 0.6783      |
| mean+attn, single             | 0.6778      |
| topk pooling, multi-scale     | 0.6771      |

More tests against the same band:

| Test                              | Held-out AP |
|-----------------------------------|-------------|
| Label protocol: vote-fraction     | 0.6918      |
| Label protocol: any-strict        | 0.6788      |
| Training set + 1,565 all-negative | 0.6843      |
| Training set + 404 all-positive   | 0.6870      |

Verdict: fourteen variants. One band: 0.677 to 0.696. No variant escapes.
Pooling, structure, motion features, label protocol, and training breadth do
not move the ranking. CORRECTION (section 6.5): the band's absolute level is
a recipe effect, not a representation ceiling - the same frozen tower under
the probe recipe reaches 0.7462. The relative ties inside the matrix stand.

## 6. The LoRA confirmation round (V11 stage-1 data)

The V11 review required: the full trajectory, a second seed, a same-budget
frozen-tower control, exposure counts, the evaluation manifest with its
hash, and checkpoint hashes. This section collects them. The operator's
warning stands: the step-12 point is probably noise.

### 6.1 Method and audit fields

LoRA r=8 on the Siglip2 attention projections; the temporal objective
backprops through the tower. Paired fields, all from disk:

| Field | Value |
|-------|-------|
| Sampler pool | 1,963 mixed training fragments (with replacement) |
| Exposure | 900 steps x 8 fragments = 7,200 instances; mean 3.67 per fragment; P(never sampled) = 0.025 |
| Mid-run eval | deterministic prefix: first 270 mixed of 1,954 held-out fragments |
| Full eval | all 1,954 mixed held-out fragments |
| Eval manifest SHA-256 | 24fae58a334bd0aed3eb398929cce39f15ce5a6616be15a27db3800738e312c6 |
| Seed-1 RNG note | ran before the --seed flag: numpy 20261003, torch UNSEEDED (audit defect, recorded) |
| Seed-2 | numpy and torch both 20261004 |
| Frozen control | same sampler, same steps, same evals, no adapters |
| Checkpoint SHA-256 | [PENDING: lora_probe.pt / lora_probe_seed2.pt / lora_probe_frozen.pt] |

### 6.2 Seed-1 trajectory (mid-run prefix, then the full set)

| Step | held-out AP | Scope |
|------|-------------|-------|
| 12 (smoke, unpaired TCN) | 0.7067 | full set, unpaired TCN - the operator's warning applied |
| 150 | 0.7048 | 270-fragment prefix |
| 300 | 0.6861 | prefix |
| 450 | 0.7011 | prefix |
| 600 | 0.6943 | prefix |
| 750 | 0.6943 | prefix |
| **900** | **0.7443** | **full 1,954 fragments** |

### 6.3 Same-budget frozen control and seed-2

| Step | frozen AP | seed-1 LoRA AP | delta |
|------|-----------|----------------|-------|
| 150 | 0.7067 | 0.7048 | -0.0019 |
| 300 | 0.7172 | 0.6861 | -0.0311 |
| 450 | 0.7209 | 0.7011 | -0.0198 |
| 600 | 0.7293 | 0.6943 | -0.0350 |
| 750 | 0.7269 | 0.6943 | -0.0326 |
| **900 full** | **0.7462** | **0.7443** | **-0.0019** |

Seed-2 (both RNGs pinned to 20261004) is still running; its prefix points
at steps 150 and 300 (0.7013, 0.6943) match seed-1's (0.7048, 0.6861)
within 0.008, with no collapse. The operator's decision rule ("if seed-2
slides early, seed-1 was an accident") did not trigger - and the frozen
control made the question moot: the LoRA-minus-frozen gap, not the seed
spread, is the measured effect.

### 6.4 Checkpoint hashes

| File | SHA-256 |
|------|---------|
| lora_probe.pt (seed-1 LoRA, 900 steps) | c55f82786e3415e2ea26522184f55228467af84ade5d5041b6e07ad07897d748 |
| lora_probe_frozen.pt (frozen control, 900 steps) | 5617b3c655b644946e3ff19e3f14cafe4c4fd53c543d0f62dc89381373c7310e |
| eval_fragment_manifest.json | 24fae58a334bd0aed3eb398929cce39f15ce5a6616be15a27db3800738e312c6 |

### 6.5 Findings, in descending order of importance

1. **The training recipe, not the tower, set the fourteen-variant wall.**
   The frozen control - same TCN, same pool, same split, no adapters -
   reaches AP 0.7462 on the full held-out set. The ten-variant matrix
   (2500 steps, batch 64, about 81 exposures per fragment) capped at
   0.696. The probe recipe (900 steps, batch 8, about 3.7 exposures per
   fragment) lifts the SAME frozen tower by +0.056. The wall at 0.677 to
   0.696 was a property of the large-batch long-schedule recipe, not of
   the Siglip2 representation. Candidate mechanisms: over-training on
   81 exposures versus 3.7, or optimization noise. Section 5's relative
   conclusions (pooling and structure variants tie under the sweep
   recipe) stand; its absolute ceiling does not.
2. **LoRA adds nothing.** Full-set delta LoRA - frozen = -0.0019, inside
   seed noise. The mid-run prefix showed -0.02 to -0.035, but the full-set
   endpoint erases it. The step-12 smoke signal (0.7067) is fully
   explained: the probe recipe alone crosses 0.70 (frozen at step 150
   reads 0.7067 on the prefix), with no tower update needed.
3. **The prefix is not the population.** The 270-fragment prefix reads
   about 0.02 to 0.03 BELOW the full set for the same model (frozen:
   0.7269 prefix at step 750, 0.7462 full at 900). Monitoring may keep the
   prefix; decisions may not cite it (V11 audit INVALID_EVAL_SET).
4. **The `escaped_band` field is void.** Both runs "escaped" 0.70; the
   field cannot separate them. The valid comparison is the paired
   LoRA-minus-frozen delta, which is zero.
5. **Verdicts.** Tower fine-tune (LoRA r=8, this objective): CLOSED as
   NO_PRACTICAL_GAIN(lora r=8, Siglip2 tower, MSE ranking objective, 900
   steps, full-set AP; paired delta -0.0019 [-noise, +noise]). The
   ranking-recipe line (batch x steps x exposure) is OPEN and is the new
   main line: it moved AP by +0.056 with zero architecture change. The
   V11 M01 video-native encoder line keeps priority for the next stage,
   but must run under the probe recipe, not the sweep recipe.
6. **Distance to the oracle.** With AP 0.746 on the held-out pool against
   the prefix-oracle 0.7804, the remaining ranking gap is about 0.034,
   and the decision layer gap (+0.116) is now the larger prize. The
   recipe finding must be re-tested against the decision layer before
   any platform extrapolation - none is made here.

### 6.4 Verdict rule (unchanged)

The signal survives only if the full seed-1 evaluation supports the
mid-run checks, the frozen control does not match the LoRA trajectory, and
seed-2 reproduces the escape. Otherwise the line closes as
NO_PRACTICAL_GAIN(lora r=8, this tower, this objective, 900 steps, AP) and
the video-native encoder line (V11 M01) takes priority.

## 7. State of all lines

| Line                       | State       | Evidence                         |
|----------------------------|-------------|----------------------------------|
| k_size / vision-only       | Closed      | 34.73 vs 34.75 on one prediction set |
| Keep-rate scan             | Closed      | Peak at 0.70 to 0.80; shipped     |
| PHD2 KD distillation       | Closed      | Local gain; platform flat         |
| QV auxiliary domain        | Closed      | Cross-domain AP 0.531 vs 0.695    |
| Decision layer, route A    | Closed      | Rule 0.568 vs fixed 0.569         |
| Decision layer, route B    | Closed      | +0.0007, CI contains zero         |
| Pooling x structure (sweep recipe) | Closed | Ten variants in one band; absolute level corrected in 6.5 |
| Label protocol             | Closed      | Three protocols in one band       |
| Training breadth           | Closed      | Boundary fragments do not help    |
| Tower fine-tune (LoRA)     | Closed      | Paired full-set delta -0.0019 vs frozen control |
| Ranking recipe (batch/steps/exposure) | OPEN, main line | Frozen tower AP 0.690 -> 0.746 under probe recipe |
| Video-native encoder (M01) | Queued      | V11 priority; must run under the probe recipe |

## 8. Rules for later reports

1. Report held-out, source-level numbers only. A training number is not a
   result.
2. A line is dead only after: three or more variants across the hypothesis
   class, one control, and a CI that contains zero. The control must match
   the recipe; a variant matrix under one recipe does not bound what
   another recipe reaches (section 6.5 finding 1 is the proof).
3. One changed factor per experiment.
4. Mark every proxy metric as a proxy. The constant-IoU F1 is a temporal
   proxy, not the platform F. AP is a ranking diagnostic, not the
   deployment metric.
5. Report negative results with the same detail as positive results.
6. Mid-run prefix numbers are monitoring only. No decision cites them
   (eval manifest sha 24fae58a...; prefix reads 0.02 to 0.03 below the
   full set).
7. A paired same-budget control runs before any "gain" claim. The smoke
   signal and the recipe effect were both unmasked by controls, not by
   more seeds.
