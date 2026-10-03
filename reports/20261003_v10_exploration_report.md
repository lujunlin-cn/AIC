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
not move the ranking. CORRECTION (section 6.5, finding 1): the band's level
versus the probe numbers is an evaluation-protocol difference, not a recipe
or representation effect - the shipped head also reads 0.7462 under the
probe protocol. The relative ties inside the matrix stand.

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

1. **The 0.626-to-0.746 gap was an evaluation-protocol artifact, not a
   model difference.** Scored under the probe protocol (same 1,954 mixed
   held-out fragments, fresh tower features), the SHIPPED head -
   tcn_s0.pt, the head inside the 34.73 package - reads AP 0.7462,
   identical to the probe frozen head. The shipped head's stored
   val_ap = 0.6263 came from an older, undocumented protocol. The probe
   recipe never improved ranking over the shipped head; the two heads tie
   on every PHD2 measure we have.
2. **Platform 33.85 (-0.88 versus the 34.73 parent) is therefore the
   deployment variance between two ranking-equivalent heads.** The new
   package changed only the temporal head; spatial boxes are byte-
   identical on kept frames; mask volume matches within 0.1 percent. Yet
   426 of 426 videos changed their kept-frame set (mask Jaccard 0.773).
   Two heads that AP cannot separate (+0.0000) differ by 0.88 platform
   points. Third directional failure, first negative one: KD gained
   locally and stayed flat, QV fused worse, and now a ranking-tied head
   swap moved the score DOWN.
3. **PHD2-domain ranking metrics have no predictive power for the
   platform direction.** Not imprecise - uninformative. Every local
   ranking number (AP, simulated keep F1, oracle distance) is blind to
   the sign of the platform delta. All prior closures and priorities that
   rested on those numbers keep their internal validity but lose their
   platform relevance.
4. **The `escaped_band` field, the step-12 signal, and the LoRA line are
   all fully explained with no tower effect.** LoRA - frozen at 900 full
   set: -0.0019. The probe protocol alone puts any trained head near
   0.746.
5. **Verdicts (revised).** Tower fine-tune (LoRA r=8): CLOSED,
   NO_PRACTICAL_GAIN, paired delta -0.0019. Ranking-recipe line: CLOSED,
   NO_PRACTICAL_GAIN - the apparent +0.056 was protocol confounding
   (finding 1). The probe-recipe head stays deployed nowhere; the
   platform-tested parent (34.73) remains the best package.
6. **The only trusted local signal is official-domain ground truth.** The
   174-video preliminary set carries official GT and is not part of the
   426-video semifinal drop. Every future package decision (head choice,
   keep rate, mask style) must first be measured as real F on the 174
   videos. Next action: score the two heads (tcn_s0 and probe frozen) on
   the 174 GT to confirm the 33.85-versus-34.73 ordering is reproducible
   locally - if yes, we finally own a trustworthy proxy.

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
| Pooling x structure        | Closed      | Ten variants in one band          |
| Label protocol             | Closed      | Three protocols in one band       |
| Training breadth           | Closed      | Boundary fragments do not help    |
| Tower fine-tune (LoRA)     | Closed      | Paired full-set delta -0.0019 vs frozen control |
| Ranking recipe             | Closed      | Probe-protocol artifact; shipped head also 0.7462 |
| PHD2 metrics as platform proxy | Closed  | 33.85 platform on a ranking-tied head swap; 3rd failure |
| 174 official-GT validation | OPEN, main line | Score both heads on 174 GT; verify the ordering |

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
   deployment metric. STRONGER (section 6.5, finding 3): PHD2-domain
   numbers do not predict the platform delta's sign. They may rank
   hypotheses inside PHD2 only; no package decision cites them.
5. Report negative results with the same detail as positive results.
6. Mid-run prefix numbers are monitoring only. No decision cites them
   (eval manifest sha 24fae58a...; prefix reads 0.02 to 0.03 below the
   full set).
7. A paired same-budget control runs before any "gain" claim. The smoke
   signal and the recipe effect were both unmasked by controls, not by
   more seeds.
8. Every package decision (head, keep rate, mask style) is scored first
   as real F on the 174-video preliminary GT. A candidate that has not
   passed the 174 gate does not get packaged.
