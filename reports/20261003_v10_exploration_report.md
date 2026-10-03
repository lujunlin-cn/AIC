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

- GT marks only highlight frames. The PHD2 positive rate is about 0.33. The
  earlier value 0.39 came from a pool with a 65 percent highlight-anchor bias.
  The corrected value changes the reachability math.
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
not move the ranking. The wall sits in the frozen Siglip2 representation, or
in the information content of the GIF-interval labels.

## 6. The LoRA probe (in progress)

Method: LoRA r=8 on the Siglip2 attention projections. Backprop the same
temporal objective through the tower. Same pool, same split, same head.

Smoke test: 12 steps. Held-out AP 0.7067 on the full 1,954 fragments. This is
above the band.

WARNING. This point is not evidence. Three reasons:

1. The paired TCN has 12 steps of training. The frozen baselines have 2500.
   The comparison is not fair.
2. One seed. One point. Seed noise is about 0.005 to 0.01.
3. The mechanism of the gain is not understood. A 12-step LoRA shift is small.

Confirmation protocol: the 900-step run is in progress. An escape counts only
if the AP at step 300 or later stays above 0.705 for two checks. Then a second
seed must repeat it. If the AP stays in the band, the signal was noise.

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
| Tower fine-tune (LoRA)     | Open        | Smoke 0.7067; unconfirmed         |
| Video-native encoder       | Not started | Next if LoRA fails                |

## 8. Rules for later reports

1. Report held-out, source-level numbers only. A training number is not a
   result.
2. A line is dead only after: three or more variants across the hypothesis
   class, one control, and a CI that contains zero.
3. One changed factor per experiment.
4. Mark every proxy metric as a proxy. The constant-IoU F1 is a temporal
   proxy, not the platform F.
5. Report negative results with the same detail as positive results.
