# DECISION_LAYER correction record (R8)

Date: 2026-10-05.  Style: ASD-STE100.  All scores RAW.
Input: GPT-6-PRO round-8 response + audit bundle, verified (see
reports/20261005_round8_plan.md).  This file records the formal
amendment of reports/r7/preregistration.yaml `DECISION_LAYER`.

## 1. What was wrong

### D0 ("ceiling of ANY monotone decision rule over frozen B3 scores") - VACUOUS

A common strictly monotone transform g over the per-frame score vector
preserves argmax:

    argmax_j g(s_j) = argmax_j s_j.

So the best "monotone decision rule over frozen scores" IS the frozen
argmax, except for ties.  As preregistered, D0 measures nothing beyond
B3 itself.  The isotonic-style ceiling idea fails for the same reason:
isotonic regression is monotone; re-ranking by it never changes the
selected candidate.

Worse, the prereg made D0 a router ("if D0 ceiling ~ B3, reduce 7B
content trials").  A monotone invariance holds for ALL inputs, so D0
cannot say anything about whether a STRONGER model (7B) gains new
information.  That routing logic is deleted.

### D1 ("argmax over monotone calibrated score g(B3)") - GAIN EXACTLY 0

Same identity.  A dev-fitted common monotone calibration changes no
argmax.  The preregistered gate (+0.015 vs B3) could only fire through
ties, numerical artifacts, or leakage (fitting a DIFFERENT g per
candidate position is not calibration - it is a new scorer).  D1 is
deleted as a training arm.  Tie-breaking must use a pre-fixed rule
that never reads GT; picking the best within a tie group by GT is
leakage.

### D2 ("per-candidate IoU quantile head + quantile-tau selection") - DEMOTED

Quantile-tau selection changes the RISK POSTURE, not the risk-neutral
objective.  There is no source-only rule that picks a universally
correct tau under an unknown official annotator protocol.  D2 survives
only as a secondary risk diagnostic (distribution mean vs single
quantile comparison).  It has no promotion role this round.

### D3 ("group-relative listwise readout, GRPO shape") - SUBSUMED

All 36 candidate rewards per frame are already computed.  A sampled
group-relative gradient estimates an expectation we can compute
EXACTLY.  Softmax policy with full reward vector r gives the exact
policy gradient p_j (r_j - r_bar), no critic, no sampling variance.
The E arm (full-information expected utility, optional KL to a frozen
reference policy) is the sampling-free superset of D3.  If E fails its
gate, a sampled variant of the same objective will not rescue it - it
only adds variance.

## 2. The deeper correction (affects our R6 reading, not just D0-D3)

Our R7 survey line "the head stops at fit-the-mean-then-argmax, and
mean regression is IoU-suboptimal (Ahmed ICCV 2015)" misfired in our
setting.  B3's labels are already candidate-utility means
u_j = mean_a IoU(b_j, Y_a) (v8_s_train_multidata.build_sample), and
argmax over the conditional mean utility IS the Bayes-optimal
risk-neutral decision.  The Ahmed gap applies to BOX-COORDINATE mean
regression, which we never did.

What actually deviates from the Bayes rule in our stack is the
TRAINING OBJECTIVE: B3 is trained with huber(delta=0.25) + 0.3 x
pairwise(best-vs-worst), not L2 on u.  The huber optimum solves
E[clip(u_hat - U, -delta, +delta) | x] = 0, which is not the
conditional mean; the pairwise term perturbs it further.  A synthetic
counterexample in the R8 bundle shows order inversion is possible
(A: 0.4 x 1 + 0.6 x 0 vs B: 0.3 constant; huber-optimal prediction
1/6 picks B, mean picks A).  This is tested, not assumed, on our
table: arms C/M/E (see reports/r8/preregistration.yaml).

Lineage note (verified 2026-10-05): the deployed B3 checkpoint IS
v8_s_train_multidata lineage - reports/r6/source_ledger.md line 21
and source_ancestry_audit.json line 10.  The code reading applies to
the deployed model.

## 3. Label-semantics corrections (record, not rerun)

- LIVE-YT-VC released labels: ONE box per frame, single human subject
  per frame (verified upstream README 2026-10-05; LIVE-YT-VC++ still
  "Coming soon").  Taxonomy entry: raw_single_box.  Our earlier
  wording "one aggregated box per frame" was wrong in semantics.
- The paper's ~0.50 IoU is between boxes on CONSECUTIVE frames
  (temporal-smoothness statistic; consecutive frames may be labeled by
  different subjects).  It is NOT same-frame inter-annotator
  agreement.  The R6 ledger line "adjacent-annotator IoU ~0.50" is
  corrected accordingly.  Consequence: the inference "B3 0.529 ~= human
  ceiling" LOSES its direct support.  The annotation-ceiling
  hypothesis stays open; Part B (estimation/decision error) is not
  yet bounded by that number.
- New anchor from the same paper (Table IV, flexible / non-height-
  preserving protocol - the protocol type our 9:16 task belongs to):
  STCAT 52.3 / CG-STVG 53.1 mIoU; from-scratch models collapse toward
  center bias.  B3 0.529 sits at published-model level on this
  dataset family.  Different splits - ballpark, not a bound.
- RetargetVid rows keep 6 per-frame annotators locally; u there is the
  6-annotator mean utility.  This is the only local multi-rater
  resource; it powers the protocol-sensitivity audit (A_PROTO).

## 4. Replacement program (amended DECISION_LAYER)

- D0 -> MONOTONE_CONTRACT unit test only (synthetic; already in the
  R8 bundle, 208/208 reproduced locally).  Never a router, never a
  ceiling for other models.
- D1 -> deleted.  Replaced by arm M (pure L2 conditional-utility
  regression) inside C/M/E.
- D2 -> secondary diagnostic only (no gate, no promotion).
- D3 -> replaced by arm E (full-information expected utility with
  optional KL; exact gradients).  RLOO/GRPO stay behind the
  enumeration-infeasibility condition, which our candidate table does
  not meet.
- A_PROTO (new, CPU): protocol-sensitivity audit on RetargetVid
  per-annotator labels - per-annotator mean utility vs coordinate-mean
  box utility vs leave-one-annotator-out utility.  Diagnostic; no
  synthetic annotators.

Gates for C/M/E and all successors: see reports/r8/preregistration.yaml.
