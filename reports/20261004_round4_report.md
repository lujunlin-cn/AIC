# Round-4 report: audit gate, the position-prior inversion, and the clean confirm set

Date: 2026-10-04.  Style: ASD-STE100.  Platform still down; both packages
(56693 VTREPLAY, 56694 EXACTDP) remain packaged and unscored.  This report
covers the round-4 cycle: the GPT-6-PRO audit gate, our verification of
every finding, the pooled-line inversion the diagnostics exposed, and the
frozen confirm set.

## 1. Audit gate (all findings verified before acceptance)

We verified every GPT-6-PRO finding against the running code before
accepting it.  The four runtime scripts on the 910A are md5-identical to
the repo snapshot, so the audit applies to what actually ran.

| Finding | Verification | Artifact |
|---|---|---|
| R1: 3-seed mean is +0.0271, not +0.0352 | confirmed; the CI quoted in day-2 was a single-seed bootstrap; a 3-sample seed-level t-interval crosses zero ([-0.0073, +0.0615]) | report correction, section 0 of day-2 report |
| R2: dp_bin call site mixes slot-unit G with frame-unit gains | confirmed: E[F] = 1.85 on the counterexample (mixture of F<=1 cannot exceed 1); dp_occ and pooled expected_f_binary are clean | round4_reward_audit.json |
| R3/R4: native seek units + missing normalization + existence-only cache skip | confirmed and MEASURED: seek offset scaled 33x-78x on all 8 audited sources, selected frames 24-1608 s from target (8-12 s fragments), 15/16 duplicate frames per window, 16/16 outside the fragment | round4_native_input_audit.json |
| provenance: running code = repo snapshot | md5 4/4 identical | commit c14860e |

Consequences: both native feature caches carry INVALID_README.json markers;
every native absolute number (the +0.0795 motion gap, dp_bin AP 0.7845,
occupancy arms, the 8-window collapse) is quarantined.  Both PACKAGES are
unaffected (VTREPLAY uses the SigLIP pipeline; EXACTDP's training objective
is slot-level self-consistent) - SUBMIT_AS_IS stands.

## 2. The position-prior inversion (the cycle's main finding)

Question raised by the audit: is the DP arm's near-zero seed variance
learning, or saturation?  Diagnostics on the frozen dev checkpoints
(round4_pooled_diagnostics.json):

1. The 3 DP seeds produce IDENTICAL top-0.80 masks on 99.97 percent of
   fragments, with 100 percent of logits beyond |8| - full saturation.
2. Content destruction: the DP head scores F1 = 0.6625 on real features,
   0.6625 on all-zero features, 0.6625 on time-shuffled features.  The head
   ignores visual content; its output is a fixed per-slot prior ordering.
   The MSE arm is similar (real 0.6354 vs zeros 0.6625 - content makes it
   WORSE on dev held-out).
3. Deconfound at lr 1e-4 (no saturation): MSE reaches 0.6622 mean and DP
   0.6625 - BOTH objectives converge to the SAME solution, matching the
   GPT-6-PRO prediction that equal top-K sets force equal F.

Reading: 0.6625 is the position-prior ceiling of this 8-slot mixed pool.
The celebrated +0.027 expected-F advantage was an artifact of MSE being
DEGRADED at lr 1e-3.  With healthy training, expected-F and MSE tie on F1
in-domain - the same tie the native line showed.  The expected-F line's
remaining differentiators are seed stability and AP, not the truncated
decision.  The EXACTDP package carries a position-prior head with weak
content; it is legal and its platform score still measures the whole
package against its parent, but mechanism attributions change accordingly.

Oracle context: fixed-budget action oracle 0.7309, fixed-ranking prefix
oracle 0.6995 on the same pool - a +0.037 gap exists for a perfect
re-ranking, but neither objective's head extracts it from these features.

## 3. The clean confirm set (frozen before any label was read)

500 NEW sources from the 10,837-source eligible pool (training.csv minus
the official testing family, minus tier0/tier2, minus every dev-exposed
source; labeled; media on disk; >=12 s).  seed 20261004,
sha256 d2695ae3...  (confirm_sources_500.json).  Known limit recorded:
ID-level exclusion only; perceptual near-duplicate audit is a listed
follow-up.

460 of 500 sources passed the fragmenting filter (90 s-2 h, fps<=30,
info.json) and were cut into 920 mixed/background fragments with the
unchanged v9 protocol (8-14 s, 8 x 1 s keyframes; 648 highlight-bearing,
272 background).  Frames extracted with the CORRECT decoder
(v9_phd2_extract: stream-time_base seek + sequential PTS match, 920/920
ok).  Features: same pooled contract as dev (valid-token grid mean, topk,
attn; fp32 CPU forward instead of fp16 NPU - noise-level difference,
recorded).

## 4. Confirm-set evaluation protocol (preregistered)

Two frozen candidate groups, one pass each, no tuning on this set:

- GROUP A (package lineage): expected_f_arm_{mse,exactdp}_s2026100{6,7,8}.pt
  - the exact checkpoints inside/nalike the EXACTDP package lineage.
- GROUP B (deconfound group, selected on DEV only): the lr1e-4 arms
  trained after the saturation finding.  Selection happened on dev; the
  confirm set did not influence any choice.  Recorded here BEFORE the
  confirm evaluation ran.

Metric: keep-0.80 binary-temporal F per fragment, paired per seed,
source-cluster bootstrap CI (10k resamples).  Decision rule (from the
round-4 plan): promote only if the seed-mean delta lower bound > 0 and
mean >= 0.007.

Results (585 mixed fragments, 389 sources evaluated):

- GROUP A - package lineage (the saturated lr1e-3 checkpoints):
  MSE F1 0.6225 mean (0.6093/0.6181/0.6402); DP F1 0.6530 (0.653 x3).
  Paired deltas +0.0437/+0.0349/+0.0129, ALL seed CIs above zero;
  source-cluster seed-mean CI95 [+0.0207, +0.0406].  PASSES the
  preregistered promote rule (lower bound > 0, mean >= 0.007).
- GROUP B - deconfound (healthy lr1e-4 checkpoints): seed-mean paired
  delta CI95 [-0.0030, +0.0010] - a CLEAN TIE, as section 2 predicted.

Interpretation.  On fresh sources the saturated DP head (a position-prior
ordering with weak content) transfers almost losslessly (0.6625 dev ->
0.6530 confirm) while the MSE content head degrades (0.6354 dev ->
0.6225 confirm).  The +0.031 GROUP A win is REAL and clean, but its
mechanism is prior robustness under distribution shift, NOT the
expected-F objective: GROUP B proves the objective itself adds nothing
at the cut.  The expected-F line closes under the preregistered rule;
its surviving assets are (a) the EXACTDP package - now carrying the only
candidate with a clean in-domain confirm (+0.031, CI above zero), even
though its mechanism label changes, and (b) the fixed-budget conditional
DP (Q3.6) as a DIFFERENT hypothesis, plus the +0.037 oracle gap.

## 4b. Consequence for the two pending packages

- EXACTDP (56694) evidence status UPGRADED: from "dev-exposed, CI
  crosses zero" to "clean confirm, +0.031 mean, CI95 [+0.021, +0.041],
  3/3 seeds" - with the honest mechanism note above.  SUBMIT_AS_IS.
- VTREPLAY (56693) unchanged: A0 mechanism fix, awaiting platform score.
- Q1 tree reading is unchanged in structure, but the D-vs-parent
  comparison now has a positive in-domain prior, raising the
  probability weight of the D-H and D-M branches (estimate, not
  calibrated probability).

## 4c. OFFICIAL SCORES (2026-10-04, platform back)

SCORE-UNIT CORRECTION (2026-10-04, operator): 34.95 and 34.74 are RAW
scores with NO size penalty applied.  The "raw F = 38.83 / 38.60 =
platform / 0.90" figures published earlier in this section were an
illegal score/k_size conversion (the rule is on record since 10-02 and
was violated again).  All scores below are read as returned.

| Package | official score (raw, no penalty) | vs parent 34.75 | verdict |
|---|---:|---:|---|
| V11_VTREPLAY | **34.95** | +0.20 | NEW HIGH; A0 recovery = 1.10/0.90 = 122 percent (recovered 1.10 of the 0.90-point defect drop, same-unit ratio) - the valid-token contract fix is PLATFORM-CONFIRMED and its head beats the parent's |
| V11_EXACTDP | 34.74 | -0.01 | tie with parent, noise level - the confirm-set +0.031 did NOT convert to the official domain |

Q1 tree cell V=H, D=H.  Actions per the tree: do NOT scale the DP line
(D-C ceiling is below noise; no matched-MSE package C will be spent), do
NOT submit a third diagnostic package now.  The confirm-set verdict and
the platform agree on the mechanism: the prior head adds nothing on the
official domain either.

Operator intelligence: the current first place reports 45 (raw, no
penalty) vs our 34.95 (raw, no penalty) - a 10.05-point gap.  No
conversion arithmetic attaches to either number.  The in-domain
fixed-ranking prefix oracle F=0.6995 is a dev-pool simulated binary-
temporal F, NOT an official-domain ceiling; the earlier "63 platform"
figure was a double conversion and is withdrawn.  What stands: our dev
pooled AP is 0.74 and content-ablation shows the pooled head is mostly
position prior; spatial IoU has never been optimized.  The two unmined
seams left: spatial IoU quality (rho) and true temporal ranking quality
(native VideoMAE line rebuilt on the corrected extractor).
Keep-structure is NOT tuned on the official set (in-domain curve peaks
at 0.70-0.80 and falls at 0.50; blind cuts would be leaderboard
overfitting).

## 5. Environment incident and recovery

The 910A container's writable layer was reset (python3.11.15, CANN
toolkit, lfm_venv base all gone; /data intact).  Recovered: Miniconda
installed AS /usr/local/python3.11.15 (revives the venv), ML stack
installed into the conda base, torch_npu removed until CANN returns (it
breaks CPU-only transformers imports).  NPU is DOWN until CANN 9.0.0
aarch64 is reinstalled - needs the run package from hiascend.com (not on
public mirrors); every NPU task (native rebuild, M01) is queued behind
it.

## 6. What this changes

1. Q1 decision tree: both packages remain single-variable vs their
   parents, but the EXPECTED-F mechanism claim for EXACTDP is withdrawn;
   its score now measures "position-prior head, trained with expected-F
   objective" vs the parent's content head.
2. The confirm evaluation (section 4) is the only in-domain evidence
   still able to separate the objectives - and section 2 predicts a tie.
   If it ties cleanly on 500 fresh sources, the expected-F line closes
   under the preregistered rule and the line's residual value is the
   fixed-budget conditional DP (Q3.6) and the oracle gap.
3. The native line restarts ONLY after: CANN reinstall, corrected
   extractor (stream-time_base seek + official normalization +
   hash-versioned cache keys), and the fair motion control (ordered vs
   shuffled vs multi-frame static).

## 7. Budget actually used this cycle

Audit + confirm build + diagnostics: ~4 NPU-equivalent hours were NOT
used (platform NPU down); CPU heavy work ~3 h; the 72 NPU.h envelope is
intact with 28 h conditional reserve untouched.
