# Prompt for GPT-6-PRO, round 6: two dead feature lines, one structural gap, and where 10.05 points can still come from

Style: ASD-STE100.  Short sentences.  Active voice.  Present tense.  Mark
every estimate.  Answer order per question: direct answer, mechanism,
cheapest deciding experiment, failure condition.

## 0. What happened since round 5 (all verified against running code)

### 0.1 Score units (correction adopted)

All scores are RAW, no size penalty.  Our champion V11_VTREPLAY = 34.95.
Leader intelligence = 45.  Real gap = 10.05 points.  All earlier
"platform vs raw" duality and the 63-point ceiling were conversions and
are withdrawn.

### 0.2 The champion verdict (P0-CONTENT, preregistered)

The VTREPLAY head is not blind (real−zeros +0.085~0.104, both CIs
positive) but its content contribution is VIDEO-level: swapping another
video's features erases it (+0.003/+0.004, confirm CI includes 0).  The
pure position prior TIES the champion.  Saturated exactdp is a pure
position prior (all ablation deltas 0.0).  A6: removing the anchored
slicing structure costs the champion −0.108 keep-F1, the same as the
prior (−0.116).  CONSEQUENCE ALREADY TAKEN: no more loss/calibration
tuning on the champion.

### 0.3 The spatial line: three probes, three negatives

| probe | result |
|---|---|
| S0 oracle audit | exploitable gap CONFIRMED: locked-out pool oracle−head = 0.247 [0.227, 0.268]; 70.2% frames gap ≥ 0.05; head−center only +0.071 |
| A5 objective audit | B3 IS ALREADY a candidate-utility head (huber on annotator-mean IoU) - the "retrain the scorer" route is a duplicate |
| S1 SEGMENTS=4 (2305→5377 features, same seed/config) | NO gain: confirmation head −0.010, oracle gap NOT narrowed (0.257 vs 0.247) |
| S4 score-smoothing TTA | NO gain: all variants ≤ 0 on the locked-out pool (top2 CI negative) |

The residual 0.247 gap is STRUCTURAL.  The surviving candidates are
candidate-geometry redesign, a stronger backbone, or label supervision
breadth.  Each moves the oracle itself, so each needs a re-measured gap.

### 0.4 The native line: contract rebuilt, content signal tested, not found

The native VideoMAEv2-B pipeline was rebuilt on the corrected contract
(stream-time_base seek, nearest-PTS, normalization; contract hash
ac3b657f; 256 dev sources; 46.5k windows; 0 hard errors).  The
preregistered P/O/S/L matrix ran (action-level labels over FULL source
sequences; lr scan picked 1e-4):

| arm | dev AP | dev F1 |
|---|---|---|
| O (ordered native) | 0.1703 | 0.1824 |
| P (position-only) | 0.1624 | 0.1849 |
| L (1 Hz sparse) | 0.1751 | 0.1807 |

O−P = +0.0079, source-cluster CI [−0.0123, +0.0305] - CROSSES ZERO.
O−L = −0.0048, crosses zero.  Under the preregistered rule the native
content signal DID NOT convert.  Two caveats we are acting on: (a) an
1800-step extension run (O/P × 3 seeds) is in flight to exclude
under-training; (b) the task is the FULL-SOURCE action sequence, whose
position prior itself only reaches AP 0.16 - the pooled 8-slot contract
remains the better-scoped task, and its own ceiling (0.6625) is a
position-prior ceiling per P0-CONTENT.

Also this cycle: the first parity run caught the NEW-STACK Siglip2
positional-embedding bug (all-zero pos embeds on torch_npu 2.9 / CANN
9.0.0 - silent wrong values; fixed by CPU-side pos-embed resize; parity
now eps 0.0132, 0/32 masks differ, F identical).  Every NPU forward now
requires the patch + a parity re-run after any stack change.

### 0.5 Assets and cadence

Fresh confirm set: 500 sources frozen (sha f43281bf), slicing protocol +
read discipline declared; NOT touched.  Anchor/Neutral stress slices:
built (920+920).  Native dev build: 256 sources.  6 NPU cards idle
(physical 2-7; 0-1 are another team's).  Platform return ~3 days.
Submission slots: 3 planned this round, ALL still unspent (nothing
passed a gate).

## 1. Questions

### Q1 - Where can 10.05 points physically come from now?

Both mined seams returned negatives: temporal content (pooled AND native
contracts) does not beat position priors; spatial scorer improvements
(features, smoothing) do not move the 0.247 gap.  Decompose the
remaining space: (a) candidate geometry redesign - what geometry would
raise the oracle itself, and how do we estimate that WITHOUT oracle
circularity?; (b) backbone upgrade inside the size tier - which encoder,
replacing which input block, at what expected gap change; (c) supervision
breadth (the y-axis has no reliable GT); (d) anything we have not
modeled (denominator/keep structure, per-video adaptivity, calibration -
all previously excluded, say if any should be re-opened and why).  Name
the single attribution with the best evidence-per-week ratio.

### Q2 - Candidate geometry redesign (the S0 gap's remaining owner)

129 equally spaced legal max-windows.  Propose 2-3 concrete alternative
geometries (e.g. content-adaptive candidate generation, multi-scale
pyramids, aspect-preserving sub-windows), each with: expected oracle
movement (estimate), the risk of just widening the gap, and the cheapest
offline read that predicts official gain without a submission.  Define
the go/no-go gate that a new geometry must pass BEFORE training a new
scorer (oracle delta on locked-out sources with re-measured head gap).

### Q3 - Backbone upgrade decision (size-tier constrained)

The vision tower is SigLIP2 NaFlex 86M (also the spatial head's feature
source).  Candidates: higher token resolution, DINOv2-class features,
InternVideo2-1B temporal features (fits the 500M tier?), or training a
lightweight temporal adapter on frozen features.  For each: expected
effect on (i) the 0.247 spatial gap and (ii) the dead temporal content
lines - and be explicit if the answer is "none, the features are not the
binding constraint".  Define the cheapest per-candidate offline probe.

### Q4 - Given both content lines are dead, is there ANY training-side move left?

Training-side debias (Anchor/Neutral re-windowing, position debias
augmentation) attacks the confirmed slicing-protocol dependence - but it
de biases a position prior, which cannot raise F above the prior ceiling
on neutral slicing.  Is there a formulation where training-side debias
PLUS the fresh confirm set produces a submission-worthy candidate, or is
the honest read "no candidate this round; bank the evidence"?  If a
candidate exists, define its exact local gate.

### Q5 - Submission strategy with an empty candidate shelf

3 slots unspent, no gated candidate.  Options: (a) hold all slots; (b)
one K70 keep-frac diagnostic (previously conditional on NPU downtime that
did not happen); (c) a "mechanism probe" package that changes a
non-scored aspect to measure platform behavior.  Which (if any) buys
information worth a slot, given the leader is at 45 and second place is
untested?  State the exact preregistered read for whatever you propose.

### Q6 - Attack the evidence chain (fourth round)

Fresh surfaces: (a) the P0-CONTENT xperm control permutes features
across videos WITHIN the dev pool - does cross-video permutation fully
decouple "video-level appearance" from "within-video position", or could
a video-similarity confound explain the residual +0.003?  (b) the L-matrix
action labels project GT intervals onto 1-second actions whose centers
come from tubelet timing - quantify the label noise this injects versus
the 8-slot contract; (c) the native dev build trained on 128 sources
whose anchor-sliced positives were ALSO the training pool of the pooled
line - is the L-matrix read contaminated by construction?  (d) anything
else that would not survive hostile review.

### Q7 - 72 NPU-hour plan for the week after

Cards idle: 6.  Programs on the table: candidate-geometry redesign
(Q2), backbone probes (Q3), fresh-confirm native features (gated, built
only if a T-gate candidate exists), augmentation experiments (gated on a
live content line, currently closed).  Give the exact allocation, what
runs on CPU in parallel, and the single measurement that would reorder
everything.

## 2. Output format

Answer Q1 first.  Direct answer, mechanism, cheapest deciding
experiment, failure condition per question.  Mark estimates.  Do not
treat any number from the anchored dev/confirm pools as official-domain
evidence.  Keep ASD-STE100.
