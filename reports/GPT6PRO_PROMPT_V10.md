# Prompt for GPT-6-PRO, round 10: the learnable-signal census — finding alpha over the position prior, and measuring the policy headroom

Style: ASD-STE100.  Short sentences.  Active voice.  Present tense.
This prompt is a DEEP-RESEARCH assignment.  Search your knowledge to the
2026-10 edge; go name-by-name; say "not sure" where your knowledge ends.
Every frontier claim needs: paper/project NAME + year + what it MEASURED.
A claim without a name is empty.  Mark every estimate.  All our scores are
RAW; no conversions of any kind.

## 0. The reframe (operator decision, 2026-10-06)

We stop asking "which optimizer / which loss / which RL algorithm is
stronger".  Rounds 6-9 measured, four independent ways, that no learned head
on our pool beats a train-fitted position prior.  Our blocking points are
now exactly two:

1. We have NOT PROVEN that a content signal strong enough to beat the
   position prior exists on our pool, in ANY modality we have touched.
2. We have NOT PROVEN that the reachable policy space is large enough for
   any such signal to convert into official F1.

So this round is a SIGNAL-FINDING assignment, not an algorithm assignment.
Operator analogy: quant factor research.  The position prior is beta.
Everything else is a candidate alpha.  We want a CENSUS of candidate signal
families, each priced for (a) residual predictive power, (b) orthogonality
to the prior, (c) extraction cost on our stack, (d) convertibility into the
deployment metric.  Literature first; then papers' own future-work sections;
then invention (tier 3: most papers are silent on a signal because they
never tried it, not because it fails - silence is weak evidence, and we
must say so).

## 0.1 The two governing experiments (your round-9 answers, adopted)

1. After a representation swap: does any head STABLY beat slotprior?
   (R9 gate: +0.007 vs champion AND vs macro-optimal prior, paired
   source-cluster CI lower > 0.  This run is executing tonight.)
2. After finite-action enumeration: can the oracle gap be DECOMPOSED into
   "pure hindsight" vs "decision gap predictable from legal inputs"?
   If either fails, RL stays frozen.  If both pass and the action space
   grows into "decide where to look -> observe -> decide what to keep /
   crop", you will support the RL main line.

## 0.2 Verified facts since round 9 (local, 2026-10-06 night)

Metric-space audit (exhaustive, actual K, frozen pool, champion scores):

- Actual K = 6 (f1_at_keep 0.80 on 8 slots rounds to 6; real keep 0.75;
  keep 0.70 maps to the same 6 - the historical "0.70 vs 0.80 curve" was
  one budget drawn twice).
- K=6: head 0.6588 / marginal prior 0.6625 / macro-optimal prior 0.6625 /
  exhaustive oracle 0.7309 / random-budget expectation 0.5561.
  Head-minus-prior paired CI [-0.0064, -0.0011] - the champion is
  significantly WORSE than the prior at the deployment point, by a hair.
- 88.5% of eval fragments: the champion's top-6 SET equals the prior's
  top-6 set.  Mean boundary swaps per fragment: 0.13.  Within-top-K
  reordering contributes exactly 0 to F1 - only boundary swaps count.
- Oracle headroom above the prior: +0.068.  NOT saturation - the metric
  has room; the signal is what is missing.
- K=2: head 0.4459 > prior 0.4398, CI [-0.010, +0.021] crosses zero.
  The head has SOME content signal; the loose budget swallows it.

Pool decomposition: 5,886 index frags -> mixed-only 3,917 frags / 1,963
sources (33.5% dropped; our earlier "67%" was a transcription error).
Eval G histogram: 1:71, 2:374, 3:521, 4:408, 5:247, 6:198, 7:135.
`_r` twin fragments share the same frozen time axis -> identical features
by construction (pool-structure fact, relevant for any per-frag analysis).

A1/A2 audits are DONE (2026-10-07, CPU, frozen pool):

- A1 label swap: under THREE label definitions - guard (frozen),
  centre-point, occupancy>0.5 - the prior/champion gap barely moves:
  champ-minus-prior = -0.0036 / -0.0024 / -0.0023, and absolute F1 falls
  (0.6625 / 0.5497 / 0.5549 for the prior).  The LABEL DEFINITION is not
  what creates the prior platform; it rescales everything uniformly.
- A2 score surgery on the frozen champion, eval F1 at K=6 / AP:
  real 0.6588 / 0.7463; temporal-permuted 0.5548 / 0.6016 (== the
  random-budget expectation 0.5561 - permutation destroys exactly the
  set information); CONSTANT score (mean or zero vector) 0.6625 / 0.7101.
  Reading: a content-FREE constant score equals slotprior (its top-6 IS
  the prior mask on this pool); the champion is +0.036 AP ABOVE the
  constant (real full-ranking content signal) yet -0.0037 F1 BELOW it
  (the signal does not land on the boundary, and its 0.13 swaps/fragment
  have negative net utility).  The cleanest "AP has signal, F1 has none"
  separation we have ever produced.

Cheap-scalar screen (the FIRST census probe, executed before this prompt):
audio track exists on 96.7% of manifest sources, but ALL 11 hand-crafted
per-slot scalars are NULL - 5 visual statistics (frame-diff energy,
Laplacian sharpness, brightness, histogram chi2, saturation-degenerate) and
6 audio scalars (RMS mean/std/max/flux, silence ratio, cheer contrast):
slot-level point-biserial |corr| < 0.05 for every column (n = 31k slots),
and the 2-parameter likelihood grid sets the signal weight to 0 for all.
Scope: the PIXEL-STATISTIC and AUDIO-SCALAR layer carries no signal for
GIF-style selections; any audio/visual cheap-feature family must jump to
LEARNED embeddings (PANNs/BEATs/CLAP-class) to be testable at all.  Treat
Q3/Q4's "cheapest probe" designs accordingly - the probe already ran and
killed the scalar tier.

## 0.3 The signal families we have ALREADY excluded, with the exact scope of the exclusion

| family | status | exact scope of the negative evidence |
|---|---|---|
| SigLIP mean-pooled slot features + TCN head (mse/exactdp) | CLOSED on this pool | four independent measurements converge to the prior |
| IV2-1B features on the QVH window pool | CLOSED on THAT pool | old SigLIP pipeline wins by +0.056 Spearman (CI [-0.001,+0.110]); P/O/L readout agrees |
| IV2-1B / Stage2 / Qwen visual tower on the PHD2 slot pool | OPEN - running tonight | results in 0.4; PHD2 slot pool has never seen any encoder but SigLIP until now |
| full-info exact policy gradient on the 36-candidate table (C/M/E) | CLOSED | best arm +0.0094 [-0.002, +0.015], gate failed |
| VLM choosing among 36 crops (7B, LoRA, SFT) | CLOSED under that recipe | 0.52411 < B3 0.53108; picks concentrate mid-range IDs |
| re-reading the crop (S-REREAD) | CLOSED | -0.0046 vs capacity control |
| k_size / keep-rate re-selection as an official-score lever | mostly spent | V8 ladder: +1.94 official raw from stripping the LM (k 0.90->0.95); per-video adaptive keep is NOT yet measured - see Q7 |

## 0.4 Overnight results (2026-10-07; all on the frozen PHD2 slot pool)

slot_head.json - IV2-1B **Stage1** on the PHD2 slot pool, 3 readouts x
{mse, exactdp} x 3 seeds, judging arm = exactdp (pre-set rule), dual gate
(+0.007 vs champion AND vs macro-prior, source-cluster CI lower > 0):

| readout | exactdp mean | delta vs champ (CI) | delta vs prior (CI) | gate |
|---|---|---|---|---|
| mean768 (clip_proj path) | 0.6557 | -0.0031 [-0.0061, -0.0001] | -0.0068 [-0.0084, -0.0053] | FAIL |
| m1408_l (last-layer mean) | 0.6515 | -0.0073 [-0.0112, -0.0036] | -0.0109 [-0.0136, -0.0085] | FAIL |
| m1408_m5 (layer-35 mean) | 0.6472 | -0.0116 [-0.0151, -0.0082] | -0.0153 [-0.0180, -0.0126] | FAIL |

Readings: (a) EVERY Stage1 readout is significantly WORSE than the SigLIP
champion - consistent with the QVH-pool negative; the CLIP-aligned SigLIP
feature stays the best representation we have for creator-choice labels.
(b) The InternVideo2 paper's "minus-5 beats last" (THUMOS14) REVERSES on
our GIF pool (last > -5 here) - layer-transfer from action-localization
does not survive the label-type change.  (c) exactdp no longer converges
to slotprior on the new features (0.6557 < 0.6625) - the new features are
not even Position-info-equivalent, they are strictly poorer here.

slot_head_s2.json - IV2-1B **Stage2** (4-frame control), same protocol:
ALL three readouts FAIL, WORSE than Stage1 - mean768 0.6435 (-0.0153 vs
champion), m1408_l 0.6416 (-0.0172), m1408_m5 0.6471 (-0.0117), CIs all
negative.  The video-text pretraining objective (Stage2) does not help
over video-only (Stage1) on this pool either.

slot_head_qwen.json - Qwen2.5-VL-7B visual tower, static 8-anchor:
{FILL IF ARRIVED - last arm of the night}

a1a2_audit.json + signal_screen.json: filled in 0.2 - label definition
excluded; AP/F1 separation measured; all 11 cheap scalars null.

## 0.5 Standing constraints (unchanged)

6x Ascend 910A (48GB, fp16, eager); daily-5 submissions; inference stack
<= 9B total params (32B teacher allowed TRAINING-TIME only, its outputs on
official media stay quarantined from training and selection); official test
media never in training/pseudo-labelling/selection; no new human labels;
single training run < 12h.  The official inference input is the VIDEO FILE
ONLY (frames + its own audio track).  Metadata outside the media file
(title, channel, captions published alongside) is NOT available at
inference - any signal family that implicitly needs it must be flagged as
non-deployable, or reduced to what is readable INSIDE the file (OCR of
burned-in text, the audio track, EXIF-like container data if present).

## Q1 - Formalize the factor framework and the residual readout

We will run every signal family through ONE preregistered protocol, on the
frozen manifest (3,917 frags, eval_sources_50 split, actual K=6):

- Baseline: the marginal slot prior p_s fitted on train.
- Family signal x_i (per fragment; vector or scalar).
- Model A: logistic(p_s).  Model B: logistic(p_s + g(x_i)).  Readouts:
  likelihood increment on eval (source-cluster CI), F1 at K=6 delta with
  CI, boundary-swap rate, and the swap NET UTILITY (incoming positives -
  outgoing positives).
- Alpha qualifies only if: CI lower > 0 on BOTH the likelihood increment
  and the F1 delta, computed on a confirm split the family never touched
  during development.
- Multiple-testing: we will screen ~10 families.  Set the per-family
  significance bar so the family-wise error stays <= 10% (Bonferroni or
  Benjamini-Hochberg - which, and why, for correlated families?).

Questions for you: 1) Is a logistic-residual screen the right FIRST filter
for alpha on binary labels with heavy position structure, or is there a
named better protocol (name + year + what it measured)?  2) What is the
MINIMUM effect size (in likelihood increment and in boundary swaps per
fragment) that can convert into +0.007 macro F1 given the K=6 oracle
structure - derive it.  3) Where does this protocol break (label noise,
G-mixing, small eval)?

## Q2 - The signal-family census (the core ask)

For EACH family: (a) mechanism - why it should predict slotprior residuals
for GIF-style creator highlight selections; (b) named evidence with
numbers (which paper measured what on which benchmark); (c) extraction
cost on our stack (910A NPU, fp16 eager; ffmpeg/librosa on 192-core CPU);
(d) expected effect size vs the +0.068 oracle headroom (mark as estimate);
(e) the cheapest falsification probe we can run in <4 CPU-hours or
<2 NPU-hours; (f) orthogonality argument vs the position prior.

Families (add your own; strike any you can kill with named evidence):

1. AUDIO (see Q3).
2. Burned-in TEXT: OCR keywords (goal/win/wow/killed/record), scoreboard
   changes, UI changes, subtitle lines (see Q4).
3. ASR of the commentary track (see Q4).
4. CHEAP MOTION: frame-difference energy, optical-flow magnitude,
   global camera motion, shot-boundary density - no learned encoder.
5. TOKEN-LEVEL / REGION features from video encoders - who moves, who
   interacts, small-region actions - instead of pooled vectors (see Q6).
6. CROSS-FRAME IDENTITY: who is tracked, who is followed by the camera,
   active speaker, subject persistence across slots (see Q5).
7. FRAME-STATISTICS SPIKES: exposure/color shifts, cut detection,
   compression-artifact changes, sharpness jumps - event-style detectors
   rather than smooth scorers.
8. ENCODER READOUT alternatives: intermediate layers, attention-pool
   tokens, temporal tokens instead of global mean (tonight's run covers
   last/-5 of Stage1/Stage2 + Qwen tower 32/24 - what ELSE is worth a
   readout that we have not cached?).
9. Semantic-event prompting: a <=9B VLM scanning longer context as a
   SIGNAL DETECTOR (event classifier per second), NOT as a crop chooser
   (that recipe failed) and NOT as a distillation target yet.
10. Anything in the video-summarization/highlight literature 2024-2026
    that predicts human "highlight" judgments from features OTHER than
    global appearance embeddings - name the feature, the paper, the number.

## Q3 - Audio deep-dive

Our media are YouTube rips; the audio track is INSIDE the file, hence legal
inference input.  We have never used it.  Plan the cheapest probe and the
scaling path:

1. Per-second scalars: RMS, spectral flux, onset rate, zero-crossing;
   speech-activity ratio; embedding novelty.  Cost to extract for 1,963
   sources (~2-6 min each) on CPU?
2. Pretrained audio encoders: PANNs / BEATs / CLAP / Perceiver-based
   audio models - which weights run on Ascend fp16 (conv/transformer ops),
   which have known NPU-port issues, and what per-second embedding dim?
3. Named evidence: audio-driven highlight detection numbers - crowd-noise
   proxies in sports highlight detection, audio-visual summarization,
   movie highlight via sound events.  Name + year + benchmark + number.
4. Cross-modal coincidence: does (audio onset) x (visual change) jointly
   beat either alone?  Named evidence for coincidence features.
5. Falsification probe: on 64 public sources, audio-only logistic screen
   vs slotprior residual - design it so a null result kills the family.

## Q4 - Text-in-frame and speech deep-dive

1. OCR stack on 910A: PaddleOCR / EasyOCR / trocr - fp16/eager status,
   fps on our cards, and whether a keyword dictionary over OCR text
   (goal/win/record/killed/wow + score-line change detection) is the
   cheapest alpha in this whole census.  Any named prior art for
   scoreboard-change -> highlight mapping with numbers?
2. ASR: whisper-small/base on 910A NPU - known op coverage?  CPU fallback
   throughput for 1,963 sources x ~4 min?  Commentary sentiment/surprise
   as a per-second scalar: named evidence (sports commentary excitement
   detection - name + number).
3. Which fraction of GIF-source videos even HAS speech vs music vs crowd?
   (If you know PHD2's source composition or GIF-selection culture,
   say what is documented; mark inference otherwise.)

## Q5 - Cross-frame identity / tracking for the SPATIAL line

The spatial decision layer is stuck at B3 0.53108 (objective choice did
not move it; a 7B VLM crop-chooser lost to a cached-feature head).  Your
hypothesis from round 9: crop preference may depend on WHO the subject is
across frames - the followed actor, the active speaker, the persistence of
the object - not on understanding one frame better.

1. Cheap stack: face detection + embedding clustering (who recurs),
   IoU/ByteTrack-style association (who persists), active-speaker
   (lip-region motion x audio activity), camera-follow detection
   (subject-centred framing stability).  Cost on 910A + CPU for the
   spatial training set (~240 train videos + 80 dev)?
2. Named evidence: cinematography / highlighting rules learned
   (name + year + number); identity-aware cropping or smart-crop papers;
   tracking-assisted highlight cropping.  What was MEASURED?
3. Probe design: does subject-persistence predict the CHAMPION's own crop
   residual (where the champion's crop disagrees with GT, is a persistent
   subject usually there)?  This turns Q5 into a residual screen on the
   SPATIAL line with zero new labelling.

## Q6 - Un-pooling: token-level / region-level features

Round-9 evidence pending, but design ahead: we can cache (8 slots x 256
tokens x 1280-d fp16) = ~5MB/frag, 20GB total.  Heads that read token MAPS
(top-k region aggregation, attention pooling, region-motion deltas between
adjacent slots) vs the pooled 768 mean.

1. Named evidence that region-level / token-level readouts beat global
   mean pooling for TEMPORAL grounding or highlight (name + number).
2. The failure mode to avoid: more head capacity on the same thin signal
   converges to the prior again (we measured this four ways).  What in
   the literature distinguishes "region features add signal" from
   "bigger head memorizes position"?  Propose the control.
3. Budget: is a token-map head worth it BEFORE the pooled readouts report,
   or only conditional on tonight's pooled results?  Give the decision rule.

## Q7 - Measuring the policy headroom (the second blocking point)

Formalize and design the local measurement:

1. Decompose the exhaustive oracle gap (+0.068 at K=6) into:
   (a) hindsight-only mass - swaps that NO legal input could predict;
   (b) decision gap - swaps correlated with ANY legal signal.
   Concretely: fit the best legal predictor of oracle-vs-head swaps;
   the predicted share is the decision gap.  What statistics guard against
   overfitting the swap predictor on 1,954 eval frags (cross-source CV)?
2. Action-space enlargement, legal levers only: per-video keep count
   (k is OUR choice per video in the submission format; the official
   arithmetic 2H/(k+n_gt) rewards k near G - has per-video adaptive k been
   optimized POST-k_size-penalty-removal?  we believe NO - confirm the
   lever and estimate its official-score range from the G histogram logic);
   multi-crop per frame; variable-length temporal intervals.
   For each: is it inside the official contract (predictions rows = frame
   + bboxes), and what is the cheapest local measurement of headroom?
3. What does "enough headroom" mean numerically before RL is unfrozen?
   Give the arithmetic: if the decision gap is D and the signal strength
   is s, the F1 ceiling is ... (derive).

## Q8 - Backtest discipline for signal mining (the quant analogy made rigorous)

We will screen ~10 families on the same dev pool.  The danger: the 11th
"signal" is noise that passed by luck; sequential peeking inflates alpha.

1. Named methodologies: how do factor-research and ML-competition
   protocols handle family-wise error, staged confirmation, and holdout
   decay?  (The Ladder, ICML 2015 is our baseline citation - what else?)
2. Design our two-stage protocol concretely: SCREEN (train-only residual
   fits, coarse) -> CONFIRM (frozen, preregistered, one shot per family
   on the confirm split).  What sample size (fragments/sources) does the
   confirm stage need to resolve +0.007 F1 at 80% power with source
   clustering?  Derive from our cluster structure (50 eval sources,
   1,954 frags).
3. When is a family killed vs kept ambiguous?  Write the rule we should
   preregister.

## Q9 - Tier-3: your own inventions

Signals ABSENT from (or nearly absent in) the literature that you would
still test, each with: mechanism, cheapest falsification, and why the
silence of the literature is weak evidence here.  Aim for 5-10 candidates
across audio/text/motion/identity/frame-statistics.  Mark every one as
UNVERIFIED INVENTION.  Examples to seed (accept, mutate, or reject):
audio silence-to-cheer contrast spike; OCR text-density timeline spikes;
sharpness/JPEG-blockiness jump at directed cuts; global-motion snap
(whip-pan) detector; face-count timeline; frame-color-histogram distance
spikes; scoreboard region change detector; comment energy slope BEFORE
the selection (anticipation) vs during.

## Q10 - Updated allocation

Given tonight's encoder results (0.4) and this census: a day-by-day plan
for the next 3 days that (a) keeps the R9 encoder gate line running, (b)
schedules the signal-family probes by information-per-cost, (c) states
which families must be killed before their modality gets more compute.
Budget: 72 NPU-hours total for 3 days, CPU essentially free at night.

## Output contract

For every question: named evidence (name + year + measured number) for
each claim; mark estimates; "not sure" where knowledge ends.  End with:
(a) the ranked signal-family leaderboard by P(alpha qualifies on confirm)
per NPU-hour; (b) the single most likely reason no content signal has
surfaced so far, committed in one paragraph; (c) the single cheapest
experiment that could surface a NEW qualifying alpha within 48 hours,
committed.  All our scores are RAW; no conversions.  Say "not sure"
freely; we verify everything locally before spending.
