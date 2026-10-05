# Prompt for GPT-6-PRO, round 6: track the frontier, compare us to it, then beat it

Style: ASD-STE100.  Short sentences.  Active voice.  Present tense.  Mark
every estimate.  Cite by NAME for every frontier claim: paper/project
name, year, and what it measured.  A frontier claim without a name is
treated as empty.  Search your knowledge to its 2026 edge; say "not
sure" where your knowledge ends.

## 0. Our position, verified this cycle

### 0.1 Scores

All scores are RAW, no size penalty.  Champion V11_VTREPLAY = 34.95.
Leader intelligence = 45.  Gap = 10.05 points.  Task: pick highlight
frames from raw video AND predict a spatial crop per kept frame
(video-macro F over joint hit IoU; no sliding window means crop IoU is
full-frame for 159 of 426 official videos).

### 0.2 Two content lines closed this cycle (all preregistered)

- TEMPORAL (pooled): the deployed head's content contribution is
  VIDEO-level only (+0.003 under cross-video permutation); the pure
  position prior TIES it; anchored slicing removal costs it −0.108, the
  same as the prior.  Dev ceiling 0.6625 = position-prior ceiling.
- TEMPORAL (native, REBUILT on a verified contract - stream-time_base
  seek, nearest-PTS, normalization, hash-versioned; 256 sources, 46.5k
  windows, 0 hard errors): preregistered P/O/L matrix at action level.
  O−P = +0.0079, CI [−0.0123, +0.0305] - crosses zero.  L ≥ O.  The
  full-sequence position prior itself only reaches AP 0.16 there.
  1800-step extension in flight.
- SPATIAL: S0 audit shows an exploitable 0.247 scorer gap (oracle 0.782
  vs head 0.535 on locked-out LIVE sources; 70% of frames ≥ 0.05).
  But: S1 richer per-candidate features (2305→5377, band pooling) = NO
  gain; S4 score-smoothing TTA = NO gain; the scorer is ALREADY a
  candidate-utility head (129 legal max-windows, huber on annotator-mean
  IoU).  Three probes, three negatives.

### 0.3 What we are, in one paragraph

A 86M SigLIP2 NaFlex tower extracts per-frame features.  A 246K TCN
consumes 8x1s pooled features and emits per-second scores that are - per
our own audits - a position prior with weak video-level content.  A
1.5M B3 head picks 1 of 129 fixed max-windows per frame from
window/outside pooling features.  No motion features (1 fps grid killed
them; native rebuild found no exploitable signal).  No temporal detector
architecture.  No test-time compute beyond trivial smoothing (failed).
Size tier: ≤ 500M params; we ship ~178 MB.

## 1. The questions - each one has a frontier obligation

For EVERY question: (1) name the 2024-2026 frontier methods with
references; (2) place us on that map honestly; (3) propose the transfer
that could BEAT the relevant frontier piece, or prove it cannot apply;
(4) give the cheapest local experiment that would show the transfer
working within 48 hours on our 192-core CPU + 6 idle NPU (910B) box.

### Q1 - The frontier map for this exact task family (HIGHEST PRIORITY)

Survey the 2024-2026 state of the art across: video highlight detection
(QVHighlights-style), temporal moment grounding/localization,
video summarization, sports/cinematic highlight spotting, smart-crop /
aesthetic retargeting, and LLM/VLM-based temporal reasoning.  For each
family: the current best method by name, its architecture idea, its
data scale, and its mechanism for making CONTENT beat POSITION priors
(this is precisely our confirmed failure mode).  Then: which single
idea, transferred to our joint highlight+crop metric, has the highest
evidence-per-week ratio?  Kill the obvious answer first: "just use a
bigger VLM" - we have 500M params and a 178 MB stack; what fits?

### Q2 - How do frontier methods kill the position-prior shortcut?

Our audits say our heads learn WHERE highlights sit in a clipped
fragment, not WHAT is happening.  Frontier moment-detection methods
(DETR-style set prediction, dual-stream text-video alignment,
LLM-guided temporal reasoning, etc.) - name the mechanisms they use
that structurally prevent position shortcuts: set-based matching?
query learnability?  long-context encoders?  ordering-free losses?
For each mechanism: would it survive OUR data reality (PHD2 GIF
fragments, 8-14 s, anchor-sliced positives, weak labels), and what is
the minimal re-implementation?  Define the 48-hour probe.

### Q3 - The spatial 0.247 gap: does the frontier already own this?

Smart crop / retargeting / saliency-crop work 2024-2026: name the best
published approaches to choosing WHERE to crop inside a sliding band
(saliency-guided, attention-guided, aesthetic-scoring, segmentation-
guided, diffusion-guided).  Our head scores 129 fixed max-windows from
window-mean features - is the frontier's lesson "richer candidate
descriptions", "continuous regression instead of discrete argmax",
"detection-style dense heads", or something else?  Name the transfer
with the best IoU-per-week, and the locked-out evaluation that gates it.

### Q4 - Motion is the one unmined modality - what does the frontier say?

Our native rebuild found no exploitable content signal at 1 s actions.
Frontier video encoders (InternVideo2-class, V-JEPA 2/2.1,
video-native LLMs): do any of them demonstrate highlight/moment gains
specifically attributable to MOTION at small scale?  What sampling and
what head do they use?  Is our 2 s window / 16 frames / tubelet-mean
protocol a known-dead shape?  If motion is genuinely dead for GIF
highlights, say so with named evidence.  If not, name the smallest
frontier-faithful probe we have not run.

### Q5 - Test-time compute: what does the 2026 frontier actually do?

Our smoothing TTA failed.  The frontier has moved to heavier test-time
compute: multiprobe ensembling, query decomposition, iterative
refinement, self-consistency over VLM captions, inverse-render/saliency
hints.  Which test-time methods apply at our size tier with zero
trained-parameter change, and what gain do they claim on tasks shaped
like ours?  Name the one test-time method worth 48 hours.

### Q6 - Anti-shortcut learning: the frontier of debiasing

Our strongest confirmed structure is a shortcut (slicing-protocol
position prior).  2024-2026 debiasing work: counterfactual attribution,
shortcut-contrastive training, environment-invariant risk, attribution-
based reweighting.  Which named method would actually move a model off
a position prior when the labels themselves are position-correlated
(our Neutral slicing showed the labels ARE the prior)?  Be honest if
the answer is "none - fix the data protocol instead", and specify the
data-protocol fix with named precedent.

### Q7 - Attack our cycle's evidence chain (fourth round)

New surfaces: (a) xperm swaps features across videos WITHIN one pool -
could video-similarity confounds explain the residual +0.003?  (b) the
L-matrix projects GT intervals onto 1 s actions - quantified label noise
vs the 8-slot contract?  (c) the native dev sources overlap the pooled
line's training ancestors - contamination or not?  (d) the parity gate
found a silent NPU bug - what OTHER silent-wrong-value risks should we
gate before trusting any future number?

### Q8 - Converge: the one-week plan that could reach a frontier-grade candidate

Given everything above: the exact 72 NPU-hour allocation, the CPU
parallel work, the gate each line must pass, and the single measurement
that would reorder everything.  Also: with 3 unspent submission slots
and no gated candidate, is there a frontier-mechanism probe package
that buys more information than holding the slots?

## 2. Output format

Answer Q1 first.  Every frontier claim: NAME + year + measured claim.
Every proposal: mechanism, transfer cost, 48-hour probe, failure
condition.  Mark estimates.  Do not treat anchored-pool numbers as
official-domain evidence.  Keep ASD-STE100.
