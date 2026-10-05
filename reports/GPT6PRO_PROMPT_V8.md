# Prompt for GPT-6-PRO, round 8: the annotator-disagreement problem and the RL toolchain

Style: ASD-STE100.  Short sentences.  Active voice.  Present tense.
This prompt is a DEEP-RESEARCH assignment: the operator judged the
round-7 survey too shallow.  Search your knowledge to its 2026-10 edge;
go name-by-name; say "not sure" where your knowledge ends.  Every
frontier claim needs: paper/project NAME + year + what it MEASURED.
A claim without a name is empty.  Mark every estimate.  All our scores
are RAW; no conversions of any kind.

## 0. New facts since round 7 (all verified, 2026-10-05 evening)

### 0.1 The S-REREAD verdict landed (four arms, dev80, video-macro IoU)

B3 zero-shot 0.5289; same-capacity control +0.0066 [0.002, 0.012];
**actual-crop observation arm -0.0046 [-0.013, +0.004]** (worse than
the capacity control); zero-train cosine fusion -0.163.  In the
Siglip2-feature + residual-head setting, seeing the crop buys nothing.
The LIVE-YT-VC paper's own adjacent-annotator IoU ~0.50 ~= our B3.
The 0.247 oracle gap may be largely mean-annotator vs single-annotator
disagreement.  Ledger: reports/r6/claim_ledger.md.

### 0.2 Operator decisions you must design around

- **Human labeling is REJECTED.**  No new annotations, ever.  Every
  method must run on existing public labels.
- Per-annotator raw labels are NOT in the public LIVE-YT-VC release
  (verified: one aggregated box per frame, sparse json + box csv;
  upstream README lists nothing more; "++ coming soon").
- Weights on hand: InternVideo2-1B stage1(+k710)/stage2 LOCAL;
  Qwen2.5-VL-7B/3B downloading now (hf-mirror; direct HF is blocked
  from our region).  Qwen3-VL-32B stays a teacher only (>9B).
- Deployment: 6 x Ascend **910A** 48GB (pinned by lspci 19e5:d801;
  CANN 9.0.0), fp16, ~34GB host RAM per NPU process (max 12
  processes), single training run < 12h.  vLLM pairing unresolved;
  transformers eager is the working path.

### 0.3 What we pre-registered overnight (your predecessors' moves - critique them)

DECISION_LAYER line (all CPU, on a cached 2,560-row crop table with
B3 scores, 36-candidate shortlists, and annotator-mean IoU labels):
- D0: ceiling of ANY monotone decision rule over frozen B3 scores.
- D1: argmax over a monotone calibrated score g(B3).
- D2: per-candidate IoU quantile head + quantile-tau selection.
- D3: group-relative (small-model GRPO-shaped) listwise readout.
Named evidence that motivated it: Ahmed et al., ICCV 2015
(Optimizing Expected IoU with Candidate-Constrained Decision-Theoretic
Models - mean regression is NOT the IoU-optimal decision under label
distributions); "Diverging Preferences" (arXiv:2410.14632 - >75% of
disagreement is individual predilection, not noise; Mean-Var
distributional rewards keep accuracy while identifying diverging
pairs); Time-R1 (GRPO, NeurIPS 2025); VideoTG-R1 (curriculum RL on
boundary-noise annotations, ACM MM 2026); CROP (DPO crop preferences).
**You must criticize this design**: what does it miss, where will it
fail, what would you cut or add before we spend CPU on it?

### 0.4 Standing rules

Official test media never enter training or selection; inference on
official videos only after a frozen recipe, argued per the round-7
compliance table.  New stacks pass the CPU/NPU parity gate before any
number is trusted.  3 submission slots unspent; champion 34.95;
leader 45; gap 10.05.  Any proposal: mechanism, cost, 48h probe,
failure condition.

## Q1 - The mean-vs-single-annotator disagreement: what actually recovers it?

### The question

Our spatial labels are one aggregate draw per frame; the official GT
is one (unknown-protocol) draw per frame; B3 already scores 0.529 ~=
"another annotator".  Which published methods recover REAL IoU from
annotator-disagreement regimes WITHOUT new annotations, and by how
much (measured, named)?

### Cover at minimum (each with named evidence and measured numbers)

1. **Decision-theoretic selection**: beyond Ahmed ICCV 2015 - who has
   QUANTIFIED the gap between "predict the mean" and "maximize
   expected IoU" on real annotation distributions?  Segmentation with
   multiple raters (STAPLE and descendants), jitter-consistent box
   evaluation, any IoU-metric-aware training.
2. **Distributional label learning**: quantile/distributional heads,
   Mean-Var reward models, evidential deep learning, label
   distributions - measured gains under annotator noise, and their
   failure modes.
3. **Annotator-as-latent-variable**: Dawid-Skene lineage, deep
   crowdsourcing models, annotator embeddings - do any work when each
   item has exactly ONE label (our case) rather than many?
4. **Style recovery without annotator IDs**: can per-video or
   per-dataset annotator-style statistics (box tightness priors,
   center bias, size bias) be estimated from single-label data and
   exploited at inference?  Named precedent or honest "none found".
5. **The protocol fork**: our official GT is EITHER a single-annotator
   draw OR a multi-annotator mean (unknown).  Derive the optimal
   strategy under each hypothesis and what training-side evidence
   could distinguish them without official labels.  Note: 159/426
   official videos have NO sliding window at all (spatial is void
   there) - the spatial metric mass is ~267 videos.

### Obligations

Rank the recoverable mechanisms by expected IoU gain on OUR table
(B3 0.529, oracle 0.78, adjacent-annotator ~0.50).  For each:
mechanism, cost, 48h probe, failure condition.  If your honest answer
is "nothing in the literature recovers X", say so with the search you
ran - an empty result set is a finding.

## Q2 - Which RL method(s) do we prepare, given OUR structure?

### Our structure (be specific against it)

- Reward is COMPUTABLE from existing labels (IoU vs known boxes): this
  is RLVR - no learned reward model, no reward hacking surface from a
  net, but label NOISE directly enters the reward.
- Action space: discrete legal candidates (129 or a 36-shortlist) per
  keyframe; natural GROUP structure within a frame (GRPO's native
  shape); or continuous boxes if we drop the candidate grid.
- Data: 2,560-row spatial table; the PHD2-family temporal pools
  (larger); 12h/run budget; 6x910A; LoRA feasibility unmeasured.
- Base models: frozen 7B verifier (post Probe 0) OR our 88M stack
  (small-model policy gradient).

### Compare, by name, with measured evidence

GRPO / RLOO / REINFORCE with leave-one-out baselines; DPO family
(IPO, KTO, cDPO robust variants, Semi-DPO, DPO-PRO NeurIPS 2025);
online/iterative DPO; best-of-n distillation; QR/distributional
RL for noisy rewards; curriculum RL (VideoTG-R1's boundary-reflection
agent - does its filtering idea transfer to single-draw boxes?).
Answer specifically:
1. Which method has the best THEORY-to-our-case fit when the reward
   noise IS the annotation noise (variance of group-relative
   advantages under high within-group reward variance)?
2. Which are safe at 88M scale (tiny policy) vs which need 7B?
3. What sample sizes did the named papers actually train with (we are
   at 2,560 rows - Time-R1's 2.5K is the closest anchor; who trained
   on less and won?)
4. Rank: the ONE method to prep first, and the cheapest 48h probe
   that produces a real go/no-go.

## Q3 - Reward function: do we need to re-tune it, and how?

Our implicit reward so far is supervised huber on annotator-mean IoU.
If we go RL, the reward design opens up.  Analyze AT LEAST:

1. **reward = IoU(pred-candidate, GT-box)** - under the hypothesis
   "official GT is a single draw from the same annotator distribution
   as our training labels", this reward's EXPECTATION is exactly the
   deployment objective.  Verify or refute this claim formally: is
   the RL objective provably better-aligned than supervised mean
   regression under label noise, or are the two asymptotically
   equivalent (and the difference is only finite-sample)?
2. **Quantile/bet rewards**: selecting at quantile-tau of a predicted
   IoU distribution (tau as a tuned decision parameter) - what do the
   distributional-RL and quantile-regression literatures say about
   tau selection under distribution shift between train and official
   annotators?
3. **Temporal-line rewards**: the same disagreement story applies to
   our temporal labels (F1 gap ~28 points lives in frame-level
   ranking).  Candidate reward forms: per-frame binary, interval
   soft-IoU, rank-based, expected-F as reward.  CONSTRAINT: we have a
   standing no-reopen rule on "unconstrained expected-F recipe
   changes" from an earlier cycle - does using expected-F as an RL
   REWARD constitute reopening it?  Argue both sides; the operator
   decides.
4. **Legality/format rewards** for candidate-ID outputs, and
   anti-hacking: what does the model gain by exploiting within-frame
   candidate statistics rather than content?  (We have documented
   position-prior failures on three data constructions - a reward
   that rewards "picking slot 2" is a real failure mode here.  Name
   the guard: baseline subtraction within frame?  content-control
   arms?  both?)
5. Deliver a concrete reward recipe table: components, weights, the
   hyperparameters to sweep, and the ablation that isolates each.

## Q4 - Are SFT and on-policy self-distillation exhausted for us?  What switches in?

### Clarify the terms first

- "SFT" for us to date: supervised regression training of tiny heads
  (huber on mean IoU; TCN temporal heads).  NO SFT has ever been run
  on a 7B.  On the tiny side we have capacity controls, feature
  controls, four-arm patterns - argue whether the SUPERVISED tiny-head
  direction is exhausted or just under-iterated.
- "OPSD" - on-policy self-distillation (e.g., "Frame Differential
  On-Policy Self-Distillation for Video", 2026-10 preprint, and the
  GKD / on-policy KD lineage).  Investigate this family on its 2026
  edge: what it measures, where it beat plain SFT and RL, and
  whether it has ANY evidence under label noise.

### Answer

1. For the 7B verifier path: rank SFT-only / SFT-then-RL / RL-only /
   SFT+OPSD mixes against our structure (small data, noisy labels,
   computable reward).  Named evidence for each ranking position.
2. Does distilling from our Qwen3-VL-32B teacher help AT ALL when the
   teacher was never trained on our labels (it only zero-shots them)?
   Is there a measured case of teacher-distillation beating direct
   RL from compute-heavy to compute-light models under noisy labels?
3. If SFT and OPSD are NOT exhausted for us, name the single cheapest
   next experiment per line (tiny-head SFT variant / 7B SFT / OPSD).
   If they ARE exhausted, say what switches in first and why.
4. End with the ONE question we should have asked you and did not
   (same convention as round 7).

## Output format

Answer Q1, Q2, Q3, Q4 in order.  Every frontier claim: NAME + year +
measured claim.  Every proposal: mechanism, transfer cost, 48h probe,
failure condition.  Mark estimates.  Criticize our DECISION_LAYER
preregistration explicitly (0.3).  Keep ASD-STE100.  Do not propose
anything that touches official test data in training.
