# Prompt for GPT-6-PRO, round 7: the 9B era - correlation, backbone, and what ports

Style: ASD-STE100.  Short sentences.  Active voice.  Present tense.  Mark
every estimate.  Cite by NAME for every frontier claim: paper/project
name, year, and what it measured.  A frontier claim without a name is
treated as empty.  Search your knowledge to its 2026 edge; say "not
sure" where your knowledge ends.

## 0. Rule change and verified position

### 0.1 The rule changed this round (operator-confirmed, 2026-10-05)

The official rules now: **NO size penalty.  The ONLY constraint is ≤9B
total inference parameters** (all deployed components sum).  The old
100-500M tier and its penalty are gone.  Score contract is unchanged:
all scores are RAW.  Our champion V11_VTREPLAY = 34.95.  Leader
intelligence = 45.  Gap = 10.05.  No score/parameter conversions - ever.

### 0.2 What this unlocks

Our current stack is ~88M (86M SigLIP2 NaFlex tower + 246K TCN + 1.5M B3
candidate-utility head).  Every 1-9B backbone that was previously
over-budget is now a legal candidate.  From your own round-6 map, these
re-enter: **InternVideo2-1B** (the SG-DETR encoder; QVHighlights MR mAP
58.8 with 150K InterVid-MR pretraining), **Qwen2.5-VL-7B** (Time-R1's
base; CROP's base), **InternVL2-8B**, and the V-JEPA 2 line.  Treat this
as a new design space, not a free lunch: our deployment reality is 6
Ascend 910B (48GB HBM each), 192-core CPU, single training run ≤12h,
and a hard rule that every NEW stack must pass the CPU/NPU parity gate
before any number is trusted (we caught a silent all-zero positional
embedding on NPU this cycle - see 0.4).

### 0.3 Verdict ledger summary (all preregistered, 2026-10-04/05)

- TEMPORAL, three independent lines CLOSED: (a) pooled xperm 5-control
  decomposition - the head reads video-level semantics + residual
  statistics, "when" contributes zero; (b) native P/O/L matrix +
  1800-step extension - O−P crosses zero; (c) T-CONTEXT 2×2 cross-eval
  - multi-position training transfers at **−0.192** vs anchor-trained
  on the common anchor-dev ground.  Every arm only reaches
  slotprior+0.008..0.031 **within its own slicing protocol**.
- The position-prior reliance is now reproduced on THREE independent
  data constructions.  Any 9B-era design must assume this shortcut is
  in our data/label structure, not in our model size.
- SPATIAL: S0 oracle gap 0.247 [0.227, 0.268] on locked-out LIVE
  sources (70% of frames ≥0.05).  S1 richer features and S4 smoothing:
  negative.  S-REREAD (actual crop re-encoding, four arms S0/S1control/
  S2/S3zero) is RUNNING NOW - four-arm reads land tonight.
- Label projection audit: e_proj 0.76%, block-constant oracle loss
  7.1%, label agree 97.9% - 1 s projection noise is NOT the blocker.
- 3 official slots unspent.  Official-domain readouts exist ONLY as
  submission feedback.  Official test data must never enter training.

### 0.4 Standing rules this prompt operates under

- All numbers are raw; no conversions of any kind.
- Official test videos: no parameter updates, no pseudo-labels.  (If a
  proposal wants ANY forward pass over official videos, it must argue
  the compliance boundary explicitly - see Q1.)
- New NPU stacks: parity gate first, numbers after.
- Deployment: 6×910B, 48GB HBM, fp16, ~34GB host RAM per NPU process
  (safe concurrency ≤12), single training ≤12h, 192-core CPU.
- **Throughput reality check (operator-corrected)**: our 86M tower
  encoding runs at ~37 crops/s/card with AICore near 0% - the limit is
  our single-threaded decode and batch-1 implementation, NOT the
  hardware.  A 7B VLM scoring 129 candidate crops per frame costs
  ~1,032 prefill passes (~350 tokens each) per video ≈ a few hours for
  all 426 official videos on 6 cards, assuming ~2,000 token/s/card
  prefill.  But we have NO vLLM on 910B (vllm-ascend lacks
  FusedInferAttentionScore; transformers runs eager attention).  The
  REAL unknowns are: (a) measured 7B batched-prefill throughput on our
  stack, (b) whether LoRA fine-tuning of a 7B fits the 12h/run limit
  on 6×910B.  Any backbone ranking MUST size against these two
  measured-to-be numbers, not against generic GPU numbers.

## Q1 - How do we build a STRONG positive correlation between our local
        validation sets and the official evaluation set? (HIGHEST PRIORITY)

### The evidence you must confront

Two-tier correlation reality, operator-confirmed:
- Temporal line (PHD2-family local pools): same task family, direction
  filtering works (the 34.95 champion came from it), but **local
  positive deltas repeatedly fail to transfer**: H3 gate2 passed locally
  and did not ship; XRERANK did not beat its parent officially; expected-F
  GROUP-B tie stayed an official tie; V4 arms all rejected.  Multiple
  "local +x → official 0".
- Spatial line (LIVE_YT_VC / RetargetVid): our weakest link.  LIVE is a
  retargeting-research dataset with researcher-annotated crop
  preferences; the official spatial GT is competition-annotated.  The
  alignment was never validated.  And we have documented OFFICIAL
  NEGATIVE transfers: scaling package **−4.00**, centering package
  **−4.29**.  Official protocol: 159/426 videos have no sliding window
  (spatial is void there) - the spatial ceiling is divided by ~1.6.
- We have ZERO official-domain development handles.  Submission
  feedback is the only official readout.  3 slots left.

### Your obligations

1. **Mechanisms**: name and rank concrete mechanisms to build local-
   official correlation under these constraints.  At minimum, evaluate:
   (a) label-free domain adaptation on official videos (feature
   distribution alignment; argue the compliance boundary of ANY forward
   pass over official videos - is a frozen-encoder feature extraction
   "entering training"?  Give the strict reading and the loose reading);
   (b) diagnostic packages (zero-parameter, frontier-mechanism packages
   whose official delta CALIBRATES the local-official transfer function
   per domain - design the information-optimal diagnostic set);
   (c) building a synthetic official-domain proxy (what public data
   best matches the official task construction: GIF-style user
   highlights with per-kept-frame crops, video-macro joint F?);
   (d) correlation measurement itself: how to predict "local +0.03 →
   official ?" with calibrated uncertainty BEFORE spending a slot.
2. **48h probe** for the single best mechanism, sized to 6×910B.
3. **Failure conditions** and what evidence would kill it.

## Q2 - With ≤9B legal, which backbone, and how does the 1-9B frontier
        raise scores in this task family?

### Your obligations

1. **The 1-9B frontier map, by name**: 2024-2026 methods in video
   highlight detection / moment retrieval / temporal grounding at this
   size: base model, params, training data scale, measured numbers
   (QVHighlights, CharadesSTA, TVSum, or closest).  Cover at minimum:
   InternVideo2 (1B and 6B), Qwen2.5-VL 3B/7B, InternVL2/2.5-8B,
   Time-R1 (7B, 2,500 samples, CharadesSTA R1@0.7 = 35.3), CROP
   (Qwen2.5-VL-7B, FLMS IoU .822→.871 with DPO), V-JEPA 2/2.1,
   VideoChat-Flash, Apollo-class video LLMs, and anything 2026-newer.
2. **The decision**: given our metric (video-macro joint F over kept
   frames + crop IoU; no sliding window on 159/426), our deployment
   reality (0.4), and our measured failure modes (0.3): rank concrete
   architectures.  At minimum argue: (a) swap the vision tower only
   (86M → InternVideo2-1B-class frozen features, keep TCN+B3 heads,
   re-run the preregistered gates); (b) add a 1-2B temporal reasoning
   VLM as a THIRD stage (crop-reading verifier on top of B3 shortlist);
   (c) full 7B VLM pipeline (keyframe selection → crop reading → joint
   scoring).  For each: params budget, NPU-hours estimate for a 48h
   probe, the parity/throughput risk, and the expected effect on the
   THREE closed temporal lines (does a 1B video encoder invalidate the
   native-line closure?  Its reopen condition was exactly "feature/
   task definition change" - say explicitly whether it fires).
3. **S-REREAD interaction**: S-REREAD's mechanism (the encoder must
   SEE the actual crop) is native to any VLM that reads cropped images.
   Say explicitly how the S-REREAD four-arm read (tonight) should
   change the Q2 ranking.
4. **48h probe**: the cheapest experiment that produces the strongest
   evidence for or against the top-ranked backbone, sized to 6×910B
   with the parity gate included.  Probe 0 (before any backbone probe):
   measure our stack's 7B batched-prefill throughput (single card,
   batch {1, 8, 32}, fp16, eager attention) - this single number
   re-prices every architecture in your ranking, because our software
   stack has no vLLM and 34GB host RAM per process caps concurrency.
5. **Failure conditions.**

## Q3 - What ports from our small-model work to the 9B era, and what
        does a 1-9B model simply do natively?

### The asset inventory you must judge

Small-model assets, each with its evidence level:
1. **Shortcut diagnostics** (xperm 5-control decomposition, slotprior
   baseline, anchor/neutral slicing audits, equivariance readouts):
   these found that our temporal heads learn the slicing protocol's
   position structure, not content.  Would a 7B VLM on the same
   anchor-sliced training data ALSO learn the position prior?  Or does
   in-context/vision-language pretraining absorb content signal that a
   246K TCN cannot?  Name evidence for or against (e.g., does the
   moment-retrieval literature report position-shortcut failures for
   VLM-based methods on weakly-sliced training data?).
2. **The candidate-utility head paradigm** (129 legal max-windows,
   huber regression on annotator-mean IoU): keep, or replace with
   direct coordinate regression / VLM grounding heads?
3. **Protocol engineering** (frame contracts, contract hashes, parity
   gates, checkpoint resumes, preregistration + claim ledgers, one-shot
   confirmation pools): we assume this ports fully - confirm or push
   back.
4. **Negative results**: three closed temporal lines, S1/S4 negatives.
   Which of these re-test for free under a new backbone (the native
   line's reopen condition is exactly "feature/task definition change")
   and which stay closed regardless of backbone?
5. **The four-arm residual-head pattern** (baseline + capacity control
   + actual-observation arm + zero-parameter fusion): does this
   experimental pattern port to VLM-era probes?

### Your obligations

1. Per asset: PORT / RE-TEST / ABSORB-BY-NATIVE-ABILITY, with the named
   evidence and the cheapest re-test.
2. The honest comparison: what does the literature say a 7B VLM does
   ZERO-SHOT on GIF-highlight-style tasks (not CharadesSTA - the
   closest measured proxy)?  Do NOT convert between metrics; name the
   closest measured claim and its task.
3. End with the ONE question we should have asked you and did not.

## Output format

Answer Q1 first.  Every frontier claim: NAME + year + measured claim.
Every proposal: mechanism, transfer cost, 48h probe, failure condition.
Mark estimates.  Keep ASD-STE100.  Do not propose anything that touches
official test data in training.
