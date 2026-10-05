# Round-6 report: R6 plan executed - three verdicts, 9B rule change, V7

Date: 2026-10-05.  Style: ASD-STE100.  All official scores are RAW, no
size penalty.  Champion V11_VTREPLAY = 34.95; leader intelligence 45;
gap 10.05.  Branch: r6-sreread-tcontext (from 9efb1e0).  Plan:
reports/20261005_round6_plan.md.  Ledger: reports/r6/claim_ledger.md.

## 0. Rule change (operator, evening): the 9B era

The official rules dropped the size penalty.  The only constraint is
≤9B total inference parameters.  Raw-score discipline is unchanged.
Everything 1-9B re-enters the candidate set (InternVideo2-1B,
Qwen2.5-VL-7B, InternVL2-8B, ...).  Memory: aic-9b-rule-change.

## 1. P0 contract layer - all six items DONE (morning)

1. Source ledger + ancestry audit: **val178 confirmed clean** for the
   champion line (vid-level: B3's 1,115 live_train clips all inside
   T5 train_index, intersection with val178 = 0; release manifests put
   val178 outside every split as the frozen holdout).  T5 (deprecated
   teacher line) built caches on val178 - recorded, non-leaking per
   Q7-C case 1.  BLOCKED_FRESH_SPATIAL_CONFIRM does not fire.
2. Q7-A xperm 5-control decomposition (20 seeds, both pools): same-src
   exclusion does NOT explain the small round-5 delta (derange still
   +0.0034/+0.0041); res_shuffle and rep_mu both ~0 - the head reads
   video-level semantics + residual statistics, "when" contributes
   zero; zeros (+0.104/+0.085) collapses scores (gamma median 0.0003)
   - the out-of-distribution inflation is confirmed.
3. Q7-B label projection audit (128 dev sources, 691k frames):
   e_proj 0.76%, e_min 0.38%, partial-block rate 1.5%, frame-truth vs
   tubelet label agreement 97.9%, perfect-block-predictor F = 0.929.
   1 s projection noise is NOT the blocker; native-line closure stands.
4. Parity gate re-run + error-injection self-test: gate PASS
   (F_cpu = F_npu = 0.77229, dF 0.0).  Injections: temporal_shift /
   layout_swap / fp16_overflow caught 32/32 each; **bad_padding is a
   finding** - appending one zero frame does not flip the top-k
   decision under the gamma>2eps rule (deployment-side low risk;
   training-side padding policy still must be pinned, AdaSpot lesson).
5. preregistration.yaml frozen (gates, lambda grid, seeds, one-shot
   val178 policy).
6. T5-era val caches audited; every number in this report runs AFTER
   the parity gate.

## 2. T-CONTEXT verdict: NO_PRACTICAL_GAIN_IN_SCOPE

Build: 300 sources (210 train / 90 dev, source-level split), 3,448
variants (anchor + front/mid/back per merged event, L = 12 s
continuous reads from the original videos), **0 decode failures**;
features via the round-4 CPU contract (fp32, 16 shards, 3,448 npz).

2×2 cross-eval (EVERY model reads BOTH dev variant sets; the common
ground is the anchor-dev variants):

| trained arm | @anchor dev | @own task |
|---|---|---|
| anchor+TCN | 0.5752 | 0.4803 |
| anchor+dense | 0.5763 | 0.4674 |
| multi+TCN | **0.3833** | 0.5243 |
| multi+dense | 0.3666 | 0.5192 |
| slotprior (no model) | 0.5672 | 0.4933 |

Gate wanted multi ≥ +0.007 vs anchor AND slotprior.  Measured:
**−0.192 [−0.215, −0.169]** and −0.184.  Every arm reaches only
slotprior+0.008..0.031 INSIDE its own slicing protocol and falls below
the prior on the other.  dense ≈ TCN (third confirmation).  The
multi-position data intervention does NOT move the model from the
position prior to content.  Scope: this intervention + these two heads;
not a claim about "content is unlearnable".

## 3. S-REREAD verdict: NO_PRACTICAL_GAIN_IN_SCOPE

Shortlist: v1 (top3+g9, ≤12/frame) FAILED the oracle gate (regret
0.0347); amended per the R6 extend-candidates rule to **top3+g33
(~36/frame), regret 0.0052/0.0056 PASS** - recorded in
preregistration.yaml BEFORE any crop encoding.  Throughput: 100-crop
probe = 36.7 crops/s/card -> full 240/80 scale kept, no reduction.

Pools frozen (seed 20261006): train240 from the B3 ancestry, dev80
from the 156 never-used train_index sources (all with GT), confirm =
val178 untouched.  dev80 grid extracted (6 cards, 2,400 frames, 78 s).
Full crop+full-image encoding: 2,560 rows x (36 crops + 1 full),
row-level checkpoints, done on 12 shards.

Four arms (dev80, video-macro IoU, source-cluster CI):

| arm | IoU | vs B3 |
|---|---|---|
| S0 B3 zero-shot | 0.5289 | - |
| S1-control (same capacity, no new observation) | 0.5355 | +0.0066 [0.0019, 0.0122] |
| **S2 actual crop observation** | **0.5243** | **−0.0046 [−0.0125, +0.0036]** |
| S3-zero λ=1 pure cosine | 0.3655 | −0.1633 |
| S3-zero λ=0.25 | 0.5299 | +0.0010 [0.00002, 0.0021] |

Gate wanted S2 ≥ +0.03 vs B3 AND vs S1-control.  Actual: S2 is WORSE
than the capacity control by 0.011.  In the Siglip2-feature +
residual-head setting, seeing the actual crop buys nothing.  The R6
failure action applies verbatim: pause this input transfer; no
"all video-crop methods are futile" claim.

**Annotator-disagreement hypothesis upgraded**: LIVE-YT-VC's own paper
reports adjacent-annotator IoU ~0.50 - B3 at 0.529 already matches
"another annotator".  Most of the 0.247 oracle gap may be unlearnable
mean-vs-single-annotator disagreement.  Cheap follow-up: on
multi-annotator sources, score B3 against EACH annotator separately to
measure the learnable ceiling directly.

## 4. Three bugs caught by sanity reads (all fixed, all auditable)

1. **dev80 u labels**: freeze read sparse boxes_xywh and fed the
   numbers to IoU as ltrb -> B3 zero-shot read 0.038 (absurd).
   Fixed from raw_ltrb; after the fix B3 = 0.529 (matches round-5
   confirmation 0.535).  Fix script recomputed 640 rows and updated
   the 12 shard .pt files in place; crop embeddings and B3 scores were
   unaffected.
2. S3zero z-score indexed by shortlist POSITION instead of candidate
   id (lam=0 read 0.095 instead of ~0.53).  Fixed; lam=0 now equals
   S0 exactly (sanity identity holds).
3. Arm evaluation filtered dev rows out of a train-row argmax (nan).
   Fixed: train on train240 rows, read on dev rows.

## 5. vLLM probe: install path found, engine pairing unresolved

- Card type PINNED by operator: **Ascend 910A** (lspci 19e5:d801;
  official CANN device identification).  npu-smi's name string is not
  a type indicator.  "910A server" in older memories was the correct
  folk name after all.
- CANN pinned at 9.0.0.  The exact vllm-ascend pairing is **v0.18.0**
  (CANN == 9.0.0, torch == 2.9.0, torch-npu == 2.9.0.post2) - no CANN
  change needed.  Installed in /data/aic/venvs/vllm18 with cmake+ninja
  fix (arctic-inference builds) and the torch_npu 2.9.0.post2 exact
  wheel (plain 2.9.0 has an undefined symbol against torch 2.9.0).
- BLOCKED at the last step: vllm main package pairing (0.11.0 lacks
  vllm.utils.math_utils; 0.10.2 pulls torch 2.8).  Official path is a
  pre-paired Docker image; nested docker is not available inside this
  container.  Status: **NOT_RUN**, recorded, not failed.  Honest
  boundary: v0.18.0's support matrix lists A2/A3/300I/950 - 910A
  operator coverage is measured, not assumed.
- 9B-era implication recorded: our 37 crops/s/card is single-thread
  decode + batch-1 implementation, not hardware; 129-crop VLM scoring
  is a few hours for 426 videos at conservative prefill rates.

## 6. Incident: OOM (operator-identified)

24 encoding processes x ~34GB host RAM each = 816GB > 755GB -> the box
went down (up 18 min on reconnect).  Fixes: safe concurrency <= 12
(memory formula in memory), row-level checkpoint resume + incremental
save every 10 rows (v11_r6_s_encode_all.py).  Memory: aic-910a-host-mem-per-proc.

## 7. V7 prompt (GPT6PRO_PROMPT_V7.md) - three questions

Q1 local-official correlation under zero official-domain handles
(incl. the compliance boundary of label-free passes over official
videos); Q2 backbone ranking at 1-9B with Probe 0 = version-paired
vLLM throughput; Q3 which small-model assets port vs get absorbed.
Tonight's three verdicts and the annotator-disagreement hypothesis are
in the evidence chain; the prompt forbids silent bets between
"residual-head capacity" and "annotator disagreement" readings.

## 8. Budget and slots

NPU-hours this round: parity ~1 + dev80 grid ~0.1 + full crop encode
~6 (12 shards x ~2.5h wall) + T features (CPU) + misc < 10 total.
Submission slots: 3, unspent.  No gated candidate exists (all three
mechanism lines returned local negatives); the first official-domain
readout should be a DIAGNOSTIC package, per the correlation analysis.
