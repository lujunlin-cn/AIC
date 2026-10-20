# Valen utility target experiment T006 — 2026-10-20

## Frozen question

The RV601--700 confirmation improved the existing head, but the four-candidate source-macro gap remains large (`oracle4 - Valen = 0.087500`). This run tests whether the SFT decision is too smooth when the native IoU utility distribution is soft.

## Single intervention

Use the same public TRAIN rows, candidate images, prompts, model, optimizer, checkpoint selection policy and temporal policy as `VALEN_2B_SFT_CROP_V1`. Replace only the target distribution temperature from `0.12` to `0.06`:

\[
p_j = \operatorname{softmax}(u_j / 0.06)
\]

where `u_j` is the native crop IoU for the four displayed candidates. The existing run used `0.12`. This is one utility-aware target objective, not a temperature sweep.

## Batch and resource contract

- GPU: remote RTX 4080 32 GB, `CUDA_VISIBLE_DEVICES=0`
- Training microbatch: `16`; token budget: `32,768`; prefetch enabled
- Evaluation: Valen adapter `decide_batch`, batch `16`; group by equal visual tensor shapes
- No official test media or labels
- Expected runtime: under 3 hours; recoverable `latest` checkpoint and `save_every=50`

## Gate

Evaluate the frozen checkpoint on the existing 168-source exposed pool and RV601--700 exposed confirmation pool. Report source-macro IoU, paired source bootstrap, ratio split, candidate recall and selection loss. Promote only if the new objective beats the T006=0.12 checkpoint on a source-level held-out comparison with no severe domain regression. Otherwise close the Valen deployment route.
