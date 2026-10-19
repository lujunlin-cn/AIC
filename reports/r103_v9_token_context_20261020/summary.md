# R103 genuine V9 token context (2026-10-20)

## Question

Does preserving spatial token layout in adjacent V9 frames add native crop utility beyond the current-frame 129-window representation? R101's pooled ResNet context was not a sufficient test of this hypothesis.

## Protocol

V9 vision grids were re-extracted for the same source-disjoint RetargetVid/LIVE native crop tables as R70. Each row received two adjacent same-unit keyframes, each pooled to a 4×4 grid of 768D V9 tokens. The head used current 129-window features plus cross-attention to 32 neighbor tokens. Control, actual context, and source-permuted context used identical training and evaluation. There were 12,822 rows and 168 held-out source videos; no audio or official labels.

## Result

| variant | held-out source macro IoU |
|---|---:|
| zero-context control | 0.5684209 |
| genuine V9 token context | 0.5677629 |
| context source-permuted at evaluation | 0.5669107 |

The context delta is −0.0006575; the permutation delta is −0.0015101. The first implementation reached an all-masked attention NaN for one-frame units; the mask was corrected to expose a neutral zero token, then the complete comparison was rerun.

## Decision

Close this exact context recipe. The result does not prove that temporal context is useless in general; it says adjacent 4×4 V9 tokens plus this readout and native crop supervision did not improve the current representation. Do not spend another run on ranks, attention width or fusion coefficients without changing the observation or supervision contract.

Evidence: `metrics.json`; code `scripts/r103_v9_token_context.py`; remote cache under `/data/aic/experiments/R103_V9_TOKEN_CONTEXT/full`.
