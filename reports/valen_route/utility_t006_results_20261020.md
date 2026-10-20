# Valen utility target T006 evaluation — 2026-10-20

## Verdict

T006 is complete and does not pass the promotion gate. It changes only the
native-crop target temperature from `0.12` to `0.06`; model, prompt, candidate
set, time policy and evaluation contract are fixed. The exact prepared dev
request evaluation gives a source-macro improvement of `+0.000791`, with a
source bootstrap 95% CI of `[-0.004341,+0.007054]`. On the exposed
RetargetVid 601–700 confirmation pool the improvement is `+0.001296`, with CI
`[-0.000436,+0.003101]`. Both intervals include zero, so this is not evidence
of a deployable gain.

The utility-temperature route is closed. Do not continue scanning target
temperature, prompt wording, calibration or fusion weights under this route.

## Reproducibility

- Remote host: RTX 4080 32GB, CUDA device 0; official test media and labels were not read.
- T006 checkpoint: `/data/aic/experiments/VALEN_2B_SFT_UTIL_T006/latest/checkpoint.pt`
- T006 checkpoint SHA256: `329cc0297d7f6ba125d072c7dd24f0269cabbdd90e5f0bda0f58f19665c418f7`
- Baseline checkpoint SHA256: `0063f07cafbb5379a1c80bfab27ef50a28e10d25932e82d7e9ad5284b20dd561`
- Base model parameters: `2,248,463,977`; checkpoint updated tensors: `411`.
- Dev data SHA256: `97525b1a12938a8a717f17be92890c51ad356c7202ac23b954ca0ce8ec19bc53`.
- Confirmation records SHA256: `7202ecd7f160b39baae99dcaf74a61732c52873b0d513f05e23ca7e6ca9d00b9`.
- Inference batch: `16` for both runs. T006 dev: 1,546/1,546 rows, 228.2 s;
  confirmation: 4,081 primary rows plus 32 order controls, 603.1 s.

## Paired checkpoint comparison

All deltas below are T006 minus the frozen `temperature=0.12` checkpoint and
use source as the independent unit.

| Pool | Sources | Old source macro | T006 source macro | Delta | 95% bootstrap CI | Positive sources | <= -0.05 |
|---|---:|---:|---:|---:|---|---:|---:|
| Public dev, all | 168 | 0.575236 | 0.576026 | +0.000791 | [-0.004341,+0.007054] | 29 | 6 |
| Public dev, LIVE | 148 | 0.558478 | 0.559603 | +0.001125 | [-0.004605,+0.008213] | 20 | 6 |
| Public dev, RetargetVid | 20 | 0.699242 | 0.697562 | -0.001680 | [-0.005393,+0.002040] | 9 | 0 |
| RV601–700 exposed | 100 | 0.675226 | 0.676522 | +0.001296 | [-0.000436,+0.003101] | 45 | 0 |
| RV601–700, 1:3 | 100 | 0.556977 | 0.557955 | +0.000978 | [-0.002047,+0.004217] | 21 | 0 |
| RV601–700, 3:1 | 100 | 0.793134 | 0.794752 | +0.001618 | [+0.000045,+0.003308] | 39 | 0 |

The single positive 3:1 ratio interval does not rescue the route: the full
confirmation CI crosses zero, and the public dev aggregate remains too small
to establish a practical gain.

## Error decomposition

On the 168-source public dev pool, T006 leaves candidate recall unchanged:
`oracle129 - oracle4 = 0.084062`. Candidate-internal selection loss changes
from `0.124249` to approximately `0.1241` (LIVE `0.131560→0.130435`, RV
`0.070150→0.071830`). On RV601–700, selection loss changes
`0.087500→0.086204` while recall loss remains `0.071254`.

Thus the sharper target slightly changes choices, but the change is too small,
unstable across sources, and does not address the larger candidate coverage
limit. This is a failed objective intervention, not evidence that native crop
supervision is useless: the earlier SFT transfer signal remains recorded as a
separate exposed diagnostic.

## Artifacts

- `utility_t006_eval_dev_source.json`
- `utility_t006_eval_confirm_analysis.json`
- `utility_t006_checkpoint_comparison.json`
- Remote raw predictions remain on `/data/aic`; they are not committed to Git.
