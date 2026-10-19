# R48 consensus gated temporal residual (2026-10-18)

## Question

The 4fps local-state readout improves the fixed-set public proxy while its
source-held-out MAE is slightly worse than the 1fps local readout. R48 tests
one fixed reliability rule: allow the 4fps residual to change the centre
ranking only when the 1fps and 4fps readouts lie on the same side of their
within-video medians.

## Public source-held-out result

Using the same 540 labels from 12 sources, fixed `eta=.20`, Ridge `alpha=10`,
and exact 85% selection:

| rule | macro proxy delta | positive views / 18 |
|---|---:|---:|
| D2, 1fps local | +0.01901 | 9 |
| D3, unrestricted 4fps local | +0.02283 | 9 |
| R48, consensus gated | **+0.02360** | 9 |
| onset only | +0.01376 | 7 |

The gate changes fewer candidate swaps when the two readouts disagree and gives
a small improvement over D3 on this public proxy. The gain is not independent
confirmation: the same 12 sources and one annotator account were used, and the
proxy has no native official joint IoU labels.

## Interpretation and limits

The result supports a narrower mechanism claim: high-frequency state is more
credible as a conditional correction than as an unconditional score. It does
not establish that either readout transfers to the official test domain. A
negative official score would reject the deployment value of this residual
family, including R48; it would not identify whether the failure is the local
descriptor, public-source shift, or the centre prior.

The official R48 package is built from the R26 full-frame B3 parent, keeps each
video's exact `round(.85*N)` budget, and changes temporal ranking only. It is
not uploaded and must pass the same 426-video, frame-count, box identity,
parameter and ZIP round-trip checks as R26/R46.

Reproduction: `scripts/r48_consensus_gated_residual_probe.py`,
`scripts/r48_consensus_gated_residual.py`; raw metrics are in
`reports/r48_consensus_probe_20261018.json`.
