# R47 cross scale consensus residual (2026-10-18)

## Decision question

The 4 fps local state residual improved the public proxy, but full D3 ranking
can perturb quiet frames. This experiment tests whether the increment is more
reliable when the 1 fps and 4 fps local readouts agree in rank direction.

The gate was fixed before reading the result: fit two source held out Ridge
heads (`alpha=10`) on the same 12 public annotated sources, rank each readout
within a held out video, and enable the signed D3 residual (`eta=.20`) only
when both ranks lie on the same side of their video median. No parameter or
official score was scanned. The public labels are a mechanism diagnostic, not
official temporal ground truth.

## Public source held out result

| rule | soft proxy | binary proxy | positive sources |
|---|---:|---:|---:|
| R26 center | 0.721916 | 0.704195 | — |
| full D3 4 fps residual | 0.756016 | 0.746158 | 9/12 |
| cross scale consensus gate | 0.747357 | 0.734698 | 7/12 |

The gate retains most of the D3 signal while reducing its unrestricted
perturbation. This is evidence for a conditional rather than universal
high-frequency effect, but the labels are small and same-account annotations;
it does not predict an official score.

## Full candidate

The fixed gate was fitted once on the public labels and applied to the 426
semifinal videos using the original frame axis. It passes through the audited
full-frame B3 boxes and keeps exactly `round(.85*N)` frames per video.

- Canonical package (same predictions): `submissions/r48_candidates_20261018/R48_CONSENSUS_GATED_RESIDUAL_B3/`
  (`submissions/R47_HIGHFREQ_CONSENSUS_B3_packaged/` is a local duplicate used
  for independent packaging verification.)
- Predictions SHA256: `e6964805eb1fc0e53a9e5e620492f929b29efa69c7bfd6d2df0c754898a6a20c`
- ZIP SHA256: `c543f4d89630c3a1535c99dc471a342386e801130c16908dc5ae43bd4c79bdb5`
- Coverage: 426 videos, 199,037 predictions
- Parameters: 450,283,377 (`0.450283377B`)
- Relative to R26 center: mean 12.65 frame swaps; 415/426 videos changed

The local contract validator and independent checker both pass. The package
has not been uploaded and has no official score.

## Competing cheap mechanism

A fixed shot aware prior (4 fps grayscale-difference 95th percentile cuts,
equal global/within-shot center ranks) gave soft `0.726447` and binary
`0.711860`, versus center `0.721916`/`0.704195`, with only 3/12 sources
positive. It is retained as a negative/weak diagnostic and was not packaged.

## Next decision

Use R47 consensus only as an exploratory fifth candidate if the submission
quota permits. If its official score does not beat the R26/R44 fluctuation
range, stop high-frequency residual variants and move to a spatial utility
readout or new event/audio supervision. Do not scan eta, alpha, gate
thresholds, or cut thresholds after this result.
