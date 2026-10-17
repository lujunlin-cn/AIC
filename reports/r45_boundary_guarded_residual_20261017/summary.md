# R45 boundary guarded residual (2026-10-17)

## Mechanism

R44 changed only the temporal set relative to the R26 center-85/B3 package,
but its small official gain was not distinguishable from a near-neighbor
exchange null. R45 therefore keeps R26 as a hard anchor. For each video it
retains only those R44 additions and removals that lie within five percent of
either edge of the R26 center window. All other R44 changes revert to R26.
The budget remains exactly `round(0.85*N)`, and the spatial B3 box is copied
from R26 on common frames and from R44 on newly admitted frames (the common
frame boxes were checked byte-for-byte equal).

This is one fixed intervention, not an eta/alpha sweep. It tests whether the
R44 signal is local to the center-window boundary and whether limiting the
displacement removes the long-video failure mode.

## Package audit

- 426/426 videos and 199,037 predictions.
- 411 videos changed relative to R26.
- 5,942 of R44's 6,352 swaps were retained; 410 were reverted by the guard.
- `450,283,377` declared parameters (`0.450283377B`).
- Predictions SHA256: `f71d233a199b81ff48adcea154e1af792fef921022272afc3d27070bca6d0940`.
- ZIP SHA256: `f50506802fa3809d41913e2f3751bddc1ca71d2e9ea60c47e067b220a6d07cb2`.
- Independent local structural audit and ZIP round-trip: PASS.

The candidate is ready for the user's upload workflow. It has no official
score yet; R26 remains the rollback package and R44 remains the highest
reported package at 36.09 until R45 is scored.
