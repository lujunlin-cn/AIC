# R46 semantic-transition probe (2026-10-17)

## Question

Does multiplying fixed sub-second local-state change channels by a frozen InternVideo2 event anchor provide an additional, source-held-out signal beyond the already tested 4 fps local-state + anchor residual?

## Contract

- 12 publicly annotated 30 s source clips, 540 rows, source-held-out Ridge (`alpha=10`), no official labels or score access.
- 4 fps local-state descriptor plus frozen event-score anchor; fixed `eta=.20`, no coefficient sweep.
- The added arm is a fixed interaction block: event anchor × local change/state channels.
- The fixed-set proxy is the same center 85% selection rule used in the preceding high-frequency probe.

## Result

The interaction arm was indistinguishable at the deployment proxy:

| arm | source-held-out macro MAE | proxy F | positive sources |
|---|---:|---:|---:|
| D4 local+HF+IV2 | 1.896765 | 0.706636 | 3/12 |
| D5 semantic × transition | 1.903116 | 0.706636 | 3/12 |
| center | — | 0.703889 | — |

The interaction changed the regression error slightly worse and changed no selected set under the fixed eta and source-level proxy. The absolute MAE here is not comparable to the earlier state probe because this script retains the duplicated annotation views in the regression rows; the selection comparison is paired and fixed by source/time.

## Decision

Do not create a submission from the semantic interaction. It does not identify an increment over the high-frequency residual. Keep the high-frequency full-frame candidate as a separate pre-registered experiment only if its package completes with the full-frame B3 contract; otherwise retain R26 and stop this interaction branch.
