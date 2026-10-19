# R86 visual observation path and R87 target transfer pilot

Date: 2026-10-19

## R86: T5 crophead on the current 426-video path

T5_CROPHEAD_V3 was fit on the preregistered public native-crop tables. Its H1
mapping was selected on the public development split and confirmed on the
source-held-out LIVE and RetargetVid pools:

| confirmation | mean change vs point-centred H0 | source bootstrap 95% CI |
|---|---:|---:|
| LIVE confirm3, 178 sources | +0.0062447 | [+0.0034429, +0.0092864] |
| RetargetVid confirm2, 100 sources | +0.0148418 | [+0.0096558, +0.0200412] |

The deployment script applies exactly this fixed H1 mapping to the current V9
observation points, then runs the existing hold and EMA path. R54's frame set,
budget, temporal scores, and model accounting remain fixed. The package covers
all 426 videos and 199,037 predictions. It changes 106,781 kept-frame boxes,
keeps the legal window width unchanged, passes the independent checker, and
uses 8,854,531,425 declared parameters. ZIP SHA256:

`308275cd2a3ac3caef0aaf3ae237e3d2552bc048500cb23f12b79c3cdd8f714a`

This is a real structural candidate, not a keep or threshold sweep. Its public
crop gains do not identify the official gain; the only unresolved measurement
is the platform score.

### Official result and root-cause audit

The user reported the official R86 score as **34.88**. Relative to its parent
R54 score of **36.22**, this is `-1.34` points. The result is not a small
random fluctuation: R86 changed `106,781 / 199,037` kept-frame boxes, so the
intervention was large enough to expose a spatial contract error.

The decisive audit found that the package was built with
`points_V9_A1raw/points`, while R54's spatial parent and the intended control
use `points_B3_vo/points`. T5 H1 was learned from the original Qwen point
observation contract, and was therefore applied to the A1raw point semantics.
R86 consequently changed both the point source and the point-to-window
mapping. It did not test “T5 H1 on B3” as its manifest description implied.

The geometry comparison makes the mismatch concrete:

| candidate | changed boxes vs R54 | changed-box mean abs delta `[x,y,w]` | max delta |
|---|---:|---:|---:|
| R86, A1raw points + H1 | 106,781 | `[17.95, 31.60, 0]` px | 748.07 px |
| corrected R88, B3 points + same H1 | 106,781 | `[9.69, 25.80, 0]` px | 244.87 px |

The corrected package `R88_T5_H1_B3POINTS_R54TIME` was generated with the
same R54 parent, exact frame set and budget, the same frozen H1 coefficients,
and `points_B3_vo/points`. It passes the local semifinal contract with
426/426 videos, 199,037 predictions and 8.854531425B declared parameters.
Its ZIP SHA256 is
`db6408cdff86649ab25913b225f61991d5e221c0af711483624131014fd71207`.
It has not been uploaded and has no official score.

This does not yet prove that H1 transfers to the B3 contract. It does prove
that the R86 score cannot be used as evidence against that properly matched
experiment. R86 is archived as a **confounded transfer**, and R88 is the
minimal corrective candidate if an upload slot is available.

## R87: AHA/PHD2 adaptation pilot

The PHD2 fragment and label assets were available locally, so the pilot used a
source-level split and kept `gif_soft`/interval supervision separate from the
known placeholder hard labels. The AHA base and adapter loaded after repairing
the dedicated environment to Transformers 4.49.0, tokenizers 0.21.4, and the
missing plotting/runtime packages.

The cloned environment initially failed a more basic contract: after the first
causal frame, AHA relevance, informativeness, and uncertainty outputs became
non-finite for many fragments. The finite score fraction in the 96/48 pilot
was only `0.1015625`. This reproduced with both the static cache call and the
repository SinkCache path; changing the empty-frame transition to an EOS token
did not remove the NaNs.

The runtime repair is now complete. Running the AHA base in BF16 on the RTX
4080, with the repository's empty-token causal transition, gives finite
processor values, visual embeddings, hidden states and all three heads for
all 8 frames in the trace; the full 96/48 pilot has `finite_fraction=1.0`.
The repair is recorded in `scripts/r87_aha_trace.py` and
`scripts/r87_aha_phd_pilot.py`.

The repaired pilot is still not a positive transfer result. On 32 held-out
sources, the frozen relevance/informative/uncertainty heads obtain row AP
`0.2884/0.2644/0.3144`; a 7-feature Ridge adapter reaches row AP `0.3486`,
but source-macro AP falls from relevance `0.1047` to `0.0952`. The labels are
weak GIF interval labels and the sample is only 48 fragments, so this is a
diagnostic with low power, not a reason to package AHA. The runtime blocker is
closed; the remaining blocker is target-domain supervision and contract
alignment.

## Reproduction record

Code commit: `27e3f1e`.

R86 was run on `root@connect.westc.seetacloud.com:22527` with the 426-video
semifinal index, the existing `obs_cache`, `points_V9_A1raw/points`, and the
frozen T5 probe metrics. The remote independent checker reported `valid=true`,
`videos=426`, and `predictions=199037`.

The corrected R88 reproduction used the same inputs except
`points_B3_vo/points`; its local semifinal contract reported `valid=true`,
`videos=426`, and `predictions=199037`.

R87 used the same host, GPU0 (RTX 4080 32GB), the local AHA base and adapter,
PHD2 `index_v2`, and `LFM_V13/frag_labels.jsonl`; no official test files were
read. The AHA environment was repaired to Transformers 4.49.0,
tokenizers 0.21.4, matplotlib, thop and h5py before the pilot.
