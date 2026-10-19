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

## R87: AHA/PHD2 adaptation pilot

The PHD2 fragment and label assets were available locally, so the pilot used a
source-level split and kept `gif_soft`/interval supervision separate from the
known placeholder hard labels. The AHA base and adapter loaded after repairing
the dedicated environment to Transformers 4.49.0, tokenizers 0.21.4, and the
missing plotting/runtime packages.

The cloned environment then failed a more basic contract: after the first
causal frame, AHA relevance, informativeness, and uncertainty outputs became
non-finite for many fragments. The finite score fraction in the 96/48 pilot
was only `0.1015625`. This reproduced with both the static cache call and the
repository SinkCache path; changing the empty-frame transition to an EOS token
did not remove the NaNs. Because the failure occurs before the target head is
trained, no AHA ranking or adaptation number is valid. The route is paused at
an implementation diagnosis, not counted as a negative transfer result and
not packaged.

The next AHA action is to compare the repository's original `LiveInferForBenchmark`
frame transition and cache implementation under the original pinned dependency
set on a one-fragment trace, checking hidden-state finiteness before any larger
run. Until that trace is finite, R86 remains the only upload candidate from
this pair of routes.
