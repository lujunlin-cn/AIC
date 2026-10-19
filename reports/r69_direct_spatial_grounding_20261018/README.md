# R69 TimeLens direct spatial grounding (high risk)

## Motivation

R54 is the current official anchor at 36.22, but its positive result changes
both the TimeLens temporal set and the V9 spatial boxes. R67 changed only the
frame set using V9/R26 geometry disagreement and scored 36.16, so unlabelled
box disagreement is not a useful temporal utility signal. R57 and R59 also
showed that global late shifts and low resolution 4-fps endpoint substitution
destroy the temporal set.

The remaining high-leverage unisolated component is the spatial readout itself.
V9 is a small intervention over B3 and there is no native AIC crop ground truth.
R69 therefore asks the same 8.767B TimeLens2 model to produce one target-ratio
crop from the complete video, while retaining the exact R54 temporal frame
set. This is deliberately a high-risk spatial replacement, not a parameter
scan or an official-test-label fit.

## Frozen hypothesis and counter-hypothesis

*H1*: complete-video language/vision grounding identifies the event's main
subject and composition better than the frozen V9 per-frame readout, so a
video-level crop can improve native IoU on the selected frames.

*H0*: native crop boxes move with the event and camera; a single video-level
crop loses temporal geometry and will lower IoU substantially. A malformed
answer or invalid coordinate must never create an illegal package.

The experiment is useful even if it fails: it isolates the spatial model while
holding R54's selected frame IDs, exact 85% budget, and temporal model fixed.

## Input and compliance

The remote run uses the public semifinal videos and the frozen TimeLens2-8B
checkpoint, with `fps=1`, `max_pixels=50176`, and a versioned direct-crop prompt.
Raw responses are cached before parsing. The final package counts TimeLens2
once (`8,767,123,696` parameters), below the `9B` rule. No test labels,
official score, or inferred test truth is used during construction.

## Acceptance checks

The package must contain all 426 videos and 199,037 selected predictions,
preserve the exact R54 frame IDs, maintain ratio-exact legal `[x,y,w]` boxes,
pass the independent checker and ZIP round-trip, and report response and
prediction SHA256 values. The official score is the only evidence of net
competition impact; there is no local native-crop oracle for the AIC videos.
