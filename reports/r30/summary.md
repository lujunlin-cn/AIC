# R30_DYNAMIC_CONTENT_PROBE — 2026-10-16

## Decision first

The first test of the new public-data route does not justify spending on a
larger temporal model or denser sampling yet. With the same 12 native PTS and
an exact 85% output budget, a TRAIN-only linear readout of adjacent VPB
changes fell below the position control on the previously exposed DEV set.
The frozen VPB embedding by itself was only slightly above the center control,
with a source-level interval crossing zero. This is an overfit/contract
diagnostic, not evidence that all visual dynamics are absent.

| DEV arm | source-macro F |
|---|---:|
| cubic position readout | 0.297399 |
| center 85% | 0.288381 |
| VPB static readout | 0.292491 |
| VPB adjacent-difference readout | 0.260556 |
| VPB static + adjacent-difference | 0.278525 |

Paired source bootstrap contrasts against center are: static +0.004109,
95% CI [-0.012593, 0.021734]; dynamic -0.027826, CI
[-0.050982, -0.005046]; static+dynamic -0.009857, CI
[-0.027008, 0.007853]. The predeclared +0.005 and positive-lower-bound gate
was not met.

## What was tested

The allowlisted 26 TRAIN / 14 DEV VideoXum sources and ten original raters
were used. VPB features came from the bound R24 12-observation cache; the
native PTS axis, split, and source hashes were checked against the R29 binding.
Only a ridge-logistic readout was fitted on TRAIN. Dynamic features were
causal adjacent VPB difference, cosine change, and second-difference norm;
all arms used the same exact 85% budget. No protected source, official test
label, artificial annotation, or GPU training was used. Reproducible outputs
are `metrics.json` and `records.json` in this directory.

## Interpretation and next action

The dynamic arm's high TRAIN F (0.304126) versus DEV F (0.260556) shows that
the 768-dimensional difference readout is too flexible for 26 sources, so its
negative DEV result cannot identify a lack of motion information. The static
arm also does not pass the source-out-of-sample gate. The robust conclusion is
that the current 12 sparse VPB observations plus an unconstrained linear
readout do not provide evidence for a deployable visual-content replacement
of center selection.

Stop this high-dimensional dynamic recipe. Do not launch a larger temporal
model from this result. The next low-cost discriminator is a frozen,
low-dimensional event-state probe (norm/cosine/temporal-window summaries with
stronger regularization and no full 768-D differences). If that also fails,
the visual-only public route is stopped and effort moves to an existing
audio/event-structure signal or to recovering the missing spatial deployment
contract. A success would need the same source-level gate on an untouched
source split before any official package is built.

This is public development evidence only; it is not an official score or an
independent confirmation of the competition distribution.
