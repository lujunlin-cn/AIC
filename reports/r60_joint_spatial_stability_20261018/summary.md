# R60 joint spatial stability control (2026-10-18)

R60 uses the same joint utility as R59 but reverses the spatial term:

```
rank(center) + 0.20 * rank(TimeLens2 interval) - 1.0 * rank(V9/R26 geometry disagreement)
```

It favours frames where the two crop readouts agree (a stability hypothesis),
while R59 favours disagreement (a transition/content hypothesis).  Both use
the same V9 boxes and exact 85% frame budget.  This is a predeclared opposite
mechanism control, not a coefficient sweep of an upload score.

Package: `submissions/R60_JOINT_V9_STABILITY_GM1/`

* 426 videos, 199,037 predictions; duplicate/range/budget checks pass.
* Relative to R54: 23,266 swaps (all 426 videos), 119 non-contiguous videos.
* Mean normalized selected position is `.44482` (R54 `.52344`), making this a
  strong early/stability intervention.
* ZIP SHA256: `daf6547fac065cc0365e8a9b22841f2438fc0af55e3ab973517a5a060230044e`.
* Prediction SHA256:
  `9afff46a93b70db203df97f09950c8e0beccb951d401740e60c258aaf225a418`.
* Declared parameters remain `8,854,531,425` (8.855B).

R60 and R59 form a high-risk pair: a positive R59/R60 separation would tell us
whether crop-model disagreement or stability is a useful frame acquisition
signal.  A negative result for both would reject this family while leaving the
R54 semantic phase and V9-center controls interpretable.
