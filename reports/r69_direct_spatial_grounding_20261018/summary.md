# R69 direct spatial grounding package

## Result of the construction audit

The remote TimeLens2 direct-crop run completed for all 426 semifinal videos.
Every response contained a parseable crop (`426/426`); no response fallback was
needed. The run used `fps=1`, `max_pixels=50176`, and the complete-video prompt
recorded in `direct_crops.json`. The raw response SHA is recorded in the package
manifest.

The package retains the exact R54 selected frame IDs and exact per-video budget:

* 426 videos and 199,037 predictions;
* zero frame-set mismatches against R54;
* 129,443 selected-frame boxes replaced, 69,594 equal to the inherited V9 box;
* 116 distinct direct crop boxes across 426 videos;
* zero malformed or fallback videos;
* TimeLens2 counted once: 8,767,123,696 parameters (8.767B).

The independent checker and local semifinal contract both pass. Package ZIP
SHA256 is `f5d6dcadbd511f286c98360a72cfb568146056eb77d8cee6d62194b9e7839871`.

## What the intervention actually changes

The direct model emits one crop per complete video, so every selected frame in a
video receives the same box. That is the intended high-risk intervention. It
changes almost every portrait selected frame (53,171/53,456) and 52.4% of
landscape selected frames. On the selected frames, average normalized width
changes from 0.9995 to 0.9024 for target `16:9`, and from 0.3615 to 0.3066 for
target `9:16` when weighted by selected frames; average normalized vertical-
origin displacement is 0.0953 and 0.0161 respectively. The corresponding
video-macro widths are 0.9994→0.9049 and 0.3520→0.3363. Thus this is a genuine
spatial policy replacement, not a renamed R54 or a tiny geometry residual.

## Decision boundary

This package is the only remaining high-risk candidate worth an official slot
after R57/R59 and R67. R63/R64 are rejected as invalid mechanism tests because
their stable ordinal rank assigned different utilities to tied geometry values,
injecting a frame-index prior. R67 fixed that bug and scored 36.16 versus
R54's 36.22, weakening unlabelled geometry disagreement as a temporal signal.

R69 can move the official score substantially in either direction because the
official metric evaluates native per-frame crop IoU and R69 intentionally
removes V9's frame-level tracking. There is no native AIC crop GT, so a local
score prediction would be fabricated. The single available slot should be used
only if accepting this downside in exchange for testing the direct-grounding
hypothesis; otherwise R54 remains the expected-score choice.
