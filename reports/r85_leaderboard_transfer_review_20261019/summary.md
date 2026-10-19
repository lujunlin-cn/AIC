# R85 official leaderboard distribution and highlight-model transfer review

Date: 2026-10-19. This report uses the public semifinal leaderboard and public
papers only. It does not use test labels, test videos, submission internals, or
any score-derived training target.

## What the leaderboard actually shows

The public ranking endpoint returned 182 rows at the snapshot time. The score
distribution was:

| quantity | value |
|---|---:|
| maximum | 47.13 |
| minimum | 0.03 |
| median | 36.44 |
| score >= 45 | 4 |
| score >= 40 | 42 |
| 38 <= score < 40 | 27 |
| 34 <= score < 38 | 52 |
| score < 30 | 25 |
| nearest rank to our 36.43 | 92 |

The public endpoint is a ranking display, so low scores may include failed or
obsolete attempts. The useful comparison is that 36.43 is close to the middle
of the current table, while 40.14 is the score at rank 40. Reaching that band
requires roughly +3.7 score points over R71. The top tail is real, but the
distribution alone does not identify the mechanism.

The official formula allows an empty prediction list in the output contract,
but the public rules do not publish the fraction of videos whose hidden GT is
empty. Therefore we cannot claim that the top teams won by detecting empty
videos. If a fraction `q` of videos were truly empty and a baseline emitted
nonempty predictions on them, correcting only that gate would add at most
`100q` macro score points. A 10.7 point gap to 47.13 would require `q=0.107`
under the unrealistically favorable assumption that every other video stayed
unchanged. This makes an empty-video prior plausible as a high-value check,
but it cannot explain the leaderboard without evidence about `q`.

Our 18 manually reviewed public clips all contained at least one frame marked
2/3 by the V1.3 protocol. This is weak evidence for the PHD/GIF source policy,
not evidence about the official hidden set. It argues against making an empty
gate the primary route before a negative pool exists.

## What the literature says is transferable

The relevant papers do not all solve the same task:

* **Highlight-CLIP (arXiv:2404.01745)** fine-tunes a multimodal encoder and a
  saliency-pooling module on QVHighlights. It is a strong representation and
  pooling candidate, but QVHighlights is query-conditioned and therefore does
  not directly specify generic GIF value.
* **AHA (arXiv:2509.16421)** uses a causal VLM with decoupled relevance,
  informativeness, and uncertainty heads. Our R25 result (35.18) tested the
  released inference behavior without target-domain adaptation. It does not
  rule out using AHA as an initialization followed by PHD/GIF adaptation; it
  does rule out treating the released zero-shot score as a solution.
* **Cross-category VHD (arXiv:2108.11770)** explicitly studies transfer from a
  labeled source category to an unlabeled target category and uses set-level
  context rather than isolated pairwise ranking. This is the closest published
  transfer formulation for our domain-shift concern, but it needs target-like
  source labels and a source-disjoint validation protocol.
* **Audiovisual Highlight Detection (arXiv:2102.05811)** reports transfer from
  video summarization to highlight detection and an improvement from adding
  audio to visual features. Audio should be an auxiliary branch with a missing
  mask, not a replacement for visual evidence.
* **Less is More (arXiv:1903.00859)** uses video duration as weak supervision;
  **audio/visual recurrence (arXiv:2407.13933)** uses cross-video recurrence
  without frame labels. These are suitable pretraining signals for PHD/GIF
  selection, not final labels.
* **MH-DETR (arXiv:2305.00355)** and **Saliency-Guided DETR
  (arXiv:2410.01615)** are strong moment/highlight architectures, but their
  query-conditioned objective is a mismatch unless we remove the query branch
  and retrain on PHD/GIF labels.

The repository already contains the important negative evidence: QV transfer
was strong in-domain but weak on PHD2, MrHiSum pairwise training reached 32.74,
InternVideo2 event-completion scoring reached 34.95, and R25 AHA zero-shot
scoring reached 35.18. These close the naive recipe “download a high-scoring
paper model and run it unchanged”. They do not close target-domain adaptation.

## Revised bottleneck hypothesis

The leaderboard gap is more consistent with a **task-policy mismatch** than a
parameter-capacity gap. PHD/GIF selection is a creator/user decision over a
short clip. Our main temporal branches learned weak interval membership,
position, or generic event completion. They did not learn a calibrated
creator-style decision that distinguishes “the event exists” from “this is the
frame a GIF editor would keep”. Spatial improvements around R70/R71 cannot
repair that missing target policy.

The leading mechanism is therefore:

```text
PHD/GIF target supervision
  + high-level visual initialization
  + set-level / pairwise temporal adaptation
  + explicit no-highlight uncertainty head
  -> calibrated frame utility and video-level keep decision
```

This is a different route from swapping frozen encoders or scanning keep
values. It changes the representation, supervision use, and decision head at
the same target-aligned boundary.

## Next route and falsification test

The first adaptation experiment should use the already available AHA assets so
that a new download is not mistaken for a new mechanism:

1. Rebuild the PHD2 target manifest from source-bound intervals. Use union
   intervals as the positive selection target, retain `gif_soft` as a separate
   continuous target, and exclude the known `gif_hard` placeholder rows from
   hard-negative training.
2. Train a small temporal adapter and one AHA relevance adapter on the same
   source-disjoint train/dev split. Freeze the base first; then allow only the
   late visual/projector LoRA if the head-only pilot improves both ranking and
   target-like set utility.
3. Add a video-level abstention head only after constructing a negative pool
   from full non-GIF/background clips. Do not infer negative prevalence from
   the official score distribution.
4. Keep B3/R70 spatial boxes fixed in the first comparison. This isolates the
   temporal policy and avoids attributing a temporal gain to crop changes.

The promotion gate is source-macro joint F on a held-out PHD/GIF pool plus the
18 public human clips. A temporal ranking metric alone is insufficient. The
route is falsified if target-adapted AHA/HL-CLIP initialization does not beat a
source-disjoint PHD2 baseline with the same output budget and if its gain is
only a position prior. If it passes, package one fixed candidate using R54/R70
spatial outputs and reserve one official submission for that structural test.

The highest-value external model to obtain next is HL-CLIP or a compatible
CLIP saliency-pooling implementation, subject to license and checkpoint
verification. It should be compared against AHA adaptation under the same PHD2
labels; the comparison is about target transfer, not paper leaderboard scores.

Snapshot data is stored in `leaderboard_snapshot.json`; it can be regenerated
with `scripts/fetch_public_leaderboard.py`.
