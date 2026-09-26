# Scaling experiments (2026-09-26)

## Baseline and size economics

The first official raw scores are A0/center `1.01`, DeiT-S/center `6.08`,
and DeiT-S/YuNet `6.64`. Relative to the current 6.64 reference, an M-tier
candidate must reach raw `>6.98947` after the 0.95 coefficient and an L-tier
candidate must reach raw `>7.37778` after the 0.90 coefficient. The project
therefore treats 100–500M as a valid search region while preserving a <=100M
fallback.

## Available training data audit

The local package at
`/home/hajimi2025/datasets/data-challenge-2026/video_highlight` contains
129.308GB of compressed QVHighlights-derived shards and 129.988GB of extracted
clips. It has 987 weak seed annotations (887 train/100 val); 889 annotated
clips are present and 98 are missing. The labels carry
`seed_weak_training_label_v1` and a recorded Doubao seed model, so they are
not native human QVHighlights GT. Full details are in
`reports/20260926_official_train_audit.md`.

The frozen A0/DeiT releases used only 27 TVSum training videos (about 6,732s,
187,216 sampled source frames, 366,499,017 source bytes), approximately 0.283%
of the compressed QVHighlights-derived archive bytes. This makes data scale a
real hypothesis, while label quality and domain mismatch remain confounds.

## First controlled data-scale probe

This probe holds DeiT-S frozen features, the Temporal U-Net, BCE, seed,
validation set and 20-epoch budget fixed. It changes only the deterministic
prefix of the 27-video TVSum train manifest (7, 14, or 27 videos); the 16-video
validation set is unchanged. The values below are video-macro F1 at the
trainer's fixed diagnostic threshold 0.5, not AIC F_video:

| run | train videos | best dev macro F1@0.5 | best epoch | wall seconds |
|---|---:|---:|---:|---:|
| DATA_SCALE_20260926_D25 | 7 | 0.0527846 | 16 | 10.95 |
| DATA_SCALE_20260926_D50 | 14 | 0.0421875 | 17 | 13.78 |
| DATA_SCALE_20260926_D100 | 27 | 0.1214247 | 11 | 19.43 |

This first curve is noisy and non-monotonic because it uses one TVSum split and
one seed. It is evidence that the current sample is too small for a smooth
scaling claim, not evidence that adding data hurts. The next data-scale run
must use source-group folds and the QVHighlights weak-label protocol before a
model is promoted.

## QVHighlights-derived frozen DeiT pilot

A deterministic 37-train/9-val subset (46 matched clips, about 516MB media) was
transferred to `/data/aic` and is undergoing 2 FPS DeiT-S extraction. Labels
are frame-indexed seed temporal segment scores; no test directory is read.
The matched 40/10 pilot (37/9 after missing-source filtering) completed on
the remote V100s. With the timeline-linear label protocol, frozen DeiT-S plus
the same Temporal U-Net reached weak-label validation macro F1 `0.67846` and
Spearman about `0.393` in 29.25 seconds. A matched ResNet18 control reached
macro F1 `0.61780` and Spearman `0.395` in 23.75 seconds. The F1 gap is not a
ranking gain (Spearman is effectively unchanged), so it is evidence of score
calibration/label interaction rather than a proven semantic superiority. These
are not official AIC metrics.

The same quick split included ViT-B/16: macro F1 `0.60388`, Spearman `0.39421`,
and 21.63 seconds of temporal-head training after extraction. It was below both
DeiT-S and ResNet18 on this weak timeline proxy. This does not rule out ViT-B on
the official evaluator, but it removes quick-pilot evidence for automatic
capacity promotion.

A second subset used segment-score labels and produced a different failure
mode: both frozen and last-block controls had validation F1 `0.0` at 0.5 and
Spearman around `0.21`; the 2-epoch last-block run had validation BCE `0.04689`
versus `0.04573` for the frozen control. This confirms that the exact weak-label
definition dominates the apparent scale result. Both protocols are retained
and explicitly versioned; they must not be mixed.

## Model scale and finetuning status

The existing ViT-B/16 bundle is the first capacity reference (87,462,401
parameters, 174,986,447 persisted bytes including the temporal head). Under the
user parameter-count interpretation it is below 100M; under the repository byte
table it is M-tier. Its local TVSum and QVHighlights quick evidence has not
established a stable advantage over DeiT-S. A fair official diagnostic candidate
remains ViT-B + the same YuNet spatial protocol, but it is not yet a promoted
winner.

The bounded raw-video probe is now complete. Unfreezing the final DeiT block
added no persistent inference parameters, but on the segment-score protocol it
did not improve validation loss or F1. This is a rejection of this single
2-epoch/37-video finetuning hypothesis, not a rejection of full-data
task-specific adaptation.

## Continuation corrections

The local asset is not organizer-provided training data (explicit user clarification). The timeline quick pilot metrics .678463/.617799/.603881 belong to **40 train / 10 val**, verified from remote manifests; 37/9 belongs to a separate segment-score pilot. Direct comparisons across these label protocols are invalid. Full 800/89 split has 23 shared original source IDs; use QVH_SOURCE_PURGE_V1 (775/89) before scaling. Current 40/10 quick split is source-disjoint.

VideoMAEv2 first normalization probe is invalid (0.5 normalization instead of model-declared ImageNet statistics); a new version with timestamps and matched clip controls is running. InternVideo2-1B passed standard-kernel FP16 V100 compatibility only; see `20260926_internvideo2_compatibility.md`.
