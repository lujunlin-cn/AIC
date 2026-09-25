# Representation Generalization Status (2026-09-25)

## Protocol boundary

The original seven-video TVSum split is now `comparison_holdout_v1`. It has
already influenced route selection and is not a pristine lockbox for new
model or post-processing choices. New TVSum claims must use repeated or nested
source-group CV. No AIC joint temporal/crop ground truth is available, so
`official_f_video` and `competition_score` remain null.

## Current evidence

| representation | bundle bytes | TVSum evidence | status |
|---|---:|---|---|
| A0 ResNet18 + Temporal U-Net | 25,685,169 | DEV macro proxy 0.151875; five-fold control 0.14631 +/- 0.03251 | engineering fallback |
| DeiT-S/16 frozen + same head | 46,618,447 | DEV 0.140241; five-fold 0.14389 +/- 0.05173; comparison holdout seeds 0.29187/0.22378/0.28128 | S-tier challenger, not replacement |
| ViT-B/16 frozen + same head | 174,986,447 | DEV 0.15928; comparison holdout 0.194505 | M-tier semantic reference |
| internal layer1 TSM | A0-sized | five-fold 0.13800 +/- 0.06299; paired delta -0.00830 | deprioritized |

These are TVSum temporal proxy values, not competition F_video. The existing
reports contain per-video calibration and seed diagnostics. Threshold-free
ranking evaluation is now implemented in `aic.temporal_metrics` with
Spearman, Kendall tau, NDCG, NDCG@15%, and top-budget average relevance. The
metrics are intentionally independent of the binary prediction threshold;
cached candidate predictions should be re-evaluated with this module before
any future route promotion.

## Interpretation

DeiT-S is not yet proven better than A0 on representation generalization. Its
large comparison-holdout advantage is compatible with split/category
composition and score calibration effects; it has no independent OOD raw-video
confirmation yet. ViT-B is useful as a semantic upper reference, but its
approximately 175 MB bundle requires a measured raw-score gain before its size
coefficient can be justified.

## OOD and spatial status

SumMe raw video remains blocked by an inactive LFS object. No cross-dataset
backbone claim is made. RetargetVid annotation/evaluator assets are available
remotely and have been audited: 200 videos, six annotators, 1:3 and 3:1
annotations, frame-wise coordinate files. DHF1K source videos are not yet
available in `/data/aic`, therefore spatial IoU for center, saliency, and
subject proxy remains null. The annotation audit is at
`/data/aic/experiments/retargetvid_annotation_audit.json`.

## Next evidence-gathering order

1. Run the new ranking metrics on identical cached A0/DeiT-S/ViT-B outputs.
2. Obtain a bounded raw OOD subset or a clearly labelled feature-level OOD
   benchmark.
3. Download a small DHF1K/RetargetVid video subset and score fixed temporal
   frames with center, saliency, and subject_proxy crops.
4. Only after those checks consider additional DeiT seeds or distillation.
