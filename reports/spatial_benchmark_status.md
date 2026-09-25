# Spatial Benchmark Status (2026-09-25)

RetargetVid annotations and the upstream evaluator are present on the remote
server under `/data/aic/datasets/RetargetVid`. An annotation-only audit found
200 videos, six annotators, both target aspect ratios (1:3 and 3:1), and
consistent frame-wise text records. The machine-readable result is
`/data/aic/experiments/retargetvid_annotation_audit.json`.

No DHF1K source videos are currently present in `/data/aic`, so this stage does
not report IoU. `spatial_iou = null` is intentional. Existing center,
saliency, and subject modes are legal-output pipelines and can be measured for
trajectory stability, but saliency is not a ground-truth metric. The next
spatial experiment requires a bounded source-video subset, fixed temporal
frames, and the official RetargetVid coordinate convention.
