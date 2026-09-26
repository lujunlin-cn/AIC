# Native QVHighlights annotation acquisition (2026-09-26)

The public Moment-DETR release files were acquired locally through the existing network path:

- `highlight_train_release.jsonl`: 7,218 query records
- `highlight_val_release.jsonl`: 1,550 query records
- fields include `vid`, query text, relevant windows, relevant clip IDs, and three-annotator saliency scores

These are public QVHighlights human/query annotations, not an AIC joint temporal+crop evaluator. The local challenge-derived asset contains 11,245 extracted clip files and 889 annotation-linked clips. A filename stem audit found 7,738 public `vid` values also present in the local extracted tree. This is only an ID intersection: media duration/segment semantics and licensing still require validation before alignment. No label alignment or training was performed. The files are retained outside the Git repository under `/home/hajimi2025/datasets/data-challenge-2026/qvh_native_annotations/`.

This acquisition demonstrates a viable source for a future OOD/native temporal benchmark, but it does not yet provide raw matching videos. No official test video was accessed.
