# R9 metric-audit bundle

CPU-only reference for the equal-cost, binary, fixed-cardinality temporal metric.
This is not a model-training result and is not an official joint-score evaluator.

Requirements: Python >=3.10 and NumPy.

Synthetic tests:

    python metric_sensitivity_audit.py --self-test --out self_test_results.json

Permitted PUBLIC labeled data only, with the actual decoder K:

    python metric_sensitivity_audit.py --train train_public.jsonl --eval dev_public.jsonl --n 8 --k 6 --out audit_k6.json

Each JSONL row: source_id, fragment_id, labels (binary array), scores (finite
array). Train/eval sources must be disjoint. K is explicit, not inferred from a
nominal keep ratio. Position priors use train labels only. Evaluation labels
are for metrics and diagnostic oracles only, never deployment decisions.

The CI resamples entire source clusters and preserves fragment-macro. It does
not quantify checkpoint selection, seed selection, target shift, or uncertainty
in the official test labels. Unequal frame costs require a different evaluator.

The synthetic inputs are not AIC data. Real experiment statuses are NOT_RUN.
