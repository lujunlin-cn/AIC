# R58 TimeLens mask + edge multi-view ensemble

- Status: packaged, not uploaded.
- Hypothesis: the generic TimeLens2 interval is a noisy event support mask; its interval endpoints carry onset/outcome evidence. Combining both views as small residuals around the exact centre-85 rank should preserve the strong budget prior while changing only event-boundary frames.
- Frozen coefficients: `eta_inside=0.20`, `eta_edge=0.50`; no official-label fitting or coefficient sweep.
- Spatial artifact: V9 full-frame B3 boxes.
- Inference parameter count: 8,854,531,425 (8.8545B), under the 9B rule if the V9 artifact is accepted.

## Public held-out proxy

Using the two R42 annotation CSVs and the frozen R47 TimeLens public source spans (18 views), centre macro F is 0.7044166126 and R58 is 0.7408776857 (`+0.0364610731`); 12/18 views improve and mean frame swaps are 2.06 per view. This is a mechanism diagnostic, not an official-score estimate.

## Reproduction

```bash
python scripts/r58_timelens_ensemble_probe.py \
  --labels reports/r42_annotation_v13_full_20261017/project1/frame_labels.csv \
           reports/r42_annotation_v13_full_20261017/project2/frame_labels.csv \
  --responses reports/r47_timelens2_public_proxy_20261017.json \
  --output reports/r58_timelens_ensemble_proxy_20261018.json
python scripts/r58_timelens_mask_edge_ensemble_package.py \
  --responses reports/r47_timelens2_full_426.json \
  --parent reports/r47_v9_parent_full_20261017.jsonl \
  --index reports/r47_timelens2_index_enriched_426.jsonl \
  --output submissions/R58_TIMELENS_MASK_EDGE_ENSEMBLE_V9_B3
python scripts/independent_submission_check.py \
  --index reports/r47_timelens2_index_enriched_426.jsonl \
  --submission submissions/R58_TIMELENS_MASK_EDGE_ENSEMBLE_V9_B3/predictions.jsonl
```

Package ZIP SHA256: `0e445cb4e7a36063e74171a16cc6ef5554ee8008dc9aa106a83ec17743e84101`
Prediction JSONL SHA256: `5716d6df0d4eeb8edf819546a860bfbd77b88e985b7d2b7cfd8eea1950622f3c`
