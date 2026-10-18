# R69 segmented crop stability gate

## Hypothesis

Independent segment queries can improve spatial framing when the crop centre
changes smoothly, but a single unstable segment can create a large native-frame
IoU loss.  A run-level gate should keep only coherent segment outputs and use
the complete-video crop for every frame of an unstable run.

The gate is fixed before any official score is available:

```text
max_global_delta = 0.12
max_run_spread   = 0.12
```

The values are in normalized legal free-axis centre coordinates.  TimeLens
responses are mostly quantized in 0.1 steps, so the rule permits one step from
the complete-video answer and rejects larger disagreement.  A run is accepted
only when all of its segment centres satisfy both limits.  This avoids an
isolated middle or endpoint jump being released independently.

## Construction and validation

Command:

```bash
python scripts/r69_segmented_stability_gate.py \
  --parent submissions/R54_TIMELENS_CENTER_RESIDUAL_V9_B3/predictions.jsonl \
  --index reports/r47_timelens2_index_enriched_426.jsonl \
  --segments reports/r69_segmented_crop_full_20261018.json \
  --direct-responses reports/r69_timelens_direct_spatial_responses_20261018.json \
  --output submissions/R69_SEGMENTED_GATE_R54TIME
```

The output retains all 426 videos and 199,037 R54-selected frames.  There are
427 runs (video 373 has two runs): 375 stable and 52 unstable.  The unstable
runs use the direct complete-video crop for 33,378 selected frames.  The output
has 111,306 boxes different from the R54 parent, 265 changed videos, and is
28,302 frames different from the ungated R69 segmented package (52 videos).
The package uses 8,767,123,696 parameters and has no fallback caused by a
malformed current response.

On the legal free-axis centre sequence, the gate reduces adjacent segment
total variation from 22.4604 to 11.9998 (46.5% reduction), and the 95th
percentile adjacent jump from 0.2000 to 0.1367.  The largest-jump 95th
percentile falls below one 0.1 quantization step; this is a structural effect
of the gate, not an estimate of native crop IoU.

Local checks:

* `independent_submission_check_v1`: valid, 426 videos, 199,037 predictions;
* `aic.contract` final validation: valid, 8.767123696B parameters;
* ZIP round-trip prediction SHA: valid;
* ZIP SHA256: `f7d17806697bd677580fcddda2fe02c441c1dacb3c7f979cd000823a57c5cf58`.

## Interpretation limits

This is a structural diagnostic and an upload candidate, not evidence of an
official gain.  No native crop GT exists for the 426 semifinal videos, so the
gate cannot be selected by official回分 or by a fabricated local score.  Its
mechanism prediction is that it should reduce abrupt spatial errors while
retaining coherent segment changes; the official metric has no smoothness
reward, so the gate can still lose if the global crop is systematically wrong.
