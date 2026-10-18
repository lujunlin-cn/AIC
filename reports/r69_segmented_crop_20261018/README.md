# R69 segmented crop query

## Question

R69 asked TimeLens2 once on a complete video for one crop center. This probe
asks the same frozen 8.767B checkpoint separately on three equal temporal
segments of the already selected R54 temporal run. The R54 frame set is held
fixed; this experiment changes only the spatial box assigned to each selected
frame.

The selected run is split by original frame number, not by sampled-model
index. Each segment receives 5% of the selected run length as read context on
both sides. The video reader receives `video_start`/`video_end` in seconds
computed from the verified input fps. The response is cached verbatim before
parsing. A later package builder projects each valid response to the same
maximum legal target-ratio window as R69 and falls back safely for malformed
responses.

## Three-video smoke

Command (remote AutoDL, RTX 4080):

```bash
CUDA_VISIBLE_DEVICES=0 /root/autodl-tmp/aha_env/bin/python \
  /root/autodl-tmp/r69_segmented_crop_smoke.py \
  --model /root/autodl-tmp/evac/data/aic/pretrained/TimeLens2-8B \
  --video-root '/root/autodl-tmp/evac/data/aic/semifinal_test/复赛-基于视频大模型的通用视频高光剪辑/video' \
  --index /root/autodl-tmp/r47_timelens2_index_enriched_426.jsonl \
  --parent /root/autodl-tmp/R54_predictions.jsonl \
  --output /root/autodl-tmp/experiments/R69_SEGMENTED_CROP_SMOKE/r69_segmented_smoke.json \
  --video-ids 0 1 2 --segments 3 --context 0.05 --fps 1.0 \
  --max-pixels 50176 --max-new-tokens 128
```

The nine responses were all valid JSON. They were:

```text
video 0: [200,100,800], [200,100,800], [200,100,800]
video 1: [100,0,900], [200,0,800], [200,0,800]
video 2: [200,0,800], [100,0,1000], [0,0,1000]
```

The global R69 responses for the same videos were respectively
`[200,0,800]`, `[0,0,1000]`, and `[100,0,1000]`. Therefore the segment
queries are not merely returning a fixed whole-video answer: video 0 changes
the vertical center, video 1 changes the first segment's horizontal extent,
and video 2 moves from a narrower crop to a full-width crop over time. This
supports feasibility of a segment-wise package while leaving official spatial
gain unknown.

## Full cache

The full 426-video run is intentionally one resumable scalar process (the
checkpoint occupies most of the 32GB GPU; CPU video decoding is the current
throughput limit):

```bash
nohup /root/autodl-tmp/aha_env/bin/python \
  /root/autodl-tmp/r69_segmented_crop_smoke.py \
  --model /root/autodl-tmp/evac/data/aic/pretrained/TimeLens2-8B \
  --video-root '/root/autodl-tmp/evac/data/aic/semifinal_test/复赛-基于视频大模型的通用视频高光剪辑/video' \
  --index /root/autodl-tmp/r47_timelens2_index_enriched_426.jsonl \
  --parent /root/autodl-tmp/R54_predictions.jsonl \
  --output /root/autodl-tmp/experiments/R69_SEGMENTED_CROP/r69_segmented_full.json \
  --segments 3 --context 0.05 --fps 1.0 --max-pixels 50176 \
  --max-new-tokens 128 > /root/autodl-tmp/experiments/R69_SEGMENTED_CROP/run.log 2>&1 &
```

At report creation the cache had 31 videos/93 segment responses and was still
running. It contains one complete video row at a time and can be resumed by
rerunning the same command; completed video IDs are skipped.

## Package path

`scripts/r69_segmented_crop_package.py` consumes the completed cache and the
R54 parent. It produces a ZIP with exactly the parent frame membership and
records the number of replaced and fallback frames. A smoke package built
locally from the nine responses (using direct-R69 fallback for the other
videos) passed ZIP round-trip and retained 426/426 coverage; it is only a
builder test, not a submission candidate.

The builder counts TimeLens2 once (`8,767,123,696` parameters), does not use
official labels or scores, and never changes the temporal budget. The full
package should be built only after the cache reports 426 rows and zero missing
segment responses; its contract/checker output and SHA256 must be recorded
before considering a submission.

## Files

- `scripts/r69_segmented_crop_smoke.py`: scalar resumable query runner.
- `scripts/r69_segmented_crop_batch.py`: optional batched runner for future
  throughput tests; it has not been used for the full cache.
- `scripts/r69_segmented_crop_package.py`: legal projection and package
  builder with direct/parent fallback accounting.
- Remote cache: `/root/autodl-tmp/experiments/R69_SEGMENTED_CROP/r69_segmented_full.json`.
