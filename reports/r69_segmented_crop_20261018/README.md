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

The initial scalar run was stopped after preserving 59 complete videos. A
batched runner was then smoke-tested on video 59 (three responses, no OOM) and
resumed the same cache. The final cache contains **426 videos and 1,281 valid
segment responses** (video 373 has two selected runs, hence six segment
queries). Host memory remained below 50 GiB and GPU memory around 17.1 GiB on
the 32 GiB RTX 4080. The cache contains one complete video row at a time and
is resumable; completed IDs are skipped.

The resumed command used the batched runner (the output path is the same, so
the 59 scalar rows were retained):

```bash
/root/autodl-tmp/aha_env/bin/python /root/autodl-tmp/r69_segmented_crop_batch.py \
  --model /root/autodl-tmp/evac/data/aic/pretrained/TimeLens2-8B \
  --video-root '/root/autodl-tmp/evac/data/aic/semifinal_test/复赛-基于视频大模型的通用视频高光剪辑/video' \
  --index /root/autodl-tmp/r47_timelens2_index_enriched_426.jsonl \
  --parent /root/autodl-tmp/R54_predictions.jsonl \
  --output /root/autodl-tmp/experiments/R69_SEGMENTED_CROP/r69_segmented_full.json \
  --segments 3 --context 0.05 --fps 1.0 --max-pixels 50176 --max-new-tokens 128
```

## Package path

`scripts/r69_segmented_crop_package.py` consumes the completed cache and the
R54 parent. It produces a ZIP with exactly the parent frame membership and
records the number of replaced and fallback frames. A smoke package built
locally from the nine responses (using direct-R69 fallback for the other
videos) passed ZIP round-trip and retained 426/426 coverage; it is only a
builder test, not a submission candidate.

The completed package is:

```text
submissions/R69_SEGMENTED_CROP_R54TIME/R69_SEGMENTED_CROP_R54TIME.zip
SHA256 0b1de985c709b2b4d31d2c1a57b3d0420a3f66afb9dbad82605ff534e10a8010
```

It has 426/426 videos and 199,037 predictions, with 111,306 spatial boxes
replaced and zero fallback frames. The independent checker and local contract
validator both pass; the exact R54 frame sequence is preserved for every
video. The package is `PACKAGED_NOT_UPLOADED` and has no official score yet.

The builder counts TimeLens2 once (`8,767,123,696` parameters), does not use
official labels or scores, and never changes the temporal budget. The recorded
checker and contract outputs are local validation only; an official score is
needed to decide whether this route is worth a submission slot.

## Files

- `scripts/r69_segmented_crop_smoke.py`: scalar resumable query runner.
- `scripts/r69_segmented_crop_batch.py`: optional batched runner for future
  throughput tests; it has not been used for the full cache.
- `scripts/r69_segmented_crop_package.py`: legal projection and package
  builder with direct/parent fallback accounting.
- Remote cache: `/root/autodl-tmp/experiments/R69_SEGMENTED_CROP/r69_segmented_full.json`.
