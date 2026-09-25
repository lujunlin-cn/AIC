# 2026-09-25 Spatial group observation and OOD acquisition update

## Scope

This continuation did not reopen TVSum, SumMe, TSM, Feature Bank v1, or old postprocessing. It tested one concrete spatial hypothesis and began a bounded independent-data acquisition path. `official_f_video` and `competition_score` remain `null`.

## Fixed spatial hypothesis

On the already exposed 20 DHF1K videos with RetargetVid human crop annotations (two target ratios, six raters), `true_face_group` selects the weighted center of up to three YuNet faces by area times confidence. `true_face_group_smooth` adds the already frozen EMA alpha `.25`. Detector threshold, crop width, shot reset, source IDs, and evaluation code were unchanged. This is a development comparison, not a new lockbox.

The group methods scored:

| method | 1:3 IoU | 3:1 IoU |
|---|---:|---:|
| true_face_group | .48004 | .75115 |
| true_face_group_smooth | .48106 | .75216 |

`true_face_group_smooth - true_face_smooth` paired source-video delta was `-.00264`, bootstrap 95% CI `[-.00726, 0]`, with zero positive sources out of 20. Relative to center it was `+.00590`, CI `[-.01690, +.02739]`. On the known multi-person failure 020, group+EMA fell from `.48228` to `.44026`. The rule is rejected; the explicit modes, source hashes, JSONL validator, and raw E2E checks remain.

The group raw path was checked on video 020: both group modes emitted 566 legal predictions and matched the fixed benchmark crops exactly. CPU-only E2E support was added to `verify_dense_e2e.py` so this integration check does not require an authorized GPU.

## New temporal OOD acquisition

A Range-capable indexer `scripts/index_youtube_highlights.py` is running against the published `jhanglee/youtube-highlights-full` tar. The archive metadata reports 417 MP4s, 315 human MTurk and 102 weak-match sources, plus explicit alignment warnings. Only small metadata files are being extracted during indexing; no model selection or training has used this data. The 9.9GB media tar is not silently downloaded wholesale.

The data source is a domain-specific highlight benchmark, so any future result will use its native segment/MTurk protocol and a frozen model comparison. It will not be relabeled as TVSum F1 or AIC F_video. If acquisition exceeds its bounded time budget, the failure and partial index stay recorded.

## Status

- A0 center remains engineering fallback.
- DeiT-S remains TVSum challenger only.
- `true_face_smooth` remains a spatial exploration result, not a replacement.
- No VLM pilot, no official AIC joint GT, and no competition score claim.
