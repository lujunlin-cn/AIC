# Dataset provenance and preparation

This file records what can legally and reproducibly enter training. It is not a
claim that the source media is owned by this project. A repository/data-card
license covers code, annotations, or packaging only when it says so; the
underlying YouTube media keeps its uploader/platform rights and must retain the
source attribution and terms. We do not use a dataset merely because a mirror
labels it `MIT`, `GPL`, or `CC`.

## TVSum (first acquisition)

- **Purpose:** 50 YouTube videos with 2-second shot importance scores from 20
  crowd workers. This is a video summarization/importance proxy. It is not the
  AIC competition's per-frame highlight + composition ground truth, and no crop
  labels are synthesized from it.
- **Version pinned:** `tvsum50_v1.1_2019-11-06` (author package timestamp
  2019-11-06).
- **Author source:** [Yale Song TVSum README](https://github.com/yalesong/tvsum)
  and direct package
  [`tvsum50_ver_1_1.tgz`](https://people.csail.mit.edu/yalesong/tvsum/tvsum50_ver_1_1.tgz).
  The author page reports a 641M package (HTTP `Content-Length` is currently
  671,779,858 bytes); the downloaded file's SHA-256 is recorded in the manifest
  summary after completion.
- **Source-media statement and license gate:** the package README says the
  collected YouTube videos came with a Creative Commons (CC-BY) license. The
  same archive includes `WebscopeReadMe.txt`, which is a stricter Yahoo
  Webscope/Data Sharing Agreement notice: use is only for approved
  non-commercial academic research by a signed-agreement recipient; commercial
  use, redistribution, network storage and archiving are prohibited. These are
  source/dataset terms, not merely a GitHub code license, and the YouTube
  uploader's rights remain separate. The project user has authorized using
  datasets and model files that are downloadable in this workspace; we retain
  the conflict and terms hash for provenance, while the default experiment
  path follows that authorization. `download_status=verified` means the bytes
  and metadata were checked; `--strict-license-gate` is available when a run
  must enforce the manifest gate independently.
- **Annotation:** `ydata-tvsum50.mat`, `user_anno` (20 rater columns; shot-level
  importance). The preparation script keeps `annotation_type` as
  `summary_importance_2s`; downstream code must use an explicit task mask and
  must not reinterpret it as composition/crop GT.
- **Splits:** `scripts/acquire_tvsum.py` uses a SHA-256 hash of
  `aic-tvsum-v1:tvsum:<video_id>` to assign groups deterministically to train,
  val, and test (70/20/10). Every occurrence of the same `source_group` gets
  one split. A future cross-dataset source-ID map must override this assignment
  before merging datasets.
- **Manifest:** JSONL, one record per source video. Required provenance includes
  dataset/version/video/source group, source URL, license URL, download status,
  explicit `license_gate`, SHA-256, ffprobe duration/fps/frame count/dimensions/rotation/audio, split,
  annotation type and annotation path. Missing or failed videos stay visible as
  `missing`/`failed`; they are never silently trained. Records retain their
  explicit `license_gate` value in experiment manifests.

Run (on a machine with the already downloaded package):

```bash
python scripts/acquire_tvsum.py \
  --archive /data/aic/datasets/TVSum/downloads/tvsum50_ver_1_1.tgz \
  --output-root /data/aic/datasets/TVSum/raw \
  --manifest /data/aic/datasets/TVSum/tvsum_manifest.jsonl \
  --extract
```

A metadata-only run can still produce an auditable record if extraction has
failed, but it must not be passed to training. The package URL is an HTTP
source that was reachable during this audit; if it becomes unavailable, retain
any partial file as `.part`, resume via the range downloader, and verify the
SHA-256 before extraction.

## Planned sources and gates

| Source | Intended capability | Current gate | Use now |
|---|---|---|---|
| VideoXum | larger summary supervision | recover original ActivityNet media and verify rights per source | no |
| RetargetVid + DHF1K | reframing/saliency auxiliary labels | preserve each author's separate terms and source IDs | no |
| LIVE-YT-VideoCropping | direct crop preference | dataset license/download still unresolved | no |
| QVHighlights / TimeLens | query-conditioned temporal auxiliary labels | do not remove query and call it generic highlight GT | no |

Competition test videos and labels are never downloaded into these datasets,
never sent to a teacher/API, and never used to construct a split.
