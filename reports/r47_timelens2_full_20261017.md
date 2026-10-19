# R47 TimeLens2 full inference and package audit (2026-10-17)

## What was run

TimeLens2-8B was run on all 426 competition videos with the fixed generic
highlight prompt in `scripts/r47_timelens2_official.py`, 2 fps input, a
320×320 per-frame pixel cap, deterministic decoding (`temperature=0.01`,
`top_p=0.001`, `top_k=1`), and no competition labels. Nine non-overlapping
shards (0–49, …, 400–425) were merged with
`scripts/merge_r47_timelens2.py`. The merged response file contains exactly
426 unique IDs and 426 non-empty responses:

```text
reports/r47_timelens2_all_20261017.json
SHA256 a48f1cd1820bd5e526801f5c9cf1b580383405f235adbc3f137c458c63be09cb
```

## Packages

Both packages use the exact `round(0.85 * frame_count)` budget (199,037
predictions) and are independently checked against the 426-video index.

| Package | Spatial decision | Total parameters | ZIP SHA256 | Eligibility |
|---|---|---:|---|---|
| `R47_TIMELENS2_CENTER_GEOMETRY` | deterministic legal center crop | 8,767,123,696 (8.767B) | `09429700c0fe201338f13c10f772e1be5ed56d68bd8cad32537a40ce7170665e` | parameter compliant |
| `R47_TIMELENS2_V9_ARTIFACT_TRUE` | inherited V9 vision-only spatial artifact | 8,854,531,425 (8.855B) | `c22d7734423240d7564ef5366308715e7afa56e89ff5f82f9b9b9c3801be1aa8` | compliant only if V9 artifact counts as deployed component |

The V9 artifact package uses the actual V9 full-frame parent (parent SHA
`95b9219676a5eece74fbce7bc108d08ba2b84759d0f754bf5660cad9a44c8e8a`) and
is retained for review because its boxes are copied from a prior spatial run;
it must not be described as a zero-parameter crop.
The center package is the unambiguous parameter-compliant candidate.

## Evidence and limits

The 12-source public diagnostic previously gave TimeLens2 interval decisions a
macro proxy of 0.721327 versus 0.704417 for the center control (+0.01691;
6/18 views improved). That result is source-held-out public evidence only. No
official score, official test label, or score-derived tuning was used here.
On the 426 package, TimeLens2 changes the selected frame set relative to R26
on 197 videos; the mean number of swaps is 11.28, median 0, and maximum 620.
This is a genuine temporal intervention, so its official result will identify
whether the public semantic signal survives domain transfer. Center geometry
may erase the temporal gain through spatial IoU; the V9 package isolates that
interaction if its artifact accounting is accepted.
