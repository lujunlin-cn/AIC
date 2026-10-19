# R54/R46 package difference audit (2026-10-18)

This audit compares the committed prediction artifacts for R26, R44, R46-local, R54 and the existing V9 center control. It uses frame IDs exactly as serialized in each `predictions.jsonl`; `n_full` is read from the R54 manifest. It does not use official test labels or scores to infer frame-level truth.

## Official score records and artifact identity

| package | repo status | reported score | prediction SHA | ZIP SHA |
|---|---|---:|---|---|
| R26_POS_CENTER_EXACT085_B3 | user-reported/uploaded | 36.04 | `4af2d6ae656450936577a5cd53ed0368986b5016a53640a2b37483034e2256a4` | `95d079da9b1f845e967509dcb5a3bebc057e8501b582cdc4858af4fc7b60a981` |
| R44_HUMAN_CALIBRATED_RESIDUAL_ETA020_B3 | user-reported/uploaded | 36.09 | `209ad1472dbb160daf91976ad9d317f59e42d9dc104d03b533887d19998a397f` | `b814699f1bdc71f2fec80e8e0e9465fe600327d44eab731f41f4d75ff6deac9e` |
| R46_HIGHFREQ_LOCAL_RESIDUAL_B3 | package only | user reports 35.88; no in-repo binding | `ebdf0f1869a83ec425a52c127535cd341e0ea5a546221e4dc4997f46c39004dc` | `be52cfd428ab53d5c87c29aedd7c331bc803ebef4accd3915522d8e944ce1b3a` |
| R54_TIMELENS_CENTER_RESIDUAL_V9_B3 | package only | user reports 36.22; no in-repo binding | `62371bc54d62b72eafd7168898ab5b23fad79a6c252ebde458b8fc78d033983e` | `1e0f92f2292da0c057aa9b45a27134783af4def37a60c95ffde6d8fa9a8ae210` |

R46/R54 `manifest.json` still says `PACKAGED_NOT_UPLOADED`; the 35.88/36.22 values therefore remain user-reported until platform score, upload SHA and package identity are recorded together.

## Selection-set changes

All four packages contain 426 videos and 199,037 predictions, with exact per-video 85% counts. Symmetric swaps means the number removed from the parent (equal to the number added when counts match).

| comparison | changed videos | swaps | mean swaps/video | mean Jaccard |
|---|---:|---:|---:|---:|
| R44 vs R26 | 419 | 6,352 | 14.91 | 0.93836 |
| R46-local vs R26 | 424 | 5,636 | 13.23 | 0.94754 |
| R54 vs R26 | 417 | 6,152 | 14.44 | 0.94758 |
| R54 vs R44 | 416 | 8,508 | 19.97 | 0.92259 |
| R54 vs R46-local | 411 | 7,955 | 18.67 | 0.92765 |

R54 and R46 therefore alter similarly small portions of the fixed center set, but they are different perturbations rather than nested versions of one another.

## Location and interval statistics

For frame position `p = frame/(n_full-1)`, selected mass is almost exactly centered for R26 (`mean .49957`), R44 (`.49717`) and R46 (`.49802`). R54 is materially later (`global mean .52344`, per-video mean `.51904`). Its mean per-video displacement from R26 is `+.01974` (median `+.02174`), and displacement correlates with the number of swaps (`r=.500`). This is the direct observable footprint of the TimeLens2 interval prior; it is not a random symmetric perturbation.

R54's manifest reports 164,591 full frames inside the TimeLens2 intervals and 152,325 selected frames inside intervals. Thus 76.53% of all R54 selected frames lie inside an interval (mean per-video fraction .7622, median .8427). This interval statistic does not imply those frames are true highlights; it only describes the decoder's behavior.

R54's mean swaps per video rise with video length: 1.88 (`n<200`), 6.35 (`200-400`), 14.51 (`400-800`), 32.72 (`800-1600`), and 83.31 (`>=1600`). Position displacement is likewise +.0051, +.0168, +.0230, +.0302 and +.0264. Long videos receive the strongest temporal intervention and are the most informative subgroup for any future score interpretation.

## Spatial confound and control

R44 and R46 have **zero** box-coordinate differences from R26 on every shared frame. Their official score deltas isolate temporal selection (under the fixed B3 parent contract), subject to the usual score binding caveat.

The existing `R47_V9_B3_CENTER85` control has exactly the same frame set as R26 but changes V9 spatial boxes on 28,383/199,037 frames (155 videos; maximum coordinate difference 143.45). R54 uses the same V9 boxes as this control: on shared R54 frames there are zero V9-vs-R47V9C box differences. Consequently, R54's score delta relative to R26 cannot be attributed to TimeLens2 alone. An official score for `R47_V9_B3_CENTER85` would identify the spatial component; an official score for a package with R54 time and R26 B3 boxes would identify the temporal component.

## Interpretation

The user-reported R54=36.22 is +.18 over R26=36.04, while R46=35.88 is -.16. The artifact audit supports these statements:

1. R54's intervention is semantically directed and late-biased, not a near-symmetric random swap. Its gain is consistent with useful event information, but the V9 spatial branch is a co-intervention.
2. R46's local residual leaves boxes identical to R26 and shifts position only -0.00227 on average; its negative score is evidence against deploying this low-level 4fps state residual on the official domain, not evidence against temporal content in general.
3. The next high-value separation is a score pair: R47-V9-center (same R26 time, V9 boxes) and R54 with R26 B3 boxes (same TimeLens2 time, old space). Without that pair, changing TimeLens2 and spatial boxes together risks overclaiming the source of +.18.

## Swap anatomy (important qualification)

The set difference shows why R54 moves later. Relative to R26, all 6,152 removed frames are early boundary frames (normalized position mean `.1294`, median `.0917`, 90th percentile `.1129`; only 276 are after `.5`). All 6,152 added frames are late boundary frames (mean `.9018`, median `.9412`, 90th percentile `.9607`; 5,858 are after `.5`). Therefore the fixed rank residual behaves primarily as a right translation of the center-85 window. Among these swaps, 4,766 added and 3,102 removed frames fall inside a TimeLens2 interval; 1,386 added and 3,050 removed are outside.

This weakens a naive claim that R54 identifies the exact event frames. The official +.18 may reflect a useful correction to a global early/late bias, an event-position prior, or V9 space; the current artifact cannot distinguish these. The next temporal control should use the same V9 boxes and an explicitly matched late-window translation (or compare R54 to a package with old B3 boxes) before attributing the change to event semantics.

## R55 phase control (available artifact)

The existing `R55_TIMELENS_EVENT_PHASE_V9_B3` replaces the binary interval rank with an explicit normalized completion phase while retaining the same V9 boxes, center anchor, and eta. It differs from R54 by only 119 swaps across 19 videos (`Jaccard .99953`, mean position difference `2.5e-5`), while differing from R26 by 6,109 swaps. Its public proxy is identical to R54. This makes an accidental within-interval rank ramp an unlikely explanation for the broad R54 behavior; both decoders are dominated by the same per-video late-window translation.

## Public fixed-shift diagnostic

On the existing 18 manually labeled public views, a pre-declared fixed center-window shift of +2 seconds (same 85% count) scores `0.736923` soft proxy (`+0.032506` over center `0.704417`), while R54 scores `0.729414` (`+0.024997`). The +2 s rule improves 10/18 views, the same count as R54. Its mean frame-set Jaccard with R54 is `.943` (median `.926`), so it is related to but not identical with the interval-driven decoder. This is a mechanism diagnostic only: the labeled views are small and cannot estimate official score. It motivates a fixed, predeclared late-window control to determine whether TimeLens2 contributes beyond phase shift.

For comparison, R46-local is structurally different: 5,636 swaps are diffuse (`removed` normalized-position mean `.5004`, median `.143`; `added` mean `.4457`, median `.0709`) and its selected sets are non-contiguous in 138/426 videos (mean 7.77 internal gaps). R44 is also non-contiguous in 144/426 videos, with late removals (median `.888`) and early additions (median `.071`). The coherent contiguous-window behavior is specific to the TimeLens2 R54 residual and is a more plausible reason it survives official scoring than the low-level frame residuals.

Available controls reinforce this interpretation: `R57_FIXED_LATE2S_V9_B3` is a fixed +2 s shift and moves the full output by `.0713` normalized position (14,015 swaps vs R26), far stronger than R54's `.0197` (`6,152` swaps). Its public proxy is higher, but the official test is unknown. `R56` edge residuals move only `.0038` and are unlike R54; they should not be treated as equivalent semantic alternatives.
