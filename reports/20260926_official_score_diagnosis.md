# Official score diagnosis (2026-09-26)

The first platform feedback is recorded as an external official result:

| submission | raw official score | loaded weight bytes | spatial mode | selected frames | empty videos |
|---|---:|---:|---|---:|---:|
| SUB_A | 1.01 | 25,685,169 | center | 1,809 | 150/174 |
| SUB_B | 6.08 | 46,618,447 | center | 8,133 | 97/174 |
| SUB_C | 6.64 | 46,851,036 | YuNet `true_face_smooth` | 8,133 | 97/174 |

The platform values are raw initial scores. All three releases are in the S
size tier under the current working interpretation, so their reported and
size-weighted values are equal (`k_size=1.00`). The M and L break-even raw
scores relative to SUB_C are `6.64/0.95 = 6.98947` and
`6.64/0.90 = 7.37778`.

## What actually changed

SUB_A and SUB_B both use the same 27 TVSum train videos, 16 validation videos,
2 FPS sampling, BCE temporal supervision, 20 epochs, seed `20260925`, and the
same Temporal U-Net family. Both visual encoders are frozen ImageNet
representations. A uses a 512D ResNet18 embedding; B uses a 384D DeiT-S/16
embedding. The historical prediction thresholds differ (`0.40` and `0.35`),
and the two feature extraction/training entry points differ in AMP usage, so
the `1.01 -> 6.08` gap is a strong system-level signal but not a pure parameter
count intervention. The cached frame indices, timestamps, masks and labels
were mechanically compared and are equal across the two representations.

SUB_B and SUB_C use the same DeiT temporal bundle and threshold. Their frame
lists are identical for all 174 test videos and all 8,133 selected frames;
8,130 of the 8,133 paired boxes differ. The isolated raw gain `6.08 -> 6.64`
(`+0.56`, `+9.21%`) is therefore evidence for the YuNet face-based spatial
observer under the official evaluator. It is not evidence that a generic
person/object crop model is solved: YuNet is face-focused. SUB_C adds 232,589
bytes and increases runtime from 742.764s to 1,959.029s because it observes
all decoded frames.

## Interpretation and limits

The official result establishes a large representation/domain interaction in
favour of the frozen DeiT release and a reproducible spatial contribution in
the B/C differential. It does not identify whether the A/B gap comes from
backbone semantics, threshold calibration, feature distribution, or the
TVSum-summary-to-AIC label mismatch. The A0 and DeiT local TVSum ranking and
SumMe evidence was mixed before this score, so no single local proxy should
be used to explain the platform ordering.

The three releases did not use the local QVHighlights-derived challenge
package. That package is now audited separately as weak seed supervision and
is being tested only on a deterministic training subset. No official test
video was used for training, tuning, or label construction.

## Size accounting note

The repository rules document records the detailed coefficient table as
actual loaded weight files, while the latest project instruction describes
the practical tiers in parameter-count language. We retain both fields in
every candidate manifest (`parameter_count` and `loaded_weight_bytes`) and do
not alter the source rules document. Under either interpretation, A/B/C are
below 100MB/parameters; future M-tier candidates must be reported with both
measurements before submission.
