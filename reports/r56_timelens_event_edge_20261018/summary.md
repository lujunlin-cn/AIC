# R56 TimeLens2 event edge residual (2026-10-18)

## Mechanism

R54 uses `rank(center)+0.20*(rank(binary_interval)-0.5)`. R56 replaces the implicit stable frame-order ramp from the binary mask with an explicit edge prior. For each predicted interval `[a,b]`, frame phase is `p=(t-a)/(b-a)`, and

```text
q(t) = max_j 2*|p_j(t)-0.5| * I[a_j <= t <= b_j]
score(t) = rank(center(t)) + 0.20*(rank(q(t))-0.5)
```

The prior rewards both an onset and a completed outcome near interval boundaries, while the static middle receives little residual. For multiple intervals the maximum edge evidence is used. Center 85% budget, TimeLens2 responses and V9 B3 spatial artifact are unchanged.

## Public diagnostic

Using the existing 12-source/18-view human proxy:

| rule | soft proxy | binary proxy |
|---|---:|---:|
| center | 0.704417 | 0.710385 |
| R54 | 0.729414 | 0.743255 |
| R56 edge | 0.731110 | 0.745839 |

R56 is `+0.001696` soft and `+0.002584` binary over R54; edge beats center on 11/18 views, loses on 2, ties 5. The aggregate is dominated by one view changing (`+0.03053`), so this is weak diagnostic evidence, not an official-score forecast.

## Package audit

- `426/426` videos, `199,037` predictions, exact `round(.85*N)` budget.
- Independent checker and local semifinal contract passed.
- Parameters `8,854,531,425` (`TimeLens2-8B + V9 vision-only B3`), under 9B.
- ZIP SHA256: `43845ef4085e65985c768a7c9391e895d28ed0c89f0d04ef69bc3801da96f455`.
- Prediction SHA256: `1ad27b800b7e58e013ee230ad85b47c5d86b7b24792c932198da7eac34dc5c46`.
