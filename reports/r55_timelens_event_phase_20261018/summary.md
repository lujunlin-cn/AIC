# R55 TimeLens2 event phase residual (2026-10-18)

## Mechanism

R54 uses `rank(center) + .20*(rank(binary_interval)-.5)`. This rank of a binary mask introduces a frame-order ramp among all inside frames and does not represent event phase. R55 replaces it with an explicit completion phase:

```text
p(t) = max_j clip((t-a_j)/(b_j-a_j), 0, 1) * 1[a_j <= t <= b_j]
score(t) = rank(center(t)) + .20*(rank(p(t))-.5)
```

The center 85% policy, V9 B3 spatial predictions, TimeLens2 prompt and outputs are unchanged. This is an offline decode change and does not read official labels or score feedback.

## Public diagnostic

On the existing 12-source/18-view public human proxy, the completion phase and R54 produce the same macro proxy (`+0.024997` soft, `+0.032869` binary over center) and 10/18 positive views. The phase rule changes the full 426-frame set on only 5 videos (up to 60 swaps), so it is a mechanism probe rather than a claimed improvement. It is useful because an official R55/R54 pair can test whether R54's gain came from semantic interval membership or an accidental within-interval frame-order ramp.

## Package audit

- `426/426` videos; `199,037` predictions; exact `round(.85*N)` per video.
- Independent checker: passed.
- Local semifinal contract: passed.
- Parameters: `8,854,531,425` (`TimeLens2-8B + V9 vision-only B3`), <=9B.
- ZIP SHA256: `9dd3ab99bf5c1ae920efd7a1d620197fb662bf64112a9bcf66a36852019f12d1`.
- Prediction SHA256: `f5f1879eaa850901cc7c96bb560b5f7dabe866cb95b6af60046fcba4d96d9f52`.
