# R47 TimeLens2 + V9 vision-only B3 candidate

TimeLens2-8B was run on the complete 426-video semifinal index using one
fixed generic-highlight prompt. Shard outputs 0--8 were merged without
changing responses. Interval membership is the primary rank and the normalized
centre rank only breaks ties; each video keeps exactly `round(.85*N)` original
frames. The official intake index supplies each video's real FPS, including
VFR entries; no 30-fps assumption is used in packaging.

Spatial predictions come from the audited V9 vision-only B3 full-frame output.
This keeps total deployed parameters within the official cap:

```
TimeLens2             8,767,123,696
V9 vision-only B3        87,407,729
total                  8,854,531,425 (8.8545B)
```

The package contains 426 videos and 199,037 predictions. The independent
checker passes at `model_size_mb=8854.531425`; ZIP extraction is byte-identical.
The candidate changes both temporal ranking and the spatial runtime, so it is
deliberately a system-level route. It must not be interpreted as a pure
TimeLens2 temporal ablation. It is packaged but not uploaded and has no official
score yet.

Candidate: `submissions/R47_TIMELENS2_V9_B3/`. This is the canonical package
for the current source: its selected boxes match the downloaded V9 full-frame
parent byte-for-byte. A separately generated `R47_TIMELENS2_V9B3` directory
has a different spatial artifact on 33,301 selected frames and should not be
treated as equivalent without a new source manifest.
