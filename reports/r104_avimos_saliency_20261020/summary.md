# R104 AViMoS visual saliency pretraining (2026-10-20)

## Protocol

Only RGB was used; the project's audio experiments are negative and AViMoS audio was excluded. A 475,521-parameter MobileNetV2 decoder was trained on 100 AViMoS source videos and evaluated on 30 held-out videos, eight sampled frames per video. The video and fixation assets were downloaded separately:

- `SaliencyTrain.zip` SHA256 `8d30723234c68e6649bdd319ae53323256ab46337eae57764143c7d6594d4789`.
- `FixationsTrain.zip` SHA256 `1b6ba74ff3877ad42eb57485dc80e14a8e0a122c79470a24f005ca1ad7b4ea41`.

## Result

The best 3-epoch decoder improved held-out saliency metrics against the empirical train mean map:

| metric | center/mean-map prior | visual decoder |
|---|---:|---:|
| CC | 0.2584 | 0.4202 |
| SIM | 0.1442 | 0.2677 |
| KL (lower) | 2.6212 | 1.9713 |
| NSS | 1.6222 | 2.6970 |

This establishes that the model learned visual fixation structure, not that fixation equals AIC composition GT.

Evidence: `metrics.json`; code `scripts/r104_avimos_saliency.py`.
