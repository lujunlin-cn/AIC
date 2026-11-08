# Position-conditioned Scale coverage

Status: blocked.

LIVE clean train: 47850 frames / 1595 sources; GNMC: 7000 train sources.

Training P1 centers are absent. Effective labels, position mismatch, bin coverage and Scale improvement space are unmeasured. GT-centered bounds in scale_bin_analysis.json are separate diagnostics and are not P1 evidence. After generating hash-bound P1 centers, run position_conditioned_coverage() on train only; calibrate and lock thresholds before validation.
