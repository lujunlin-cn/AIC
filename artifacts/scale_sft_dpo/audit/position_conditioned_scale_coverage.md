# Current status: full P1 coverage still pending

A real GPU pilot now has 64 frame/view centers (one LIVE source, two GNMC images), with 512 legal Scale projections and maximum drift 0.5 px. This does not determine effective full-data training labels, quality thresholds or source-level uncertainty. Full-data center inference has not started. See `../gpu_preflight_35092.md`.

Historical CPU snapshot follows.

# Position-conditioned Scale coverage

Status: blocked.

LIVE clean train: 47850 frames / 1595 sources; GNMC: 7000 train sources.

Training P1 centers are absent. Effective labels, position mismatch, bin coverage and Scale improvement space are unmeasured. GT-centered bounds in scale_bin_analysis.json are separate diagnostics and are not P1 evidence. After generating hash-bound P1 centers, run position_conditioned_coverage() on train only; calibrate and lock thresholds before validation.
