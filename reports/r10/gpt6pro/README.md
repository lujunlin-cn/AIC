# R10 audit bundle

This bundle does not run video models, use official media, or create predictions.

Files:
- signal_census_audit.py: equal-cost binary top-K, exchange decomposition, train-only
  prior utilities, source-cluster inference, Holm adjustment, power planning.
- numeric_reference.json: arithmetic from the supplied source counts and G histogram.
- selftest_results.json: actual synthetic test results.
- preregistration.json: proposed real experiments; all status fields remain NOT_RUN.
- research.md: the full Chinese research and execution document.

Run:
    python signal_census_audit.py --self-test --output selftest_results.json

Dependencies: numpy, scipy.

Limits:
- NOT the official joint IoU evaluator.
- Exchange diagnostics require equal per-fragment K.
- No actual AIC per-fragment prediction arrays are included.
- Statistical intervals are asymptotic and conditional on a frozen candidate.
- Zero policy variance does not prove a feature family has no information.
- Fit priors and preprocessing on training sources only.
- Repeated holdout selection invalidates unadjusted significance claims.
