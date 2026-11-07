import pytest

from aic.scale_budget import empty_budget, record_gpu_run


def test_budget_records_gpu_hours_and_enforces_per_stage_and_total_limits():
    budget = empty_budget()
    record_gpu_run(budget, "scale_sft", "2026-11-08T00:00:00+00:00",
                   "2026-11-08T01:30:00+00:00", "RTX 4090", "sft-1", "sft.log")
    assert budget["total_used_hours"] == pytest.approx(1.5)
    with pytest.raises(ValueError, match="scale_sft GPU budget"):
        record_gpu_run(budget, "scale_sft", "2026-11-08T02:00:00+00:00",
                       "2026-11-08T04:45:00+00:00", "RTX 4090", "sft-2", "sft2.log")


def test_budget_rejects_invalid_time_and_unknown_stages():
    budget = empty_budget()
    with pytest.raises(ValueError, match="unknown GPU stage"):
        record_gpu_run(budget, "sweep", "2026-11-08T00:00:00+00:00",
                       "2026-11-08T00:10:00+00:00", "GPU", "", "")
    with pytest.raises(ValueError, match="after start"):
        record_gpu_run(budget, "spatial_dpo", "2026-11-08T00:00:00+00:00",
                       "2026-11-08T00:00:00+00:00", "GPU", "", "")
