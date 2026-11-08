"""The final evidence validator must reject resets and inconsistent timestamps."""
import pytest

from scripts.validate_scale_final_preflight import validate_budget


def ledger(seconds=3590):
    return {'cap_seconds': 3600, 'within_budget': True, 'cumulative_GPU_seconds': seconds,
            'runs': [{'started_at': '2026-11-08T03:00:00+00:00',
                      'ended_at': '2026-11-08T03:59:50+00:00', 'elapsed_seconds': seconds,
                      'optimizer_steps': 0, 'includes_model_load_warmup_and_inference': True,
                      'formal_training': False}]}


def test_sealed_final_budget_accepts_bounded_inference():
    assert validate_budget(ledger()) == 3590


def test_sealed_final_budget_rejects_cumulative_overrun():
    with pytest.raises(ValueError, match='overrun'):
        validate_budget(ledger(3601))


def test_sealed_final_budget_rejects_unclosed_or_reset_timestamps():
    value = ledger(); value['runs'][0]['ended_at'] = None
    with pytest.raises(ValueError, match='not closed'):
        validate_budget(value)
    value = ledger(100)
    with pytest.raises(ValueError, match='timestamps'):
        validate_budget(value)


def test_sealed_final_budget_rejects_optimizer_steps():
    value = ledger(); value['runs'][0]['optimizer_steps'] = 1
    with pytest.raises(ValueError, match='optimizer'):
        validate_budget(value)
