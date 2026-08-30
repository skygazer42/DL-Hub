import math

import pytest

from dlhub.training.early_stop import EarlyStopping


def test_early_stopping_min_mode_triggers_after_patience() -> None:
    stopper = EarlyStopping(patience=2, min_delta=0.0, mode="min")

    # Improvements reset the counter.
    assert stopper.update(1.0) is False
    assert stopper.update(0.9) is False

    # No improvement for 2 consecutive updates -> stop.
    assert stopper.update(0.9) is False
    assert stopper.update(0.91) is True


@pytest.mark.parametrize("patience", [True, 1.5])
def test_early_stopping_rejects_non_integer_patience(patience: object) -> None:
    with pytest.raises(TypeError, match="patience must be an integer"):
        EarlyStopping(patience=patience)  # type: ignore[arg-type]


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_early_stopping_rejects_nonfinite_updates_without_mutating_state(value: float) -> None:
    stopper = EarlyStopping(patience=2)
    assert stopper.update(1.0) is False

    with pytest.raises(ValueError, match="finite"):
        stopper.update(value)

    assert stopper.best == 1.0
    assert stopper.bad_epochs == 0


def test_early_stopping_validates_initial_state() -> None:
    with pytest.raises(ValueError, match="best must be finite"):
        EarlyStopping(best=math.nan)
    with pytest.raises(ValueError, match="bad_epochs"):
        EarlyStopping(bad_epochs=-1)
