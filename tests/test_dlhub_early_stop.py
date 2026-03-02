from dlhub.training.early_stop import EarlyStopping


def test_early_stopping_min_mode_triggers_after_patience() -> None:
    stopper = EarlyStopping(patience=2, min_delta=0.0, mode="min")

    # Improvements reset the counter.
    assert stopper.update(1.0) is False
    assert stopper.update(0.9) is False

    # No improvement for 2 consecutive updates -> stop.
    assert stopper.update(0.9) is False
    assert stopper.update(0.91) is True

