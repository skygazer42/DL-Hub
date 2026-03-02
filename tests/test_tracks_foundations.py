import pytest


torch = pytest.importorskip("torch")


def test_foundations_lesson_02_regression_dataloaders_smoke() -> None:
    from tracks.foundations.lesson_02_linear_regression_autograd.data import (
        DataConfig,
        make_regression_dataloaders,
    )

    train_loader, eval_loader = make_regression_dataloaders(
        DataConfig(num_samples=64, batch_size=16, noise_std=0.1)
    )
    x_batch, y_batch = next(iter(train_loader))
    assert tuple(x_batch.shape) == (16, 2)
    assert tuple(y_batch.shape) == (16, 1)

    x_eval, y_eval = next(iter(eval_loader))
    assert tuple(x_eval.shape[1:]) == (2,)
    assert tuple(y_eval.shape[1:]) == (1,)

