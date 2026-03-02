import math

import numpy as np

from optimization.python.losses import (
    binary_cross_entropy,
    categorical_cross_entropy,
    mean_absolute_error,
    mean_squared_error,
)
from optimization.python.lr_schedulers import (
    CosineAnnealing,
    ExponentialDecay,
    StepDecay,
    WarmupCosine,
)
from optimization.python.metrics import accuracy_score, precision_recall_f1, r2_score
from optimization.python.optimizers import SGD, Adagrad, Adam, Momentum, RMSProp


def test_losses_match_simple_expected_values() -> None:
    assert mean_squared_error([1, 2], [1, 4]) == 2.0
    assert mean_absolute_error([1, 2], [1, 4]) == 1.0

    bce = binary_cross_entropy([1, 0], [0.9, 0.1])
    assert math.isclose(bce, -math.log(0.9), rel_tol=1e-6, abs_tol=1e-6)

    y_true = np.array([[1, 0, 0], [0, 1, 0]])
    y_pred = np.array([[0.8, 0.1, 0.1], [0.2, 0.7, 0.1]])
    cce = categorical_cross_entropy(y_true, y_pred)
    expected = (-math.log(0.8) - math.log(0.7)) / 2.0
    assert math.isclose(cce, expected, rel_tol=1e-6, abs_tol=1e-6)


def test_metrics_match_simple_expected_values() -> None:
    assert accuracy_score([1, 0, 1], [1, 1, 1]) == 2.0 / 3.0

    precision, recall, f1 = precision_recall_f1([1, 0, 1, 0], [1, 1, 0, 0])
    assert math.isclose(precision, 0.5, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(recall, 0.5, rel_tol=1e-6, abs_tol=1e-6)
    assert math.isclose(f1, 0.5, rel_tol=1e-6, abs_tol=1e-6)

    assert math.isclose(r2_score([1, 2, 3], [1, 2, 3]), 1.0, rel_tol=1e-6, abs_tol=1e-6)


def test_lr_schedulers_are_deterministic_and_bounded() -> None:
    step = StepDecay(base_lr=1.0, drop_every=2, drop_factor=0.1)
    lrs = [step.step() for _ in range(5)]
    assert lrs[0] == 1.0
    assert math.isclose(lrs[1], 0.1, rel_tol=1e-12, abs_tol=1e-12)
    assert math.isclose(lrs[3], 0.01, rel_tol=1e-12, abs_tol=1e-12)

    exp = ExponentialDecay(base_lr=1.0, decay_rate=0.9)
    assert math.isclose(exp.step(), 0.9, rel_tol=1e-12)
    assert math.isclose(exp.step(), 0.81, rel_tol=1e-12)

    cosine = CosineAnnealing(base_lr=1.0, max_steps=10, min_lr=0.1)
    for _ in range(25):
        lr = cosine.step()
        assert 0.1 <= lr <= 1.0
    assert math.isclose(lr, 0.1, rel_tol=1e-12)

    warmup = WarmupCosine(base_lr=1.0, warmup_steps=2, max_steps=10, min_lr=0.0)
    assert math.isclose(warmup.step(), 0.5, rel_tol=1e-12)
    assert math.isclose(warmup.step(), 1.0, rel_tol=1e-12)
    for _ in range(20):
        lr = warmup.step()
        assert 0.0 <= lr <= 1.0


def test_optimizers_update_params_in_expected_direction() -> None:
    grads = {"w": np.array([0.1, -0.2], dtype=np.float64)}

    sgd = SGD(learning_rate=0.5)
    params = {"w": np.array([1.0, -1.0], dtype=np.float64)}
    updated = sgd.step(params, grads)
    assert np.allclose(updated["w"], np.array([0.95, -0.9], dtype=np.float64))
    assert sgd.state.step == 1

    for Optimizer in (Momentum, RMSProp, Adagrad, Adam):
        opt = Optimizer(learning_rate=0.1)  # type: ignore[call-arg]
        params = {"w": np.array([1.0, -1.0], dtype=np.float64)}
        before = params["w"].copy()
        after = opt.step(params, grads)["w"]

        assert np.all(np.sign(after - before) == -np.sign(grads["w"]))
        assert opt.state.step == 1
