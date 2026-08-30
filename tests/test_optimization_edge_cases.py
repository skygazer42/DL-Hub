import warnings

import numpy as np
import pytest

from optimization.python.losses import (
    binary_cross_entropy,
    categorical_cross_entropy,
    mean_absolute_error,
    mean_squared_error,
)
from optimization.python.lr_schedulers import CosineAnnealing, StepDecay, WarmupCosine
from optimization.python.metrics import accuracy_score, precision_recall_f1, r2_score
from optimization.python.optimizers import SGD, Adagrad, Adam, Momentum, RMSProp


@pytest.mark.parametrize("loss", [mean_squared_error, mean_absolute_error])
def test_pointwise_losses_reject_empty_or_broadcastable_mismatched_inputs(loss) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        loss([], [])

    with pytest.raises(ValueError, match="same shape"):
        loss(np.ones((2, 1)), np.ones((2,)))


def test_cross_entropy_validates_probability_contracts() -> None:
    with pytest.raises(ValueError, match="same shape"):
        binary_cross_entropy([0, 1], [[0.2], [0.8]])
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        binary_cross_entropy([0, 2], [0.2, 0.8])
    with pytest.raises(ValueError, match="eps"):
        binary_cross_entropy([0, 1], [0.2, 0.8], eps=0.0)

    y_true = np.eye(2)
    with pytest.raises(ValueError, match="2D"):
        categorical_cross_entropy([1, 0], [0.8, 0.2])
    with pytest.raises(ValueError, match="sum to 1"):
        categorical_cross_entropy(y_true, [[0.8, 0.3], [0.2, 0.8]])
    with pytest.raises(ValueError, match="finite"):
        categorical_cross_entropy(y_true, [[np.nan, 0.0], [0.2, 0.8]])


def test_metrics_reject_ambiguous_shapes_and_empty_inputs() -> None:
    with pytest.raises(ValueError, match="same shape"):
        accuracy_score(np.array([[1], [0]]), np.array([1, 0]))
    with pytest.raises(ValueError, match="non-empty"):
        accuracy_score([], [])
    with pytest.raises(ValueError, match="finite"):
        accuracy_score([np.nan], [np.nan])
    with pytest.raises(ValueError, match="binary labels"):
        precision_recall_f1([0, 2], [0, 1])


def test_r2_has_defined_constant_target_semantics() -> None:
    assert r2_score([2.0, 2.0], [2.0, 2.0]) == 1.0
    assert r2_score([2.0, 2.0], [1.0, 1.0]) == 0.0


@pytest.mark.parametrize(
    "factory",
    [
        lambda: SGD(learning_rate=-0.1),
        lambda: SGD(learning_rate=0.1, weight_decay=-0.1),
        lambda: Adam(beta1=1.0),
        lambda: Adam(beta2=-0.1),
        lambda: Adam(epsilon=0.0),
    ],
)
def test_optimizers_reject_invalid_hyperparameters(factory) -> None:
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("optimizer", [SGD(0.1), Adam(0.1)])
def test_optimizer_validation_is_transactional(optimizer) -> None:
    params = {"first": np.array([1.0]), "second": np.array([2.0])}
    before = {name: value.copy() for name, value in params.items()}

    with pytest.raises(ValueError, match="same keys"):
        optimizer.step(params, {"first": np.array([0.5])})

    assert optimizer.state.step == 0
    for name in params:
        np.testing.assert_array_equal(params[name], before[name])


def test_optimizer_rejects_shape_mismatch_and_nonfinite_gradients_before_update() -> None:
    optimizer = SGD(0.1)
    params = {"weight": np.array([1.0, 2.0])}

    with pytest.raises(ValueError, match="same shape"):
        optimizer.step(params, {"weight": np.array([0.5])})
    with pytest.raises(ValueError, match="finite"):
        optimizer.step(params, {"weight": np.array([np.nan, 0.5])})

    np.testing.assert_array_equal(params["weight"], np.array([1.0, 2.0]))
    assert optimizer.state.step == 0


def test_optimizer_rejects_empty_parameter_sets_without_advancing_state() -> None:
    optimizer = SGD(0.1)

    with pytest.raises(ValueError, match="non-empty"):
        optimizer.step({}, {})

    assert optimizer.state.step == 0


@pytest.mark.parametrize(
    "optimizer",
    [RMSProp(0.1, decay=0.0), Adagrad(0.1), Adam(0.1, beta2=0.0)],
)
def test_optimizer_state_overflow_does_not_commit_any_changes(optimizer) -> None:
    params = {"weight": np.array([1.0])}

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(FloatingPointError, match="state"):
            optimizer.step(params, {"weight": np.array([1e308])})

    np.testing.assert_array_equal(params["weight"], np.array([1.0]))
    assert optimizer.state.step == 0
    assert optimizer.state.slots == {}
    if isinstance(optimizer, Adam):
        assert optimizer._moment1 == {}
        assert optimizer._moment2 == {}


def test_momentum_overflow_preserves_preexisting_state() -> None:
    optimizer = Momentum(learning_rate=0.0, momentum=0.9)
    params = {"weight": np.array([1.0])}
    optimizer.step(params, {"weight": np.array([1e308])})
    velocity_before = optimizer.state.slots["weight"]["value"].copy()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        with pytest.raises(FloatingPointError, match="state"):
            optimizer.step(params, {"weight": np.array([1e308])})

    np.testing.assert_array_equal(params["weight"], np.array([1.0]))
    np.testing.assert_array_equal(optimizer.state.slots["weight"]["value"], velocity_before)
    assert optimizer.state.step == 1


def test_scheduler_parameters_prevent_division_by_zero() -> None:
    with pytest.raises(ValueError, match="drop_every"):
        StepDecay(base_lr=1.0, drop_every=0)
    with pytest.raises(ValueError, match="max_steps"):
        CosineAnnealing(base_lr=1.0, max_steps=0)
    with pytest.raises(ValueError, match="warmup_steps"):
        WarmupCosine(base_lr=1.0, warmup_steps=5, max_steps=4)


def test_warmup_cosine_stays_at_minimum_after_schedule_ends() -> None:
    scheduler = WarmupCosine(base_lr=1.0, warmup_steps=0, max_steps=4, min_lr=0.0)

    rates = [scheduler.step() for _ in range(8)]

    assert rates[3] == pytest.approx(0.0)
    assert rates[4:] == pytest.approx([0.0] * 4)
