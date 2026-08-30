"""Lightweight NumPy implementations of common deep learning optimizers."""

from dataclasses import dataclass, field

import numpy as np

ArrayDict = dict[str, np.ndarray]


def _finite_float(name: str, value: float) -> float:
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _non_negative_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result < 0.0:
        raise ValueError(f"{name} must be >= 0")
    return result


def _positive_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if result <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return result


def _unit_interval_float(name: str, value: float) -> float:
    result = _finite_float(name, value)
    if not 0.0 <= result < 1.0:
        raise ValueError(f"{name} must be in [0, 1)")
    return result


def _all_finite(name: str, value: np.ndarray) -> bool:
    try:
        return bool(np.all(np.isfinite(value)))
    except TypeError as exc:
        raise ValueError(f"{name} must contain numeric values") from exc


def _ensure_finite_update(name: str, value: np.ndarray) -> None:
    if not _all_finite(name, value):
        raise FloatingPointError(f"optimizer update for parameter {name!r} is not finite")


def _ensure_finite_state(name: str, state_name: str, value: np.ndarray) -> None:
    if not _all_finite(state_name, value):
        raise FloatingPointError(
            f"optimizer state {state_name!r} for parameter {name!r} is not finite"
        )


def _effective_gradient(
    name: str, value: np.ndarray, grad: np.ndarray, weight_decay: float
) -> np.ndarray:
    if not weight_decay:
        return grad
    with np.errstate(over="ignore", invalid="ignore"):
        result = grad + weight_decay * value
    if not _all_finite(name, result):
        raise FloatingPointError(f"effective gradient for parameter {name!r} is not finite")
    return result


@dataclass
class OptimizerState:
    step: int = 0
    slots: dict[str, ArrayDict] = field(default_factory=dict)

    def get_slot(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        if name not in self.slots:
            self.slots[name] = {}
        if "value" not in self.slots[name]:
            self.slots[name]["value"] = np.zeros(shape, dtype=np.float64)
        value = self.slots[name]["value"]
        if value.shape != shape:
            raise ValueError(
                f"optimizer state for parameter {name!r} has shape {value.shape}, expected {shape}"
            )
        return value


class Optimizer:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = _non_negative_float("learning_rate", learning_rate)
        self.state = OptimizerState()

    def step(self, params: ArrayDict, grads: ArrayDict) -> ArrayDict:
        raise NotImplementedError

    def _validate_hyperparameters(self) -> None:
        self.learning_rate = _non_negative_float("learning_rate", self.learning_rate)

    def _prepare_step(
        self, params: ArrayDict, grads: ArrayDict
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        self._validate_hyperparameters()
        if set(params) != set(grads):
            raise ValueError("params and grads must have the same keys")
        if not params:
            raise ValueError("params and grads must be non-empty")

        arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, value in params.items():
            param = np.asarray(value)
            grad = np.asarray(grads[name])
            if param.shape != grad.shape:
                raise ValueError(
                    f"parameter {name!r} and its gradient must have the same shape, "
                    f"got {param.shape} and {grad.shape}"
                )
            if not _all_finite(f"parameter {name!r}", param):
                raise ValueError(f"parameter {name!r} must contain only finite values")
            if not _all_finite(f"gradient {name!r}", grad):
                raise ValueError(f"gradient {name!r} must contain only finite values")
            arrays[name] = (param, grad)
        return arrays

    def _validate_slot_shapes(self, arrays: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        for name, (value, _) in arrays.items():
            slot = self.state.slots.get(name, {}).get("value")
            if slot is not None and slot.shape != value.shape:
                raise ValueError(
                    f"optimizer state for parameter {name!r} has shape {slot.shape}, "
                    f"expected {value.shape}"
                )
            if slot is not None:
                _ensure_finite_state(name, "value", slot)


class SGD(Optimizer):
    def __init__(self, learning_rate: float, weight_decay: float = 0.0) -> None:
        super().__init__(learning_rate)
        self.weight_decay = _non_negative_float("weight_decay", weight_decay)

    def _validate_hyperparameters(self) -> None:
        super()._validate_hyperparameters()
        self.weight_decay = _non_negative_float("weight_decay", self.weight_decay)

    def step(self, params: ArrayDict, grads: ArrayDict) -> ArrayDict:
        arrays = self._prepare_step(params, grads)
        updates: ArrayDict = {}
        for name, (value, grad) in arrays.items():
            grad = _effective_gradient(name, value, grad, self.weight_decay)
            with np.errstate(over="ignore", invalid="ignore"):
                updates[name] = value - self.learning_rate * grad
            _ensure_finite_update(name, updates[name])
        params.update(updates)
        self.state.step += 1
        return params


class Momentum(Optimizer):
    def __init__(
        self, learning_rate: float, momentum: float = 0.9, weight_decay: float = 0.0
    ) -> None:
        super().__init__(learning_rate)
        self.momentum = _unit_interval_float("momentum", momentum)
        self.weight_decay = _non_negative_float("weight_decay", weight_decay)

    def _validate_hyperparameters(self) -> None:
        super()._validate_hyperparameters()
        self.momentum = _unit_interval_float("momentum", self.momentum)
        self.weight_decay = _non_negative_float("weight_decay", self.weight_decay)

    def step(self, params: ArrayDict, grads: ArrayDict) -> ArrayDict:
        arrays = self._prepare_step(params, grads)
        self._validate_slot_shapes(arrays)
        updates: ArrayDict = {}
        velocities: ArrayDict = {}
        for name, (value, grad) in arrays.items():
            grad = _effective_gradient(name, value, grad, self.weight_decay)
            previous = self.state.slots.get(name, {}).get("value")
            if previous is None:
                previous = np.zeros(value.shape, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                velocities[name] = self.momentum * previous + grad
            _ensure_finite_state(name, "velocity", velocities[name])
            with np.errstate(over="ignore", invalid="ignore"):
                updates[name] = value - self.learning_rate * velocities[name]
            _ensure_finite_update(name, updates[name])
        for name, velocity in velocities.items():
            self.state.get_slot(name, velocity.shape)[:] = velocity
        params.update(updates)
        self.state.step += 1
        return params


class RMSProp(Optimizer):
    def __init__(
        self,
        learning_rate: float,
        decay: float = 0.9,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(learning_rate)
        self.decay = _unit_interval_float("decay", decay)
        self.epsilon = _positive_float("epsilon", epsilon)
        self.weight_decay = _non_negative_float("weight_decay", weight_decay)

    def _validate_hyperparameters(self) -> None:
        super()._validate_hyperparameters()
        self.decay = _unit_interval_float("decay", self.decay)
        self.epsilon = _positive_float("epsilon", self.epsilon)
        self.weight_decay = _non_negative_float("weight_decay", self.weight_decay)

    def step(self, params: ArrayDict, grads: ArrayDict) -> ArrayDict:
        arrays = self._prepare_step(params, grads)
        self._validate_slot_shapes(arrays)
        updates: ArrayDict = {}
        caches: ArrayDict = {}
        for name, (value, grad) in arrays.items():
            grad = _effective_gradient(name, value, grad, self.weight_decay)
            previous = self.state.slots.get(name, {}).get("value")
            if previous is None:
                previous = np.zeros(value.shape, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                caches[name] = self.decay * previous + (1.0 - self.decay) * (grad**2)
            _ensure_finite_state(name, "cache", caches[name])
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                updates[name] = value - self.learning_rate * grad / (
                    np.sqrt(caches[name]) + self.epsilon
                )
            _ensure_finite_update(name, updates[name])
        for name, cache in caches.items():
            self.state.get_slot(name, cache.shape)[:] = cache
        params.update(updates)
        self.state.step += 1
        return params


class Adagrad(Optimizer):
    def __init__(
        self, learning_rate: float, epsilon: float = 1e-8, weight_decay: float = 0.0
    ) -> None:
        super().__init__(learning_rate)
        self.epsilon = _positive_float("epsilon", epsilon)
        self.weight_decay = _non_negative_float("weight_decay", weight_decay)

    def _validate_hyperparameters(self) -> None:
        super()._validate_hyperparameters()
        self.epsilon = _positive_float("epsilon", self.epsilon)
        self.weight_decay = _non_negative_float("weight_decay", self.weight_decay)

    def step(self, params: ArrayDict, grads: ArrayDict) -> ArrayDict:
        arrays = self._prepare_step(params, grads)
        self._validate_slot_shapes(arrays)
        updates: ArrayDict = {}
        caches: ArrayDict = {}
        for name, (value, grad) in arrays.items():
            grad = _effective_gradient(name, value, grad, self.weight_decay)
            previous = self.state.slots.get(name, {}).get("value")
            if previous is None:
                previous = np.zeros(value.shape, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                caches[name] = previous + grad**2
            _ensure_finite_state(name, "cache", caches[name])
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                updates[name] = value - self.learning_rate * grad / (
                    np.sqrt(caches[name]) + self.epsilon
                )
            _ensure_finite_update(name, updates[name])
        for name, cache in caches.items():
            self.state.get_slot(name, cache.shape)[:] = cache
        params.update(updates)
        self.state.step += 1
        return params


class Adam(Optimizer):
    def __init__(
        self,
        learning_rate: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(learning_rate)
        self.beta1 = _unit_interval_float("beta1", beta1)
        self.beta2 = _unit_interval_float("beta2", beta2)
        self.epsilon = _positive_float("epsilon", epsilon)
        self.weight_decay = _non_negative_float("weight_decay", weight_decay)
        self._moment1: dict[str, np.ndarray] = {}
        self._moment2: dict[str, np.ndarray] = {}

    def _validate_hyperparameters(self) -> None:
        super()._validate_hyperparameters()
        self.beta1 = _unit_interval_float("beta1", self.beta1)
        self.beta2 = _unit_interval_float("beta2", self.beta2)
        self.epsilon = _positive_float("epsilon", self.epsilon)
        self.weight_decay = _non_negative_float("weight_decay", self.weight_decay)

    def step(self, params: ArrayDict, grads: ArrayDict) -> ArrayDict:
        arrays = self._prepare_step(params, grads)
        for name, (value, _) in arrays.items():
            for moments in (self._moment1, self._moment2):
                if name in moments and moments[name].shape != value.shape:
                    raise ValueError(
                        f"optimizer state for parameter {name!r} has shape "
                        f"{moments[name].shape}, expected {value.shape}"
                    )
                if name in moments:
                    _ensure_finite_state(name, "moment", moments[name])

        t = self.state.step + 1
        updates: ArrayDict = {}
        moment1_updates: ArrayDict = {}
        moment2_updates: ArrayDict = {}
        for name, (value, grad) in arrays.items():
            grad = _effective_gradient(name, value, grad, self.weight_decay)
            previous_m1 = self._moment1.get(name)
            previous_m2 = self._moment2.get(name)
            if previous_m1 is None:
                previous_m1 = np.zeros_like(value, dtype=np.float64)
                previous_m2 = np.zeros_like(value, dtype=np.float64)
            with np.errstate(over="ignore", invalid="ignore"):
                moment1_updates[name] = self.beta1 * previous_m1 + (1.0 - self.beta1) * grad
                moment2_updates[name] = self.beta2 * previous_m2 + (1.0 - self.beta2) * (grad**2)
            _ensure_finite_state(name, "first moment", moment1_updates[name])
            _ensure_finite_state(name, "second moment", moment2_updates[name])
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                m1_hat = moment1_updates[name] / (1.0 - self.beta1**t)
                m2_hat = moment2_updates[name] / (1.0 - self.beta2**t)
                updates[name] = value - self.learning_rate * m1_hat / (
                    np.sqrt(m2_hat) + self.epsilon
                )
            _ensure_finite_update(name, updates[name])

        self._moment1.update(moment1_updates)
        self._moment2.update(moment2_updates)
        params.update(updates)
        self.state.step = t
        return params
