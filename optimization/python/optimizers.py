"""Lightweight NumPy implementations of common deep learning optimizers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ArrayDict = dict[str, np.ndarray]


@dataclass
class OptimizerState:
    step: int = 0
    slots: dict[str, ArrayDict] = field(default_factory=dict)

    def get_slot(self, name: str, shape: tuple[int, ...]) -> np.ndarray:
        if name not in self.slots:
            self.slots[name] = {}
        if "value" not in self.slots[name]:
            self.slots[name]["value"] = np.zeros(shape, dtype=np.float64)
        return self.slots[name]["value"]


class Optimizer:
    def __init__(self, learning_rate: float) -> None:
        self.learning_rate = float(learning_rate)
        self.state = OptimizerState()

    def step(self, params: ArrayDict, grads: ArrayDict) -> ArrayDict:
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, learning_rate: float, weight_decay: float = 0.0) -> None:
        super().__init__(learning_rate)
        self.weight_decay = float(weight_decay)

    def step(self, params: ArrayDict, grads: ArrayDict) -> ArrayDict:
        for name, value in params.items():
            grad = grads[name]
            if self.weight_decay:
                grad = grad + self.weight_decay * value
            params[name] = value - self.learning_rate * grad
        self.state.step += 1
        return params


class Momentum(Optimizer):
    def __init__(
        self, learning_rate: float, momentum: float = 0.9, weight_decay: float = 0.0
    ) -> None:
        super().__init__(learning_rate)
        self.momentum = float(momentum)
        self.weight_decay = float(weight_decay)

    def step(self, params: ArrayDict, grads: ArrayDict) -> ArrayDict:
        for name, value in params.items():
            grad = grads[name]
            if self.weight_decay:
                grad = grad + self.weight_decay * value
            velocity = self.state.get_slot(name, value.shape)
            velocity[:] = self.momentum * velocity + grad
            params[name] = value - self.learning_rate * velocity
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
        self.decay = float(decay)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)

    def step(self, params: ArrayDict, grads: ArrayDict) -> ArrayDict:
        for name, value in params.items():
            grad = grads[name]
            if self.weight_decay:
                grad = grad + self.weight_decay * value
            cache = self.state.get_slot(name, value.shape)
            cache[:] = self.decay * cache + (1.0 - self.decay) * (grad**2)
            params[name] = value - self.learning_rate * grad / (np.sqrt(cache) + self.epsilon)
        self.state.step += 1
        return params


class Adagrad(Optimizer):
    def __init__(
        self, learning_rate: float, epsilon: float = 1e-8, weight_decay: float = 0.0
    ) -> None:
        super().__init__(learning_rate)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)

    def step(self, params: ArrayDict, grads: ArrayDict) -> ArrayDict:
        for name, value in params.items():
            grad = grads[name]
            if self.weight_decay:
                grad = grad + self.weight_decay * value
            cache = self.state.get_slot(name, value.shape)
            cache[:] = cache + grad**2
            params[name] = value - self.learning_rate * grad / (np.sqrt(cache) + self.epsilon)
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
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.weight_decay = float(weight_decay)
        self._moment1: dict[str, np.ndarray] = {}
        self._moment2: dict[str, np.ndarray] = {}

    def step(self, params: ArrayDict, grads: ArrayDict) -> ArrayDict:
        self.state.step += 1
        t = self.state.step
        for name, value in params.items():
            grad = grads[name]
            if self.weight_decay:
                grad = grad + self.weight_decay * value
            if name not in self._moment1:
                self._moment1[name] = np.zeros_like(value, dtype=np.float64)
                self._moment2[name] = np.zeros_like(value, dtype=np.float64)
            m1 = self._moment1[name]
            m2 = self._moment2[name]
            m1[:] = self.beta1 * m1 + (1.0 - self.beta1) * grad
            m2[:] = self.beta2 * m2 + (1.0 - self.beta2) * (grad**2)
            m1_hat = m1 / (1.0 - self.beta1**t)
            m2_hat = m2 / (1.0 - self.beta2**t)
            params[name] = value - self.learning_rate * m1_hat / (np.sqrt(m2_hat) + self.epsilon)
        return params
