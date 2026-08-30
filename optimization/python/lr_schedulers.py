"""Learning-rate schedulers commonly used in deep learning."""

import math
import operator
from dataclasses import dataclass


@dataclass
class SchedulerState:
    step: int = 0


def _finite_non_negative(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and >= 0")
    return result


def _step_count(name: str, value: int, *, allow_zero: bool) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an integer")
    try:
        result = operator.index(value)
    except TypeError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    minimum = 0 if allow_zero else 1
    if result < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return result


def _minimum_lr(base_lr: float, min_lr: float) -> float:
    result = _finite_non_negative("min_lr", min_lr)
    if result > base_lr:
        raise ValueError("min_lr must be <= base_lr")
    return result


class LRScheduler:
    def __init__(self, base_lr: float) -> None:
        self.base_lr = _finite_non_negative("base_lr", base_lr)
        self.state = SchedulerState()

    def step(self) -> float:
        self.state.step += 1
        return self.get_lr()

    def get_lr(self) -> float:
        raise NotImplementedError


class StepDecay(LRScheduler):
    def __init__(self, base_lr: float, drop_every: int, drop_factor: float = 0.1) -> None:
        super().__init__(base_lr)
        self.drop_every = _step_count("drop_every", drop_every, allow_zero=False)
        self.drop_factor = _finite_non_negative("drop_factor", drop_factor)

    def get_lr(self) -> float:
        base_lr = _finite_non_negative("base_lr", self.base_lr)
        drop_every = _step_count("drop_every", self.drop_every, allow_zero=False)
        drop_factor = _finite_non_negative("drop_factor", self.drop_factor)
        drops = self.state.step // drop_every
        return base_lr * (drop_factor**drops)


class ExponentialDecay(LRScheduler):
    def __init__(self, base_lr: float, decay_rate: float) -> None:
        super().__init__(base_lr)
        self.decay_rate = _finite_non_negative("decay_rate", decay_rate)

    def get_lr(self) -> float:
        base_lr = _finite_non_negative("base_lr", self.base_lr)
        decay_rate = _finite_non_negative("decay_rate", self.decay_rate)
        return base_lr * (decay_rate**self.state.step)


class CosineAnnealing(LRScheduler):
    def __init__(self, base_lr: float, max_steps: int, min_lr: float = 0.0) -> None:
        super().__init__(base_lr)
        self.max_steps = _step_count("max_steps", max_steps, allow_zero=False)
        self.min_lr = _minimum_lr(self.base_lr, min_lr)

    def get_lr(self) -> float:
        base_lr = _finite_non_negative("base_lr", self.base_lr)
        max_steps = _step_count("max_steps", self.max_steps, allow_zero=False)
        min_lr = _minimum_lr(base_lr, self.min_lr)
        step = min(self.state.step, max_steps)
        cosine = (1.0 + math.cos(math.pi * step / max_steps)) / 2.0
        return min_lr + (base_lr - min_lr) * cosine


class WarmupCosine(LRScheduler):
    def __init__(
        self, base_lr: float, warmup_steps: int, max_steps: int, min_lr: float = 0.0
    ) -> None:
        super().__init__(base_lr)
        self.warmup_steps = _step_count("warmup_steps", warmup_steps, allow_zero=True)
        self.max_steps = _step_count("max_steps", max_steps, allow_zero=False)
        if self.warmup_steps > self.max_steps:
            raise ValueError("warmup_steps must be <= max_steps")
        self.min_lr = _minimum_lr(self.base_lr, min_lr)

    def get_lr(self) -> float:
        base_lr = _finite_non_negative("base_lr", self.base_lr)
        warmup_steps = _step_count("warmup_steps", self.warmup_steps, allow_zero=True)
        max_steps = _step_count("max_steps", self.max_steps, allow_zero=False)
        if warmup_steps > max_steps:
            raise ValueError("warmup_steps must be <= max_steps")
        min_lr = _minimum_lr(base_lr, self.min_lr)

        if self.state.step <= warmup_steps and warmup_steps > 0:
            return base_lr * (self.state.step / warmup_steps)
        duration = max(max_steps - warmup_steps, 1)
        step = min(max(self.state.step - warmup_steps, 0), duration)
        cosine = (1.0 + math.cos(math.pi * step / duration)) / 2.0
        return min_lr + (base_lr - min_lr) * cosine
