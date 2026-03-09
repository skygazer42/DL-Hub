"""Learning-rate schedulers commonly used in deep learning."""


import math
from dataclasses import dataclass


@dataclass
class SchedulerState:
    step: int = 0


class LRScheduler:
    def __init__(self, base_lr: float) -> None:
        self.base_lr = float(base_lr)
        self.state = SchedulerState()

    def step(self) -> float:
        self.state.step += 1
        return self.get_lr()

    def get_lr(self) -> float:
        raise NotImplementedError


class StepDecay(LRScheduler):
    def __init__(self, base_lr: float, drop_every: int, drop_factor: float = 0.1) -> None:
        super().__init__(base_lr)
        self.drop_every = int(drop_every)
        self.drop_factor = float(drop_factor)

    def get_lr(self) -> float:
        drops = self.state.step // self.drop_every
        return self.base_lr * (self.drop_factor**drops)


class ExponentialDecay(LRScheduler):
    def __init__(self, base_lr: float, decay_rate: float) -> None:
        super().__init__(base_lr)
        self.decay_rate = float(decay_rate)

    def get_lr(self) -> float:
        return self.base_lr * (self.decay_rate**self.state.step)


class CosineAnnealing(LRScheduler):
    def __init__(self, base_lr: float, max_steps: int, min_lr: float = 0.0) -> None:
        super().__init__(base_lr)
        self.max_steps = int(max_steps)
        self.min_lr = float(min_lr)

    def get_lr(self) -> float:
        step = min(self.state.step, self.max_steps)
        cosine = (1.0 + math.cos(math.pi * step / self.max_steps)) / 2.0
        return self.min_lr + (self.base_lr - self.min_lr) * cosine


class WarmupCosine(LRScheduler):
    def __init__(
        self, base_lr: float, warmup_steps: int, max_steps: int, min_lr: float = 0.0
    ) -> None:
        super().__init__(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.max_steps = int(max_steps)
        self.min_lr = float(min_lr)

    def get_lr(self) -> float:
        if self.state.step <= self.warmup_steps and self.warmup_steps > 0:
            return self.base_lr * (self.state.step / self.warmup_steps)
        step = max(self.state.step - self.warmup_steps, 0)
        max_steps = max(self.max_steps - self.warmup_steps, 1)
        cosine = (1.0 + math.cos(math.pi * step / max_steps)) / 2.0
        return self.min_lr + (self.base_lr - self.min_lr) * cosine
