from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True)
class BuildConfig:
    vocab_size: int
    pad_id: int
    max_length: int
    num_classes: int
    width_mult: float = 1.0
    dropout: float = 0.1


Builder = Callable[[BuildConfig], nn.Module]

