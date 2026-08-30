from dataclasses import dataclass
import math
import operator
from typing import Literal

Mode = Literal["min", "max"]


@dataclass
class EarlyStopping:
    """A tiny, dependency-free early stopping helper.

    Usage:
      stopper = EarlyStopping(patience=3, mode="min")
      for epoch in ...:
          ...
          if stopper.update(val_loss):
              break
    """

    patience: int = 3
    min_delta: float = 0.0
    mode: Mode = "min"

    best: float | None = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        self.patience = self._integer("patience", self.patience)
        if self.patience < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        self.min_delta = float(self.min_delta)
        if not math.isfinite(self.min_delta) or self.min_delta < 0.0:
            raise ValueError(f"min_delta must be >= 0, got {self.min_delta}")
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")
        if self.best is not None:
            self.best = float(self.best)
            if not math.isfinite(self.best):
                raise ValueError(f"best must be finite, got {self.best}")
        self.bad_epochs = self._integer("bad_epochs", self.bad_epochs)
        if self.bad_epochs < 0:
            raise ValueError(f"bad_epochs must be >= 0, got {self.bad_epochs}")

    @staticmethod
    def _integer(name: str, value: int) -> int:
        if isinstance(value, bool):
            raise TypeError(f"{name} must be an integer, not bool")
        try:
            return operator.index(value)
        except TypeError as exc:
            raise TypeError(f"{name} must be an integer") from exc

    def update(self, value: float) -> bool:
        """Update with the monitored value. Returns True if training should stop."""

        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"value must be finite, got {value}")
        self.__post_init__()

        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False

        improved = False
        if self.mode == "min":
            improved = value < (self.best - self.min_delta)
        else:
            improved = value > (self.best + self.min_delta)

        if improved:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
