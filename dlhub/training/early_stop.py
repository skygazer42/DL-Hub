from dataclasses import dataclass
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
        if int(self.patience) < 1:
            raise ValueError(f"patience must be >= 1, got {self.patience}")
        if float(self.min_delta) < 0.0:
            raise ValueError(f"min_delta must be >= 0, got {self.min_delta}")
        if self.mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {self.mode!r}")

    def update(self, value: float) -> bool:
        """Update with the monitored value. Returns True if training should stop."""

        value = float(value)

        if self.best is None:
            self.best = value
            self.bad_epochs = 0
            return False

        improved = False
        if self.mode == "min":
            improved = value < (self.best - float(self.min_delta))
        else:
            improved = value > (self.best + float(self.min_delta))

        if improved:
            self.best = value
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= int(self.patience)
