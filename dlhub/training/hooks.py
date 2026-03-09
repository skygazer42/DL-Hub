
from dataclasses import dataclass


@dataclass(frozen=True)
class BatchLog:
    stage: str  # "train" | "eval"
    batch_idx: int
    loss: float
    accuracy: float | None = None


class Hook:
    """A minimal hook interface for training/evaluation loops.

    Hooks are intentionally tiny to keep this repo "learnable":
    - no hidden global state
    - no complex callback ordering rules
    """

    def on_batch_end(self, log: BatchLog) -> None:  # noqa: D401 (simple hook)
        return None

