from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LabelPropagationConfig:
    num_layers: int = 10
    alpha: float = 0.9
    clamp_labeled: bool = True


class LabelPropagation(torch.nn.Module):
    """Classic label propagation on a graph.

    Uses row-normalized adjacency (D^{-1}A) and iterates:

        Y_{t+1} = alpha * A_row @ Y_t + (1 - alpha) * Y0

    where Y0 contains one-hot labels only on labeled nodes.
    """

    def __init__(self, cfg: LabelPropagationConfig) -> None:
        super().__init__()
        if cfg.num_layers <= 0:
            raise ValueError("num_layers must be > 0")
        if not (0.0 <= cfg.alpha <= 1.0):
            raise ValueError("alpha must be in [0, 1]")
        self.cfg = cfg

    @torch.no_grad()
    def forward(
        self,
        *,
        adj_row: torch.Tensor,
        labels: torch.Tensor,
        idx_labeled: torch.Tensor,
        num_classes: int | None = None,
    ) -> torch.Tensor:
        """Return propagated label distribution `(N, C)`."""

        if labels.ndim != 1:
            raise ValueError("labels must be a 1D tensor of shape (N,)")
        if idx_labeled.ndim != 1:
            raise ValueError("idx_labeled must be a 1D index tensor")

        n = int(labels.shape[0])
        if num_classes is None:
            num_classes = int(labels.max().item()) + 1

        y0 = torch.zeros((n, num_classes), dtype=torch.float32, device=labels.device)
        y0[idx_labeled] = torch.nn.functional.one_hot(labels[idx_labeled], num_classes=num_classes).to(
            torch.float32
        )

        y = y0.clone()
        for _ in range(self.cfg.num_layers):
            y = self.cfg.alpha * torch.sparse.mm(adj_row, y) + (1.0 - self.cfg.alpha) * y0
            if self.cfg.clamp_labeled:
                y[idx_labeled] = y0[idx_labeled]
            y = y.clamp(min=0.0)
            y = y / y.sum(dim=1, keepdim=True).clamp(min=1e-12)

        return y

