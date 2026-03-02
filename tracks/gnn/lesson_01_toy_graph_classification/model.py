from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_adj(adj: torch.Tensor) -> torch.Tensor:
    """Symmetric normalization for a batch of adjacency matrices.

    adj: (B, N, N) float tensor with 0/1 entries.
    """

    b, n, _ = adj.shape
    eye = torch.eye(n, device=adj.device, dtype=adj.dtype).expand(b, n, n)
    a_hat = adj + eye
    deg = a_hat.sum(dim=-1)  # (B, N)
    deg_inv_sqrt = deg.clamp(min=1e-6).pow(-0.5)
    return deg_inv_sqrt.unsqueeze(-1) * a_hat * deg_inv_sqrt.unsqueeze(-2)


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, Fin), adj: (B, N, N)
        norm_adj = _normalize_adj(adj)
        h = torch.bmm(norm_adj, x)
        return self.linear(h)


@dataclass(frozen=True)
class ModelConfig:
    in_features: int = 2
    hidden_features: int = 32
    num_classes: int = 2


class GCNGraphClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.gcn1 = GCNLayer(config.in_features, config.hidden_features)
        self.gcn2 = GCNLayer(config.hidden_features, config.hidden_features)
        self.head = nn.Linear(config.hidden_features, config.num_classes)

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, adj = inputs
        h = F.relu(self.gcn1(x, adj))
        h = F.relu(self.gcn2(h, adj))
        g = h.mean(dim=1)  # global mean pooling
        return self.head(g)

