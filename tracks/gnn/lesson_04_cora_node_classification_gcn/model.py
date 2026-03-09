
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    in_features: int
    hidden_features: int = 16
    num_classes: int = 7
    dropout: float = 0.5


class GCNLayer(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(int(in_features), int(out_features), bias=False)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # adj: sparse (N, N), x: dense (N, F)
        support = self.linear(x)
        return torch.sparse.mm(adj, support)


class GCN(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.gc1 = GCNLayer(config.in_features, config.hidden_features)
        self.gc2 = GCNLayer(config.hidden_features, config.num_classes)
        self.dropout = nn.Dropout(float(config.dropout))

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gc1(x, adj))
        h = self.dropout(h)
        return self.gc2(h, adj)

