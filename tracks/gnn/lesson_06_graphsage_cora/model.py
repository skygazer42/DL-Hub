
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_features: int
    hidden_features: int
    num_classes: int
    dropout: float = 0.5


class GraphSAGELayer(nn.Module):
    """Mean-aggregator GraphSAGE layer using row-normalized adjacency."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features * 2, out_features)

    def forward(self, x: torch.Tensor, adj_row: torch.Tensor) -> torch.Tensor:
        neigh = torch.sparse.mm(adj_row, x)
        h = torch.cat([x, neigh], dim=1)
        return self.linear(h)


class GraphSAGE(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layer1 = GraphSAGELayer(cfg.in_features, cfg.hidden_features)
        self.layer2 = GraphSAGELayer(cfg.hidden_features, cfg.num_classes)
        self.dropout = float(cfg.dropout)

    def forward(self, x: torch.Tensor, adj_row: torch.Tensor) -> torch.Tensor:
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = torch.nn.functional.relu(self.layer1(x, adj_row))
        x = torch.nn.functional.dropout(x, p=self.dropout, training=self.training)
        x = self.layer2(x, adj_row)
        return x

