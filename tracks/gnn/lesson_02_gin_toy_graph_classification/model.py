from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    in_features: int = 2
    hidden_features: int = 32
    num_layers: int = 3
    num_mlp_layers: int = 2
    num_classes: int = 2

    neighbor_pooling: str = "sum"  # sum | mean | max
    graph_pooling: str = "mean"  # sum | mean | max
    learn_eps: bool = False
    dropout: float = 0.0


class MLP(nn.Module):
    def __init__(self, *, in_features: int, hidden_features: int, out_features: int, num_layers: int) -> None:
        super().__init__()
        num_layers = int(num_layers)
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")

        if num_layers == 1:
            self.net = nn.Linear(in_features, out_features)
            self.out_features = out_features
            return

        layers: list[nn.Module] = []
        layers.append(nn.Linear(in_features, hidden_features))
        layers.append(nn.ReLU())
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_features, hidden_features))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)
        self.out_features = out_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _neighbor_aggregate(h: torch.Tensor, adj: torch.Tensor, *, mode: str) -> torch.Tensor:
    """Aggregate neighbor node representations for each node.

    h: (B, N, F)
    adj: (B, N, N) with 0/1 entries and no self loops.
    """

    mode = mode.lower().strip()
    if mode in {"sum", "mean"}:
        neigh = torch.bmm(adj, h)
        if mode == "mean":
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
            neigh = neigh / deg
        return neigh

    if mode == "max":
        b, n, f = h.shape
        # For each target node v, take max over its neighbor set u.
        neigh_feat = h.unsqueeze(1).expand(b, n, n, f)
        mask = adj.unsqueeze(-1).to(torch.bool)
        neg_inf = torch.finfo(h.dtype).min
        masked = neigh_feat.masked_fill(~mask, neg_inf)
        out = masked.max(dim=2).values
        # If a node has no neighbors (shouldn't happen here), replace -inf with 0.
        out = torch.where(torch.isfinite(out), out, torch.zeros_like(out))
        return out

    raise ValueError(f"Unsupported neighbor_pooling: {mode!r} (expected sum/mean/max)")


def _graph_pool(h: torch.Tensor, *, mode: str) -> torch.Tensor:
    mode = mode.lower().strip()
    if mode == "mean":
        return h.mean(dim=1)
    if mode == "sum":
        return h.sum(dim=1)
    if mode == "max":
        return h.max(dim=1).values
    raise ValueError(f"Unsupported graph_pooling: {mode!r} (expected sum/mean/max)")


class GINLayer(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        num_mlp_layers: int,
        neighbor_pooling: str,
        learn_eps: bool,
    ) -> None:
        super().__init__()
        self.neighbor_pooling = neighbor_pooling
        self.learn_eps = bool(learn_eps)
        if self.learn_eps:
            self.eps = nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer("eps", torch.zeros(1), persistent=False)

        self.mlp = MLP(
            in_features=in_features,
            hidden_features=out_features,
            out_features=out_features,
            num_layers=int(num_mlp_layers),
        )

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        neigh = _neighbor_aggregate(h, adj, mode=self.neighbor_pooling)
        out = (1.0 + self.eps) * h + neigh
        return self.mlp(out)


class GINGraphClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        layers: list[GINLayer] = []
        in_f = int(config.in_features)
        for _ in range(int(config.num_layers)):
            layers.append(
                GINLayer(
                    in_features=in_f,
                    out_features=int(config.hidden_features),
                    num_mlp_layers=int(config.num_mlp_layers),
                    neighbor_pooling=config.neighbor_pooling,
                    learn_eps=bool(config.learn_eps),
                )
            )
            in_f = int(config.hidden_features)
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(float(config.dropout))
        self.head = nn.Linear(int(config.hidden_features), int(config.num_classes))

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, adj = inputs
        h = x
        for layer in self.layers:
            h = F.relu(layer(h, adj))
        g = _graph_pool(h, mode=self.config.graph_pooling)
        g = self.dropout(g)
        return self.head(g)

