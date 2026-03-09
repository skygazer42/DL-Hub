
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class ModelConfig:
    in_features: int = 2
    hidden_features: int = 32
    num_heads: int = 4
    num_classes: int = 2
    dropout: float = 0.1
    alpha: float = 0.2  # LeakyReLU negative slope


class GATLayer(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        num_heads: int,
        dropout: float,
        alpha: float,
        concat: bool,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_heads = int(num_heads)
        self.concat = bool(concat)

        self.w = nn.Linear(self.in_features, self.out_features * self.num_heads, bias=False)
        self.a_src = nn.Parameter(torch.empty(self.num_heads, self.out_features))
        self.a_dst = nn.Parameter(torch.empty(self.num_heads, self.out_features))
        self.leaky_relu = nn.LeakyReLU(float(alpha))
        self.dropout = nn.Dropout(float(dropout))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.w.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        # x: (B, N, Fin), adj: (B, N, N) with 0/1 entries
        b, n, _ = x.shape

        # Add self loops (GAT normally includes i -> i).
        eye = torch.eye(n, device=adj.device, dtype=adj.dtype).expand(b, n, n)
        adj = (adj + eye).clamp(max=1.0)
        mask = adj > 0

        h = self.w(x)  # (B, N, H*out)
        h = h.view(b, n, self.num_heads, self.out_features)  # (B, N, H, F)

        # Additive attention: e_ij = LeakyReLU(a_src^T h_i + a_dst^T h_j)
        f1 = (h * self.a_src).sum(dim=-1)  # (B, N, H)
        f2 = (h * self.a_dst).sum(dim=-1)  # (B, N, H)
        e = self.leaky_relu(f1.unsqueeze(2) + f2.unsqueeze(1))  # (B, N, N, H)

        neg_inf = torch.finfo(e.dtype).min
        e = e.masked_fill(~mask.unsqueeze(-1), neg_inf)

        alpha = torch.softmax(e, dim=2)  # (B, N, N, H)
        alpha = self.dropout(alpha)

        # Weighted sum of neighbor representations.
        alpha = alpha.permute(0, 3, 1, 2)  # (B, H, N, N)
        h_heads = h.permute(0, 2, 1, 3)  # (B, H, N, F)
        out = torch.matmul(alpha, h_heads)  # (B, H, N, F)
        out = out.permute(0, 2, 1, 3)  # (B, N, H, F)

        if self.concat:
            return out.reshape(b, n, self.num_heads * self.out_features)
        return out.mean(dim=2)


class GATGraphClassifier(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.dropout = nn.Dropout(float(config.dropout))
        self.gat1 = GATLayer(
            in_features=config.in_features,
            out_features=config.hidden_features,
            num_heads=config.num_heads,
            dropout=config.dropout,
            alpha=config.alpha,
            concat=True,
        )
        self.gat2 = GATLayer(
            in_features=config.hidden_features * config.num_heads,
            out_features=config.hidden_features,
            num_heads=1,
            dropout=config.dropout,
            alpha=config.alpha,
            concat=False,
        )
        self.head = nn.Linear(int(config.hidden_features), int(config.num_classes))

    def forward(self, inputs: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        x, adj = inputs
        h = self.dropout(x)
        h = F.elu(self.gat1(h, adj))
        h = self.dropout(h)
        h = F.elu(self.gat2(h, adj))
        g = h.mean(dim=1)  # graph-level mean pooling
        return self.head(g)

