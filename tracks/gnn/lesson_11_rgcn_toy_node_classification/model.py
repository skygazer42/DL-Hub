from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ModelConfig:
    in_features: int
    hidden_features: int
    num_classes: int
    num_rels: int
    num_bases: int = -1
    dropout: float = 0.1


class RGCNLayer(nn.Module):
    def __init__(
        self,
        *,
        in_features: int,
        out_features: int,
        num_rels: int,
        num_bases: int,
        activation: nn.Module | None,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.num_rels = int(num_rels)

        if int(num_bases) <= 0 or int(num_bases) > int(num_rels):
            num_bases = int(num_rels)
        self.num_bases = int(num_bases)

        # Basis decomposition: W_r = sum_b a_{r,b} V_b
        self.bases = nn.Parameter(torch.empty((self.num_bases, self.in_features, self.out_features)))
        self.coeff: nn.Parameter | None = None
        if self.num_bases < self.num_rels:
            self.coeff = nn.Parameter(torch.empty((self.num_rels, self.num_bases)))

        self.bias = nn.Parameter(torch.zeros((self.out_features,)))
        self.activation = activation
        self.dropout = nn.Dropout(float(dropout))

        nn.init.xavier_uniform_(self.bases)
        if self.coeff is not None:
            nn.init.xavier_uniform_(self.coeff)

    def _relation_weights(self) -> torch.Tensor:
        if self.coeff is None:
            return self.bases  # (R, in, out) when num_bases == num_rels

        # (R, B) @ (B, in, out) -> (R, in, out)
        b = self.bases.view(self.num_bases, -1)  # (B, in*out)
        w = self.coeff @ b  # (R, in*out)
        return w.view(self.num_rels, self.in_features, self.out_features)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor,
        edge_norm: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x.to(torch.float32)
        edge_index = edge_index.to(torch.long)
        edge_type = edge_type.to(torch.long)
        if edge_norm is not None:
            edge_norm = edge_norm.to(torch.float32)

        num_nodes = int(x.shape[0])
        weights = self._relation_weights()

        out = torch.zeros((num_nodes, self.out_features), device=x.device, dtype=torch.float32)
        src_all = edge_index[0]
        dst_all = edge_index[1]

        for rel in range(self.num_rels):
            mask = edge_type == int(rel)
            if not torch.any(mask):
                continue
            src = src_all[mask]
            dst = dst_all[mask]
            msg = x[src] @ weights[rel]  # (E_r, out)
            if edge_norm is not None:
                msg = msg * edge_norm[mask].unsqueeze(1)
            out.index_add_(0, dst, msg)

        out = out + self.bias
        if self.activation is not None:
            out = self.activation(out)
        out = self.dropout(out)
        return out


class RGCN(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.layer1 = RGCNLayer(
            in_features=cfg.in_features,
            out_features=cfg.hidden_features,
            num_rels=cfg.num_rels,
            num_bases=cfg.num_bases,
            activation=nn.ReLU(),
            dropout=cfg.dropout,
        )
        self.layer2 = RGCNLayer(
            in_features=cfg.hidden_features,
            out_features=cfg.num_classes,
            num_rels=cfg.num_rels,
            num_bases=cfg.num_bases,
            activation=None,
            dropout=0.0,
        )

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, edge_norm: torch.Tensor | None
    ) -> torch.Tensor:
        h = self.layer1(x, edge_index=edge_index, edge_type=edge_type, edge_norm=edge_norm)
        return self.layer2(h, edge_index=edge_index, edge_type=edge_type, edge_norm=edge_norm)


__all__ = ["RGCN", "ModelConfig"]

