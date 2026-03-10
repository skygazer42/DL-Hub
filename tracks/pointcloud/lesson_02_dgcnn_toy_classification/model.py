from dataclasses import dataclass

import torch
from torch import nn


def _knn_indices(x: torch.Tensor, k: int) -> torch.Tensor:
    """Return kNN indices for x.

    Args:
        x: (B, C, N) features
        k: number of neighbors, must be < N

    Returns:
        idx: (B, N, k) indices
    """

    if x.ndim != 3:
        raise ValueError(f"Expected x shape (B, C, N), got {tuple(x.shape)}")
    b, _, n = x.shape
    k = int(k)
    if k <= 0 or k >= n:
        raise ValueError(f"k must be in [1, N-1], got k={k} with N={n}")

    xt = x.transpose(1, 2)  # (B, N, C)
    dist = torch.cdist(xt, xt)  # (B, N, N)
    dist = dist + torch.eye(n, device=x.device, dtype=dist.dtype).unsqueeze(0) * 1e9
    idx = dist.topk(k=k, dim=-1, largest=False).indices  # (B, N, k)
    return idx


def _graph_feature(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Build edge features.

    Args:
        x: (B, C, N)
        idx: (B, N, k)

    Returns:
        edge: (B, 2C, N, k) where edge = concat(x_i, x_j - x_i)
    """

    if x.ndim != 3:
        raise ValueError(f"Expected x shape (B, C, N), got {tuple(x.shape)}")
    if idx.ndim != 3:
        raise ValueError(f"Expected idx shape (B, N, k), got {tuple(idx.shape)}")

    b, c, n = x.shape
    _, n2, k = idx.shape
    if n2 != n:
        raise ValueError("idx N dimension must match x")

    x_t = x.transpose(1, 2).contiguous()  # (B, N, C)
    idx_base = (torch.arange(b, device=x.device).view(-1, 1, 1) * n).to(idx.dtype)
    idx_flat = (idx + idx_base).view(-1)  # (B*N*k,)

    neighbors = x_t.view(b * n, c)[idx_flat].view(b, n, k, c)
    central = x_t.view(b, n, 1, c).expand(-1, -1, k, -1)
    edge = torch.cat([central, neighbors - central], dim=-1)  # (B, N, k, 2C)
    return edge.permute(0, 3, 1, 2).contiguous()  # (B, 2C, N, k)


class EdgeConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, edge_feat: torch.Tensor) -> torch.Tensor:
        # edge_feat: (B, 2C, N, k)
        x = self.net(edge_feat)  # (B, out, N, k)
        x = torch.max(x, dim=-1).values  # (B, out, N)
        return x


@dataclass(frozen=True)
class ModelConfig:
    k: int = 10
    hidden_features: int = 64
    dropout: float = 0.1
    num_classes: int = 2
    dynamic_graph: bool = True


class DGCNNClassifier(nn.Module):
    """A small DGCNN-style classifier for toy point clouds."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        h = int(cfg.hidden_features)

        self.edge1 = EdgeConv(in_channels=3 * 2, out_channels=h, dropout=cfg.dropout)
        self.edge2 = EdgeConv(in_channels=h * 2, out_channels=h, dropout=cfg.dropout)
        self.edge3 = EdgeConv(in_channels=h * 2, out_channels=h * 2, dropout=cfg.dropout)

        self.fuse = nn.Sequential(
            nn.Conv1d(h + h + h * 2, h * 4, kernel_size=1, bias=False),
            nn.BatchNorm1d(h * 4),
            nn.LeakyReLU(0.2),
        )

        self.head = nn.Sequential(
            nn.Linear(h * 4, h * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(h * 2, int(cfg.num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected points shape (B, N, 3), got {tuple(points.shape)}")

        # (B, 3, N)
        x = points.transpose(1, 2).contiguous()

        idx = _knn_indices(x, k=int(self.cfg.k))
        e = _graph_feature(x, idx)
        x1 = self.edge1(e)

        x_for_graph = x1 if self.cfg.dynamic_graph else x
        idx = _knn_indices(x_for_graph, k=int(self.cfg.k))
        e = _graph_feature(x1, idx)
        x2 = self.edge2(e)

        x_for_graph = x2 if self.cfg.dynamic_graph else x
        idx = _knn_indices(x_for_graph, k=int(self.cfg.k))
        e = _graph_feature(x2, idx)
        x3 = self.edge3(e)

        x_cat = torch.cat([x1, x2, x3], dim=1)  # (B, C, N)
        x_feat = self.fuse(x_cat)  # (B, C2, N)
        pooled = torch.max(x_feat, dim=2).values  # (B, C2)
        return self.head(pooled)


__all__ = ["DGCNNClassifier", "ModelConfig"]
