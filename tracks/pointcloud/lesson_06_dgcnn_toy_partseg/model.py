from dataclasses import dataclass

import torch
from torch import nn


def _knn_indices(x: torch.Tensor, k: int) -> torch.Tensor:
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
            nn.Conv2d(int(in_channels), int(out_channels), kernel_size=1, bias=False),
            nn.BatchNorm2d(int(out_channels)),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=float(dropout)),
        )

    def forward(self, edge_feat: torch.Tensor) -> torch.Tensor:
        x = self.net(edge_feat)  # (B, out, N, k)
        return torch.max(x, dim=-1).values  # (B, out, N)


@dataclass(frozen=True)
class ModelConfig:
    k: int = 10
    hidden_features: int = 64
    dropout: float = 0.1
    num_classes: int = 2
    dynamic_graph: bool = True


class DGCNNPartSeg(nn.Module):
    """Toy DGCNN-style part segmentation.

    Input: points (B, N, 3)
    Output: logits (B, N, num_classes)
    """

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

        self.seg_head = nn.Sequential(
            nn.Conv1d(h * 8, h * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(h * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Conv1d(h * 2, int(cfg.num_classes), kernel_size=1, bias=True),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected points shape (B, N, 3), got {tuple(points.shape)}")

        x0 = points.to(torch.float32).transpose(1, 2).contiguous()  # (B, 3, N)

        idx = _knn_indices(x0, k=int(self.cfg.k))
        x1 = self.edge1(_graph_feature(x0, idx))  # (B, h, N)

        x_for_graph = x1 if self.cfg.dynamic_graph else x0
        idx = _knn_indices(x_for_graph, k=int(self.cfg.k))
        x2 = self.edge2(_graph_feature(x1, idx))  # (B, h, N)

        x_for_graph = x2 if self.cfg.dynamic_graph else x0
        idx = _knn_indices(x_for_graph, k=int(self.cfg.k))
        x3 = self.edge3(_graph_feature(x2, idx))  # (B, 2h, N)

        x_cat = torch.cat([x1, x2, x3], dim=1)
        x_feat = self.fuse(x_cat)  # (B, 4h, N)
        global_feat = torch.max(x_feat, dim=2, keepdim=True).values  # (B, 4h, 1)
        global_feat = global_feat.expand(-1, -1, x_feat.shape[2])  # (B, 4h, N)
        fused = torch.cat([x_feat, global_feat], dim=1)  # (B, 8h, N)
        logits = self.seg_head(fused)  # (B, C, N)
        return logits.transpose(1, 2).contiguous()  # (B, N, C)


__all__ = ["DGCNNPartSeg", "ModelConfig"]
