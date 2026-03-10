from dataclasses import dataclass

import torch
from torch import nn


def _index_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Index points with idx.

    points: (B, N, C)
    idx: (B, S) or (B, S, K)
    """

    b = points.shape[0]
    batch_indices = torch.arange(b, device=points.device).view(b, 1, 1)
    if idx.ndim == 2:
        batch_indices = batch_indices[:, :, 0]
        return points[batch_indices, idx, :]
    if idx.ndim == 3:
        batch_indices = batch_indices.expand(-1, idx.shape[1], idx.shape[2])
        return points[batch_indices, idx, :]
    raise ValueError(f"idx must be 2D or 3D, got {idx.ndim}D")


def _farthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest point sampling indices.

    xyz: (B, N, 3)
    returns: (B, npoint) indices
    """

    if xyz.ndim != 3 or xyz.shape[-1] != 3:
        raise ValueError(f"Expected xyz shape (B, N, 3), got {tuple(xyz.shape)}")
    b, n, _ = xyz.shape
    npoint = int(npoint)
    if npoint <= 0 or npoint > n:
        raise ValueError(f"npoint must be in [1, N], got {npoint} with N={n}")

    centroids = torch.zeros((b, npoint), dtype=torch.long, device=xyz.device)
    distance = torch.full((b, n), 1e10, device=xyz.device, dtype=xyz.dtype)
    farthest = torch.randint(0, n, (b,), device=xyz.device)
    batch_indices = torch.arange(b, device=xyz.device)

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(b, 1, 3)
        dist = ((xyz - centroid) ** 2).sum(dim=-1)
        mask = dist < distance
        distance = torch.where(mask, dist, distance)
        farthest = torch.max(distance, dim=1).indices

    return centroids


def _knn_point(k: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
    """kNN search.

    xyz: (B, N, 3)
    new_xyz: (B, S, 3)
    returns idx: (B, S, k)
    """

    if xyz.ndim != 3 or new_xyz.ndim != 3:
        raise ValueError("xyz and new_xyz must be 3D tensors")
    b, n, _ = xyz.shape
    _, s, _ = new_xyz.shape
    k = int(k)
    if k <= 0 or k > n:
        raise ValueError(f"k must be in [1, N], got k={k} with N={n}")

    dist = torch.cdist(new_xyz, xyz)  # (B, S, N)
    idx = dist.topk(k=k, dim=-1, largest=False).indices
    return idx


class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_c: int, out_c: int) -> None:
        super().__init__(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
        )


class SetAbstraction(nn.Module):
    def __init__(self, *, npoint: int, k: int, in_channels: int, mlp: list[int]) -> None:
        super().__init__()
        self.npoint = int(npoint)
        self.k = int(k)

        layers: list[nn.Module] = []
        last_c = 3 + int(in_channels)
        for out_c in mlp:
            layers.append(_ConvBNReLU(last_c, int(out_c)))
            last_c = int(out_c)
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # xyz: (B, N, 3), features: (B, N, C) or None
        fps_idx = _farthest_point_sample(xyz, self.npoint)  # (B, S)
        new_xyz = _index_points(xyz, fps_idx)  # (B, S, 3)

        idx = _knn_point(self.k, xyz, new_xyz)  # (B, S, k)
        grouped_xyz = _index_points(xyz, idx)  # (B, S, k, 3)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if features is not None:
            grouped_features = _index_points(features, idx)  # (B, S, k, C)
            new_points = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)
        else:
            new_points = grouped_xyz_norm

        # (B, C_in, S, k)
        new_points = new_points.permute(0, 3, 1, 2).contiguous()
        new_points = self.mlp(new_points)  # (B, C_out, S, k)
        new_points = torch.max(new_points, dim=-1).values  # (B, C_out, S)
        new_features = new_points.transpose(1, 2).contiguous()  # (B, S, C_out)
        return new_xyz, new_features


@dataclass(frozen=True)
class ModelConfig:
    npoint1: int = 32
    k1: int = 16
    npoint2: int = 8
    k2: int = 8

    hidden_features: int = 64
    dropout: float = 0.1
    num_classes: int = 2


class PointNet2Classifier(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        h = int(cfg.hidden_features)
        self.sa1 = SetAbstraction(npoint=cfg.npoint1, k=cfg.k1, in_channels=0, mlp=[h, h])
        self.sa2 = SetAbstraction(npoint=cfg.npoint2, k=cfg.k2, in_channels=h, mlp=[h * 2, h * 4])

        self.head = nn.Sequential(
            nn.Linear(h * 4, h * 2),
            nn.ReLU(),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(h * 2, int(cfg.num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"Expected points shape (B, N, 3), got {tuple(points.shape)}")

        xyz = points
        features = None
        xyz, features = self.sa1(xyz, features)  # (B, S1, 3), (B, S1, h)
        xyz, features = self.sa2(xyz, features)  # (B, S2, 3), (B, S2, h*4)

        global_feat = torch.max(features, dim=1).values  # (B, h*4)
        return self.head(global_feat)


__all__ = ["PointNet2Classifier", "ModelConfig"]
