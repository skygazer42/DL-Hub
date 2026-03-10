from dataclasses import dataclass

import torch
from torch import nn

from ..ops import farthest_point_sample, index_points, knn_query
from .utils import ConvBNAct1d, ConvBNAct2d, _c, global_max_pool


@dataclass(frozen=True)
class PointNetConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    use_tnet: bool = False


class TNet(nn.Module):
    """A tiny transform network to predict an affine matrix (k x k)."""

    def __init__(self, k: int, *, width_mult: float, dropout: float) -> None:
        super().__init__()
        kk = int(k)
        if kk <= 0:
            raise ValueError("k must be > 0")
        self.k = kk

        h = _c(64, float(width_mult), min_ch=16, divisor=8)
        h2 = _c(128, float(width_mult), min_ch=32, divisor=8)
        h3 = _c(256, float(width_mult), min_ch=64, divisor=8)
        self.mlp = nn.Sequential(
            ConvBNAct1d(kk, h, act="relu"),
            ConvBNAct1d(h, h2, act="relu"),
            ConvBNAct1d(h2, h3, act="relu", dropout=float(dropout)),
        )
        self.fc = nn.Sequential(
            nn.Linear(h3, max(8, h3 // 2)),
            nn.ReLU(inplace=True),
            nn.Linear(max(8, h3 // 2), kk * kk),
        )

        # Initialize close to identity.
        with torch.no_grad():
            last = self.fc[-1]
            if isinstance(last, nn.Linear):
                nn.init.zeros_(last.weight)
                nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, k, N)
        if x.ndim != 3 or int(x.shape[1]) != int(self.k):
            raise ValueError(f"Expected x shape (B, {self.k}, N), got {tuple(x.shape)}")
        y = self.mlp(x.to(torch.float32))
        y = global_max_pool(y)  # (B, C)
        y = self.fc(y).view(x.shape[0], self.k, self.k)
        eye = torch.eye(self.k, device=y.device, dtype=y.dtype).unsqueeze(0)
        return y + eye


class PointNetClassifier(nn.Module):
    """A minimal PointNet-style classifier.

    Input: (B, N, 3) or (B, N, C) where C == cfg.in_channels.
    """

    def __init__(self, cfg: PointNetConfig) -> None:
        super().__init__()
        self.cfg = cfg

        c_in = int(cfg.in_channels)
        h = _c(64, float(cfg.width_mult), min_ch=16, divisor=8)
        h2 = _c(128, float(cfg.width_mult), min_ch=32, divisor=8)
        h3 = _c(256, float(cfg.width_mult), min_ch=64, divisor=8)

        self.mlp = nn.Sequential(
            ConvBNAct1d(c_in, h, act="relu", dropout=0.0),
            ConvBNAct1d(h, h2, act="relu", dropout=0.0),
            ConvBNAct1d(h2, h3, act="relu", dropout=float(cfg.dropout)),
        )
        self.head = nn.Sequential(
            nn.Linear(h3, _c(256, float(cfg.width_mult), min_ch=64, divisor=8)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(_c(256, float(cfg.width_mult), min_ch=64, divisor=8), int(cfg.num_classes)),
        )
        self.tnet = (
            TNet(3, width_mult=float(cfg.width_mult), dropout=float(cfg.dropout))
            if bool(cfg.use_tnet)
            else None
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError(f"Expected points shape (B, N, C), got {tuple(points.shape)}")
        if int(points.shape[-1]) != int(self.cfg.in_channels):
            raise ValueError(
                f"Expected in_channels={self.cfg.in_channels}, got C={points.shape[-1]}"
            )

        p = points.to(torch.float32)
        if self.tnet is not None:
            if p.shape[-1] < 3:
                raise ValueError("pointnet_tnet requires at least 3 input dims for xyz")
            xyz = p[..., :3].transpose(1, 2).contiguous()  # (B, 3, N)
            t = self.tnet(xyz)  # (B, 3, 3)
            xyz_t = torch.bmm(p[..., :3], t.transpose(1, 2))  # (B, N, 3)
            p = torch.cat([xyz_t, p[..., 3:]], dim=-1)

        x = p.transpose(1, 2).contiguous()  # (B, C, N)
        x = self.mlp(x)
        x = global_max_pool(x)
        return self.head(x)


def build_pointnet_classifier(
    *,
    in_channels: int,
    num_classes: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "pointnet",
) -> nn.Module:
    name = str(variant).lower().strip()
    use_tnet = name in {"pointnet_tnet", "tnet", "pointnet+tnet"}
    return PointNetClassifier(
        PointNetConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            use_tnet=bool(use_tnet),
        )
    )


@dataclass(frozen=True)
class DeepSetsConfig:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    pool: str = "max"  # max|mean


class DeepSetsClassifier(nn.Module):
    """DeepSets baseline: phi(points) -> pool -> rho."""

    def __init__(self, cfg: DeepSetsConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c_in = int(cfg.in_channels)
        h = _c(64, float(cfg.width_mult), min_ch=16, divisor=8)
        h2 = _c(128, float(cfg.width_mult), min_ch=32, divisor=8)
        self.phi = nn.Sequential(
            ConvBNAct1d(c_in, h, act="relu"),
            ConvBNAct1d(h, h2, act="relu", dropout=float(cfg.dropout)),
        )
        self.rho = nn.Sequential(
            nn.Linear(h2, h2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(h2, int(cfg.num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError(f"Expected points shape (B, N, C), got {tuple(points.shape)}")
        if int(points.shape[-1]) != int(self.cfg.in_channels):
            raise ValueError(
                f"Expected in_channels={self.cfg.in_channels}, got C={points.shape[-1]}"
            )

        x = points.to(torch.float32).transpose(1, 2).contiguous()  # (B, C, N)
        x = self.phi(x)  # (B, C2, N)
        if str(self.cfg.pool).lower().strip() == "mean":
            pooled = x.mean(dim=-1)
        else:
            pooled = global_max_pool(x)
        return self.rho(pooled)


def build_deepsets_classifier(
    *,
    in_channels: int,
    num_classes: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str = "deepsets",
) -> nn.Module:
    name = str(variant).lower().strip()
    pool = "max"
    if name.endswith("_mean"):
        pool = "mean"
    return DeepSetsClassifier(
        DeepSetsConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            pool=str(pool),
        )
    )


class _ConvBNReLU(nn.Sequential):
    def __init__(self, in_c: int, out_c: int, *, dropout: float) -> None:
        super().__init__(
            ConvBNAct2d(int(in_c), int(out_c), act="relu", dropout=float(dropout)),
        )


class SetAbstraction(nn.Module):
    def __init__(
        self, *, npoint: int, k: int, in_channels: int, mlp: list[int], dropout: float
    ) -> None:
        super().__init__()
        self.npoint = int(npoint)
        self.k = int(k)

        layers: list[nn.Module] = []
        last_c = 3 + int(in_channels)
        for out_c in mlp:
            layers.append(_ConvBNReLU(last_c, int(out_c), dropout=float(dropout)))
            last_c = int(out_c)
        self.mlp = nn.Sequential(*layers)

    def forward(
        self, xyz: torch.Tensor, features: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # xyz: (B, N, 3), features: (B, N, C) or None
        fps_idx = farthest_point_sample(xyz, self.npoint)  # (B, S)
        new_xyz = index_points(xyz, fps_idx)  # (B, S, 3)

        idx = knn_query(self.k, xyz, new_xyz)  # (B, S, k)
        grouped_xyz = index_points(xyz, idx)  # (B, S, k, 3)
        grouped_xyz_norm = grouped_xyz - new_xyz.unsqueeze(2)

        if features is not None:
            grouped_features = index_points(features, idx)  # (B, S, k, C)
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
class PointNet2Config:
    in_channels: int
    num_classes: int
    width_mult: float
    dropout: float
    npoint1: int
    k1: int
    npoint2: int
    k2: int


class PointNet2Classifier(nn.Module):
    def __init__(self, cfg: PointNet2Config) -> None:
        super().__init__()
        self.cfg = cfg

        h = _c(64, float(cfg.width_mult), min_ch=16, divisor=8)
        self.sa1 = SetAbstraction(
            npoint=cfg.npoint1, k=cfg.k1, in_channels=0, mlp=[h, h], dropout=cfg.dropout
        )
        self.sa2 = SetAbstraction(
            npoint=cfg.npoint2,
            k=cfg.k2,
            in_channels=h,
            mlp=[h * 2, h * 4],
            dropout=cfg.dropout,
        )

        out_dim = int(h * 4)
        self.head = nn.Sequential(
            nn.Linear(out_dim, max(8, out_dim // 2)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(cfg.dropout)),
            nn.Linear(max(8, out_dim // 2), int(cfg.num_classes)),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3 or points.shape[-1] != int(self.cfg.in_channels):
            raise ValueError(
                f"Expected points shape (B, N, C={self.cfg.in_channels}), got {tuple(points.shape)}"
            )

        xyz = points[..., :3].to(torch.float32)
        features = None
        xyz, features = self.sa1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        global_feat = torch.max(features, dim=1).values
        return self.head(global_feat)


def build_pointnet2_classifier(
    *,
    in_channels: int,
    num_classes: int,
    num_points: int,
    width_mult: float = 1.0,
    dropout: float = 0.1,
    variant: str,
) -> nn.Module:
    name = str(variant).lower().strip()
    n = int(num_points)
    if n <= 0:
        raise ValueError("num_points must be > 0")

    if name in {"pointnet2_ssg", "ssg", "pointnet2"}:
        npoint1, k1, npoint2, k2 = max(16, n // 2), 16, max(8, n // 8), 8
    elif name in {"pointnet2_msg", "msg"}:
        # Keep a similar footprint; MSG is handled as a slightly larger SSG here.
        npoint1, k1, npoint2, k2 = max(16, n // 2), 24, max(8, n // 8), 12
    else:
        raise ValueError("Unknown PointNet2 variant. Supported: pointnet2_ssg|pointnet2_msg")

    return PointNet2Classifier(
        PointNet2Config(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            width_mult=float(width_mult),
            dropout=float(dropout),
            npoint1=int(npoint1),
            k1=int(k1),
            npoint2=int(npoint2),
            k2=int(k2),
        )
    )


__all__ = [
    "build_deepsets_classifier",
    "build_pointnet2_classifier",
    "build_pointnet_classifier",
]
