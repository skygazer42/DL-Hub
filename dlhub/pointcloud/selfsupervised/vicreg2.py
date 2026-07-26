import torch
import torch.nn.functional as F
from torch import nn


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("Expected square matrix for off-diagonal extraction")
    n = int(x.shape[0])
    mask = ~torch.eye(n, device=x.device, dtype=torch.bool)
    return x[mask]


def vicreg2_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    inv_weight: float = 25.0,
    var_weight: float = 25.0,
    cov_weight: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    """VICReg loss (invariance + variance + covariance).

    Args:
        z1, z2: (B, D) projected features for two augmented views.
    """

    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError(
            f"Expected z1/z2 shapes (B, D), got {tuple(z1.shape)} and {tuple(z2.shape)}"
        )
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape")
    b, d = z1.shape
    if int(b) < 2:
        raise ValueError("Batch size must be >= 2 for VICReg variance/covariance terms")
    if int(d) < 2:
        raise ValueError("Feature dim must be >= 2 for covariance term")

    inv = F.mse_loss(z1, z2)

    g = float(gamma)
    e = float(eps)
    std1 = torch.sqrt(z1.var(dim=0, unbiased=False) + e)
    std2 = torch.sqrt(z2.var(dim=0, unbiased=False) + e)
    var = 0.5 * (F.relu(g - std1).mean() + F.relu(g - std2).mean())

    z1c = z1 - z1.mean(dim=0)
    z2c = z2 - z2.mean(dim=0)
    cov1 = (z1c.t() @ z1c) / float(b - 1)
    cov2 = (z2c.t() @ z2c) / float(b - 1)
    cov = 0.5 * (
        _off_diagonal(cov1).pow(2).sum() / float(d) + _off_diagonal(cov2).pow(2).sum() / float(d)
    )

    return float(inv_weight) * inv + float(var_weight) * var + float(cov_weight) * cov


class PointNetGlobalEncoder(nn.Module):
    """PointNet-style global encoder that outputs an embedding (B, D)."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_features: int,
        embed_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        h = int(hidden_features)
        d = int(embed_dim)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if h <= 0:
            raise ValueError("hidden_features must be > 0")
        if d <= 0:
            raise ValueError("embed_dim must be > 0")

        self.mlp = nn.Sequential(
            nn.Conv1d(c_in, h, kernel_size=1, bias=False),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Conv1d(h, h * 2, kernel_size=1, bias=False),
            nn.BatchNorm1d(h * 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(h * 2, h * 4, kernel_size=1, bias=False),
            nn.BatchNorm1d(h * 4),
            nn.ReLU(inplace=True),
        )
        self.drop = nn.Dropout(p=float(dropout))
        self.proj = nn.Linear(h * 4, d)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError(f"Expected points shape (B, N, C), got {tuple(points.shape)}")
        x = points.to(torch.float32).transpose(1, 2).contiguous()  # (B, C, N)
        feat = self.mlp(x)  # (B, F, N)
        pooled = torch.max(feat, dim=2).values  # (B, F)
        pooled = self.drop(pooled)
        return self.proj(pooled)  # (B, D)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int, *, hidden_dim: int | None = None) -> None:
        super().__init__()
        d_in = int(in_dim)
        d_out = int(proj_dim)
        h = int(hidden_dim) if hidden_dim is not None else d_in
        if d_in <= 0 or d_out <= 0 or h <= 0:
            raise ValueError("in_dim/proj_dim/hidden_dim must be > 0")
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x shape (B, D), got {tuple(x.shape)}")
        return self.net(x)


class VICReg2PointNet(nn.Module):
    """VICReg model: PointNet encoder + projection head."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_features: int = 64,
        embed_dim: int = 128,
        proj_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = PointNetGlobalEncoder(
            in_channels=int(in_channels),
            hidden_features=int(hidden_features),
            embed_dim=int(embed_dim),
            dropout=float(dropout),
        )
        self.projector = ProjectionHead(int(embed_dim), int(proj_dim), hidden_dim=int(proj_dim))

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(points)
        z = self.projector(h)
        return {"h": h, "z": z}


_VARIANTS: dict[str, dict] = {
    "vicreg2_pointnet_tiny": {"hidden": 32, "embed": 64, "proj": 64, "dropout": 0.0},
    "vicreg2_pointnet_small": {"hidden": 64, "embed": 128, "proj": 128, "dropout": 0.0},
    "vicreg2_pointnet_base": {"hidden": 96, "embed": 192, "proj": 192, "dropout": 0.0},
}


def build_vicreg2_pointnet(
    *,
    in_channels: int,
    variant: str = "vicreg2_pointnet_small",
    dropout: float | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown VICReg-PointNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return VICReg2PointNet(
        in_channels=int(in_channels),
        hidden_features=int(spec["hidden"]),
        embed_dim=int(spec["embed"]),
        proj_dim=int(spec["proj"]),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    v1 = torch.randn(8, 128, 3)
    v2 = v1 + torch.randn_like(v1) * 0.01
    m = build_vicreg2_pointnet(in_channels=3, variant="vicreg2_pointnet_tiny")
    out1 = m(v1)
    out2 = m(v2)
    loss = vicreg2_loss(out1["z"], out2["z"])
    loss.backward()
    print("ok", float(loss.item()))

