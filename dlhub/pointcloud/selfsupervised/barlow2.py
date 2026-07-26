import torch
from torch import nn


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    if x.ndim != 2 or x.shape[0] != x.shape[1]:
        raise ValueError("Expected square matrix for off-diagonal extraction")
    n = int(x.shape[0])
    mask = ~torch.eye(n, device=x.device, dtype=torch.bool)
    return x[mask]


def barlow_twins_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    *,
    lambda_offdiag: float = 0.005,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Barlow Twins redundancy reduction loss.

    Args:
        z1, z2: (B, D) projected features for two augmented views.
        lambda_offdiag: weight for off-diagonal terms.
        eps: numeric stability term for variance normalization.
    """

    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError(
            f"Expected z1/z2 shapes (B, D), got {tuple(z1.shape)} and {tuple(z2.shape)}"
        )
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape")

    b, d = z1.shape
    if int(b) < 2:
        raise ValueError("Batch size must be >= 2 for Barlow Twins")
    if int(d) < 2:
        raise ValueError("Feature dim must be >= 2 for off-diagonal term")

    e = float(eps)
    z1 = z1.to(torch.float32)
    z2 = z2.to(torch.float32)

    # Normalize each dimension across the batch (similar to BN without affine).
    z1 = (z1 - z1.mean(dim=0)) / torch.sqrt(z1.var(dim=0, unbiased=False) + e)
    z2 = (z2 - z2.mean(dim=0)) / torch.sqrt(z2.var(dim=0, unbiased=False) + e)

    c = (z1.t() @ z2) / float(b)  # (D, D)

    diag = torch.diagonal(c)
    on_diag = (diag - 1.0).pow(2).sum()
    off_diag = _off_diagonal(c).pow(2).sum()

    return on_diag + float(lambda_offdiag) * off_diag


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
    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        *,
        hidden_dim: int | None = None,
        num_layers: int = 3,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_in = int(in_dim)
        d_out = int(proj_dim)
        h = int(hidden_dim) if hidden_dim is not None else d_out
        nl = int(num_layers)
        if d_in <= 0 or d_out <= 0 or h <= 0:
            raise ValueError("in_dim/proj_dim/hidden_dim must be > 0")
        if nl < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        for i in range(nl - 1):
            layers += [
                nn.Linear(d_in if i == 0 else h, h, bias=False),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
            ]
            p = float(dropout)
            if p > 0:
                layers.append(nn.Dropout(p=p))
        layers.append(nn.Linear(h if nl > 1 else d_in, d_out, bias=True))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x shape (B, D), got {tuple(x.shape)}")
        return self.net(x)


class Barlow2PointNet(nn.Module):
    """Barlow Twins model: PointNet encoder + projector."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_features: int = 64,
        embed_dim: int = 128,
        proj_dim: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = PointNetGlobalEncoder(
            in_channels=int(in_channels),
            hidden_features=int(hidden_features),
            embed_dim=int(embed_dim),
            dropout=float(dropout),
        )
        self.projector = ProjectionHead(
            int(embed_dim),
            int(proj_dim),
            hidden_dim=int(proj_dim),
            num_layers=3,
            dropout=0.0,
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(points)
        z = self.projector(h)
        return {"h": h, "z": z}


_VARIANTS: dict[str, dict] = {
    "barlow2_pointnet_tiny": {"hidden": 32, "embed": 64, "proj": 128, "dropout": 0.0},
    "barlow2_pointnet_small": {"hidden": 64, "embed": 128, "proj": 256, "dropout": 0.0},
    "barlow2_pointnet_base": {"hidden": 96, "embed": 192, "proj": 384, "dropout": 0.0},
}


def build_barlow2_pointnet(
    *,
    in_channels: int,
    variant: str = "barlow2_pointnet_small",
    dropout: float | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown BarlowTwins-PointNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return Barlow2PointNet(
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

    m = build_barlow2_pointnet(in_channels=3, variant="barlow2_pointnet_tiny")
    o1 = m(v1)
    o2 = m(v2)
    loss = barlow_twins_loss(o1["z"], o2["z"], lambda_offdiag=0.005)
    loss.backward()
    print("ok", float(loss.item()))
