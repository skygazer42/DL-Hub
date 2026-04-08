import torch
import torch.nn.functional as F
from torch import nn


def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, *, temperature: float = 0.2) -> torch.Tensor:
    """NT-Xent (SimCLR) loss for two views.

    Args:
        z1, z2: (B, D) projected features for two augmented views.
        temperature: softmax temperature.
    """

    if z1.ndim != 2 or z2.ndim != 2:
        raise ValueError(
            f"Expected z1/z2 shapes (B, D), got {tuple(z1.shape)} and {tuple(z2.shape)}"
        )
    if z1.shape != z2.shape:
        raise ValueError("z1 and z2 must have the same shape")

    b, d = z1.shape
    if b < 2:
        raise ValueError("Batch size must be >= 2 for contrastive loss")

    t = float(temperature)
    if t <= 0:
        raise ValueError("temperature must be > 0")

    z = torch.cat([z1, z2], dim=0)  # (2B, D)
    z = F.normalize(z, dim=1)

    sim = torch.matmul(z, z.t()) / t  # (2B, 2B)
    # Mask self-similarity.
    sim = sim - torch.eye(2 * b, device=sim.device, dtype=sim.dtype) * 1e9

    # Positive pairs: i <-> i+B.
    targets = torch.arange(2 * b, device=sim.device)
    targets = (targets + b) % (2 * b)

    loss = F.cross_entropy(sim, targets)
    return loss


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
        self.net = nn.Sequential(
            nn.Linear(d_in, h),
            nn.ReLU(inplace=True),
            nn.Linear(h, d_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SimCLR2PointNet(nn.Module):
    """SimCLR model: PointNet encoder + projection head.

    Forward returns a dict:
    - h: (B, embed_dim) encoder features (recommended for downstream)
    - z: (B, proj_dim) projected features (use for contrastive loss)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_features: int = 64,
        embed_dim: int = 128,
        proj_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = PointNetGlobalEncoder(
            in_channels=int(in_channels),
            hidden_features=int(hidden_features),
            embed_dim=int(embed_dim),
            dropout=float(dropout),
        )
        self.projector = ProjectionHead(int(embed_dim), int(proj_dim), hidden_dim=int(embed_dim))

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(points)
        z = self.projector(h)
        return {"h": h, "z": z}


_VARIANTS: dict[str, dict] = {
    "simclr2_pointnet_tiny": {"hidden": 32, "embed": 64, "proj": 32, "dropout": 0.0},
    "simclr2_pointnet_small": {"hidden": 64, "embed": 128, "proj": 64, "dropout": 0.0},
    "simclr2_pointnet_base": {"hidden": 96, "embed": 192, "proj": 96, "dropout": 0.0},
}


def build_simclr2_pointnet(
    *,
    in_channels: int,
    variant: str = "simclr2_pointnet_small",
    dropout: float | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown SimCLR-PointNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return SimCLR2PointNet(
        in_channels=int(in_channels),
        hidden_features=int(spec["hidden"]),
        embed_dim=int(spec["embed"]),
        proj_dim=int(spec["proj"]),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    pts = torch.randn(8, 128, 3)
    m = build_simclr2_pointnet(in_channels=3, variant="simclr2_pointnet_tiny")
    out1 = m(pts)
    out2 = m(pts + torch.randn_like(pts) * 0.01)
    loss = nt_xent_loss(out1["z"], out2["z"], temperature=0.2)
    loss.backward()
    print("ok", float(loss.item()))

