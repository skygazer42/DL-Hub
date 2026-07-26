import torch
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(
        self, *, in_channels: int, hidden_features: int, latent_dim: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        h = int(hidden_features)
        z = int(latent_dim)
        if c_in <= 0:
            raise ValueError("in_channels must be > 0")
        if h <= 0:
            raise ValueError("hidden_features must be > 0")
        if z <= 0:
            raise ValueError("latent_dim must be > 0")

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
        self.to_latent = nn.Linear(h * 4, z)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        if points.ndim != 3:
            raise ValueError(f"Expected points shape (B, N, C), got {tuple(points.shape)}")
        x = points.to(torch.float32).transpose(1, 2).contiguous()  # (B, C, N)
        feat = self.mlp(x)  # (B, F, N)
        global_feat = torch.max(feat, dim=2).values  # (B, F)
        global_feat = self.drop(global_feat)
        return self.to_latent(global_feat)  # (B, Z)


class PointNetDecoder(nn.Module):
    def __init__(
        self, *, latent_dim: int, hidden_features: int, num_points: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        z = int(latent_dim)
        h = int(hidden_features)
        n = int(num_points)
        if z <= 0:
            raise ValueError("latent_dim must be > 0")
        if h <= 0:
            raise ValueError("hidden_features must be > 0")
        if n <= 0:
            raise ValueError("num_points must be > 0")

        self.num_points = n
        self.mlp = nn.Sequential(
            nn.Linear(z, h * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(h * 4, h * 8),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(h * 8, n * 3),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2:
            raise ValueError(f"Expected latent shape (B, Z), got {tuple(latent.shape)}")
        b = latent.shape[0]
        pts = self.mlp(latent.to(torch.float32)).view(b, self.num_points, 3)
        return pts


class ClipReconAutoEncoder(nn.Module):
    """PointNet-style autoencoder for point cloud reconstruction."""

    def __init__(
        self,
        *,
        in_channels: int,
        num_points: int,
        hidden_features: int = 64,
        latent_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = PointNetEncoder(
            in_channels=int(in_channels),
            hidden_features=int(hidden_features),
            latent_dim=int(latent_dim),
            dropout=float(dropout),
        )
        self.decoder = PointNetDecoder(
            latent_dim=int(latent_dim),
            hidden_features=int(hidden_features),
            num_points=int(num_points),
            dropout=float(dropout),
        )

    def encode(self, points: torch.Tensor) -> torch.Tensor:
        return self.encoder(points)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        z = self.encode(points)
        return self.decode(z)


_VARIANTS: dict[str, dict] = {
    "clip_recon_tiny": {"hidden": 32, "latent": 64, "dropout": 0.0},
    "clip_recon_small": {"hidden": 64, "latent": 128, "dropout": 0.0},
    "clip_recon_base": {"hidden": 96, "latent": 192, "dropout": 0.0},
}


def build_clip_recon_autoencoder(
    *,
    in_channels: int,
    num_points: int,
    variant: str = "clip_recon_small",
    dropout: float | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown PointNet-AE variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return ClipReconAutoEncoder(
        in_channels=int(in_channels),
        num_points=int(num_points),
        hidden_features=int(spec["hidden"]),
        latent_dim=int(spec["latent"]),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(2, 128, 3)
    m = build_clip_recon_autoencoder(in_channels=3, num_points=128, variant="clip_recon_tiny")
    y = m(x)
    print("clip_recon_tiny", tuple(y.shape))
    loss = y.mean()
    loss.backward()
    print("ok")


