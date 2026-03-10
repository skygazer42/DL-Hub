import torch
import torch.nn.functional as F
from torch import nn


def negative_cosine_similarity(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """SimSiam negative cosine similarity (stop-grad on z).

    Args:
        p: (B, D) predictor output.
        z: (B, D) projector output (will be detached).
    """

    if p.ndim != 2 or z.ndim != 2:
        raise ValueError(f"Expected p/z shapes (B, D), got {tuple(p.shape)} and {tuple(z.shape)}")
    if p.shape != z.shape:
        raise ValueError("p and z must have the same shape")
    if int(p.shape[0]) < 2:
        raise ValueError("Batch size must be >= 2 for stable SimSiam loss")

    p = F.normalize(p, dim=1)
    z = F.normalize(z.detach(), dim=1)
    return -(p * z).sum(dim=1).mean()


def simsiam_loss(
    p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """Symmetric SimSiam loss for two views."""

    return 0.5 * (negative_cosine_similarity(p1, z2) + negative_cosine_similarity(p2, z1))


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


class MLPHead(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.0,
        final_bn: bool = False,
    ) -> None:
        super().__init__()
        d_in = int(in_dim)
        d_out = int(out_dim)
        h = int(hidden_dim) if hidden_dim is not None else d_in
        nl = int(num_layers)
        if d_in <= 0 or d_out <= 0 or h <= 0:
            raise ValueError("in_dim/out_dim/hidden_dim must be > 0")
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
        if bool(final_bn):
            layers.append(nn.BatchNorm1d(d_out, affine=False))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x shape (B, D), got {tuple(x.shape)}")
        return self.net(x)


class SimSiamPointNet(nn.Module):
    """SimSiam model: encoder + projector + predictor (no negatives, no momentum encoder)."""

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_features: int = 64,
        embed_dim: int = 128,
        proj_dim: int = 128,
        pred_dim: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        pd = int(pred_dim) if pred_dim is not None else int(proj_dim)

        self.encoder = PointNetGlobalEncoder(
            in_channels=int(in_channels),
            hidden_features=int(hidden_features),
            embed_dim=int(embed_dim),
            dropout=float(dropout),
        )
        # SimSiam uses a stronger projector; keep a 3-layer MLP toy version.
        self.projector = MLPHead(
            int(embed_dim),
            int(proj_dim),
            hidden_dim=int(proj_dim),
            num_layers=3,
            dropout=0.0,
            final_bn=True,
        )
        self.predictor = MLPHead(
            int(proj_dim),
            int(pd),
            hidden_dim=int(pd),
            num_layers=2,
            dropout=0.0,
            final_bn=False,
        )

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(points)
        z = self.projector(h)
        p = self.predictor(z)
        return {"h": h, "z": z, "p": p}


_VARIANTS: dict[str, dict] = {
    "simsiam_pointnet_tiny": {"hidden": 32, "embed": 64, "proj": 64, "dropout": 0.0},
    "simsiam_pointnet_small": {"hidden": 64, "embed": 128, "proj": 128, "dropout": 0.0},
    "simsiam_pointnet_base": {"hidden": 96, "embed": 192, "proj": 192, "dropout": 0.0},
}


def build_simsiam_pointnet(
    *,
    in_channels: int,
    variant: str = "simsiam_pointnet_small",
    dropout: float | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown SimSiam-PointNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return SimSiamPointNet(
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

    m = build_simsiam_pointnet(in_channels=3, variant="simsiam_pointnet_tiny")
    o1 = m(v1)
    o2 = m(v2)
    loss = simsiam_loss(o1["p"], o2["z"], o2["p"], o1["z"])
    loss.backward()
    print("ok", float(loss.item()))
