import torch
import torch.nn.functional as F
from torch import nn


def cosine_similarity_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """BYOL cosine similarity loss for one direction.

    Args:
        p: (B, D) online predictor output.
        z: (B, D) target projector output (will be stop-grad / detached).
    """

    if p.ndim != 2 or z.ndim != 2:
        raise ValueError(f"Expected p/z shapes (B, D), got {tuple(p.shape)} and {tuple(z.shape)}")
    if p.shape != z.shape:
        raise ValueError("p and z must have the same shape")
    if int(p.shape[0]) < 2:
        raise ValueError("Batch size must be >= 2 for stable BYOL loss")

    p = F.normalize(p, dim=1)
    z = F.normalize(z.detach(), dim=1)
    # 2 - 2*cos is in [0, 4].
    return (2.0 - 2.0 * (p * z).sum(dim=1)).mean()


def byol2_loss(
    p1: torch.Tensor, z2: torch.Tensor, p2: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """Symmetric BYOL loss for two views."""

    return 0.5 * (cosine_similarity_loss(p1, z2) + cosine_similarity_loss(p2, z1))


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
        use_final_bn: bool = False,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_in = int(in_dim)
        d_out = int(out_dim)
        h = int(hidden_dim) if hidden_dim is not None else d_in
        if d_in <= 0 or d_out <= 0 or h <= 0:
            raise ValueError("in_dim/out_dim/hidden_dim must be > 0")

        layers: list[nn.Module] = [
            nn.Linear(d_in, h, bias=False),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
        ]
        p = float(dropout)
        if p > 0:
            layers.append(nn.Dropout(p=p))
        layers.append(nn.Linear(h, d_out, bias=True))
        if bool(use_final_bn):
            # BYOL paper uses a BN without affine parameters at the end of projector.
            layers.append(nn.BatchNorm1d(d_out, affine=False))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x shape (B, D), got {tuple(x.shape)}")
        return self.net(x)


class BYOL2PointNet(nn.Module):
    """BYOL model: online encoder+projector+predictor + target encoder+projector (EMA)."""

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
        self.online_encoder = PointNetGlobalEncoder(
            in_channels=int(in_channels),
            hidden_features=int(hidden_features),
            embed_dim=int(embed_dim),
            dropout=float(dropout),
        )
        self.online_projector = MLPHead(
            int(embed_dim), int(proj_dim), hidden_dim=int(proj_dim), use_final_bn=True
        )
        self.online_predictor = MLPHead(
            int(proj_dim), int(proj_dim), hidden_dim=int(proj_dim), use_final_bn=False
        )

        self.target_encoder = PointNetGlobalEncoder(
            in_channels=int(in_channels),
            hidden_features=int(hidden_features),
            embed_dim=int(embed_dim),
            dropout=float(dropout),
        )
        self.target_projector = MLPHead(
            int(embed_dim), int(proj_dim), hidden_dim=int(proj_dim), use_final_bn=True
        )
        self.reset_target()

    @torch.no_grad()
    def reset_target(self) -> None:
        """Initialize target params from online params (and freeze target gradients)."""

        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_projector.load_state_dict(self.online_projector.state_dict())
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        for p in self.target_projector.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update_target(self, *, ema_decay: float = 0.99) -> None:
        """EMA update: target = m*target + (1-m)*online."""

        m = float(ema_decay)
        if not (0.0 <= m < 1.0):
            raise ValueError("ema_decay must be in [0, 1)")

        def _ema_update(online: nn.Module, target: nn.Module) -> None:
            for po, pt in zip(online.parameters(), target.parameters(), strict=True):
                pt.data.mul_(m).add_(po.data, alpha=(1.0 - m))

        _ema_update(self.online_encoder, self.target_encoder)
        _ema_update(self.online_projector, self.target_projector)

    def forward_online(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.online_encoder(points)
        z = self.online_projector(h)
        p = self.online_predictor(z)
        return {"h": h, "z": z, "p": p}

    @torch.no_grad()
    def forward_target(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.target_encoder(points)
        z = self.target_projector(h)
        return {"h": h, "z": z}

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_online(points)


_VARIANTS: dict[str, dict] = {
    "byol2_pointnet_tiny": {"hidden": 32, "embed": 64, "proj": 64, "dropout": 0.0},
    "byol2_pointnet_small": {"hidden": 64, "embed": 128, "proj": 128, "dropout": 0.0},
    "byol2_pointnet_base": {"hidden": 96, "embed": 192, "proj": 192, "dropout": 0.0},
}


def build_byol2_pointnet(
    *,
    in_channels: int,
    variant: str = "byol2_pointnet_small",
    dropout: float | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown BYOL-PointNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return BYOL2PointNet(
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

    m = build_byol2_pointnet(in_channels=3, variant="byol2_pointnet_tiny")
    o1 = m.forward_online(v1)
    o2 = m.forward_online(v2)
    t1 = m.forward_target(v1)
    t2 = m.forward_target(v2)
    loss = byol2_loss(o1["p"], t2["z"], o2["p"], t1["z"])
    loss.backward()
    m.update_target(ema_decay=0.99)
    print("ok", float(loss.item()))
