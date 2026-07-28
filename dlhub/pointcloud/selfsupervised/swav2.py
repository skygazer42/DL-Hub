import torch
import torch.nn.functional as F
from torch import nn


@torch.no_grad()
def sinkhorn_knopp(
    scores: torch.Tensor,
    *,
    epsilon: float = 0.05,
    iters: int = 3,
) -> torch.Tensor:
    """Balanced assignments via Sinkhorn-Knopp (SwAV).

    Args:
        scores: (B, K) prototype scores / logits.
        epsilon: temperature for exponentiation (smaller => sharper assignments).
        iters: number of normalization iterations.

    Returns:
        q: (B, K) soft assignments, each row sums to 1.
    """

    if scores.ndim != 2:
        raise ValueError(f"Expected scores shape (B, K), got {tuple(scores.shape)}")
    b, k = int(scores.shape[0]), int(scores.shape[1])
    if b < 2:
        raise ValueError("Batch size must be >= 2 for Sinkhorn")
    if k < 2:
        raise ValueError("num_prototypes must be >= 2")
    eps = float(epsilon)
    if eps <= 0:
        raise ValueError("epsilon must be > 0")

    n_iters = int(iters)
    if n_iters < 1:
        raise ValueError("iters must be >= 1")

    # Stabilize exponentiation: subtract max per sample.
    x = (scores / eps).to(torch.float32)
    x = x - x.max(dim=1, keepdim=True).values

    q = torch.exp(x).t()  # (K, B)
    q = q / q.sum()  # sum=1

    kf = float(k)
    bf = float(b)

    for _ in range(n_iters):
        # Row normalization: each prototype has mass 1/K.
        q = q / q.sum(dim=1, keepdim=True).clamp(min=1e-12)
        q = q / kf
        # Col normalization: each sample has mass 1/B.
        q = q / q.sum(dim=0, keepdim=True).clamp(min=1e-12)
        q = q / bf

    q = q * bf
    return q.t().contiguous()  # (B, K)


def swav2_loss(
    scores1: torch.Tensor,
    scores2: torch.Tensor,
    *,
    temperature: float = 0.1,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 3,
) -> torch.Tensor:
    """Two-crop SwAV loss (swap assignments between two views).

    Args:
        scores1, scores2: (B, K) prototype logits for view1/view2.
    """

    if scores1.ndim != 2 or scores2.ndim != 2:
        raise ValueError(
            f"Expected scores shapes (B, K), got {tuple(scores1.shape)} and {tuple(scores2.shape)}"
        )
    if scores1.shape != scores2.shape:
        raise ValueError("scores1 and scores2 must have the same shape")

    t = float(temperature)
    if t <= 0:
        raise ValueError("temperature must be > 0")

    with torch.no_grad():
        q1 = sinkhorn_knopp(
            scores1.detach(), epsilon=float(sinkhorn_epsilon), iters=int(sinkhorn_iters)
        )
        q2 = sinkhorn_knopp(
            scores2.detach(), epsilon=float(sinkhorn_epsilon), iters=int(sinkhorn_iters)
        )

    p1 = F.log_softmax(scores1 / t, dim=1)
    p2 = F.log_softmax(scores2 / t, dim=1)

    loss12 = -(q1 * p2).sum(dim=1).mean()
    loss21 = -(q2 * p1).sum(dim=1).mean()
    return 0.5 * (loss12 + loss21)


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
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_in = int(in_dim)
        d_out = int(proj_dim)
        h = int(hidden_dim) if hidden_dim is not None else d_in
        nl = int(num_layers)
        if d_in <= 0 or d_out <= 0 or h <= 0:
            raise ValueError("in_dim/proj_dim/hidden_dim must be > 0")
        if nl < 1:
            raise ValueError("num_layers must be >= 1")

        layers: list[nn.Module] = []
        for i in range(nl - 1):
            layers += [
                nn.Linear(d_in if i == 0 else h, h),
                nn.BatchNorm1d(h),
                nn.ReLU(inplace=True),
            ]
            p = float(dropout)
            if p > 0:
                layers.append(nn.Dropout(p=p))
        layers.append(nn.Linear(h if nl > 1 else d_in, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x shape (B, D), got {tuple(x.shape)}")
        return self.net(x)


class SwAV2PointNet(nn.Module):
    """SwAV model: encoder + projector + prototypes (compact-first, 2-crop).

    Forward returns:
      - h: (B, embed_dim) encoder features
      - z: (B, proj_dim) projected features (L2-normalized)
      - scores: (B, K) prototype logits
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_features: int = 64,
        embed_dim: int = 128,
        proj_dim: int = 128,
        num_prototypes: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        k = int(num_prototypes)
        if k < 2:
            raise ValueError("num_prototypes must be >= 2")

        self.encoder = PointNetGlobalEncoder(
            in_channels=int(in_channels),
            hidden_features=int(hidden_features),
            embed_dim=int(embed_dim),
            dropout=float(dropout),
        )
        self.projector = ProjectionHead(
            int(embed_dim), int(proj_dim), hidden_dim=int(proj_dim), num_layers=2
        )
        self.prototypes = nn.Linear(int(proj_dim), k, bias=False)
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self) -> None:
        w = self.prototypes.weight.data
        self.prototypes.weight.data = F.normalize(w, dim=1)

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(points)
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        scores = self.prototypes(z)
        return {"h": h, "z": z, "scores": scores}


_VARIANTS: dict[str, dict] = {
    "swav2_pointnet_tiny": {
        "hidden": 32,
        "embed": 64,
        "proj": 64,
        "prototypes": 32,
        "dropout": 0.0,
    },
    "swav2_pointnet_small": {
        "hidden": 64,
        "embed": 128,
        "proj": 128,
        "prototypes": 64,
        "dropout": 0.0,
    },
    "swav2_pointnet_base": {
        "hidden": 96,
        "embed": 192,
        "proj": 192,
        "prototypes": 128,
        "dropout": 0.0,
    },
}


def build_swav2_pointnet(
    *,
    in_channels: int,
    variant: str = "swav2_pointnet_small",
    dropout: float | None = None,
    num_prototypes: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown SwAV-PointNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    k = int(spec["prototypes"]) if num_prototypes is None else int(num_prototypes)
    return SwAV2PointNet(
        in_channels=int(in_channels),
        hidden_features=int(spec["hidden"]),
        embed_dim=int(spec["embed"]),
        proj_dim=int(spec["proj"]),
        num_prototypes=int(k),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    v1 = torch.randn(8, 128, 3)
    v2 = v1 + torch.randn_like(v1) * 0.01

    m = build_swav2_pointnet(in_channels=3, variant="swav2_pointnet_tiny")
    o1 = m(v1)
    o2 = m(v2)
    loss = swav2_loss(
        o1["scores"], o2["scores"], temperature=0.1, sinkhorn_epsilon=0.05, sinkhorn_iters=3
    )
    loss.backward()
    m.normalize_prototypes()
    print("ok", float(loss.item()))
