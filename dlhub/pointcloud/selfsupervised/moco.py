import torch
import torch.nn.functional as F
from torch import nn


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


def moco_logits(
    q: torch.Tensor,
    k: torch.Tensor,
    queue: torch.Tensor,
    *,
    temperature: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute MoCo logits and labels.

    Args:
        q: (B, D) normalized query features.
        k: (B, D) normalized key features.
        queue: (D, K) normalized queue of negative keys.

    Returns:
        logits: (B, 1+K)
        labels: (B,) all zeros (positive key is at index 0)
    """

    if q.ndim != 2 or k.ndim != 2:
        raise ValueError(f"Expected q/k shapes (B, D), got {tuple(q.shape)} and {tuple(k.shape)}")
    if q.shape != k.shape:
        raise ValueError("q and k must have same shape")
    if queue.ndim != 2 or queue.shape[0] != q.shape[1]:
        raise ValueError(
            f"Expected queue shape (D, K) with D={q.shape[1]}, got {tuple(queue.shape)}"
        )

    t = float(temperature)
    if t <= 0:
        raise ValueError("temperature must be > 0")

    # Positive: (B, 1)
    pos = torch.sum(q * k, dim=1, keepdim=True)
    # Negative: (B, K)
    neg = torch.matmul(q, queue)
    logits = torch.cat([pos, neg], dim=1) / t
    labels = torch.zeros((q.shape[0],), device=q.device, dtype=torch.long)
    return logits, labels


class MoCoPointNet(nn.Module):
    """MoCo v2-style model for point clouds (toy-first).

    - Query encoder (online) is trained by gradient.
    - Key encoder is updated by EMA (momentum) from query encoder.
    - Maintains a queue of negative keys to enable small batch sizes.

    Forward returns:
      - q: (B, D) normalized query projections
      - k: (B, D) normalized key projections (no grad)
      - logits: (B, 1+K)
      - labels: (B,) zeros
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_features: int = 64,
        embed_dim: int = 128,
        proj_dim: int = 128,
        queue_size: int = 1024,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(proj_dim)
        k = int(queue_size)
        if k <= 0:
            raise ValueError("queue_size must be > 0")

        self.encoder_q = PointNetGlobalEncoder(
            in_channels=int(in_channels),
            hidden_features=int(hidden_features),
            embed_dim=int(embed_dim),
            dropout=float(dropout),
        )
        self.projector_q = ProjectionHead(int(embed_dim), int(proj_dim), hidden_dim=int(proj_dim))

        self.encoder_k = PointNetGlobalEncoder(
            in_channels=int(in_channels),
            hidden_features=int(hidden_features),
            embed_dim=int(embed_dim),
            dropout=float(dropout),
        )
        self.projector_k = ProjectionHead(int(embed_dim), int(proj_dim), hidden_dim=int(proj_dim))

        self.register_buffer("queue", torch.randn(d, k, dtype=torch.float32))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.reset_key_encoder()

    @torch.no_grad()
    def reset_key_encoder(self) -> None:
        self.encoder_k.load_state_dict(self.encoder_q.state_dict())
        self.projector_k.load_state_dict(self.projector_q.state_dict())
        for p in self.encoder_k.parameters():
            p.requires_grad_(False)
        for p in self.projector_k.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def momentum_update_key_encoder(self, *, ema_decay: float = 0.99) -> None:
        """EMA update: key = m*key + (1-m)*query."""

        m = float(ema_decay)
        if not (0.0 <= m < 1.0):
            raise ValueError("ema_decay must be in [0, 1)")

        def _ema_update(q: nn.Module, k: nn.Module) -> None:
            for pq, pk in zip(q.parameters(), k.parameters(), strict=True):
                pk.data.mul_(m).add_(pq.data, alpha=(1.0 - m))

        _ema_update(self.encoder_q, self.encoder_k)
        _ema_update(self.projector_q, self.projector_k)

    @torch.no_grad()
    def dequeue_and_enqueue(self, keys: torch.Tensor) -> None:
        """Enqueue keys and dequeue the oldest ones."""

        if keys.ndim != 2:
            raise ValueError(f"Expected keys shape (B, D), got {tuple(keys.shape)}")
        keys = F.normalize(keys.detach(), dim=1)  # (B, D)

        bsz = int(keys.shape[0])
        k = int(self.queue.shape[1])
        if bsz == 0:
            return

        # If batch is larger than queue, keep the last K keys (still deterministic for toy).
        if bsz >= k:
            keys = keys[-k:]
            bsz = int(keys.shape[0])

        ptr = int(self.queue_ptr.item())
        end = ptr + bsz
        keys_t = keys.t().contiguous()  # (D, B)

        if end <= k:
            self.queue[:, ptr:end] = keys_t
        else:
            first = k - ptr
            self.queue[:, ptr:] = keys_t[:, :first]
            self.queue[:, : end - k] = keys_t[:, first:]

        self.queue_ptr[0] = (ptr + bsz) % k

    def forward(
        self, v1: torch.Tensor, v2: torch.Tensor, *, temperature: float = 0.2
    ) -> dict[str, torch.Tensor]:
        hq = self.encoder_q(v1)
        q = self.projector_q(hq)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            hk = self.encoder_k(v2)
            k = self.projector_k(hk)
            k = F.normalize(k, dim=1)

        logits, labels = moco_logits(q, k, self.queue, temperature=float(temperature))
        return {"q": q, "k": k, "logits": logits, "labels": labels}


_VARIANTS: dict[str, dict] = {
    "moco_pointnet_tiny": {"hidden": 32, "embed": 64, "proj": 64, "queue": 512, "dropout": 0.0},
    "moco_pointnet_small": {"hidden": 64, "embed": 128, "proj": 128, "queue": 1024, "dropout": 0.0},
    "moco_pointnet_base": {"hidden": 96, "embed": 192, "proj": 192, "queue": 2048, "dropout": 0.0},
}


def build_moco_pointnet(
    *,
    in_channels: int,
    variant: str = "moco_pointnet_small",
    dropout: float | None = None,
    queue_size: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown MoCo-PointNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    q = int(spec["queue"]) if queue_size is None else int(queue_size)
    return MoCoPointNet(
        in_channels=int(in_channels),
        hidden_features=int(spec["hidden"]),
        embed_dim=int(spec["embed"]),
        proj_dim=int(spec["proj"]),
        queue_size=int(q),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    v1 = torch.randn(8, 128, 3)
    v2 = v1 + torch.randn_like(v1) * 0.01
    m = build_moco_pointnet(in_channels=3, variant="moco_pointnet_tiny")

    m.momentum_update_key_encoder(ema_decay=0.99)
    out = m(v1, v2, temperature=0.2)
    loss = F.cross_entropy(out["logits"], out["labels"])
    loss.backward()
    m.dequeue_and_enqueue(out["k"])
    print("ok", float(loss.item()))
