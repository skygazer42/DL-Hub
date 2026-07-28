import torch
import torch.nn.functional as F
from torch import nn


def ressl_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    """ReSSL relational distillation loss.

    Given similarity logits over a shared key set, match the student's
    similarity distribution to the teacher's (stop-grad) distribution.

    Args:
        student_logits: (B, M) similarity logits from the student.
        teacher_logits: (B, M) similarity logits from the teacher.
    """

    if student_logits.ndim != 2 or teacher_logits.ndim != 2:
        raise ValueError(
            f"Expected student/teacher logits shapes (B, M), got {tuple(student_logits.shape)} and {tuple(teacher_logits.shape)}"
        )
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student_logits and teacher_logits must have the same shape")

    with torch.no_grad():
        t = teacher_logits.detach()
        t = t - t.max(dim=1, keepdim=True).values
        p_t = F.softmax(t, dim=1)

    s = student_logits
    s = s - s.max(dim=1, keepdim=True).values
    logp_s = F.log_softmax(s, dim=1)
    return -(p_t * logp_s).sum(dim=1).mean()


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
        self, in_dim: int, out_dim: int, *, hidden_dim: int | None = None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        d_in = int(in_dim)
        d_out = int(out_dim)
        h = int(hidden_dim) if hidden_dim is not None else d_in
        if d_in <= 0 or d_out <= 0 or h <= 0:
            raise ValueError("in_dim/out_dim/hidden_dim must be > 0")

        p = float(dropout)
        layers: list[nn.Module] = [
            nn.Linear(d_in, h),
            nn.ReLU(inplace=True),
        ]
        if p > 0:
            layers.append(nn.Dropout(p=p))
        layers.append(nn.Linear(h, d_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 2:
            raise ValueError(f"Expected x shape (B, D), got {tuple(x.shape)}")
        return self.net(x)


class ReSSLPointNet(nn.Module):
    """ReSSL-style relational distillation with a momentum teacher + queue.

    Compact-first design:
    - student sees "strong" augmentation
    - teacher sees "weak" augmentation (EMA updated from student)
    - loss matches student distribution over (batch teacher keys + queue) to teacher distribution
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
        qsz = int(queue_size)
        if qsz < 1:
            raise ValueError("queue_size must be >= 1")

        self.encoder_s = PointNetGlobalEncoder(
            in_channels=int(in_channels),
            hidden_features=int(hidden_features),
            embed_dim=int(embed_dim),
            dropout=float(dropout),
        )
        self.projector_s = ProjectionHead(
            int(embed_dim), int(proj_dim), hidden_dim=int(proj_dim), dropout=float(dropout)
        )

        self.encoder_t = PointNetGlobalEncoder(
            in_channels=int(in_channels),
            hidden_features=int(hidden_features),
            embed_dim=int(embed_dim),
            dropout=float(dropout),
        )
        self.projector_t = ProjectionHead(
            int(embed_dim), int(proj_dim), hidden_dim=int(proj_dim), dropout=float(dropout)
        )

        d = int(proj_dim)
        self.register_buffer("queue", torch.randn(d, qsz, dtype=torch.float32))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.reset_teacher()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # Keep teacher in eval mode (important if any dropout/BN exists).
        self.encoder_t.eval()
        self.projector_t.eval()
        return self

    @torch.no_grad()
    def reset_teacher(self) -> None:
        self.encoder_t.load_state_dict(self.encoder_s.state_dict())
        self.projector_t.load_state_dict(self.projector_s.state_dict())
        for p in self.encoder_t.parameters():
            p.requires_grad_(False)
        for p in self.projector_t.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def momentum_update_teacher(self, *, ema_decay: float = 0.99) -> None:
        m = float(ema_decay)
        if not (0.0 <= m < 1.0):
            raise ValueError("ema_decay must be in [0, 1)")

        def _ema_update(student: nn.Module, teacher: nn.Module) -> None:
            for ps, pt in zip(student.parameters(), teacher.parameters(), strict=True):
                pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))

        _ema_update(self.encoder_s, self.encoder_t)
        _ema_update(self.projector_s, self.projector_t)

    @torch.no_grad()
    def dequeue_and_enqueue(self, teacher_z: torch.Tensor) -> None:
        """Enqueue teacher projections (normalized) and dequeue the oldest ones."""

        if teacher_z.ndim != 2:
            raise ValueError(f"Expected teacher_z shape (B, D), got {tuple(teacher_z.shape)}")
        keys = F.normalize(teacher_z.detach(), dim=1)  # (B, D)

        bsz = int(keys.shape[0])
        k = int(self.queue.shape[1])
        if bsz == 0:
            return

        # If batch is larger than queue, keep the last K keys (still deterministic for compact).
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
        self,
        v_student: torch.Tensor,
        v_teacher: torch.Tensor,
        *,
        student_temperature: float = 0.2,
        teacher_temperature: float = 0.04,
    ) -> dict[str, torch.Tensor]:
        t_s = float(student_temperature)
        t_t = float(teacher_temperature)
        if t_s <= 0 or t_t <= 0:
            raise ValueError("Temperatures must be > 0")

        hs = self.encoder_s(v_student)
        zs = self.projector_s(hs)
        zs = F.normalize(zs, dim=1)

        with torch.no_grad():
            ht = self.encoder_t(v_teacher)
            zt = self.projector_t(ht)
            zt = F.normalize(zt, dim=1)

        # Keys: current batch teacher embeddings + queue negatives.
        keys = torch.cat([zt.detach(), self.queue.t().detach()], dim=0)  # (B+K, D)
        student_logits = torch.matmul(zs, keys.t()) / t_s  # (B, B+K)
        teacher_logits = torch.matmul(zt, keys.t()) / t_t  # (B, B+K)
        return {
            "h": hs,
            "student_z": zs,
            "teacher_z": zt,
            "student_logits": student_logits,
            "teacher_logits": teacher_logits,
        }


_VARIANTS: dict[str, dict] = {
    "ressl_pointnet_tiny": {"hidden": 32, "embed": 64, "proj": 64, "queue": 512, "dropout": 0.0},
    "ressl_pointnet_small": {
        "hidden": 64,
        "embed": 128,
        "proj": 128,
        "queue": 1024,
        "dropout": 0.0,
    },
    "ressl_pointnet_base": {"hidden": 96, "embed": 192, "proj": 192, "queue": 2048, "dropout": 0.0},
}


def build_ressl_pointnet(
    *,
    in_channels: int,
    variant: str = "ressl_pointnet_small",
    dropout: float | None = None,
    queue_size: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown ReSSL-PointNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    q = int(spec["queue"]) if queue_size is None else int(queue_size)
    return ReSSLPointNet(
        in_channels=int(in_channels),
        hidden_features=int(spec["hidden"]),
        embed_dim=int(spec["embed"]),
        proj_dim=int(spec["proj"]),
        queue_size=int(q),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    v_strong = torch.randn(8, 64, 3)
    v_weak = v_strong + torch.randn_like(v_strong) * 0.01

    m = build_ressl_pointnet(in_channels=3, variant="ressl_pointnet_tiny", queue_size=128)
    out = m(v_strong, v_weak, student_temperature=0.2, teacher_temperature=0.04)
    loss = ressl_loss(out["student_logits"], out["teacher_logits"])
    loss.backward()
    m.momentum_update_teacher(ema_decay=0.99)
    m.dequeue_and_enqueue(out["teacher_z"])
    print("ok", float(loss.item()))
