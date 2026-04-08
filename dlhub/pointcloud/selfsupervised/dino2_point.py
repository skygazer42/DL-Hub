import torch
import torch.nn.functional as F
from torch import nn


def dino2_point_loss(
    student_logits: list[torch.Tensor],
    teacher_logits: list[torch.Tensor],
    *,
    student_temperature: float = 0.1,
    teacher_temperature: float = 0.04,
    center: torch.Tensor,
) -> torch.Tensor:
    """DINO cross-view self-distillation loss (2-crop or multi-crop).

    For teacher view i, we supervise all student views j != i.

    Args:
        student_logits: list of (B, K) logits.
        teacher_logits: list of (B, K) logits (stop-grad).
        center: (1, K) running center buffer.
    """

    if len(student_logits) < 2 or len(teacher_logits) < 2:
        raise ValueError("DINO expects at least 2 views for both student and teacher")

    t_s = float(student_temperature)
    t_t = float(teacher_temperature)
    if t_s <= 0 or t_t <= 0:
        raise ValueError("Temperatures must be > 0")

    if center.ndim != 2 or center.shape[0] != 1:
        raise ValueError(f"Expected center shape (1, K), got {tuple(center.shape)}")

    loss = 0.0
    n_terms = 0

    for i, t_logits in enumerate(teacher_logits):
        if t_logits.ndim != 2:
            raise ValueError("teacher_logits must be 2D tensors (B, K)")
        with torch.no_grad():
            t_prob = F.softmax((t_logits.detach() - center) / t_t, dim=1)

        for j, s_logits in enumerate(student_logits):
            if j == i:
                continue
            if s_logits.ndim != 2:
                raise ValueError("student_logits must be 2D tensors (B, K)")

            s_logprob = F.log_softmax(s_logits / t_s, dim=1)
            loss = loss + (-(t_prob * s_logprob).sum(dim=1).mean())
            n_terms += 1

    if n_terms == 0:
        raise RuntimeError("No cross-view terms computed (check view list lengths)")
    return loss / float(n_terms)


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
                nn.GELU(),
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


class DINOHead(nn.Module):
    """DINO head: projector + normalized prototypes layer."""

    def __init__(self, in_dim: int, proj_dim: int, out_dim: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        self.projector = ProjectionHead(
            int(in_dim),
            int(proj_dim),
            hidden_dim=int(proj_dim),
            num_layers=3,
            dropout=float(dropout),
        )
        self.prototypes = nn.Linear(int(proj_dim), int(out_dim), bias=False)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.projector(h)
        z = F.normalize(z, dim=1)
        w = F.normalize(self.prototypes.weight, dim=1)
        logits = F.linear(z, w)
        return z, logits


class DINO2PointNet(nn.Module):
    """DINO for point clouds (toy-first, 2-crop compatible).

    - Student network is trained by gradient.
    - Teacher network is updated by EMA from the student.
    - Maintains a running `center` to stabilize teacher outputs.

    Default `forward(points)` returns **student** features/logits for downstream use:
      {"h": (B,D), "z": (B,P), "logits": (B,K)}
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_features: int = 64,
        embed_dim: int = 128,
        proj_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        k = int(out_dim)
        if k < 2:
            raise ValueError("out_dim (num prototypes) must be >= 2")

        self.student_encoder = PointNetGlobalEncoder(
            in_channels=int(in_channels),
            hidden_features=int(hidden_features),
            embed_dim=int(embed_dim),
            dropout=float(dropout),
        )
        self.student_head = DINOHead(
            int(embed_dim), int(proj_dim), int(out_dim), dropout=float(dropout)
        )

        self.teacher_encoder = PointNetGlobalEncoder(
            in_channels=int(in_channels),
            hidden_features=int(hidden_features),
            embed_dim=int(embed_dim),
            dropout=float(dropout),
        )
        self.teacher_head = DINOHead(
            int(embed_dim), int(proj_dim), int(out_dim), dropout=float(dropout)
        )

        self.register_buffer("center", torch.zeros(1, int(out_dim), dtype=torch.float32))
        self.reset_teacher()

    @torch.no_grad()
    def reset_teacher(self) -> None:
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_encoder.parameters():
            p.requires_grad_(False)
        for p in self.teacher_head.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def momentum_update_teacher(self, *, ema_decay: float = 0.996) -> None:
        m = float(ema_decay)
        if not (0.0 <= m < 1.0):
            raise ValueError("ema_decay must be in [0, 1)")

        def _ema_update(student: nn.Module, teacher: nn.Module) -> None:
            for ps, pt in zip(student.parameters(), teacher.parameters(), strict=True):
                pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))

        _ema_update(self.student_encoder, self.teacher_encoder)
        _ema_update(self.student_head, self.teacher_head)

    @torch.no_grad()
    def update_center(
        self, teacher_logits: list[torch.Tensor], *, center_momentum: float = 0.9
    ) -> None:
        cm = float(center_momentum)
        if not (0.0 <= cm < 1.0):
            raise ValueError("center_momentum must be in [0, 1)")
        if len(teacher_logits) == 0:
            return

        all_logits = torch.cat([t.detach() for t in teacher_logits], dim=0)  # (V*B, K)
        batch_center = all_logits.mean(dim=0, keepdim=True)
        self.center.mul_(cm).add_(batch_center, alpha=(1.0 - cm))

    def forward_student(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.student_encoder(points)
        z, logits = self.student_head(h)
        return {"h": h, "z": z, "logits": logits}

    @torch.no_grad()
    def forward_teacher(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.teacher_encoder(points)
        z, logits = self.teacher_head(h)
        return {"h": h, "z": z, "logits": logits}

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        return self.forward_student(points)


_VARIANTS: dict[str, dict] = {
    "dino2_point_pointnet_tiny": {"hidden": 32, "embed": 64, "proj": 128, "out": 64, "dropout": 0.0},
    "dino2_point_pointnet_small": {"hidden": 64, "embed": 128, "proj": 256, "out": 128, "dropout": 0.0},
    "dino2_point_pointnet_base": {"hidden": 96, "embed": 192, "proj": 384, "out": 256, "dropout": 0.0},
}


def build_dino2_point_pointnet(
    *,
    in_channels: int,
    variant: str = "dino2_point_pointnet_small",
    dropout: float | None = None,
    out_dim: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(
            f"Unknown DINO-PointNet variant: {variant!r}. Supported: {sorted(_VARIANTS)}"
        )
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    out = int(spec["out"]) if out_dim is None else int(out_dim)
    return DINO2PointNet(
        in_channels=int(in_channels),
        hidden_features=int(spec["hidden"]),
        embed_dim=int(spec["embed"]),
        proj_dim=int(spec["proj"]),
        out_dim=int(out),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    v1 = torch.randn(8, 128, 3)
    v2 = v1 + torch.randn_like(v1) * 0.01

    m = build_dino2_point_pointnet(in_channels=3, variant="dino2_point_pointnet_tiny")
    s1 = m.forward_student(v1)["logits"]
    s2 = m.forward_student(v2)["logits"]
    t1 = m.forward_teacher(v1)["logits"]
    t2 = m.forward_teacher(v2)["logits"]
    loss = dino2_point_loss(
        [s1, s2], [t1, t2], student_temperature=0.1, teacher_temperature=0.04, center=m.center
    )
    loss.backward()
    m.update_center([t1, t2], center_momentum=0.9)
    m.momentum_update_teacher(ema_decay=0.99)
    print("ok", float(loss.item()))

