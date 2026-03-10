import torch
import torch.nn.functional as F
from torch import nn

from dlhub.pointcloud.selfsupervised.dinov2 import PatchTransformer


def msn_loss(
    *,
    student_logits: list[torch.Tensor],
    teacher_logits: list[torch.Tensor],
    student_temperature: float,
    teacher_temperature: float,
    center: torch.Tensor,
    entropy_weight: float = 1.0,
) -> torch.Tensor:
    """Masked Siamese Networks (MSN)-style distillation loss.

    Toy-first variant:
    - Cross-view distillation on prototype logits (teacher -> student).
    - Prototype balance regularization (KL(mean_student || uniform)) to avoid collapse.
    """

    if len(student_logits) < 2 or len(teacher_logits) < 2:
        raise ValueError("Need >=2 student views and >=2 teacher views for MSN loss")

    t_s = float(student_temperature)
    t_t = float(teacher_temperature)
    if t_s <= 0 or t_t <= 0:
        raise ValueError("Temperatures must be > 0")

    if center.ndim != 2 or int(center.shape[0]) != 1:
        raise ValueError(f"Expected center shape (1, K), got {tuple(center.shape)}")

    k = int(center.shape[1])
    if k < 2:
        raise ValueError("center must have K>=2 classes/prototypes")

    # Cross-view distillation (exclude same-view pairs when the view counts match).
    distill = 0.0
    n_terms = 0
    same_view_exclusion = len(student_logits) == len(teacher_logits)
    for i, t_logits in enumerate(teacher_logits):
        if t_logits.ndim != 2 or int(t_logits.shape[1]) != k:
            raise ValueError("Each teacher_logits must be (B, K) matching center")
        with torch.no_grad():
            t_prob = F.softmax((t_logits.detach() - center) / t_t, dim=1)

        for j, s_logits in enumerate(student_logits):
            if s_logits.ndim != 2 or int(s_logits.shape[1]) != k:
                raise ValueError("Each student_logits must be (B, K) matching center")
            if same_view_exclusion and i == j:
                continue
            s_logprob = F.log_softmax(s_logits / t_s, dim=1)
            distill = distill + -(t_prob * s_logprob).sum(dim=1).mean()
            n_terms += 1

    if n_terms == 0:
        raise ValueError("No valid (teacher, student) pairs to compute MSN loss")
    loss = distill / float(n_terms)

    # Prototype balance regularization:
    # KL(mean_student || uniform) = sum p log p + log K
    # (>=0, equals 0 when uniform).
    w = float(entropy_weight)
    if w < 0:
        raise ValueError("entropy_weight must be >= 0")
    if w > 0:
        probs = [F.softmax(s / t_s, dim=1) for s in student_logits]
        p_mean = torch.cat(probs, dim=0).mean(dim=0)  # (K,)
        p_mean = p_mean.to(torch.float32).clamp(min=1e-6)
        kl_to_uniform = (p_mean * torch.log(p_mean)).sum() + torch.log(p_mean.new_tensor(float(k)))
        loss = loss + w * kl_to_uniform

    return loss


class MSNHead(nn.Module):
    """Projector + normalized prototypes (DINO-style head) for MSN."""

    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        out_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d_in = int(in_dim)
        d_proj = int(proj_dim)
        d_out = int(out_dim)
        if d_in <= 0 or d_proj <= 0 or d_out <= 1:
            raise ValueError("in_dim/proj_dim must be > 0 and out_dim must be >= 2")

        self.projector = nn.Sequential(
            nn.Linear(d_in, d_proj, bias=False),
            nn.BatchNorm1d(d_proj),
            nn.GELU(),
            nn.Linear(d_proj, d_proj, bias=False),
            nn.BatchNorm1d(d_proj),
            nn.GELU(),
            nn.Linear(d_proj, d_proj, bias=True),
        )
        self.prototypes = nn.Linear(d_proj, d_out, bias=False)
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim != 2:
            raise ValueError(f"Expected x shape (B, D), got {tuple(x.shape)}")
        z = self.projector(self.drop(x))
        z = F.normalize(z, dim=1)
        w = F.normalize(self.prototypes.weight, dim=1)
        logits = F.linear(z, w)
        return {"z": z, "logits": logits}


class MSNPointMAE(nn.Module):
    """Toy-first MSN-style self-supervised learning for point clouds.

    - Student: patch transformer with mask tokens -> prototype logits.
    - Teacher: same backbone/head without masking (EMA updated from student).
    - Loss: cross-view teacher->student distillation on prototypes + balance regularization.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        embed_dim: int = 128,
        num_patches: int = 16,
        group_size: int = 16,
        depth: int = 4,
        heads: int = 4,
        proj_dim: int = 256,
        out_dim: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(embed_dim)
        k = int(out_dim)
        if k < 2:
            raise ValueError("out_dim must be >= 2")

        self.student_backbone = PatchTransformer(
            in_channels=int(in_channels),
            embed_dim=int(d),
            num_patches=int(num_patches),
            group_size=int(group_size),
            depth=int(depth),
            heads=int(heads),
            dropout=float(dropout),
        )
        self.teacher_backbone = PatchTransformer(
            in_channels=int(in_channels),
            embed_dim=int(d),
            num_patches=int(num_patches),
            group_size=int(group_size),
            depth=int(depth),
            heads=int(heads),
            dropout=float(dropout),
        )

        self.student_head = MSNHead(int(d), int(proj_dim), int(k), dropout=float(dropout))
        self.teacher_head = MSNHead(int(d), int(proj_dim), int(k), dropout=float(dropout))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(d)))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.register_buffer("center", torch.zeros(1, int(k), dtype=torch.float32))
        self.reset_teacher()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # Keep teacher in eval mode (important if any dropout/BN exists).
        self.teacher_backbone.eval()
        self.teacher_head.eval()
        return self

    @torch.no_grad()
    def reset_teacher(self) -> None:
        self.teacher_backbone.load_state_dict(self.student_backbone.state_dict())
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for p in self.teacher_backbone.parameters():
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

        _ema_update(self.student_backbone, self.teacher_backbone)
        _ema_update(self.student_head, self.teacher_head)

    @torch.no_grad()
    def update_center(
        self, teacher_logits: list[torch.Tensor], *, center_momentum: float = 0.9
    ) -> None:
        cm = float(center_momentum)
        if not (0.0 <= cm < 1.0):
            raise ValueError("center_momentum must be in [0, 1)")
        if not teacher_logits:
            return
        cat = torch.cat([t.detach() for t in teacher_logits], dim=0)  # (V*B, K)
        c = cat.mean(dim=0, keepdim=True)
        self.center.mul_(cm).add_(c, alpha=(1.0 - cm))

    def forward_student(
        self, points: torch.Tensor, *, mask_ratio: float = 0.5
    ) -> dict[str, torch.Tensor]:
        out = self.student_backbone(
            points, mask_ratio=float(mask_ratio), mask_token=self.mask_token
        )
        head = self.student_head(out["cls"])
        return {
            "cls": out["cls"],
            "z": head["z"],
            "cls_logits": head["logits"],
            "mask_idx": out["mask_idx"],
        }

    @torch.no_grad()
    def forward_teacher(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.teacher_backbone(points, mask_ratio=0.0, mask_token=None)
        head = self.teacher_head(out["cls"])
        return {"cls": out["cls"], "z": head["z"], "cls_logits": head["logits"]}

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.student_backbone(points, mask_ratio=0.0, mask_token=None)
        head = self.student_head(out["cls"])
        return {"cls": out["cls"], "z": head["z"], "cls_logits": head["logits"]}


_VARIANTS: dict[str, dict] = {
    "msn_pointmae_tiny": {
        "embed_dim": 96,
        "num_patches": 12,
        "group_size": 16,
        "depth": 2,
        "heads": 4,
        "proj_dim": 192,
        "out_dim": 64,
        "dropout": 0.0,
    },
    "msn_pointmae_small": {
        "embed_dim": 128,
        "num_patches": 16,
        "group_size": 16,
        "depth": 4,
        "heads": 4,
        "proj_dim": 256,
        "out_dim": 128,
        "dropout": 0.0,
    },
    "msn_pointmae_base": {
        "embed_dim": 192,
        "num_patches": 24,
        "group_size": 24,
        "depth": 6,
        "heads": 6,
        "proj_dim": 384,
        "out_dim": 256,
        "dropout": 0.0,
    },
}


def build_msn_pointmae(
    *,
    in_channels: int,
    variant: str = "msn_pointmae_small",
    dropout: float | None = None,
    out_dim: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown MSN variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    out = int(spec["out_dim"]) if out_dim is None else int(out_dim)
    return MSNPointMAE(
        in_channels=int(in_channels),
        embed_dim=int(spec["embed_dim"]),
        num_patches=int(spec["num_patches"]),
        group_size=int(spec["group_size"]),
        depth=int(spec["depth"]),
        heads=int(spec["heads"]),
        proj_dim=int(spec["proj_dim"]),
        out_dim=int(out),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    v1 = torch.randn(4, 96, 3)
    v2 = v1 + torch.randn_like(v1) * 0.01

    m = build_msn_pointmae(in_channels=3, variant="msn_pointmae_tiny", out_dim=64)
    with torch.no_grad():
        t1 = m.forward_teacher(v1)["cls_logits"]
        t2 = m.forward_teacher(v2)["cls_logits"]
    s1 = m.forward_student(v1, mask_ratio=0.5)["cls_logits"]
    s2 = m.forward_student(v2, mask_ratio=0.5)["cls_logits"]

    loss = msn_loss(
        student_logits=[s1, s2],
        teacher_logits=[t1, t2],
        student_temperature=0.1,
        teacher_temperature=0.04,
        center=m.center,
        entropy_weight=1.0,
    )
    loss.backward()
    m.update_center([t1, t2], center_momentum=0.9)
    m.momentum_update_teacher(ema_decay=0.99)
    print("ok", float(loss.item()))
