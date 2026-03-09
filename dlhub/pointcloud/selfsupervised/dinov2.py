
import torch
from torch import nn
import torch.nn.functional as F

from dlhub.pointcloud.ops import farthest_point_sample, index_points, knn_query


def _gather_batch(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather x along dim=1 with per-batch indices.

    x: (B, S, ...)
    idx: (B, M)
    """

    if x.ndim < 2:
        raise ValueError("x must be at least 2D")
    if idx.ndim != 2:
        raise ValueError("idx must be (B, M)")
    b = int(x.shape[0])
    if int(idx.shape[0]) != b:
        raise ValueError("Batch mismatch in _gather_batch")
    batch = torch.arange(b, device=x.device).unsqueeze(1)
    return x[batch, idx]


class PatchEmbed(nn.Module):
    """Group points into patches and embed them into tokens.

    Returns:
        tokens: (B, S, D)
        centers: (B, S, 3)
    """

    def __init__(
        self,
        *,
        in_channels: int,
        embed_dim: int,
        num_patches: int,
        group_size: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.embed_dim = int(embed_dim)
        self.num_patches = int(num_patches)
        self.group_size = int(group_size)

        d = int(embed_dim)
        self.point_embed = nn.Sequential(
            nn.Linear(int(in_channels), d),
            nn.ReLU(inplace=True),
        )
        self.group_mlp = nn.Sequential(
            nn.Linear(d + 3, d),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(d, d),
        )
        self.pos = nn.Sequential(
            nn.Linear(3, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
        )

    def forward(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if points.ndim != 3 or int(points.shape[-1]) != int(self.in_channels):
            raise ValueError(f"Expected points shape (B, N, C={self.in_channels}), got {tuple(points.shape)}")

        xyz = points[..., :3].to(torch.float32)  # (B, N, 3)
        b, n, _ = xyz.shape
        s = int(self.num_patches)
        k = int(self.group_size)

        if n < k:
            raise ValueError(f"group_size must be <= num_points. Got group_size={k}, num_points={n}")
        if s <= 0:
            raise ValueError("num_patches must be > 0")
        if s > n:
            s = int(n)

        feat = self.point_embed(points.to(torch.float32))  # (B, N, D)
        fps_idx = farthest_point_sample(xyz, s)  # (B, S)
        centers = index_points(xyz, fps_idx)  # (B, S, 3)

        idx = knn_query(k, xyz, centers)  # (B, S, K)
        grouped_xyz = index_points(xyz, idx) - centers.unsqueeze(2)  # (B, S, K, 3)
        grouped_feat = index_points(feat, idx)  # (B, S, K, D)

        x = torch.cat([grouped_feat, grouped_xyz], dim=-1)  # (B, S, K, D+3)
        x = self.group_mlp(x)  # (B, S, K, D)
        tokens = x.max(dim=2).values  # (B, S, D)
        tokens = tokens + self.pos(centers)
        return tokens, centers


class PatchTransformer(nn.Module):
    """Patch transformer backbone that returns cls + patch tokens."""

    def __init__(
        self,
        *,
        in_channels: int,
        embed_dim: int,
        num_patches: int,
        group_size: int,
        depth: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        d = int(embed_dim)
        self.patch = PatchEmbed(
            in_channels=int(in_channels),
            embed_dim=int(d),
            num_patches=int(num_patches),
            group_size=int(group_size),
            dropout=float(dropout),
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.cls_pos = nn.Parameter(torch.zeros(1, 1, d))
        self.pos = nn.Sequential(
            nn.Linear(3, d),
            nn.ReLU(inplace=True),
            nn.Linear(d, d),
        )

        enc_layer = nn.TransformerEncoderLayer(
            d_model=int(d),
            nhead=int(heads),
            dim_feedforward=int(d) * 4,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(depth))
        self.norm = nn.LayerNorm(int(d))

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.cls_pos, std=0.02)

    def forward(
        self,
        points: torch.Tensor,
        *,
        mask_ratio: float = 0.0,
        mask_token: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        tokens, centers = self.patch(points)  # (B, S, D), (B, S, 3)
        b, s, d = tokens.shape

        mask_ratio_f = float(mask_ratio)
        if not (0.0 <= mask_ratio_f < 1.0):
            raise ValueError("mask_ratio must be in [0, 1)")

        mask_idx = torch.empty((b, 0), device=tokens.device, dtype=torch.long)
        if mask_ratio_f > 0.0:
            m = int(round(float(s) * mask_ratio_f))
            m = max(1, min(s - 1, m))

            scores = torch.rand((b, s), device=tokens.device, dtype=torch.float32)
            order = torch.argsort(scores, dim=1)
            mask_idx = order[:, :m]  # (B, M)

            if mask_token is None:
                raise ValueError("mask_token is required when mask_ratio > 0")
            if mask_token.ndim != 3 or int(mask_token.shape[0]) != 1 or int(mask_token.shape[1]) != 1:
                raise ValueError("mask_token must have shape (1, 1, D)")

            batch = torch.arange(b, device=tokens.device).unsqueeze(1)
            masked = tokens.clone()
            masked[batch, mask_idx] = mask_token.expand(b, m, d).to(tokens.dtype)
            tokens = masked

        cls = self.cls_token.expand(b, 1, d) + self.cls_pos
        pos = self.pos(centers.to(tokens.dtype))
        seq = torch.cat([cls, tokens + pos], dim=1)  # (B, 1+S, D)

        x = self.encoder(seq)
        x = self.norm(x)
        cls_feat = x[:, 0]
        patch_feat = x[:, 1:]
        return {"cls": cls_feat, "patch": patch_feat, "centers": centers, "mask_idx": mask_idx}


class DINOHead(nn.Module):
    """Projector + normalized prototypes for DINO-style losses."""

    def __init__(
        self,
        in_dim: int,
        proj_dim: int,
        out_dim: int,
        *,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(int(in_dim), int(proj_dim), bias=False),
            nn.BatchNorm1d(int(proj_dim)),
            nn.GELU(),
            nn.Linear(int(proj_dim), int(proj_dim), bias=False),
            nn.BatchNorm1d(int(proj_dim)),
            nn.GELU(),
            nn.Linear(int(proj_dim), int(proj_dim), bias=True),
        )
        self.prototypes = nn.Linear(int(proj_dim), int(out_dim), bias=False)
        self.drop = nn.Dropout(p=float(dropout))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if x.ndim == 2:
            flat = x
            shape = None
        elif x.ndim == 3:
            b, s, d = x.shape
            flat = x.reshape(b * s, d)
            shape = (b, s)
        else:
            raise ValueError(f"Expected x shape (B, D) or (B, S, D), got {tuple(x.shape)}")

        z = self.projector(self.drop(flat))
        z = F.normalize(z, dim=1)
        w = F.normalize(self.prototypes.weight, dim=1)
        logits = F.linear(z, w)

        if shape is not None:
            b, s = shape
            z = z.view(b, s, -1)
            logits = logits.view(b, s, -1)
        return {"z": z, "logits": logits}


def dino_cross_view_loss(
    student_logits: list[torch.Tensor],
    teacher_logits: list[torch.Tensor],
    *,
    student_temperature: float,
    teacher_temperature: float,
    center: torch.Tensor,
) -> torch.Tensor:
    """DINO cross-view loss on global (cls) logits."""

    if len(student_logits) < 2 or len(teacher_logits) < 2:
        raise ValueError("Need >=2 views for DINO cross-view loss")
    t_s = float(student_temperature)
    t_t = float(teacher_temperature)
    if t_s <= 0 or t_t <= 0:
        raise ValueError("Temperatures must be > 0")
    if center.ndim != 2 or int(center.shape[0]) != 1:
        raise ValueError(f"Expected center shape (1, K), got {tuple(center.shape)}")

    loss = 0.0
    n_terms = 0
    for i, t_logits in enumerate(teacher_logits):
        with torch.no_grad():
            t_prob = F.softmax((t_logits.detach() - center) / t_t, dim=1)
        for j, s_logits in enumerate(student_logits):
            if j == i:
                continue
            s_logprob = F.log_softmax(s_logits / t_s, dim=1)
            loss = loss + (-(t_prob * s_logprob).sum(dim=1).mean())
            n_terms += 1
    if n_terms == 0:
        raise RuntimeError("No cross-view terms computed")
    return loss / float(n_terms)


def ibot_patch_loss(
    student_patch_logits: torch.Tensor,
    teacher_patch_logits: torch.Tensor,
    mask_idx: torch.Tensor,
    *,
    student_temperature: float,
    teacher_temperature: float,
    center: torch.Tensor,
) -> torch.Tensor:
    """iBOT-style patch loss on masked patches (same view).

    Args:
        student_patch_logits: (B, S, K)
        teacher_patch_logits: (B, S, K) (stop-grad)
        mask_idx: (B, M) indices of masked patches
    """

    if student_patch_logits.ndim != 3 or teacher_patch_logits.ndim != 3:
        raise ValueError("Expected patch logits shapes (B, S, K)")
    if student_patch_logits.shape != teacher_patch_logits.shape:
        raise ValueError("student/teacher patch logits must have the same shape")
    if mask_idx.ndim != 2:
        raise ValueError("mask_idx must be (B, M)")
    if int(mask_idx.shape[1]) == 0:
        return student_patch_logits.sum() * 0.0

    t_s = float(student_temperature)
    t_t = float(teacher_temperature)
    if t_s <= 0 or t_t <= 0:
        raise ValueError("Temperatures must be > 0")
    if center.ndim != 2 or int(center.shape[0]) != 1:
        raise ValueError(f"Expected center shape (1, K), got {tuple(center.shape)}")

    with torch.no_grad():
        t_prob = F.softmax((teacher_patch_logits.detach() - center.unsqueeze(1)) / t_t, dim=-1)
    s_logprob = F.log_softmax(student_patch_logits / t_s, dim=-1)
    per_patch = -(t_prob * s_logprob).sum(dim=-1)  # (B, S)
    masked = _gather_batch(per_patch, mask_idx)  # (B, M)
    return masked.mean()


class DINOV2PointMAE(nn.Module):
    """Toy-first DINOv2-style model for point clouds.

    Combines:
      - DINO loss on global [CLS] token (cross-view student/teacher)
      - iBOT-style loss on masked patch tokens (same-view teacher->student)
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
        k = int(out_dim)
        if k < 2:
            raise ValueError("out_dim must be >= 2")

        self.student_backbone = PatchTransformer(
            in_channels=int(in_channels),
            embed_dim=int(embed_dim),
            num_patches=int(num_patches),
            group_size=int(group_size),
            depth=int(depth),
            heads=int(heads),
            dropout=float(dropout),
        )
        self.teacher_backbone = PatchTransformer(
            in_channels=int(in_channels),
            embed_dim=int(embed_dim),
            num_patches=int(num_patches),
            group_size=int(group_size),
            depth=int(depth),
            heads=int(heads),
            dropout=float(dropout),
        )

        self.student_head = DINOHead(int(embed_dim), int(proj_dim), int(out_dim), dropout=float(dropout))
        self.teacher_head = DINOHead(int(embed_dim), int(proj_dim), int(out_dim), dropout=float(dropout))

        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(embed_dim)))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        self.register_buffer("center_cls", torch.zeros(1, int(out_dim), dtype=torch.float32))
        self.register_buffer("center_patch", torch.zeros(1, int(out_dim), dtype=torch.float32))

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
    def update_centers(
        self,
        *,
        teacher_cls_logits: list[torch.Tensor],
        teacher_patch_logits: list[torch.Tensor],
        center_momentum: float = 0.9,
    ) -> None:
        cm = float(center_momentum)
        if not (0.0 <= cm < 1.0):
            raise ValueError("center_momentum must be in [0, 1)")

        if teacher_cls_logits:
            cls_cat = torch.cat([t.detach() for t in teacher_cls_logits], dim=0)  # (V*B, K)
            cls_center = cls_cat.mean(dim=0, keepdim=True)
            self.center_cls.mul_(cm).add_(cls_center, alpha=(1.0 - cm))

        if teacher_patch_logits:
            patch_cat = torch.cat([t.detach().reshape(-1, t.shape[-1]) for t in teacher_patch_logits], dim=0)
            patch_center = patch_cat.mean(dim=0, keepdim=True)
            self.center_patch.mul_(cm).add_(patch_center, alpha=(1.0 - cm))

    def forward_student(self, points: torch.Tensor, *, mask_ratio: float = 0.5) -> dict[str, torch.Tensor]:
        feats = self.student_backbone(points, mask_ratio=float(mask_ratio), mask_token=self.mask_token)
        cls = feats["cls"]
        patch = feats["patch"]
        cls_out = self.student_head(cls)
        patch_out = self.student_head(patch)
        return {
            "h": cls,
            "cls_logits": cls_out["logits"],
            "patch_logits": patch_out["logits"],
            "mask_idx": feats["mask_idx"],
        }

    @torch.no_grad()
    def forward_teacher(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        feats = self.teacher_backbone(points, mask_ratio=0.0, mask_token=None)
        cls = feats["cls"]
        patch = feats["patch"]
        cls_out = self.teacher_head(cls)
        patch_out = self.teacher_head(patch)
        return {
            "h": cls,
            "cls_logits": cls_out["logits"],
            "patch_logits": patch_out["logits"],
        }

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        # Default forward: student global features/logits without masking (useful for downstream).
        feats = self.student_backbone(points, mask_ratio=0.0, mask_token=None)
        cls = feats["cls"]
        cls_out = self.student_head(cls)
        return {"h": cls, "logits": cls_out["logits"]}


_VARIANTS: dict[str, dict] = {
    "dinov2_pointmae_tiny": {
        "embed_dim": 96,
        "num_patches": 12,
        "group_size": 16,
        "depth": 2,
        "heads": 4,
        "proj_dim": 192,
        "out_dim": 64,
        "dropout": 0.0,
    },
    "dinov2_pointmae_small": {
        "embed_dim": 128,
        "num_patches": 16,
        "group_size": 16,
        "depth": 4,
        "heads": 4,
        "proj_dim": 256,
        "out_dim": 128,
        "dropout": 0.0,
    },
    "dinov2_pointmae_base": {
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


def build_dinov2_pointmae(
    *,
    in_channels: int,
    variant: str = "dinov2_pointmae_small",
    dropout: float | None = None,
    out_dim: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown DINOv2 variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    out = int(spec["out_dim"]) if out_dim is None else int(out_dim)
    return DINOV2PointMAE(
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

    m = build_dinov2_pointmae(in_channels=3, variant="dinov2_pointmae_tiny")
    t1 = m.forward_teacher(v1)
    t2 = m.forward_teacher(v2)
    s1 = m.forward_student(v1, mask_ratio=0.5)
    s2 = m.forward_student(v2, mask_ratio=0.5)

    loss_cls = dino_cross_view_loss(
        [s1["cls_logits"], s2["cls_logits"]],
        [t1["cls_logits"], t2["cls_logits"]],
        student_temperature=0.1,
        teacher_temperature=0.04,
        center=m.center_cls,
    )
    loss_patch = 0.5 * (
        ibot_patch_loss(
            s1["patch_logits"],
            t1["patch_logits"],
            s1["mask_idx"],
            student_temperature=0.1,
            teacher_temperature=0.04,
            center=m.center_patch,
        )
        + ibot_patch_loss(
            s2["patch_logits"],
            t2["patch_logits"],
            s2["mask_idx"],
            student_temperature=0.1,
            teacher_temperature=0.04,
            center=m.center_patch,
        )
    )
    loss = loss_cls + loss_patch
    loss.backward()
    m.update_centers(
        teacher_cls_logits=[t1["cls_logits"], t2["cls_logits"]],
        teacher_patch_logits=[t1["patch_logits"], t2["patch_logits"]],
        center_momentum=0.9,
    )
    m.momentum_update_teacher(ema_decay=0.99)
    print("ok", float(loss.item()))

