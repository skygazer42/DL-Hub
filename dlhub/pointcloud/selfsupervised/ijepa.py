import torch
import torch.nn.functional as F
from torch import nn

from dlhub.pointcloud.selfsupervised.dinov2 import PatchTransformer


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


def ijepa_patch_loss(
    pred: torch.Tensor, target: torch.Tensor, mask_idx: torch.Tensor
) -> torch.Tensor:
    """I-JEPA-style masked prediction loss on patch embeddings.

    Args:
        pred: (B, S, D) predicted patch embeddings (student predictor output).
        target: (B, S, D) target patch embeddings (teacher backbone output).
        mask_idx: (B, M) masked patch indices (from student).
    """

    if pred.ndim != 3 or target.ndim != 3:
        raise ValueError(
            f"Expected pred/target shapes (B, S, D), got {tuple(pred.shape)} and {tuple(target.shape)}"
        )
    if pred.shape != target.shape:
        raise ValueError("pred and target must have the same shape")
    if mask_idx.ndim != 2:
        raise ValueError("mask_idx must be (B, M)")
    if int(mask_idx.shape[1]) == 0:
        return pred.sum() * 0.0

    pred_m = _gather_batch(pred, mask_idx)  # (B, M, D)
    target_m = _gather_batch(target.detach(), mask_idx)  # (B, M, D)
    pred_m = F.normalize(pred_m, dim=-1)
    target_m = F.normalize(target_m, dim=-1)
    loss = 2.0 - 2.0 * (pred_m * target_m).sum(dim=-1)
    return loss.mean()


class PatchPredictor(nn.Module):
    """Lightweight predictor that maps patch embeddings -> predicted target embeddings."""

    def __init__(self, dim: int, *, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        h = int(hidden_dim) if hidden_dim is not None else d
        if d <= 0 or h <= 0:
            raise ValueError("dim/hidden_dim must be > 0")
        self.net = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, h),
            nn.GELU(),
            nn.Dropout(p=float(dropout)),
            nn.Linear(h, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return self.net(x)
        if x.ndim == 3:
            b, s, d = x.shape
            y = self.net(x.reshape(b * s, d))
            return y.view(b, s, d)
        raise ValueError(f"Expected x shape (B, D) or (B, S, D), got {tuple(x.shape)}")


class IJEPAPointMAE(nn.Module):
    """Toy-first I-JEPA-style masked prediction for point clouds.

    - Student: patch transformer with mask tokens + predictor head.
    - Teacher: patch transformer without masking (EMA updated from student).
    - Loss: predict teacher patch embeddings on masked patches (cosine loss).
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
        predictor_hidden: int | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        d = int(embed_dim)

        self.student = PatchTransformer(
            in_channels=int(in_channels),
            embed_dim=int(d),
            num_patches=int(num_patches),
            group_size=int(group_size),
            depth=int(depth),
            heads=int(heads),
            dropout=float(dropout),
        )
        self.teacher = PatchTransformer(
            in_channels=int(in_channels),
            embed_dim=int(d),
            num_patches=int(num_patches),
            group_size=int(group_size),
            depth=int(depth),
            heads=int(heads),
            dropout=float(dropout),
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, int(d)))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.predictor = PatchPredictor(int(d), hidden_dim=predictor_hidden, dropout=float(dropout))

        self.reset_teacher()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # Keep teacher in eval mode (important if any dropout/BN exists).
        self.teacher.eval()
        return self

    @torch.no_grad()
    def reset_teacher(self) -> None:
        self.teacher.load_state_dict(self.student.state_dict())
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def momentum_update_teacher(self, *, ema_decay: float = 0.996) -> None:
        m = float(ema_decay)
        if not (0.0 <= m < 1.0):
            raise ValueError("ema_decay must be in [0, 1)")

        for ps, pt in zip(self.student.parameters(), self.teacher.parameters(), strict=True):
            pt.data.mul_(m).add_(ps.data, alpha=(1.0 - m))

    def forward_student(
        self, points: torch.Tensor, *, mask_ratio: float = 0.5
    ) -> dict[str, torch.Tensor]:
        out = self.student(points, mask_ratio=float(mask_ratio), mask_token=self.mask_token)
        patch = out["patch"]  # (B, S, D)
        pred = self.predictor(patch)  # (B, S, D)
        return {"h": out["cls"], "patch": patch, "pred": pred, "mask_idx": out["mask_idx"]}

    @torch.no_grad()
    def forward_teacher(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.teacher(points, mask_ratio=0.0, mask_token=None)
        return {"h": out["cls"], "patch": out["patch"]}

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        # Default forward: student features without masking (useful for downstream).
        out = self.student(points, mask_ratio=0.0, mask_token=None)
        return {"h": out["cls"], "patch": out["patch"]}


_VARIANTS: dict[str, dict] = {
    "ijepa_pointmae_tiny": {
        "embed_dim": 96,
        "num_patches": 12,
        "group_size": 16,
        "depth": 2,
        "heads": 4,
        "dropout": 0.0,
    },
    "ijepa_pointmae_small": {
        "embed_dim": 128,
        "num_patches": 16,
        "group_size": 16,
        "depth": 4,
        "heads": 4,
        "dropout": 0.0,
    },
    "ijepa_pointmae_base": {
        "embed_dim": 192,
        "num_patches": 24,
        "group_size": 24,
        "depth": 6,
        "heads": 6,
        "dropout": 0.0,
    },
}


def build_ijepa_pointmae(
    *,
    in_channels: int,
    variant: str = "ijepa_pointmae_small",
    dropout: float | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown I-JEPA variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    return IJEPAPointMAE(
        in_channels=int(in_channels),
        embed_dim=int(spec["embed_dim"]),
        num_patches=int(spec["num_patches"]),
        group_size=int(spec["group_size"]),
        depth=int(spec["depth"]),
        heads=int(spec["heads"]),
        predictor_hidden=int(spec["embed_dim"]) * 2,
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    pts = torch.randn(4, 96, 3)
    m = build_ijepa_pointmae(in_channels=3, variant="ijepa_pointmae_tiny", dropout=0.0)

    with torch.no_grad():
        t = m.forward_teacher(pts)["patch"]
    s = m.forward_student(pts, mask_ratio=0.5)
    loss = ijepa_patch_loss(s["pred"], t, s["mask_idx"])
    loss.backward()
    m.momentum_update_teacher(ema_decay=0.99)
    print("ok", float(loss.item()))
