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


def data2vec_loss(
    *,
    pred_cls: torch.Tensor,
    target_cls: torch.Tensor,
    pred_patch: torch.Tensor,
    target_patch: torch.Tensor,
    mask_idx: torch.Tensor,
    cls_weight: float = 1.0,
    patch_weight: float = 1.0,
    loss: str = "mse",
) -> torch.Tensor:
    """data2vec-style regression loss (CLS + masked patches).

    Args:
        pred_cls: (B, D) student predicted cls embedding.
        target_cls: (B, D) teacher target cls embedding (usually detached).
        pred_patch: (B, S, D) student predicted patch embeddings.
        target_patch: (B, S, D) teacher target patch embeddings.
        mask_idx: (B, M) masked patch indices (from student).
    """

    if pred_cls.ndim != 2 or target_cls.ndim != 2:
        raise ValueError(
            f"Expected pred_cls/target_cls shapes (B, D), got {tuple(pred_cls.shape)} and {tuple(target_cls.shape)}"
        )
    if pred_cls.shape != target_cls.shape:
        raise ValueError("pred_cls and target_cls must have the same shape")

    if pred_patch.ndim != 3 or target_patch.ndim != 3:
        raise ValueError(
            f"Expected pred_patch/target_patch shapes (B, S, D), got {tuple(pred_patch.shape)} and {tuple(target_patch.shape)}"
        )
    if pred_patch.shape != target_patch.shape:
        raise ValueError("pred_patch and target_patch must have the same shape")

    if mask_idx.ndim != 2:
        raise ValueError(f"Expected mask_idx shape (B, M), got {tuple(mask_idx.shape)}")
    if int(mask_idx.shape[0]) != int(pred_patch.shape[0]):
        raise ValueError("Batch mismatch between mask_idx and patch tensors")

    loss_name = str(loss).lower().strip()
    if loss_name not in {"mse", "smooth_l1"}:
        raise ValueError("loss must be one of: 'mse', 'smooth_l1'")

    cw = float(cls_weight)
    pw = float(patch_weight)
    if cw < 0 or pw < 0:
        raise ValueError("cls_weight and patch_weight must be >= 0")

    target_cls = target_cls.detach()
    target_patch = target_patch.detach()

    if loss_name == "mse":
        cls_loss = F.mse_loss(pred_cls, target_cls)
    else:
        cls_loss = F.smooth_l1_loss(pred_cls, target_cls)

    if int(mask_idx.shape[1]) == 0:
        patch_loss = pred_patch.new_zeros(())
    else:
        pred_m = _gather_batch(pred_patch, mask_idx)  # (B, M, D)
        tgt_m = _gather_batch(target_patch, mask_idx)  # (B, M, D)
        if loss_name == "mse":
            patch_loss = F.mse_loss(pred_m, tgt_m)
        else:
            patch_loss = F.smooth_l1_loss(pred_m, tgt_m)

    return cw * cls_loss + pw * patch_loss


class PredictorMLP(nn.Module):
    """Predictor MLP that supports (B, D) and (B, S, D)."""

    def __init__(self, dim: int, *, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        d = int(dim)
        h = int(hidden_dim) if hidden_dim is not None else d * 2
        if d <= 0 or h <= 0:
            raise ValueError("dim and hidden_dim must be > 0")

        self.net = nn.Sequential(
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
            flat = x.reshape(b * s, d)
            y = self.net(flat)
            return y.view(b, s, d)
        raise ValueError(f"Expected x shape (B, D) or (B, S, D), got {tuple(x.shape)}")


class Data2VecPointMAE(nn.Module):
    """Toy-first data2vec-style masked representation regression for point clouds.

    - Student: patch transformer with mask tokens + predictor MLP.
    - Teacher: patch transformer without masking (EMA updated from student).
    - Loss: regress teacher representations (CLS + masked patches).
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
        self.predictor = PredictorMLP(int(d), hidden_dim=predictor_hidden, dropout=float(dropout))

        self.reset_teacher()

    def train(self, mode: bool = True):  # type: ignore[override]
        super().train(mode)
        # Keep teacher in eval mode (important if any dropout/BN exists).
        self.teacher.eval()
        return self

    @staticmethod
    def _norm_rep(x: torch.Tensor) -> torch.Tensor:
        # Feature-scale normalization without learnable params.
        d = int(x.shape[-1])
        return F.layer_norm(x.to(torch.float32), (d,))

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
        cls = self._norm_rep(out["cls"])
        patch = self._norm_rep(out["patch"])
        pred_cls = self._norm_rep(self.predictor(cls))
        pred_patch = self._norm_rep(self.predictor(patch))
        return {
            "h": cls,
            "cls": cls,
            "patch": patch,
            "pred_cls": pred_cls,
            "pred_patch": pred_patch,
            "mask_idx": out["mask_idx"],
        }

    @torch.no_grad()
    def forward_teacher(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self.teacher(points, mask_ratio=0.0, mask_token=None)
        cls = self._norm_rep(out["cls"])
        patch = self._norm_rep(out["patch"])
        return {"h": cls, "cls": cls, "patch": patch}

    def forward(self, points: torch.Tensor) -> dict[str, torch.Tensor]:
        # Default forward: student features without masking (useful for downstream).
        out = self.student(points, mask_ratio=0.0, mask_token=None)
        cls = self._norm_rep(out["cls"])
        patch = self._norm_rep(out["patch"])
        return {"h": cls, "patch": patch}


_VARIANTS: dict[str, dict] = {
    "data2vec_pointmae_tiny": {
        "embed_dim": 96,
        "num_patches": 12,
        "group_size": 16,
        "depth": 2,
        "heads": 4,
        "predictor_hidden": 192,
        "dropout": 0.0,
    },
    "data2vec_pointmae_small": {
        "embed_dim": 128,
        "num_patches": 16,
        "group_size": 16,
        "depth": 4,
        "heads": 4,
        "predictor_hidden": 256,
        "dropout": 0.0,
    },
    "data2vec_pointmae_base": {
        "embed_dim": 192,
        "num_patches": 24,
        "group_size": 24,
        "depth": 6,
        "heads": 6,
        "predictor_hidden": 384,
        "dropout": 0.0,
    },
}


def build_data2vec_pointmae(
    *,
    in_channels: int,
    variant: str = "data2vec_pointmae_small",
    dropout: float | None = None,
    predictor_hidden: int | None = None,
) -> nn.Module:
    name = str(variant).lower().strip()
    if name not in _VARIANTS:
        raise ValueError(f"Unknown data2vec variant: {variant!r}. Supported: {sorted(_VARIANTS)}")
    spec = _VARIANTS[name]
    p = float(spec["dropout"]) if dropout is None else float(dropout)
    ph = int(spec["predictor_hidden"]) if predictor_hidden is None else int(predictor_hidden)
    return Data2VecPointMAE(
        in_channels=int(in_channels),
        embed_dim=int(spec["embed_dim"]),
        num_patches=int(spec["num_patches"]),
        group_size=int(spec["group_size"]),
        depth=int(spec["depth"]),
        heads=int(spec["heads"]),
        predictor_hidden=int(ph),
        dropout=float(p),
    )


if __name__ == "__main__":
    torch.manual_seed(0)
    v1 = torch.randn(4, 96, 3)
    v2 = v1 + torch.randn_like(v1) * 0.01

    m = build_data2vec_pointmae(in_channels=3, variant="data2vec_pointmae_tiny")
    with torch.no_grad():
        t1 = m.forward_teacher(v1)
        t2 = m.forward_teacher(v2)
    s1 = m.forward_student(v1, mask_ratio=0.5)
    s2 = m.forward_student(v2, mask_ratio=0.5)

    loss = 0.5 * (
        data2vec_loss(
            pred_cls=s1["pred_cls"],
            target_cls=t1["cls"],
            pred_patch=s1["pred_patch"],
            target_patch=t1["patch"],
            mask_idx=s1["mask_idx"],
            cls_weight=1.0,
            patch_weight=1.0,
            loss="mse",
        )
        + data2vec_loss(
            pred_cls=s2["pred_cls"],
            target_cls=t2["cls"],
            pred_patch=s2["pred_patch"],
            target_patch=t2["patch"],
            mask_idx=s2["mask_idx"],
            cls_weight=1.0,
            patch_weight=1.0,
            loss="mse",
        )
    )
    loss.backward()
    m.momentum_update_teacher(ema_decay=0.99)
    print("ok", float(loss.item()))
