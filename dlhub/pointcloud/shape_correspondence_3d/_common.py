from __future__ import annotations
import torch
from torch import nn


def check_pair(
    src: torch.Tensor, tgt: torch.Tensor, in_channels: int
) -> tuple[torch.Tensor, torch.Tensor]:
    src = src.to(torch.float32)
    tgt = tgt.to(torch.float32)
    if src.ndim != 3 or tgt.ndim != 3:
        raise ValueError(
            f"Expected paired point sets with shape (B, N, C), got {tuple(src.shape)} and {tuple(tgt.shape)}"
        )
    if src.shape[-1] != int(in_channels) or tgt.shape[-1] != int(in_channels):
        raise ValueError(f"Expected {int(in_channels)} channels")
    return src, tgt


class TinyShapeCorrespondenceModel(nn.Module):
    def __init__(self, *, family: str, mode: str, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.in_channels = int(in_channels)
        self.src_proj = nn.Linear(int(in_channels), int(width))
        self.tgt_proj = nn.Linear(int(in_channels), int(width))
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(int(width)), nn.Linear(int(width), int(width)), nn.GELU()
                )
                for _ in range(max(1, int(depth)))
            ]
        )

    def encode(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:
        feat = proj(x)
        for block in self.blocks:
            feat = feat + block(feat)
        return feat

    def forward(
        self, src_points: torch.Tensor, tgt_points: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        src, tgt = check_pair(src_points, tgt_points, self.in_channels)
        src_feat = self.encode(src, self.src_proj)
        tgt_feat = self.encode(tgt, self.tgt_proj)
        scores = torch.matmul(src_feat, tgt_feat.transpose(1, 2)) / max(
            1.0, src_feat.shape[-1] ** 0.5
        )
        return {"scores": scores, "matches": scores.argmax(dim=-1)}


def build_baseline_shape_correspondence_model(
    *,
    family: str,
    mode: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
) -> nn.Module:
    cfg = variants[str(variant)]
    width = max(16, int(int(cfg["width"]) * float(width_mult)))
    return TinyShapeCorrespondenceModel(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
    )


def smoke_test_shape_correspondence_model(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 24, 3), torch.randn(2, 20, 3))
    print(variant, tuple(out["scores"].shape))
