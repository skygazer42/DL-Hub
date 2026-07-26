from __future__ import annotations
import torch
from torch import nn


def check_points(points: torch.Tensor, in_channels: int) -> torch.Tensor:
    points = points.to(torch.float32)
    if points.ndim != 3:
        raise ValueError(f"Expected input shape (B, N, C), got {tuple(points.shape)}")
    if points.shape[-1] != int(in_channels):
        raise ValueError(f"Expected {int(in_channels)} channels, got {int(points.shape[-1])}")
    return points


class TinyOpenVocabulary3DModel(nn.Module):
    def __init__(self, *, family: str, mode: str, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.family = str(family)
        self.mode = str(mode)
        self.in_channels = int(in_channels)
        self.point_proj = nn.Linear(int(in_channels), int(width))
        self.text_proj = nn.Sequential(
            nn.Linear(32, int(width)), nn.GELU(), nn.Linear(int(width), int(width))
        )
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(int(width)), nn.Linear(int(width), int(width)), nn.GELU()
                )
                for _ in range(max(1, int(depth)))
            ]
        )
        self.logit_head = nn.Linear(int(width), 4)

    def forward(
        self, points: torch.Tensor, text: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        pts = check_points(points, self.in_channels)
        batch = pts.shape[0]
        text = (
            torch.zeros(batch, 32, dtype=pts.dtype, device=pts.device)
            if text is None
            else text.to(torch.float32)
        )
        point_feat = self.point_proj(pts)
        text_feat = self.text_proj(text).unsqueeze(1)
        feat = point_feat + text_feat
        for block in self.blocks:
            feat = feat + block(feat)
        return {
            "point_logits": self.logit_head(feat),
            "point_embedding": feat,
            "text_embedding": text_feat.squeeze(1),
        }


def build_toy_open_vocabulary_3d_model(
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
    return TinyOpenVocabulary3DModel(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
    )


def smoke_test_open_vocabulary_3d_model(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 24, 3), torch.randn(2, 32))
    print(variant, tuple(out["point_logits"].shape))
