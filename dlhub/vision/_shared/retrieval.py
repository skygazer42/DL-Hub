from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    return x


class TinyRetrievalEncoder(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        c = int(width)
        layers = [nn.Conv2d(int(in_channels), c, 3, 2, 1), nn.ReLU(inplace=True)]
        for _ in range(max(1, int(depth))):
            layers += [nn.Conv2d(c, c * 2, 3, 2, 1), nn.ReLU(inplace=True)]
            c *= 2
        self.net = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, x):
        feat = self.net(check_nchw(x))
        return F.normalize(F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1), dim=1)


class CompactRetrievalModel(nn.Module):
    def __init__(
        self, *, family: str, in_channels: int, width: int, depth: int, embed_dim: int
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.encoder = TinyRetrievalEncoder(
            in_channels=int(in_channels), width=int(width), depth=int(depth)
        )
        self.proj = nn.Linear(int(self.encoder.out_channels), int(embed_dim))

    def forward(
        self, image: torch.Tensor, gallery: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        query = F.normalize(self.proj(self.encoder(image)), dim=1)
        out = {"embedding": query}
        if gallery is not None:
            gal = F.normalize(self.proj(self.encoder(gallery)), dim=1)
            out["gallery_embedding"] = gal
            out["similarity"] = query @ gal.t()
        return out


def build_baseline_retrieval_model(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
) -> nn.Module:
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    embed = max(64, int(int(spec["embed"]) * float(width_mult)))
    return CompactRetrievalModel(
        family=str(family),
        in_channels=int(in_channels),
        width=width,
        depth=int(spec["depth"]),
        embed_dim=embed,
    )


def smoke_test_retrieval(builder, variant: str):
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    x = torch.randn(2, 3, 64, 64)
    g = torch.randn(3, 3, 64, 64)
    out = model(x, g)
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
