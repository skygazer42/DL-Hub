from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    return x


class TinyReIDEncoder(nn.Module):
    def __init__(self, *, in_channels: int, width: int, depth: int, dropout: float = 0.0) -> None:
        super().__init__()
        c = int(width)
        layers: list[nn.Module] = [
            nn.Conv2d(int(in_channels), c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
        ]
        cur = c
        for _ in range(max(1, int(depth))):
            layers += [
                nn.Conv2d(cur, cur * 2, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(cur * 2),
                nn.ReLU(inplace=True),
            ]
            if float(dropout) > 0:
                layers.append(nn.Dropout2d(float(dropout)))
            cur *= 2
        self.net = nn.Sequential(*layers)
        self.out_channels = int(cur)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(check_nchw(x))


class ToyReIdentifier(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        num_classes: int,
        width: int,
        depth: int,
        embed_dim: int,
        pooling: str = 'avg',
        part_branches: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.pooling = str(pooling)
        self.part_branches = int(part_branches)
        self.encoder = TinyReIDEncoder(
            in_channels=int(in_channels), width=int(width), depth=int(depth), dropout=float(dropout)
        )
        proj_in = int(self.encoder.out_channels) * (1 + max(0, self.part_branches))
        self.proj = nn.Linear(proj_in, int(embed_dim))
        self.cls = nn.Linear(int(embed_dim), int(num_classes))

    def _pool(self, feat: torch.Tensor) -> torch.Tensor:
        if self.pooling == 'max':
            pooled = F.adaptive_max_pool2d(feat, (1, 1)).flatten(1)
        elif self.pooling == 'gem':
            pooled = F.adaptive_avg_pool2d(feat.clamp_min(1e-6).pow(3.0), (1, 1)).pow(1.0 / 3.0).flatten(1)
        else:
            pooled = F.adaptive_avg_pool2d(feat, (1, 1)).flatten(1)
        if self.part_branches > 0:
            parts = F.adaptive_avg_pool2d(feat, (self.part_branches, 1)).flatten(2).reshape(feat.shape[0], -1)
            pooled = torch.cat([pooled, parts], dim=1)
        return pooled

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        feat = self.encoder(image)
        pooled = self._pool(feat)
        embedding = F.normalize(self.proj(pooled), dim=1)
        logits = self.cls(embedding)
        return {'embedding': embedding, 'logits': logits}


def build_toy_reidentifier(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.0,
    pooling: str = 'avg',
    part_branches: int = 0,
) -> nn.Module:
    spec = variants[str(variant)]
    width = max(8, int(int(spec['width']) * float(width_mult)))
    embed = max(32, int(int(spec['embed']) * float(width_mult)))
    return ToyReIdentifier(
        family=str(family),
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        width=width,
        depth=int(spec['depth']),
        embed_dim=embed,
        pooling=str(pooling),
        part_branches=int(part_branches),
        dropout=float(dropout),
    )


def smoke_test_reid(builder, variant: str) -> None:
    model = builder(in_channels=3, num_classes=8, variant=variant, width_mult=0.5, dropout=0.0)
    x = torch.randn(2, 3, 128, 64)
    out = model(x)
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
    print('ok')
