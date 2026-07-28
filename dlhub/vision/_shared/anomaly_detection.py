from __future__ import annotations

import torch
from torch import nn


def check_nchw(x: torch.Tensor) -> torch.Tensor:
    x = x.to(torch.float32)
    if x.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
    return x


class TinyAnomalyDetector(nn.Module):
    def __init__(self, *, family: str, in_channels: int, width: int, depth: int) -> None:
        super().__init__()
        self.family = str(family)
        c = int(width)
        layers = [nn.Conv2d(int(in_channels), c, 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(max(1, int(depth))):
            layers += [nn.Conv2d(c, c, 3, 1, 1), nn.ReLU(inplace=True)]
        self.encoder = nn.Sequential(*layers)
        self.decoder = nn.Conv2d(c, int(in_channels), 3, 1, 1)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(image)
        feat = self.encoder(x)
        recon = self.decoder(feat)
        anomaly_map = (recon - x).abs().mean(dim=1, keepdim=True)
        score = anomaly_map.mean(dim=(1, 2, 3), keepdim=False)
        return {"reconstruction": recon, "anomaly_map": anomaly_map, "score": score}


def build_baseline_anomaly_detector(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
) -> nn.Module:
    spec = variants[str(variant)]
    width = max(16, int(int(spec["width"]) * float(width_mult)))
    return TinyAnomalyDetector(
        family=str(family), in_channels=int(in_channels), width=width, depth=int(spec["depth"])
    )


def smoke_test_anomaly(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 64, 64))
    print(
        variant,
        {
            k: (
                tuple(v.shape)
                if torch.is_tensor(v) and v.ndim > 0
                else v.shape if hasattr(v, "shape") else type(v).__name__
            )
            for k, v in out.items()
        },
    )
