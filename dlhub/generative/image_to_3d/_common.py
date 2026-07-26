from __future__ import annotations
import torch
from torch import nn


def check_nchw(image: torch.Tensor) -> torch.Tensor:
    image = image.to(torch.float32)
    if image.ndim != 4:
        raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(image.shape)}")
    return image


class ToyImageTo3DGenerator(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        mode: str,
        in_channels: int,
        width: int,
        depth: int,
        voxel_size: int = 10,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.voxel_size = int(voxel_size)
        layers = [nn.Conv2d(int(in_channels), int(width), 3, 1, 1), nn.ReLU(inplace=True)]
        for _ in range(max(0, int(depth) - 1)):
            layers.extend([nn.Conv2d(int(width), int(width), 3, 1, 1), nn.ReLU(inplace=True)])
        self.encoder = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        plane_dim = 3 * int(in_channels) * self.voxel_size * self.voxel_size
        self.plane_head = nn.Linear(int(width), plane_dim)
        self.density_head = nn.Linear(int(width), self.voxel_size**3)
        self.mesh_head = nn.Linear(int(width), self.voxel_size * 3)

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        x = check_nchw(image)
        pooled = self.pool(self.encoder(x)).flatten(1)
        triplanes = self.plane_head(pooled).view(
            x.shape[0], 3, x.shape[1], self.voxel_size, self.voxel_size
        )
        density = torch.sigmoid(self.density_head(pooled)).view(
            x.shape[0], 1, self.voxel_size, self.voxel_size, self.voxel_size
        )
        mesh_tokens = self.mesh_head(pooled).view(x.shape[0], self.voxel_size, 3)
        return {"triplanes": triplanes, "density": density, "mesh_tokens": mesh_tokens}


def build_toy_image_to_3d_generator(
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
    return ToyImageTo3DGenerator(
        family=str(family),
        mode=str(mode),
        in_channels=int(in_channels),
        width=width,
        depth=int(cfg["depth"]),
        voxel_size=int(cfg.get("voxel_size", 10)),
    )


def smoke_test_image_to_3d(builder, variant: str) -> None:
    model = builder(in_channels=3, variant=variant, width_mult=0.5)
    out = model(torch.randn(2, 3, 32, 32))
    print(variant, {k: tuple(v.shape) for k, v in out.items()})
