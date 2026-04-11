from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


class ToyText3DGenerator(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        latent_dim: int,
        width: int,
        depth: int,
        voxel_size: int = 12,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.in_channels = int(in_channels)
        self.latent_dim = int(latent_dim)
        self.voxel_size = int(voxel_size)
        self.text_proj = nn.Sequential(
            nn.Linear(32, int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(width)),
        )
        self.latent_proj = nn.Sequential(
            nn.Linear(int(latent_dim), int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(width)),
        )
        layers: list[nn.Module] = []
        cur = int(width) * 2
        for _ in range(max(1, int(depth))):
            layers.extend([nn.Linear(cur, int(width)), nn.GELU()])
            cur = int(width)
        self.backbone = nn.Sequential(*layers)
        plane_dim = 3 * self.in_channels * self.voxel_size * self.voxel_size
        self.plane_head = nn.Linear(int(width), plane_dim)
        self.density_head = nn.Linear(int(width), self.voxel_size**3)
        self.mesh_head = nn.Linear(int(width), self.voxel_size * 3)

    def sample(self, *, batch_size: int = 2, device: torch.device | str | None = None) -> torch.Tensor:
        return self.forward(batch_size=batch_size, device=device)["triplanes"]

    def forward(
        self,
        *,
        batch_size: int = 2,
        device: torch.device | str | None = None,
        text: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        dev = _resolve_device(device)
        batch = int(batch_size)
        if text is None:
            text = torch.randn(batch, 32, device=dev)
        elif text.shape != (batch, 32):
            raise ValueError(f"text must have shape {(batch, 32)}, got {tuple(text.shape)}")
        latent = torch.randn(batch, self.latent_dim, device=dev)
        fused = self.backbone(torch.cat([self.text_proj(text), self.latent_proj(latent)], dim=1))
        triplanes = self.plane_head(fused).view(
            batch, 3, self.in_channels, self.voxel_size, self.voxel_size
        )
        density = torch.sigmoid(self.density_head(fused)).view(
            batch, 1, self.voxel_size, self.voxel_size, self.voxel_size
        )
        mesh_tokens = self.mesh_head(fused).view(batch, self.voxel_size, 3)
        return {"triplanes": triplanes, "density": density, "mesh_tokens": mesh_tokens}


def build_toy_text3d_family(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    latent_dim: int = 64,
    variant: str,
    width_mult: float = 1.0,
) -> nn.Module:
    cfg = variants[str(variant)]
    width = max(16, int(int(cfg["width"]) * float(width_mult)))
    latent = max(int(latent_dim), int(cfg["latent"]))
    return ToyText3DGenerator(
        family=str(family),
        in_channels=int(in_channels),
        latent_dim=latent,
        width=width,
        depth=int(cfg["depth"]),
    )


def smoke_test_text3d(builder: Callable[..., nn.Module], variant: str) -> None:
    model = builder(in_channels=3, latent_dim=64, variant=variant, width_mult=0.5)
    out = model.forward(batch_size=2)
    shapes = {k: tuple(v.shape) for k, v in out.items()}
    print(variant, shapes)


__all__ = ["ToyText3DGenerator", "build_toy_text3d_family", "smoke_test_text3d"]
