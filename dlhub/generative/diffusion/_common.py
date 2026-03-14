from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn


def _make_mlp(
    *,
    in_dim: int,
    out_dim: int,
    width: int,
    depth: int,
    dropout: float,
    final_tanh: bool,
) -> nn.Module:
    layers: list[nn.Module] = []
    cur = int(in_dim)
    for _ in range(max(1, int(depth))):
        layers.append(nn.Linear(cur, int(width)))
        layers.append(nn.GELU())
        if float(dropout) > 0:
            layers.append(nn.Dropout(float(dropout)))
        cur = int(width)
    layers.append(nn.Linear(cur, int(out_dim)))
    if final_tanh:
        layers.append(nn.Tanh())
    return nn.Sequential(*layers)


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if device is None:
        return torch.device("cpu")
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


class ToyDiffusion(nn.Module):
    def __init__(
        self,
        *,
        family: str,
        in_channels: int,
        image_size: int,
        latent_dim: int,
        width: int,
        depth: int,
        dropout: float = 0.0,
        num_classes: int = 0,
        use_condition: bool = False,
        latent_space: bool = False,
        prediction_mode: str = "eps",
        step_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.family = str(family)
        self.in_channels = int(in_channels)
        self.image_size = int(image_size)
        self.latent_dim = int(latent_dim)
        self.num_classes = int(max(0, num_classes))
        self.use_condition = bool(use_condition and self.num_classes > 0)
        self.latent_space = bool(latent_space)
        self.prediction_mode = str(prediction_mode).strip().lower()
        self.step_scale = float(step_scale)
        self.flat_dim = self.in_channels * self.image_size * self.image_size
        self.cond_dim = int(max(8, min(64, int(width) // 2))) if self.use_condition else 0

        self.label_embed: nn.Module | None
        if self.use_condition:
            self.label_embed = nn.Embedding(self.num_classes, self.cond_dim)
        else:
            self.label_embed = None

        self.time_proj = nn.Sequential(
            nn.Linear(1, int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(width)),
        )
        self.latent_proj = nn.Sequential(
            nn.Linear(self.latent_dim, int(width)),
            nn.GELU(),
            nn.Linear(int(width), int(width)),
        )

        core_dim = self.latent_dim if self.latent_space else self.flat_dim
        aux_dim = int(width) + int(width) + self.cond_dim
        if self.latent_space:
            bottleneck_depth = max(1, int(depth) - 1)
            self.image_to_latent = _make_mlp(
                in_dim=self.flat_dim,
                out_dim=self.latent_dim,
                width=int(width),
                depth=bottleneck_depth,
                dropout=float(dropout),
                final_tanh=False,
            )
            self.latent_to_image = _make_mlp(
                in_dim=self.latent_dim,
                out_dim=self.flat_dim,
                width=int(width),
                depth=bottleneck_depth,
                dropout=float(dropout),
                final_tanh=False,
            )
        else:
            self.image_to_latent = None
            self.latent_to_image = None

        self.denoiser = _make_mlp(
            in_dim=core_dim + aux_dim,
            out_dim=core_dim,
            width=int(width),
            depth=int(depth),
            dropout=float(dropout),
            final_tanh=False,
        )

    def _prepare_labels(
        self,
        *,
        batch_size: int,
        device: torch.device,
        labels: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.use_condition:
            return None
        if labels is None:
            return torch.randint(0, self.num_classes, (int(batch_size),), device=device)
        if labels.ndim != 1:
            raise ValueError(f"labels must be 1D, got shape {tuple(labels.shape)}")
        if int(labels.shape[0]) != int(batch_size):
            raise ValueError(
                f"labels batch mismatch: expected {batch_size}, got {int(labels.shape[0])}"
            )
        return labels.to(device=device, dtype=torch.long)

    def _condition_parts(
        self,
        *,
        batch_size: int,
        device: torch.device,
        labels: torch.Tensor | None,
        timesteps: torch.Tensor,
    ) -> list[torch.Tensor]:
        parts = [
            self.time_proj(timesteps),
            self.latent_proj(torch.randn(int(batch_size), self.latent_dim, device=device)),
        ]
        if self.use_condition:
            assert self.label_embed is not None
            assert labels is not None
            parts.append(self.label_embed(labels))
        return parts

    def _apply_sampler(self, x_t: torch.Tensor, pred_noise: torch.Tensor) -> torch.Tensor:
        mode = self.prediction_mode
        if mode == "eps":
            return x_t - pred_noise
        if mode == "score":
            return x_t - 0.5 * torch.tanh(pred_noise)
        if mode == "x0":
            return torch.tanh(x_t - pred_noise)
        if mode == "consistency":
            return 0.5 * x_t + 0.5 * torch.tanh(x_t - pred_noise)
        if mode == "flow":
            return x_t + self.step_scale * torch.tanh(pred_noise)
        return x_t - pred_noise

    def sample(
        self,
        *,
        batch_size: int = 2,
        device: torch.device | str | None = None,
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.forward(batch_size=int(batch_size), device=device, labels=labels)
        return out["sample"]

    def forward(
        self,
        *,
        batch_size: int = 2,
        device: torch.device | str | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        dev = _resolve_device(device)
        batch = int(batch_size)
        x_t = torch.randn(batch, self.in_channels, self.image_size, self.image_size, device=dev)
        flat = x_t.view(batch, -1)
        timesteps = torch.rand(batch, 1, device=dev)
        cond = self._prepare_labels(batch_size=batch, device=dev, labels=labels)
        parts = self._condition_parts(
            batch_size=batch,
            device=dev,
            labels=cond,
            timesteps=timesteps,
        )

        latent_out: torch.Tensor | None = None
        if self.latent_space:
            assert self.image_to_latent is not None
            assert self.latent_to_image is not None
            latent = self.image_to_latent(flat)
            latent_out = self.denoiser(torch.cat([latent, *parts], dim=1))
            pred_noise = self.latent_to_image(latent_out).view(
                batch, self.in_channels, self.image_size, self.image_size
            )
        else:
            pred_noise = self.denoiser(torch.cat([flat, *parts], dim=1)).view(
                batch, self.in_channels, self.image_size, self.image_size
            )

        sample = self._apply_sampler(x_t, pred_noise)
        out: dict[str, torch.Tensor] = {
            "sample": sample,
            "pred_noise": pred_noise,
            "timesteps": timesteps.view(-1),
        }
        if latent_out is not None:
            out["latent"] = latent_out
        if cond is not None:
            out["labels"] = cond.to(torch.float32)
        return out


def build_toy_diffusion_family(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    image_size: int = 32,
    latent_dim: int = 64,
    num_classes: int = 0,
    variant: str,
    width_mult: float = 1.0,
    dropout: float = 0.0,
    use_condition: bool = False,
    latent_space: bool = False,
    prediction_mode: str = "eps",
    step_scale: float = 1.0,
) -> nn.Module:
    cfg = variants[str(variant)]
    width = max(16, int(int(cfg["width"]) * float(width_mult)))
    latent = max(int(latent_dim), int(cfg["latent"]))
    return ToyDiffusion(
        family=str(family),
        in_channels=int(in_channels),
        image_size=int(image_size),
        latent_dim=latent,
        width=width,
        depth=int(cfg["depth"]),
        dropout=float(dropout),
        num_classes=int(num_classes),
        use_condition=bool(use_condition),
        latent_space=bool(latent_space),
        prediction_mode=str(prediction_mode),
        step_scale=float(step_scale),
    )


def smoke_test_diffusion(builder: Callable[..., nn.Module], variant: str) -> None:
    model = builder(
        in_channels=3,
        image_size=32,
        latent_dim=64,
        num_classes=10,
        variant=variant,
        width_mult=0.5,
        dropout=0.0,
    )
    out = model.forward(batch_size=2)
    shapes = {k: tuple(v.shape) for k, v in out.items() if torch.is_tensor(v)}
    print(variant, shapes)
    assert "sample" in out and "pred_noise" in out
    print("ok")


__all__ = ["ToyDiffusion", "build_toy_diffusion_family", "smoke_test_diffusion"]
