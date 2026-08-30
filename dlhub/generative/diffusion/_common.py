from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


_TRANSFORMER_FAMILIES = {
    "aura_flow",
    "dit",
    "flux",
    "hunyuan_dit",
    "lumina_next",
    "omni_gen",
    "pixart",
    "pixart_alpha",
    "pixart_sigma",
    "sana",
    "sd3",
    "uvit",
    "vision_diffusion",
}


def _resolve_device(
    device: torch.device | str | None,
    *,
    fallback: torch.device,
) -> torch.device:
    if device is None:
        return fallback
    if isinstance(device, torch.device):
        return device
    return torch.device(device)


class _TimeEmbedding(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(5, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        time = timesteps.flatten()
        features = torch.stack(
            (
                time,
                torch.sin(torch.pi * time),
                torch.cos(torch.pi * time),
                torch.sin(2.0 * torch.pi * time),
                torch.cos(2.0 * torch.pi * time),
            ),
            dim=-1,
        )
        return self.projection(features)


class _FiLMResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float):
        super().__init__()
        self.norm = nn.GroupNorm(4, width)
        self.conv1 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.modulation = nn.Linear(width, width * 2)
        self.dropout = nn.Dropout2d(float(dropout))

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(conditioning).chunk(2, dim=-1)
        update = self.norm(x)
        update = update * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        update = self.conv1(F.silu(update))
        update = self.dropout(update)
        return x + self.conv2(F.silu(update))


class _LatentFiLMBlock(nn.Module):
    def __init__(self, latent_dim: int, width: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(latent_dim)
        self.update = nn.Sequential(
            nn.Linear(latent_dim, width),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(width, latent_dim),
        )
        self.modulation = nn.Linear(width, latent_dim * 2)

    def forward(self, latent: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(conditioning).chunk(2, dim=-1)
        normalized = self.norm(latent) * (1.0 + scale) + shift
        return latent + self.update(normalized)


def _transformer(width: int, depth: int, dropout: float) -> nn.TransformerEncoder:
    layer = nn.TransformerEncoderLayer(
        d_model=width,
        nhead=4,
        dim_feedforward=width * 2,
        dropout=float(dropout),
        activation="gelu",
        batch_first=True,
        norm_first=False,
    )
    return nn.TransformerEncoder(
        layer,
        num_layers=max(1, int(depth)),
        enable_nested_tensor=False,
    )


class CompactDiffusion(nn.Module):
    """Compact one-step denoiser with explicit inputs and iterative sampling."""

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
        if self.prediction_mode not in {"eps", "score", "x0", "consistency", "flow"}:
            raise ValueError(f"Unsupported diffusion prediction mode: {prediction_mode!r}")
        if self.step_scale <= 0.0:
            raise ValueError("step_scale must be positive")
        dropout_rate = float(dropout)
        if not 0.0 <= dropout_rate < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        hidden = max(16, int(width))
        hidden = ((hidden + 3) // 4) * 4
        self.width = hidden
        if self.latent_space:
            self.architecture = "latent-autoencoder-denoiser"
        elif self.family in _TRANSFORMER_FAMILIES:
            self.architecture = "patch-transformer-denoiser"
        else:
            self.architecture = "spatial-convolutional-denoiser"
        self.mechanism = f"{self.architecture}:{self.prediction_mode}"

        self.time_embedding = _TimeEmbedding(hidden)
        if self.use_condition:
            self.label_embedding = nn.Embedding(self.num_classes, hidden)
        else:
            self.label_embedding = None

        if self.architecture == "spatial-convolutional-denoiser":
            self.input_projection = nn.Conv2d(
                self.in_channels, hidden, kernel_size=3, padding=1
            )
            self.conv_blocks = nn.ModuleList(
                [_FiLMResidualBlock(hidden, dropout_rate) for _ in range(depth)]
            )
            self.output_projection = nn.Conv2d(
                hidden, self.in_channels, kernel_size=3, padding=1
            )
        else:
            self.input_projection = None
            self.conv_blocks = nn.ModuleList()
            self.output_projection = None

        if self.architecture == "patch-transformer-denoiser":
            self.patch_embedding = nn.Conv2d(
                self.in_channels, hidden, kernel_size=4, stride=4
            )
            self.position_projection = nn.Linear(2, hidden)
            self.token_encoder = _transformer(hidden, depth, dropout_rate)
            self.patch_decoder = nn.ConvTranspose2d(
                hidden, self.in_channels, kernel_size=4, stride=4
            )
        else:
            self.patch_embedding = None
            self.position_projection = None
            self.token_encoder = None
            self.patch_decoder = None

        if self.architecture == "latent-autoencoder-denoiser":
            self.latent_encoder = nn.Sequential(
                nn.Conv2d(self.in_channels, hidden // 2, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.Conv2d(hidden // 2, hidden, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(hidden, self.latent_dim),
            )
            self.latent_blocks = nn.ModuleList(
                [
                    _LatentFiLMBlock(self.latent_dim, hidden, dropout_rate)
                    for _ in range(depth)
                ]
            )
            self.latent_decoder_input = nn.Linear(self.latent_dim, hidden * 4 * 4)
            self.latent_decoder = nn.Sequential(
                nn.ConvTranspose2d(hidden, hidden // 2, kernel_size=4, stride=2, padding=1),
                nn.SiLU(),
                nn.ConvTranspose2d(
                    hidden // 2,
                    self.in_channels,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
            )
        else:
            self.latent_encoder = None
            self.latent_blocks = nn.ModuleList()
            self.latent_decoder_input = None
            self.latent_decoder = None

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
            return torch.randint(0, self.num_classes, (batch_size,), device=device)
        if labels.ndim != 1 or labels.shape[0] != batch_size:
            raise ValueError(
                f"labels must have shape ({batch_size},), got {tuple(labels.shape)}"
            )
        return labels.to(device=device, dtype=torch.long)

    def _prepare_timesteps(
        self,
        timesteps: float | torch.Tensor | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if timesteps is None:
            time = torch.rand(batch_size, device=device, dtype=dtype)
        else:
            time = torch.as_tensor(timesteps, device=device, dtype=dtype)
            if time.ndim == 0:
                time = time.expand(batch_size)
            elif time.ndim == 2 and time.shape[-1] == 1:
                time = time.flatten()
        if time.shape != (batch_size,):
            raise ValueError(
                f"timesteps must be scalar or shape ({batch_size},), got {tuple(time.shape)}"
            )
        return time.clamp(0.0, 1.0)

    def _conditioning(
        self,
        timesteps: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> torch.Tensor:
        conditioning = self.time_embedding(timesteps)
        if labels is not None:
            assert self.label_embedding is not None
            conditioning = conditioning + self.label_embedding(labels)
        return conditioning

    @staticmethod
    def _positions(features: torch.Tensor) -> torch.Tensor:
        height, width = features.shape[-2:]
        rows = torch.linspace(-1.0, 1.0, height, device=features.device, dtype=features.dtype)
        columns = torch.linspace(
            -1.0, 1.0, width, device=features.device, dtype=features.dtype
        )
        row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
        return torch.stack((column_grid, row_grid), dim=-1).reshape(1, -1, 2)

    def _predict(
        self,
        x_t: torch.Tensor,
        conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.architecture == "spatial-convolutional-denoiser":
            assert self.input_projection is not None
            assert self.output_projection is not None
            features = self.input_projection(x_t)
            for block in self.conv_blocks:
                features = block(features, conditioning)
            return self.output_projection(F.silu(features)), None

        if self.architecture == "patch-transformer-denoiser":
            assert self.patch_embedding is not None
            assert self.position_projection is not None
            assert self.token_encoder is not None
            assert self.patch_decoder is not None
            patches = self.patch_embedding(x_t)
            batch, channels, height, width = patches.shape
            tokens = patches.flatten(2).transpose(1, 2)
            tokens = tokens + self.position_projection(self._positions(patches))
            tokens = self.token_encoder(tokens + conditioning[:, None])
            features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
            prediction = self.patch_decoder(features)
            if prediction.shape[-2:] != x_t.shape[-2:]:
                prediction = F.interpolate(
                    prediction,
                    size=x_t.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            return prediction, None

        assert self.latent_encoder is not None
        assert self.latent_decoder_input is not None
        assert self.latent_decoder is not None
        latent = self.latent_encoder(x_t)
        for block in self.latent_blocks:
            latent = block(latent, conditioning)
        decoded = self.latent_decoder_input(latent).reshape(
            x_t.shape[0], self.width, 4, 4
        )
        prediction = self.latent_decoder(decoded)
        prediction = F.interpolate(
            prediction,
            size=x_t.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return prediction, latent

    def _apply_sampler(
        self,
        x_t: torch.Tensor,
        prediction: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time = timesteps[:, None, None, None]
        if self.prediction_mode == "eps":
            step = self.step_scale * (0.05 + 0.15 * time)
            return x_t - step * prediction
        if self.prediction_mode == "score":
            step = self.step_scale * (0.02 + 0.08 * time)
            return x_t + step * prediction
        if self.prediction_mode == "x0":
            return torch.tanh(prediction)
        if self.prediction_mode == "consistency":
            return time * x_t + (1.0 - time) * torch.tanh(prediction)
        return x_t + self.step_scale * (1.05 - time) * prediction

    def forward(
        self,
        *,
        batch_size: int = 2,
        device: torch.device | str | None = None,
        labels: torch.Tensor | None = None,
        x_t: torch.Tensor | None = None,
        timesteps: float | torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        parameter_device = next(self.parameters()).device
        target_device = _resolve_device(device, fallback=parameter_device)
        if target_device != parameter_device:
            raise ValueError(
                f"model parameters are on {parameter_device}; move the model to {target_device} first"
            )
        batch = int(batch_size)
        if x_t is not None:
            if x_t.ndim != 4 or x_t.shape[1] != self.in_channels:
                raise ValueError(
                    "x_t must have shape "
                    f"(B, {self.in_channels}, H, W), got {tuple(x_t.shape)}"
                )
            batch = int(x_t.shape[0])
            x_t = x_t.to(device=target_device, dtype=torch.float32)
        else:
            x_t = torch.randn(
                batch,
                self.in_channels,
                self.image_size,
                self.image_size,
                device=target_device,
            )
        time = self._prepare_timesteps(
            timesteps,
            batch_size=batch,
            device=target_device,
            dtype=x_t.dtype,
        )
        condition_labels = self._prepare_labels(
            batch_size=batch,
            device=target_device,
            labels=labels,
        )
        conditioning = self._conditioning(time, condition_labels)
        prediction, latent = self._predict(x_t, conditioning)
        sample = self._apply_sampler(x_t, prediction, time)

        output: dict[str, torch.Tensor] = {
            "sample": sample,
            "pred_noise": prediction,
            "timesteps": time,
        }
        if latent is not None:
            output["latent"] = latent
        if condition_labels is not None:
            output["labels"] = condition_labels
        return output

    def sample(
        self,
        *,
        batch_size: int = 2,
        device: torch.device | str | None = None,
        labels: torch.Tensor | None = None,
        initial_noise: torch.Tensor | None = None,
        num_steps: int = 4,
    ) -> torch.Tensor:
        if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
            raise ValueError("num_steps must be a positive integer")
        parameter_device = next(self.parameters()).device
        target_device = _resolve_device(device, fallback=parameter_device)
        if initial_noise is None:
            current = torch.randn(
                int(batch_size),
                self.in_channels,
                self.image_size,
                self.image_size,
                device=target_device,
            )
        else:
            current = initial_noise.to(device=target_device, dtype=torch.float32)
            batch_size = int(current.shape[0])
        condition_labels = self._prepare_labels(
            batch_size=int(batch_size),
            device=target_device,
            labels=labels,
        )
        schedule = torch.linspace(1.0, 0.0, num_steps, device=target_device)
        for timestep in schedule:
            current = self.forward(
                batch_size=int(batch_size),
                device=target_device,
                labels=condition_labels,
                x_t=current,
                timesteps=timestep,
            )["sample"]
        return current


def build_compact_diffusion_family(
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
    if variant not in variants:
        raise KeyError(f"Unknown {family} diffusion variant: {variant!r}")
    config = variants[variant]
    width = max(16, int(int(config["width"]) * float(width_mult)))
    latent = max(int(latent_dim), int(config["latent"]))
    return CompactDiffusion(
        family=str(family),
        in_channels=int(in_channels),
        image_size=int(image_size),
        latent_dim=latent,
        width=width,
        depth=int(config["depth"]),
        dropout=float(dropout),
        num_classes=int(num_classes),
        use_condition=bool(use_condition),
        latent_space=bool(latent_space),
        prediction_mode=str(prediction_mode),
        step_scale=float(step_scale),
    )


def build_baseline_diffusion_family(**kwargs: object) -> nn.Module:
    """Compatibility entrypoint for labels still mapped to shared diffusion modes."""

    return build_compact_diffusion_family(**kwargs)


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
    output = model.forward(
        x_t=torch.randn(2, 3, 32, 32),
        timesteps=torch.tensor([0.25, 0.75]),
    )
    shapes = {key: tuple(value.shape) for key, value in output.items()}
    print(variant, model.mechanism, shapes)


__all__ = [
    "CompactDiffusion",
    "build_baseline_diffusion_family",
    "build_compact_diffusion_family",
    "smoke_test_diffusion",
]
