from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn
from torch.nn import functional as F


def _scaled_width(width: int, width_mult: float) -> int:
    scaled = max(8, int(int(width) * float(width_mult)))
    return ((scaled + 3) // 4) * 4


def _coordinates(x: torch.Tensor) -> torch.Tensor:
    height, width = x.shape[-2:]
    rows = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
    columns = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
    row_grid, column_grid = torch.meshgrid(rows, columns, indexing="ij")
    return torch.stack((column_grid, row_grid), dim=0).expand(x.shape[0], -1, -1, -1)


def _finish(logits: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
    if logits.shape[-2:] != source.shape[-2:]:
        logits = F.interpolate(
            logits,
            size=source.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
    return torch.sigmoid(logits + 0.1 * source)


class _ResidualConvBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(width, width, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(width, width, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CompactLayoutGANGenerator(nn.Module):
    mechanism = "latent-residual-generator"

    def __init__(self, *, width: int, depth: int, in_channels: int):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.latent_projection = nn.Linear(width, width)
        self.generator = nn.Sequential(
            *[_ResidualConvBlock(width) for _ in range(depth)]
        )
        self.head = nn.Conv2d(width, in_channels, kernel_size=1)

    def forward(
        self, x: torch.Tensor, latent: torch.Tensor | None = None
    ) -> torch.Tensor:
        features = self.stem(x)
        if latent is not None:
            if latent.shape != (x.shape[0], features.shape[1]):
                raise ValueError(
                    "latent must have shape "
                    f"({x.shape[0]}, {features.shape[1]}), got {tuple(latent.shape)}"
                )
            features = features + self.latent_projection(latent)[:, :, None, None]
        return _finish(self.head(self.generator(features)), x)


class CompactLayoutVAE(nn.Module):
    mechanism = "variational-spatial-bottleneck"

    def __init__(self, *, width: int, depth: int, in_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            *[_ResidualConvBlock(width) for _ in range(depth)],
        )
        self.to_mean = nn.Conv2d(width, width, kernel_size=1)
        self.to_log_variance = nn.Conv2d(width, width, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(width, width, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width, in_channels, kernel_size=1),
        )
        self.register_buffer("last_kl", torch.tensor(0.0), persistent=False)

    def forward(self, x: torch.Tensor, *, sample: bool = False) -> torch.Tensor:
        encoded = self.encoder(x)
        mean = self.to_mean(encoded)
        log_variance = self.to_log_variance(encoded).clamp(-8.0, 8.0)
        kl = 0.5 * (mean.square() + log_variance.exp() - 1.0 - log_variance).mean()
        self.last_kl.copy_(kl.detach())
        if sample:
            latent = mean + torch.exp(0.5 * log_variance) * torch.randn_like(mean)
        else:
            latent = mean
        return _finish(self.decoder(latent), x)


class CompactLayoutTransformer(nn.Module):
    mechanism = "spatial-self-attention"

    def __init__(self, *, width: int, depth: int, in_channels: int):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels, width, kernel_size=4, stride=4
        )
        self.position_projection = nn.Linear(2, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=4,
            dim_feedforward=width * 2,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=depth,
            enable_nested_tensor=False,
        )
        self.head = nn.Conv2d(width, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(x)
        batch, channels, height, width = patches.shape
        positions = _coordinates(patches)[:1].flatten(2).transpose(1, 2)
        tokens = patches.flatten(2).transpose(1, 2)
        tokens = tokens + self.position_projection(positions)
        tokens = self.encoder(tokens)
        features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return _finish(self.head(features), x)


class CompactBBoxLayoutGenerator(nn.Module):
    mechanism = "coordinate-objectness-gating"

    def __init__(self, *, width: int, depth: int, in_channels: int):
        super().__init__()
        self.coord_stem = nn.Conv2d(in_channels + 2, width, kernel_size=3, padding=1)
        self.box_features = nn.Sequential(
            *[_ResidualConvBlock(width) for _ in range(depth)]
        )
        self.box_head = nn.Conv2d(width, in_channels, kernel_size=1)
        self.objectness_head = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.coord_stem(torch.cat((x, _coordinates(x)), dim=1))
        features = self.box_features(features)
        logits = self.box_head(features) * torch.sigmoid(self.objectness_head(features))
        return _finish(logits, x)


class CompactPosterLayoutNet(nn.Module):
    mechanism = "multiscale-pyramid-fusion"

    def __init__(self, *, width: int, depth: int, in_channels: int):
        super().__init__()
        self.fine = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            *[_ResidualConvBlock(width) for _ in range(depth)],
        )
        self.medium = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1)
        self.coarse = nn.Conv2d(width, width, kernel_size=3, stride=2, padding=1)
        self.fusion = nn.Sequential(
            nn.Conv2d(width * 3, width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(width, in_channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        fine = self.fine(x)
        medium = F.gelu(self.medium(fine))
        coarse = F.gelu(self.coarse(medium))
        medium = F.interpolate(medium, size=fine.shape[-2:], mode="nearest")
        coarse = F.interpolate(coarse, size=fine.shape[-2:], mode="nearest")
        return _finish(self.fusion(torch.cat((fine, medium, coarse), dim=1)), x)


class _AxialBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.row_mixer = nn.Conv2d(
            width,
            width,
            kernel_size=(1, 7),
            padding=(0, 3),
            groups=width,
        )
        self.column_mixer = nn.Conv2d(
            width,
            width,
            kernel_size=(7, 1),
            padding=(3, 0),
            groups=width,
        )
        self.projection = nn.Conv2d(width, width, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.projection(F.gelu(self.row_mixer(x) + self.column_mixer(x)))


class CompactDocumentLayoutGenerator(nn.Module):
    mechanism = "axial-row-column-mixing"

    def __init__(self, *, width: int, depth: int, in_channels: int):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.axial_blocks = nn.Sequential(*[_AxialBlock(width) for _ in range(depth)])
        self.head = nn.Conv2d(width, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _finish(self.head(self.axial_blocks(self.stem(x))), x)


class CompactConstraintLayoutGenerator(nn.Module):
    mechanism = "constraint-projection"

    def __init__(self, *, width: int, depth: int, in_channels: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
            *[_ResidualConvBlock(width) for _ in range(depth)],
        )
        self.proposal = nn.Conv2d(width, in_channels, kernel_size=1)
        self.feasibility = nn.Conv2d(width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        feasibility = torch.sigmoid(self.feasibility(features))
        logits = self.proposal(features) * feasibility
        return _finish(logits, x)


class _RelationBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.attention_norm = nn.LayerNorm(width)
        self.attention = nn.MultiheadAttention(
            width,
            num_heads=4,
            dropout=0.0,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * 2),
            nn.GELU(),
            nn.Linear(width * 2, width),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        normalized = self.attention_norm(tokens)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )
        tokens = tokens + attended
        return tokens + self.mlp(self.mlp_norm(tokens))


class CompactRelationLayoutGenerator(nn.Module):
    mechanism = "object-relation-attention"

    def __init__(self, *, width: int, depth: int, in_channels: int):
        super().__init__()
        self.object_embedding = nn.Conv2d(
            in_channels, width, kernel_size=4, stride=4
        )
        self.position_projection = nn.Linear(2, width)
        self.relation_blocks = nn.ModuleList(
            [_RelationBlock(width) for _ in range(depth)]
        )
        self.head = nn.Conv2d(width, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        objects = self.object_embedding(x)
        batch, channels, height, width = objects.shape
        positions = _coordinates(objects)[:1].flatten(2).transpose(1, 2)
        tokens = objects.flatten(2).transpose(1, 2)
        tokens = tokens + self.position_projection(positions)
        for block in self.relation_blocks:
            tokens = block(tokens)
        features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return _finish(self.head(features), x)


class _TimeConditionedBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.conv = nn.Conv2d(width, width, kernel_size=3, padding=1)
        self.modulation = nn.Linear(width, width * 2)

    def forward(self, x: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        scale, shift = self.modulation(time_embedding).chunk(2, dim=-1)
        update = self.conv(x)
        update = update * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        return x + F.gelu(update)


class CompactDiffusionLayoutGenerator(nn.Module):
    mechanism = "time-conditioned-denoising"

    def __init__(self, *, width: int, depth: int, in_channels: int):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)
        self.time_embedding = nn.Sequential(
            nn.Linear(3, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.blocks = nn.ModuleList(
            [_TimeConditionedBlock(width) for _ in range(depth)]
        )
        self.head = nn.Conv2d(width, in_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        timestep: float | torch.Tensor = 0.5,
    ) -> torch.Tensor:
        time = torch.as_tensor(timestep, device=x.device, dtype=x.dtype)
        if time.ndim == 0:
            time = time.expand(x.shape[0])
        if time.shape != (x.shape[0],):
            raise ValueError(
                f"timestep must be scalar or shape ({x.shape[0]},), got {tuple(time.shape)}"
            )
        time_features = torch.stack(
            (time, torch.sin(torch.pi * time), torch.cos(torch.pi * time)), dim=-1
        )
        embedding = self.time_embedding(time_features)
        features = self.stem(x)
        for block in self.blocks:
            features = block(features, embedding)
        return _finish(self.head(features), x)


class _SelectiveScanBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.input_projection = nn.Linear(width, width * 3)
        self.output_norm = nn.LayerNorm(width)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        gate_logits, candidates, skip = self.input_projection(tokens).chunk(3, dim=-1)
        gates = torch.sigmoid(gate_logits)
        candidates = torch.tanh(candidates)
        state = torch.zeros_like(tokens[:, 0])
        outputs = []
        for gate, candidate, residual in zip(
            gates.unbind(dim=1),
            candidates.unbind(dim=1),
            skip.unbind(dim=1),
            strict=True,
        ):
            state = gate * state + (1.0 - gate) * candidate
            outputs.append(state + residual)
        return self.output_norm(tokens + torch.stack(outputs, dim=1))


class CompactMambaLayoutGenerator(nn.Module):
    mechanism = "input-dependent-selective-scan"

    def __init__(self, *, width: int, depth: int, in_channels: int):
        super().__init__()
        self.patch_embedding = nn.Conv2d(
            in_channels, width, kernel_size=4, stride=4
        )
        self.scan_blocks = nn.ModuleList(
            [_SelectiveScanBlock(width) for _ in range(depth)]
        )
        self.head = nn.Conv2d(width, in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(x)
        batch, channels, height, width = patches.shape
        tokens = patches.flatten(2).transpose(1, 2)
        for block in self.scan_blocks:
            tokens = block(tokens)
        features = tokens.transpose(1, 2).reshape(batch, channels, height, width)
        return _finish(self.head(features), x)


LayoutBuilder = Callable[..., nn.Module]

_FAMILY_BUILDERS: dict[str, LayoutBuilder] = {
    "layoutgan_baseline": CompactLayoutGANGenerator,
    "layoutvae_baseline": CompactLayoutVAE,
    "layouttransformer": CompactLayoutTransformer,
    "bbox_generator": CompactBBoxLayoutGenerator,
    "poster_layout_net": CompactPosterLayoutNet,
    "doc_layout_gen": CompactDocumentLayoutGenerator,
    "constraint_layout": CompactConstraintLayoutGenerator,
    "relation_layout": CompactRelationLayoutGenerator,
    "diffusion_layout": CompactDiffusionLayoutGenerator,
    "mamba_layout_gen": CompactMambaLayoutGenerator,
}


def build_compact_layout_generator(
    *,
    family: str,
    variants: dict[str, dict[str, int]],
    in_channels: int,
    variant: str,
    width_mult: float = 1.0,
) -> nn.Module:
    if variant not in variants:
        raise KeyError(f"Unknown {family} layout variant: {variant!r}")
    try:
        builder = _FAMILY_BUILDERS[family]
    except KeyError as error:
        raise KeyError(f"Unknown layout mechanism family: {family!r}") from error
    spec = variants[variant]
    return builder(
        width=_scaled_width(spec["width"], width_mult),
        depth=int(spec["depth"]),
        in_channels=int(in_channels),
    )


def validate_layout_generator(builder: Callable[..., nn.Module], variant: str) -> None:
    model = builder(in_channels=3, variant=variant)
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(variant, model.mechanism, tuple(y.shape))


__all__ = ["build_compact_layout_generator", "validate_layout_generator"]
