"""Compact DETR baseline shared by legacy paper-label aliases."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypeVar

import torch
from torch import nn

from dlhub.vision.backbones._blocks import ConvBNAct, scale_channels
from dlhub.vision.detection._detr_utils import (
    MLP,
    SimpleTransformer,
    flatten_hw,
    sine_positional_encoding_1d,
)


class _CompactConvBackbone(nn.Module):
    """Produce a stride-8 spatial feature map for the DETR baseline."""

    def __init__(
        self, *, in_channels: int, stem_channels: int, feat_channels: int, depth: int
    ) -> None:
        super().__init__()
        c_in = int(in_channels)
        stem = int(stem_channels)
        feat = int(feat_channels)
        d = int(depth)
        if d <= 0:
            raise ValueError("depth must be > 0")
        layers: list[nn.Module] = [
            ConvBNAct(c_in, stem, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(stem, stem, kernel_size=3, stride=2, act="relu"),
            ConvBNAct(stem, feat, kernel_size=3, stride=2, act="relu"),
        ]
        for _ in range(d):
            layers.append(ConvBNAct(feat, feat, kernel_size=3, stride=1, act="relu"))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompactDetrBaseline(nn.Module):
    """Query-based detector used when a paper-specific mechanism is not implemented."""

    REGISTERED_ALIAS = "compact_detr"

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        stem_channels: int = 32,
        feat_channels: int = 128,
        backbone_depth: int = 2,
        d_model: int = 128,
        num_heads: int = 4,
        num_queries: int = 50,
        enc_layers: int = 2,
        dec_layers: int = 2,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        nc = int(num_classes)
        if nc <= 0:
            raise ValueError("num_classes must be > 0")
        dm = int(d_model)
        if dm <= 0:
            raise ValueError("d_model must be > 0")
        q = int(num_queries)
        if q <= 0:
            raise ValueError("num_queries must be > 0")
        if dm % 2 != 0:
            raise ValueError("d_model must be even for sinusoidal encoding")

        self.registered_alias = type(self).REGISTERED_ALIAS
        self.backbone = _CompactConvBackbone(
            in_channels=int(in_channels),
            stem_channels=int(stem_channels),
            feat_channels=int(feat_channels),
            depth=int(backbone_depth),
        )
        self.proj = nn.Conv2d(int(feat_channels), dm, kernel_size=1)
        self.transformer = SimpleTransformer(
            dim=dm,
            num_heads=int(num_heads),
            num_encoder_layers=int(enc_layers),
            num_decoder_layers=int(dec_layers),
            mlp_ratio=float(mlp_ratio),
            dropout=float(dropout),
        )
        self.query_embed = nn.Parameter(torch.randn(q, dm) * 0.02)
        self.class_head = nn.Linear(dm, nc)
        self.box_head = MLP(dm, dm, 4, num_layers=3, act="relu")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = x.to(torch.float32)
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (B, C, H, W), got {tuple(x.shape)}")
        batch_size = x.shape[0]
        feat = self.proj(self.backbone(x))
        memory = flatten_hw(feat)
        positions = sine_positional_encoding_1d(
            memory.shape[1], memory.shape[2], device=memory.device
        )
        memory = memory + positions.unsqueeze(0)
        queries = self.query_embed.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        decoded = self.transformer(memory, queries)
        return {
            "class_logits": self.class_head(decoded),
            "boxes": torch.sigmoid(self.box_head(decoded)),
        }


_BASE_VARIANTS: dict[str, dict[str, int]] = {
    "tiny": {
        "stem": 24,
        "feat": 96,
        "depth": 1,
        "d_model": 96,
        "heads": 4,
        "q": 32,
        "enc": 1,
        "dec": 1,
    },
    "small": {
        "stem": 32,
        "feat": 128,
        "depth": 2,
        "d_model": 128,
        "heads": 4,
        "q": 50,
        "enc": 2,
        "dec": 2,
    },
    "base": {
        "stem": 48,
        "feat": 192,
        "depth": 2,
        "d_model": 192,
        "heads": 6,
        "q": 80,
        "enc": 3,
        "dec": 3,
    },
}


def make_detr_baseline_variants(registered_alias: str) -> dict[str, dict[str, int]]:
    """Return independent variant metadata namespaced by a compatibility alias."""

    alias = str(registered_alias).lower().strip()
    if not alias:
        raise ValueError("registered_alias must not be empty")
    return {f"{alias}_{size}": dict(spec) for size, spec in _BASE_VARIANTS.items()}


DetectorT = TypeVar("DetectorT", bound=CompactDetrBaseline)


def build_compact_detr_baseline(
    *,
    detector_type: type[DetectorT],
    variants: Mapping[str, Mapping[str, int]],
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
) -> DetectorT:
    """Build a compact DETR baseline for one registered compatibility alias."""

    name = str(variant).lower().strip()
    if name not in variants:
        raise ValueError(f"Unknown DETR variant: {variant!r}. Supported: {sorted(variants)}")
    spec = variants[name]
    stem = scale_channels(int(spec["stem"]), float(width_mult), min_ch=16, divisor=8)
    feat = scale_channels(int(spec["feat"]), float(width_mult), min_ch=16, divisor=8)
    d_model = scale_channels(int(spec["d_model"]), float(width_mult), min_ch=32, divisor=8)
    if d_model % 2 != 0:
        d_model += 8

    return detector_type(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        stem_channels=stem,
        feat_channels=feat,
        backbone_depth=int(spec["depth"]),
        d_model=d_model,
        num_heads=int(spec["heads"]),
        num_queries=int(spec["q"]),
        enc_layers=int(spec["enc"]),
        dec_layers=int(spec["dec"]),
    )


def smoke_test_detr_builder(builder: Callable[..., CompactDetrBaseline], variant: str) -> None:
    """Run the local shape/gradient check used by module entrypoints."""

    torch.manual_seed(0)
    model = builder(in_channels=3, num_classes=2, variant=variant, width_mult=0.5)
    output = model(torch.randn(2, 3, 128, 128))
    print(variant, {key: tuple(value.shape) for key, value in output.items()})
    loss = output["class_logits"].mean() + output["boxes"].mean()
    loss.backward()
    print("ok")


__all__ = [
    "CompactDetrBaseline",
    "build_compact_detr_baseline",
    "make_detr_baseline_variants",
    "smoke_test_detr_builder",
]
