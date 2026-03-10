from __future__ import annotations

from collections.abc import Callable, Mapping

import torch
from torch import nn

Builder = Callable[..., nn.Module]


def build_aliased_detector(
    *,
    family: str,
    variants: Mapping[str, str],
    base_builder: Builder,
    in_channels: int,
    num_classes: int,
    variant: str,
    width_mult: float = 1.0,
) -> nn.Module:
    name = str(variant).lower().strip()
    base_variant = variants.get(name)
    if base_variant is None:
        raise ValueError(f"Unknown {family} variant: {variant!r}. Supported: {sorted(variants)}")
    return base_builder(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        variant=str(base_variant),
        width_mult=float(width_mult),
    )


def sum_output_means(x) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(torch.float32).mean()
    if isinstance(x, dict):
        return sum((sum_output_means(v) for v in x.values()), start=torch.tensor(0.0))
    if isinstance(x, list | tuple):
        return sum((sum_output_means(v) for v in x), start=torch.tensor(0.0))
    raise TypeError(f"Unsupported output type in detection alias smoke: {type(x)!r}")


def smoke_aliased_detector(
    *,
    label: str,
    builder: Builder,
    variant: str,
    image_size: int = 128,
) -> None:
    torch.manual_seed(0)
    x = torch.randn(2, 3, int(image_size), int(image_size))
    model = builder(in_channels=3, num_classes=3, variant=str(variant), width_mult=0.5)
    out = model(x)
    print(label, type(out).__name__)
    loss = sum_output_means(out)
    loss.backward()
    print("ok")
