from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from .backbones import (
    build_convmixer_classifier,
    build_convnext_classifier,
    build_densenet_classifier,
    build_efficientnet_classifier,
    build_mlp_mixer_classifier,
    build_mobilenet_v1_classifier,
    build_mobilenet_v2_classifier,
    build_mobilenet_v3_classifier,
    build_repvgg_classifier,
    build_resnet_classifier,
    build_shufflenet_v2_classifier,
    build_squeezenet_classifier,
    build_vgg_classifier,
    build_vit_classifier,
)


@dataclass(frozen=True)
class BuildConfig:
    in_channels: int
    num_classes: int
    image_size: int = 64
    width_mult: float = 1.0
    dropout: float = 0.1


class UnknownLocalArch(ValueError):
    pass


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    arch_id = str(arch_id).strip()
    if ":" not in arch_id:
        return "dl", arch_id
    prefix, name = arch_id.split(":", 1)
    prefix = prefix.strip().lower()
    name = name.strip()
    if not prefix or not name:
        raise ValueError(f"Invalid arch id: {arch_id!r}")
    return prefix, name


Builder = Callable[[BuildConfig], nn.Module]


def _registry() -> dict[str, Builder]:
    r: dict[str, Builder] = {}

    # --- Classic CNN families
    for vgg in ["vgg11", "vgg13", "vgg16", "vgg19"]:
        r[vgg] = lambda cfg, vgg=vgg: build_vgg_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            variant=vgg,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
        )

    # ResNet (basic)
    r["resnet18"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(2, 2, 2, 2),
        variant="basic",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    )
    r["resnet34"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(3, 4, 6, 3),
        variant="basic",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    )
    for name, layers in {
        "resnet50": (3, 4, 6, 3),
        "resnet101": (3, 4, 23, 3),
        "resnet152": (3, 8, 36, 3),
    }.items():
        r[name] = lambda cfg, layers=layers: build_resnet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            layers=layers,
            variant="bottleneck",
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            groups=1,
            width_per_group=64,
        )

    # SE-ResNet
    for name, layers, variant in [
        ("se_resnet50", (3, 4, 6, 3), "se_bottleneck"),
        ("se_resnet101", (3, 4, 23, 3), "se_bottleneck"),
    ]:
        r[name] = lambda cfg, layers=layers, variant=variant: build_resnet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            layers=layers,
            variant=variant,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            groups=1,
            width_per_group=64,
        )

    # PreAct-ResNet
    for name, layers, variant in [
        ("preact_resnet18", (2, 2, 2, 2), "preact_basic"),
        ("preact_resnet34", (3, 4, 6, 3), "preact_basic"),
        ("preact_resnet50", (3, 4, 6, 3), "preact_bottleneck"),
        ("preact_resnet101", (3, 4, 23, 3), "preact_bottleneck"),
        ("preact_resnet152", (3, 8, 36, 3), "preact_bottleneck"),
    ]:
        r[name] = lambda cfg, layers=layers, variant=variant: build_resnet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            layers=layers,
            variant=variant,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
        )

    # ResNeXt + Wide-ResNet
    r["resnext50_32x4d"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(3, 4, 6, 3),
        variant="bottleneck",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        groups=32,
        width_per_group=4,
    )
    r["resnext101_32x8d"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(3, 4, 23, 3),
        variant="bottleneck",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        groups=32,
        width_per_group=8,
    )
    r["wide_resnet50_2"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(3, 4, 6, 3),
        variant="bottleneck",
        width_mult=cfg.width_mult * 2.0,
        dropout=cfg.dropout,
        groups=1,
        width_per_group=64,
    )
    r["wide_resnet101_2"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(3, 4, 23, 3),
        variant="bottleneck",
        width_mult=cfg.width_mult * 2.0,
        dropout=cfg.dropout,
        groups=1,
        width_per_group=64,
    )

    # DenseNet
    for dn in ["densenet121", "densenet169", "densenet201", "densenet264"]:
        r[dn] = lambda cfg, dn=dn: build_densenet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            variant=dn,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
        )

    # SqueezeNet
    r["squeezenet1_0"] = lambda cfg: build_squeezenet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        variant="1_0",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    )
    r["squeezenet1_1"] = lambda cfg: build_squeezenet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        variant="1_1",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    )

    # ShuffleNetV2 (use width_mult to pick stage widths; expose friendly ids)
    for name, wm in [
        ("shufflenetv2_0_5", 0.5),
        ("shufflenetv2_1_0", 1.0),
        ("shufflenetv2_1_5", 1.5),
        ("shufflenetv2_2_0", 2.0),
    ]:
        r[name] = lambda cfg, wm=wm: build_shufflenet_v2_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            width_mult=float(cfg.width_mult) * float(wm),
            dropout=cfg.dropout,
        )

    # MobileNet
    r["mobilenet_v1"] = lambda cfg: build_mobilenet_v1_classifier(
        in_channels=cfg.in_channels, num_classes=cfg.num_classes, width_mult=cfg.width_mult, dropout=cfg.dropout
    )
    r["mobilenet_v2"] = lambda cfg: build_mobilenet_v2_classifier(
        in_channels=cfg.in_channels, num_classes=cfg.num_classes, width_mult=cfg.width_mult, dropout=cfg.dropout
    )
    r["mobilenet_v3_small"] = lambda cfg: build_mobilenet_v3_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        variant="small",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    )
    r["mobilenet_v3_large"] = lambda cfg: build_mobilenet_v3_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        variant="large",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    )

    # EfficientNet
    for b in ["b0", "b1", "b2", "b3", "b4"]:
        r[f"efficientnet_{b}"] = lambda cfg, b=b: build_efficientnet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            variant=b,
            width_mult=cfg.width_mult,
            dropout=None,
        )

    # RepVGG
    for v in ["a0", "a1", "a2", "b0", "b1", "b2", "b3"]:
        r[f"repvgg_{v}"] = lambda cfg, v=v: build_repvgg_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            variant=v,
            width_mult=cfg.width_mult,
            dropout=0.0,
            deploy=False,
        )
    r["repvgg"] = r["repvgg_a0"]

    # ConvNeXt
    for v in ["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"]:
        r[v] = lambda cfg, v=v: build_convnext_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            variant=v,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
        )

    # --- Transformer-ish families (many variants to hit >= 100 ids)
    vit_specs: dict[str, tuple[int, int, int]] = {
        # name: (embed_dim, num_heads, num_layers)
        "vit_tiny": (192, 3, 6),
        "vit_small": (384, 6, 8),
        "vit_base": (512, 8, 10),
    }
    for base_name, (embed_dim, num_heads, num_layers) in vit_specs.items():
        for patch_size in [4, 8, 16]:
            name = base_name if patch_size == 8 else f"{base_name}_p{patch_size}"
            r[name] = lambda cfg, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, patch_size=patch_size: build_vit_classifier(
                in_channels=cfg.in_channels,
                num_classes=cfg.num_classes,
                image_size=cfg.image_size,
                patch_size=int(patch_size),
                embed_dim=int(embed_dim),
                num_heads=int(num_heads),
                num_layers=int(num_layers),
                ff_dim=int(embed_dim) * 4,
                dropout=cfg.dropout,
            )

    # MLP-Mixer grid
    mixer_bases: dict[str, tuple[int, int]] = {
        "mixer_tiny": (128, 4),
        "mixer_small": (256, 6),
        "mixer_base": (384, 8),
    }
    for base_name, (embed_dim, num_layers) in mixer_bases.items():
        for patch_size in [4, 8, 16]:
            for tdim_mul in [2, 4]:
                name = f"{base_name}_p{patch_size}_t{tdim_mul}"
                r[name] = lambda cfg, embed_dim=embed_dim, num_layers=num_layers, patch_size=patch_size, tdim_mul=tdim_mul: build_mlp_mixer_classifier(
                    in_channels=cfg.in_channels,
                    num_classes=cfg.num_classes,
                    image_size=cfg.image_size,
                    patch_size=int(patch_size),
                    embed_dim=int(embed_dim),
                    num_layers=int(num_layers),
                    token_mlp_dim=int((cfg.image_size // int(patch_size)) ** 2) * int(tdim_mul),
                    channel_mlp_dim=int(embed_dim) * 4,
                    dropout=cfg.dropout,
                )

    # ConvMixer grid
    for embed_dim in [128, 192, 256]:
        for depth in [4, 8, 12]:
            for patch_size in [4, 8, 16]:
                name = f"convmixer_d{depth}_c{embed_dim}_p{patch_size}"
                r[name] = lambda cfg, embed_dim=embed_dim, depth=depth, patch_size=patch_size: build_convmixer_classifier(
                    in_channels=cfg.in_channels,
                    num_classes=cfg.num_classes,
                    image_size=cfg.image_size,
                    patch_size=int(patch_size),
                    embed_dim=int(embed_dim),
                    depth=int(depth),
                    kernel_size=9,
                    dropout=cfg.dropout,
                )

    # Aliases
    r["vgg"] = r["vgg11"]
    r["densenet"] = r["densenet121"]
    r["squeezenet"] = r["squeezenet1_0"]
    r["shufflenetv2"] = r["shufflenetv2_1_0"]
    r["mobilenetv1"] = r["mobilenet_v1"]
    r["mobilenetv2"] = r["mobilenet_v2"]
    r["efficientnetb0"] = r["efficientnet_b0"]
    r["resnext50"] = r["resnext50_32x4d"]
    r["revgg"] = r["repvgg"]

    return r


_REGISTRY = _registry()


def list_local_arches() -> list[str]:
    """List locally implemented architecture ids (namespaced with `dl:`)."""

    return [f"dl:{name}" for name in sorted(_REGISTRY)]


def build_local_model(
    arch_id: str,
    *,
    in_channels: int,
    num_classes: int,
    image_size: int = 64,
    width_mult: float = 1.0,
    dropout: float = 0.1,
) -> nn.Module:
    prefix, name = _split_arch_id(arch_id)
    if prefix not in {"dl", "local"}:
        raise ValueError(f"Unsupported local prefix: {prefix!r} (arch_id={arch_id!r})")

    builder = _REGISTRY.get(name)
    if builder is None:
        raise UnknownLocalArch(
            f"Unknown local arch: {arch_id!r}. Tip: see `list_local_arches()` or `python scripts/vision_zoo.py --list --search dl:`."
        )
    return builder(
        BuildConfig(
            in_channels=int(in_channels),
            num_classes=int(num_classes),
            image_size=int(image_size),
            width_mult=float(width_mult),
            dropout=float(dropout),
        )
    )


__all__ = [
    "BuildConfig",
    "UnknownLocalArch",
    "build_local_model",
    "list_local_arches",
]
