from collections.abc import Callable
from dataclasses import dataclass

from torch import nn

from .backbones import (
    build_alexnet_classifier,
    build_coatnet_classifier,
    build_convmixer_classifier,
    build_convnext_classifier,
    build_cspdarknet_classifier,
    build_darknet_classifier,
    build_densenet_classifier,
    build_efficientnet_classifier,
    build_fnet_classifier,
    build_ghostnet_classifier,
    build_gmlp_classifier,
    build_googlenet_classifier,
    build_lenet_classifier,
    build_mlp_mixer_classifier,
    build_mnasnet_classifier,
    build_mobilenet_v1_classifier,
    build_mobilenet_v2_classifier,
    build_mobilenet_v3_classifier,
    build_mobileone_classifier,
    build_mobilevit_classifier,
    build_nin_classifier,
    build_poolformer_classifier,
    build_pvt_classifier,
    build_regnet_classifier,
    build_repvgg_classifier,
    build_resmlp_classifier,
    build_resnet_classifier,
    build_shufflenet_v1_classifier,
    build_shufflenet_v2_classifier,
    build_squeezenet_classifier,
    build_swin_classifier,
    build_vgg_classifier,
    build_vit_classifier,
    build_xception_classifier,
    build_zfnet_classifier,
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
    from dlhub.zoo_registry import split_arch_id

    return split_arch_id(arch_id, default_prefix="dl")


Builder = Callable[[BuildConfig], nn.Module]


def _extract_variants_from_source(src: str) -> list[str] | None:
    """Extract `_VARIANTS` keys from a backbone module source without importing it."""

    import ast

    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    for node in tree.body:
        target_name: str | None = None
        value = None

        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target_name = node.target.id
            value = node.value
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    target_name = t.id
                    break
            value = node.value

        if target_name != "_VARIANTS" or not isinstance(value, ast.Dict):
            continue

        keys: list[str] = []
        for k in value.keys:
            if isinstance(k, ast.Constant) and isinstance(k.value, str):
                keys.append(k.value)
        return keys or None

    return None


def _make_lazy_backbone_builder(module_name: str, *, variant: str | None) -> Builder:
    """Create a registry Builder that imports the module on first use."""

    module_name = str(module_name).strip()
    builder_name = f"build_{module_name}_classifier"

    def _builder(cfg: BuildConfig) -> nn.Module:
        import importlib
        import inspect

        mod = importlib.import_module(f"dlhub.vision.backbones.{module_name}")
        fn = getattr(mod, builder_name, None)
        if fn is None:
            raise RuntimeError(f"Backbone module {module_name!r} missing {builder_name}()")

        kwargs: dict[str, object] = {
            "in_channels": int(cfg.in_channels),
            "num_classes": int(cfg.num_classes),
        }

        sig = None
        try:
            sig = inspect.signature(fn)
        except (TypeError, ValueError):
            sig = None

        if sig is not None:
            params = sig.parameters

            if "width_mult" in params:
                kwargs["width_mult"] = float(cfg.width_mult)
            if "dropout" in params:
                kwargs["dropout"] = float(cfg.dropout)

            if "image_size" in params:
                kwargs["image_size"] = int(cfg.image_size)
            elif "img_size" in params:
                kwargs["img_size"] = int(cfg.image_size)
            elif "input_size" in params:
                kwargs["input_size"] = int(cfg.image_size)

            if variant is not None:
                for key in ("variant", "arch", "model", "name", "version"):
                    if key in params:
                        kwargs[key] = str(variant)
                        break
        elif variant is not None:
            kwargs["variant"] = str(variant)

        try:
            return fn(**kwargs)
        except TypeError:
            # Fallback: drop optional kwargs progressively for compatibility.
            for k in ("image_size", "img_size", "input_size", "width_mult", "dropout"):
                if k in kwargs:
                    kwargs.pop(k, None)
                    try:
                        return fn(**kwargs)
                    except TypeError:
                        continue
            raise

    return _builder


def _extend_registry_with_discovered_backbones(r: dict[str, Builder]) -> None:
    """Add (module, variant) backbones to the registry as `dl:<variant_name>` ids.

    Notes:
    - This is best-effort and never overwrites existing names in `r`.
    - Builders are lazy-imported so `dlhub.vision.local_zoo` stays cheap to import.
    """

    from pathlib import Path

    here = Path(__file__).resolve().parent
    backbones_dir = here / "backbones"

    hidden = {
        "__init__",
        "catalog",
        "_blocks",
        "_transformer",
        # legacy aggregator modules
        "cnn",
        "extra_cnn",
        "transformers",
        "mixers",
        "hybrids",
    }

    generic_variants = {
        "tiny",
        "small",
        "base",
        "large",
        "xlarge",
        "xxlarge",
        "huge",
        "giant",
        "b0",
        "b1",
        "b2",
        "b3",
        "b4",
        "b5",
        "b6",
        "b7",
        "s",
        "m",
        "l",
        "xl",
        "xs",
        "xxs",
    }

    for py in sorted(backbones_dir.glob("*.py")):
        module_name = py.stem
        if module_name in hidden or module_name.startswith("_"):
            continue

        try:
            src = py.read_text(encoding="utf-8-sig")
        except OSError:
            continue

        if f"def build_{module_name}_classifier" not in src:
            continue

        variants = _extract_variants_from_source(src)
        has_explicit_variants = variants is not None
        if variants is None:
            variants = [module_name]

        for v in variants:
            base = str(v).lower().strip()
            if not base:
                continue

            name = base
            # Avoid extremely generic ids like `dl:tiny` / `dl:b0`.
            if module_name not in base and (base in generic_variants or len(base) <= 4):
                name = f"{module_name}_{base}"
            if name in r:
                name = f"{module_name}_{base}"
            if name in r:
                continue

            r[name] = _make_lazy_backbone_builder(
                module_name, variant=v if has_explicit_variants else None
            )


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

    # Early / classic CNNs
    r["lenet5"] = lambda cfg: build_lenet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    )
    r["alexnet"] = lambda cfg: build_alexnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    )
    r["zfnet"] = lambda cfg: build_zfnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    )
    r["nin"] = lambda cfg: build_nin_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    )
    r["googlenet"] = lambda cfg: build_googlenet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    )

    # Xception
    for v in ["tiny", "small", "base"]:
        name = "xception" if v == "base" else f"xception_{v}"
        r[name] = lambda cfg, v=v: build_xception_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            variant=v,
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

    # Res2Net / Res2NeXt (multi-scale bottleneck)
    for name, layers in {
        "res2net50": (3, 4, 6, 3),
        "res2net101": (3, 4, 23, 3),
        "res2net152": (3, 8, 36, 3),
    }.items():
        r[name] = lambda cfg, layers=layers: build_resnet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            layers=layers,
            variant="res2net_bottleneck",
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            groups=1,
            width_per_group=64,
        )
    r["res2next50_32x4d"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(3, 4, 6, 3),
        variant="res2net_bottleneck",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        groups=32,
        width_per_group=4,
    )

    # SKNet / ResNeSt (attention bottlenecks)
    for name, layers in {
        "sk_resnet50": (3, 4, 6, 3),
        "sk_resnet101": (3, 4, 23, 3),
        "sk_resnet152": (3, 8, 36, 3),
    }.items():
        r[name] = lambda cfg, layers=layers: build_resnet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            layers=layers,
            variant="sk_bottleneck",
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            groups=1,
            width_per_group=64,
        )
    r["sk_resnext50_32x4d"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(3, 4, 6, 3),
        variant="sk_bottleneck",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        groups=32,
        width_per_group=4,
    )

    for name, layers in {
        "resnest50": (3, 4, 6, 3),
        "resnest101": (3, 4, 23, 3),
    }.items():
        r[name] = lambda cfg, layers=layers: build_resnet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            layers=layers,
            variant="resnest_bottleneck",
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
            groups=1,
            width_per_group=64,
        )
    r["resnest50_32x4d"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(3, 4, 6, 3),
        variant="resnest_bottleneck",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        groups=32,
        width_per_group=4,
    )

    # ECA/CBAM ResNet
    r["eca_resnet18"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(2, 2, 2, 2),
        variant="eca_basic",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    )
    r["cbam_resnet18"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(2, 2, 2, 2),
        variant="cbam_basic",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    )
    r["eca_resnet50"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(3, 4, 6, 3),
        variant="eca_bottleneck",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        groups=1,
        width_per_group=64,
    )
    r["cbam_resnet50"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(3, 4, 6, 3),
        variant="cbam_bottleneck",
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
    r["eca_resnext50_32x4d"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(3, 4, 6, 3),
        variant="eca_bottleneck",
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
        groups=32,
        width_per_group=4,
    )
    r["cbam_resnext50_32x4d"] = lambda cfg: build_resnet_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        layers=(3, 4, 6, 3),
        variant="cbam_bottleneck",
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

    # RegNet (X/Y)
    for size in [
        "200mf",
        "400mf",
        "600mf",
        "800mf",
        "1_6gf",
        "2_4gf",
        "3_2gf",
        "4gf",
        "6_4gf",
        "8gf",
        "12gf",
        "16gf",
        "32gf",
    ]:
        r[f"regnetx_{size}"] = lambda cfg, size=size: build_regnet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            variant=f"regnetx_{size}",
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
        )
        r[f"regnety_{size}"] = lambda cfg, size=size: build_regnet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            variant=f"regnety_{size}",
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
        )

    # DarkNet / CSPDarkNet
    for v in ["darknet_tiny", "darknet19", "darknet53"]:
        r[v] = lambda cfg, v=v: build_darknet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            variant=v,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
        )
    for v in ["cspdarknet53", "cspdarknet_small", "cspdarknet_tiny"]:
        r[v] = lambda cfg, v=v: build_cspdarknet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            variant=v,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
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

    # ShuffleNetV1
    for v in ["shufflenetv1_0_5", "shufflenetv1_1_0", "shufflenetv1_1_5", "shufflenetv1_2_0"]:
        r[v] = lambda cfg, v=v: build_shufflenet_v1_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            variant=v,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
        )

    # MobileNet
    r["mobilenet_v1"] = lambda cfg: build_mobilenet_v1_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
    )
    r["mobilenet_v2"] = lambda cfg: build_mobilenet_v2_classifier(
        in_channels=cfg.in_channels,
        num_classes=cfg.num_classes,
        width_mult=cfg.width_mult,
        dropout=cfg.dropout,
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

    # MNASNet
    for v in ["mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3"]:
        r[v] = lambda cfg, v=v: build_mnasnet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            variant=v,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
        )

    # GhostNet
    for v in [
        "ghostnet_0_5",
        "ghostnet_0_75",
        "ghostnet_1_0",
        "ghostnet_1_3",
        "ghostnet_1_5",
        "ghostnetv2_1_0",
        "ghostnetv2_1_3",
    ]:
        r[v] = lambda cfg, v=v: build_ghostnet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            variant=v,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
        )

    # MobileOne
    for v in [
        "mobileone_s0",
        "mobileone_s1",
        "mobileone_s2",
        "mobileone_s3",
        "mobileone_s4",
        "mobileone_s1_se",
        "mobileone_s2_se",
        "mobileone_s3_se",
        "mobileone_s4_se",
    ]:
        r[v] = lambda cfg, v=v: build_mobileone_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            variant=v,
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
    for v in ["swin_tiny", "swin_small", "swin_base", "swin_tiny_w2"]:
        r[v] = lambda cfg, v=v: build_swin_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            image_size=cfg.image_size,
            variant=v,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
        )

    # PoolFormer / gMLP / ResMLP
    for base_name in ["poolformer_tiny", "poolformer_small", "poolformer_base"]:
        for patch_size in [4, 8]:
            name = base_name if patch_size == 4 else f"{base_name}_p{patch_size}"
            r[name] = lambda cfg, name=name: build_poolformer_classifier(
                in_channels=cfg.in_channels,
                num_classes=cfg.num_classes,
                image_size=cfg.image_size,
                variant=name,
                width_mult=cfg.width_mult,
                dropout=cfg.dropout,
            )

    for base_name in ["gmlp_tiny", "gmlp_small", "gmlp_base"]:
        for patch_size in [4, 8, 16]:
            name = base_name if patch_size == 8 else f"{base_name}_p{patch_size}"
            r[name] = lambda cfg, name=name: build_gmlp_classifier(
                in_channels=cfg.in_channels,
                num_classes=cfg.num_classes,
                image_size=cfg.image_size,
                variant=name,
                width_mult=cfg.width_mult,
                dropout=cfg.dropout,
            )

    for base_name in ["resmlp_tiny", "resmlp_small", "resmlp_base"]:
        for patch_size in [4, 8, 16]:
            name = base_name if patch_size == 8 else f"{base_name}_p{patch_size}"
            r[name] = lambda cfg, name=name: build_resmlp_classifier(
                in_channels=cfg.in_channels,
                num_classes=cfg.num_classes,
                image_size=cfg.image_size,
                variant=name,
                width_mult=cfg.width_mult,
                dropout=cfg.dropout,
            )

    # Hybrid-ish families
    for v in ["mobilevit_tiny", "mobilevit_small", "mobilevit_base"]:
        r[v] = lambda cfg, v=v: build_mobilevit_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            image_size=cfg.image_size,
            variant=v,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
        )

    for v in ["coatnet_tiny", "coatnet_small", "coatnet_base"]:
        r[v] = lambda cfg, v=v: build_coatnet_classifier(
            in_channels=cfg.in_channels,
            num_classes=cfg.num_classes,
            image_size=cfg.image_size,
            variant=v,
            width_mult=cfg.width_mult,
            dropout=cfg.dropout,
        )

    for base_name in ["fnet_tiny", "fnet_small", "fnet_base"]:
        for patch_size in [4, 8, 16]:
            name = base_name if patch_size == 8 else f"{base_name}_p{patch_size}"
            r[name] = lambda cfg, name=name: build_fnet_classifier(
                in_channels=cfg.in_channels,
                num_classes=cfg.num_classes,
                image_size=cfg.image_size,
                variant=name,
                width_mult=cfg.width_mult,
                dropout=cfg.dropout,
            )

    # PVT (pyramid transformer)
    for base_name in ["pvt_tiny", "pvt_small", "pvt_base"]:
        for patch_size in [4, 8]:
            name = base_name if patch_size == 4 else f"{base_name}_p{patch_size}"
            r[name] = lambda cfg, name=name: build_pvt_classifier(
                in_channels=cfg.in_channels,
                num_classes=cfg.num_classes,
                image_size=cfg.image_size,
                variant=name,
                width_mult=cfg.width_mult,
                dropout=cfg.dropout,
            )

    vit_specs: dict[str, tuple[int, int, int]] = {
        # name: (embed_dim, num_heads, num_layers)
        "vit_tiny": (192, 3, 6),
        "vit_small": (384, 6, 8),
        "vit_base": (512, 8, 10),
    }
    for base_name, (embed_dim, num_heads, num_layers) in vit_specs.items():
        for patch_size in [4, 8, 16]:
            name = base_name if patch_size == 8 else f"{base_name}_p{patch_size}"
            r[name] = (
                lambda cfg,
                embed_dim=embed_dim,
                num_heads=num_heads,
                num_layers=num_layers,
                patch_size=patch_size: build_vit_classifier(
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
                r[name] = (
                    lambda cfg,
                    embed_dim=embed_dim,
                    num_layers=num_layers,
                    patch_size=patch_size,
                    tdim_mul=tdim_mul: build_mlp_mixer_classifier(
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
                )

    # ConvMixer grid
    for embed_dim in [128, 192, 256]:
        for depth in [4, 8, 12]:
            for patch_size in [4, 8, 16]:
                name = f"convmixer_d{depth}_c{embed_dim}_p{patch_size}"
                r[name] = (
                    lambda cfg,
                    embed_dim=embed_dim,
                    depth=depth,
                    patch_size=patch_size: build_convmixer_classifier(
                        in_channels=cfg.in_channels,
                        num_classes=cfg.num_classes,
                        image_size=cfg.image_size,
                        patch_size=int(patch_size),
                        embed_dim=int(embed_dim),
                        depth=int(depth),
                        kernel_size=9,
                        dropout=cfg.dropout,
                    )
                )

    # Aliases
    r["lenet"] = r["lenet5"]
    r["inception"] = r["googlenet"]
    r["inception_v1"] = r["googlenet"]
    r["vgg"] = r["vgg11"]
    r["densenet"] = r["densenet121"]
    r["squeezenet"] = r["squeezenet1_0"]
    r["shufflenetv1"] = r["shufflenetv1_1_0"]
    r["shufflenetv2"] = r["shufflenetv2_1_0"]
    r["mobilenetv1"] = r["mobilenet_v1"]
    r["mobilenetv2"] = r["mobilenet_v2"]
    r["mnasnet"] = r["mnasnet1_0"]
    r["ghostnet"] = r["ghostnet_1_0"]
    r["mobileone"] = r["mobileone_s0"]
    r["efficientnetb0"] = r["efficientnet_b0"]
    r["resnext50"] = r["resnext50_32x4d"]
    r["res2net"] = r["res2net50"]
    r["res2next50"] = r["res2next50_32x4d"]
    r["revgg"] = r["repvgg"]
    r["darknet"] = r["darknet53"]
    r["cspdarknet"] = r["cspdarknet53"]
    r["swin"] = r["swin_tiny"]
    r["poolformer"] = r["poolformer_tiny"]
    r["gmlp"] = r["gmlp_tiny"]
    r["resmlp"] = r["resmlp_tiny"]
    r["mobilevit"] = r["mobilevit_tiny"]
    r["coatnet"] = r["coatnet_tiny"]
    r["fnet"] = r["fnet_tiny"]
    r["pvt"] = r["pvt_tiny"]
    r["sk_resnet"] = r["sk_resnet50"]
    r["sk_resnext50"] = r["sk_resnext50_32x4d"]
    r["resnest"] = r["resnest50"]
    r["eca_resnet"] = r["eca_resnet18"]
    r["cbam_resnet"] = r["cbam_resnet18"]

    _extend_registry_with_discovered_backbones(r)

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
