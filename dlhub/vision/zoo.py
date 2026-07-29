import inspect
from dataclasses import dataclass


class DependencyNotAvailable(RuntimeError):
    pass


@dataclass(frozen=True)
class TorchvisionArches:
    image_classification: list[str]
    segmentation: list[str]
    detection: list[str]
    optical_flow: list[str]
    video: list[str]
    quantization: list[str]

    def all(self) -> list[str]:
        return (
            list(self.image_classification)
            + list(self.segmentation)
            + list(self.detection)
            + list(self.optical_flow)
            + list(self.video)
            + list(self.quantization)
        )


def _import_torchvision():
    try:
        import torchvision

        return torchvision
    except Exception as exc:  # pragma: no cover - exercised by environments without torchvision
        raise DependencyNotAvailable(f"torchvision import failed: {exc}") from exc


def _import_timm():
    try:
        import timm

        return timm
    except Exception as exc:  # pragma: no cover - exercised by environments without timm
        raise DependencyNotAvailable(f"timm import failed: {exc}") from exc


def _call_with_supported_kwargs(fn, /, *args, **kwargs):
    params = inspect.signature(fn).parameters
    accepts_extra_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    filtered = kwargs if accepts_extra_kwargs else {k: v for k, v in kwargs.items() if k in params}
    return fn(*args, **filtered)


def list_torchvision_arches() -> TorchvisionArches:
    """List torchvision model names by task family (no instantiation, no downloads)."""

    torchvision = _import_torchvision()
    from torchvision.models import list_models

    quant_mod = getattr(torchvision.models, "quantization", None)
    quant_models = list_models(quant_mod) if quant_mod is not None else []

    return TorchvisionArches(
        image_classification=list_models(torchvision.models),
        segmentation=list_models(torchvision.models.segmentation),
        detection=list_models(torchvision.models.detection),
        optical_flow=list_models(torchvision.models.optical_flow),
        video=list_models(torchvision.models.video),
        quantization=quant_models,
    )


def list_timm_arches() -> list[str]:
    """List timm model names (no instantiation, no downloads)."""

    timm = _import_timm()
    list_models = getattr(timm, "list_models", None)
    if list_models is None:
        raise DependencyNotAvailable("timm.list_models not found")

    try:
        return list(_call_with_supported_kwargs(list_models, pretrained=False))
    except TypeError:
        return list(list_models())


def list_vision_arches() -> list[str]:
    """Return a unified list of known vision architecture ids.

    The ids are namespaced to keep them unambiguous:
    - `tv:<name>`: torchvision image classification models
    - `tvseg:<name>`: torchvision segmentation models
    - `tvdet:<name>`: torchvision detection models
    - `tvflow:<name>`: torchvision optical flow models
    - `tvvideo:<name>`: torchvision video models
    - `tvq:<name>`: torchvision quantized image classification models
    - `timm:<name>`: timm image models (if installed)

    Note: Some families have different expected input/output conventions and are not all
    compatible with a single training loop.
    """

    out: list[str] = []

    try:
        arches = list_torchvision_arches()
    except DependencyNotAvailable:
        arches = None

    if arches is not None:
        out.extend([f"tv:{name}" for name in arches.image_classification])
        out.extend([f"tvseg:{name}" for name in arches.segmentation])
        out.extend([f"tvdet:{name}" for name in arches.detection])
        out.extend([f"tvflow:{name}" for name in arches.optical_flow])
        out.extend([f"tvvideo:{name}" for name in arches.video])
        for name in arches.quantization:
            alias = name.removeprefix("quantized_")
            out.append(f"tvq:{alias}")

    try:
        timm_names = list_timm_arches()
    except DependencyNotAvailable:
        timm_names = []
    out.extend([f"timm:{name}" for name in timm_names])

    # Locally implemented backbones (no downloads, no external model zoo required).
    try:
        from .local_zoo import list_local_arches

        out.extend(list_local_arches())
    except Exception:
        pass
    return out


def _split_arch_id(arch_id: str) -> tuple[str, str]:
    arch_id = str(arch_id).strip()
    if ":" not in arch_id:
        raise ValueError(f"Expected a namespaced arch id like 'tv:resnet18', got: {arch_id!r}")
    prefix, name = arch_id.split(":", 1)
    prefix = prefix.strip().lower()
    name = name.strip()
    if not prefix or not name:
        raise ValueError(f"Invalid arch id: {arch_id!r}")
    return prefix, name


def build_torchvision_model(
    arch_id: str,
    *,
    num_classes: int | None = None,
    **extra_config: object,
):
    """Instantiate a torchvision model with weights disabled (no downloads).

    Args:
        arch_id: A namespaced id from `list_vision_arches()` (e.g., `tv:resnet18`).
        num_classes: Optional override for models that expose a `num_classes` constructor arg.
    """

    prefix, name = _split_arch_id(arch_id)
    if prefix == "tvclf":
        prefix = "tv"  # backward-compatible alias
    if prefix not in {"tv", "tvseg", "tvdet", "tvflow", "tvvideo", "tvq"}:
        raise ValueError(f"Unsupported torchvision prefix: {prefix!r} (arch_id={arch_id!r})")

    _import_torchvision()
    from torchvision.models import get_model

    cfg: dict[str, object] = {"weights": None}
    if prefix in {"tvseg", "tvdet"}:
        # These families often have a separate backbone weights default that would otherwise download.
        cfg["weights_backbone"] = None
    if num_classes is not None:
        cfg["num_classes"] = int(num_classes)

    # Allow users/lessons to tune builder kwargs (e.g., detection min_size/max_size),
    # but never allow weights args to override our no-download policy.
    forbidden = {"weights", "weights_backbone"}
    for k, v in extra_config.items():
        if k in forbidden:
            continue
        cfg[k] = v

    # Keep outputs simple for models that otherwise return auxiliary structures in train mode.
    if name in {"googlenet", "inception_v3"}:
        cfg.setdefault("aux_logits", False)

    model_name = name
    if prefix == "tvq" and not model_name.startswith("quantized_"):
        model_name = f"quantized_{model_name}"

    return get_model(model_name, **cfg)


def build_timm_model(
    arch_id: str,
    *,
    num_classes: int | None = None,
):
    """Instantiate a timm model with pretrained weights disabled (no downloads)."""

    prefix, name = _split_arch_id(arch_id)
    if prefix != "timm":
        raise ValueError(f"Unsupported timm prefix: {prefix!r} (arch_id={arch_id!r})")

    timm = _import_timm()
    create_model = getattr(timm, "create_model", None)
    if create_model is None:
        raise DependencyNotAvailable("timm.create_model not found")

    cfg: dict[str, object] = {"pretrained": False, "in_chans": 3}
    if num_classes is not None:
        cfg["num_classes"] = int(num_classes)

    try:
        return _call_with_supported_kwargs(create_model, name, **cfg)
    except TypeError:
        return create_model(name, **cfg)


__all__ = [
    "DependencyNotAvailable",
    "TorchvisionArches",
    "build_timm_model",
    "build_torchvision_model",
    "list_timm_arches",
    "list_torchvision_arches",
    "list_vision_arches",
]
