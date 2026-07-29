"""Optional experiment-artifact writers used by lesson entrypoints."""

from __future__ import annotations

from pathlib import Path


def _load_torchvision_save_image():
    try:
        from torchvision.utils import save_image
    except (ImportError, OSError, RuntimeError):
        return None
    return save_image


def save_image_if_available(
    image: object,
    path: str | Path,
    *,
    nrow: int | None = None,
) -> bool:
    """Save an image when torchvision is usable; return whether a file was written."""

    save_image = _load_torchvision_save_image()
    if save_image is None:
        return False
    kwargs = {} if nrow is None else {"nrow": int(nrow)}
    save_image(image, path, **kwargs)
    return True


__all__ = ["save_image_if_available"]
