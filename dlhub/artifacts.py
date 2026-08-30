"""Optional experiment-artifact writers used by lesson entrypoints."""

from __future__ import annotations

import operator
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

    target = Path(path)
    if target.exists() and target.is_dir():
        raise IsADirectoryError(f"Image destination is a directory: {target}")

    validated_nrow: int | None = None
    if nrow is not None:
        if isinstance(nrow, bool):
            raise TypeError("nrow must be a positive integer, not bool")
        try:
            validated_nrow = operator.index(nrow)
        except TypeError as exc:
            raise TypeError("nrow must be a positive integer") from exc
        if validated_nrow < 1:
            raise ValueError(f"nrow must be >= 1, got {validated_nrow}")

    save_image = _load_torchvision_save_image()
    if save_image is None:
        return False
    kwargs = {} if validated_nrow is None else {"nrow": validated_nrow}
    save_image(image, path, **kwargs)
    return True


__all__ = ["save_image_if_available"]
