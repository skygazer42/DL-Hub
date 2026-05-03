"""Optional framework probes for DL-Hub topic coverage.

The project keeps heavyweight framework integrations optional.  This module gives
framework-themed topics a real runtime surface without importing those frameworks
or adding new dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class FrameworkAdapter:
    name: str
    import_name: str
    display_name: str
    topics: tuple[str, ...]
    purpose: str


@dataclass(frozen=True)
class FrameworkProbe:
    name: str
    import_name: str
    available: bool
    purpose: str


_ADAPTERS: tuple[FrameworkAdapter, ...] = (
    FrameworkAdapter(
        name="pytorch",
        import_name="torch",
        display_name="PyTorch",
        topics=("PyTorch",),
        purpose="Primary tensor and model implementation backend used by DL-Hub zoos.",
    ),
    FrameworkAdapter(
        name="tensorflow",
        import_name="tensorflow",
        display_name="TensorFlow",
        topics=("TensorFlow",),
        purpose="Optional compatibility target for framework-comparison topics.",
    ),
    FrameworkAdapter(
        name="mxnet",
        import_name="mxnet",
        display_name="MXNet",
        topics=("MXNet",),
        purpose="Optional compatibility target for framework-comparison topics.",
    ),
    FrameworkAdapter(
        name="tensorrt",
        import_name="tensorrt",
        display_name="TensorRT",
        topics=("TensorRT",),
        purpose="Optional deployment/acceleration probe; no engine build is attempted.",
    ),
    FrameworkAdapter(
        name="opencv",
        import_name="cv2",
        display_name="OpenCV",
        topics=("OpenCV",),
        purpose="Optional classical computer-vision utility probe.",
    ),
    FrameworkAdapter(
        name="numpy",
        import_name="numpy",
        display_name="NumPy",
        topics=("Numpy", "NumPy"),
        purpose="Array backend for the NumPy ML algorithm implementations.",
    ),
    FrameworkAdapter(
        name="python",
        import_name="sys",
        display_name="Python",
        topics=("Python",),
        purpose="Runtime language surface for scripts, lessons, and package modules.",
    ),
)


def _normalize(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def list_framework_adapters() -> list[FrameworkAdapter]:
    """Return optional framework adapters without importing heavyweight packages."""

    return list(_ADAPTERS)


def get_framework_adapter(name: str) -> FrameworkAdapter:
    key = _normalize(name)
    for adapter in _ADAPTERS:
        aliases = (adapter.name, adapter.import_name, adapter.display_name, *adapter.topics)
        if key in {_normalize(alias) for alias in aliases}:
            return adapter
    raise KeyError(f"Unknown framework adapter: {name!r}")


def probe_framework(name: str) -> FrameworkProbe:
    """Check whether a framework import target is installed.

    The probe uses `find_spec` and never imports the framework module, so it is
    safe for optional packages with expensive import-time side effects.
    """

    adapter = get_framework_adapter(name)
    available = find_spec(adapter.import_name) is not None
    return FrameworkProbe(
        name=adapter.name,
        import_name=adapter.import_name,
        available=bool(available),
        purpose=adapter.purpose,
    )


__all__ = [
    "FrameworkAdapter",
    "FrameworkProbe",
    "get_framework_adapter",
    "list_framework_adapters",
    "probe_framework",
]
