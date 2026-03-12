"""Lane detection models (pure torch, toy-first).

Conventions:
- One algorithm family per file (variants live in that file).
- Each file exposes a `build_<name>_lane_detector(...)` factory and a `__main__` smoke test.

This package uses lazy imports so `import dlhub.vision.lane_detection` stays lightweight.
"""

from importlib import import_module
from typing import Any


def _import_attr(name: str) -> Any:
    if name.startswith("build_") and name.endswith("_lane_detector"):
        stem = name[len("build_") : -len("_lane_detector")]
        module = import_module(f"{__name__}.{stem}")
        attr = getattr(module, name)
        globals()[name] = attr
        return attr
    raise AttributeError(name)


def __getattr__(name: str) -> Any:  # pragma: no cover
    try:
        return _import_attr(name)
    except AttributeError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc


def __dir__() -> list[str]:  # pragma: no cover
    return sorted(list(globals().keys()))


__all__ = [
    "build_anchor3dlane_lane_detector",
    "build_bezierlanenet_lane_detector",
    "build_bevlanedet_lane_detector",
    "build_clrnet_lane_detector",
    "build_condlanenet_lane_detector",
    "build_enet_sad_lane_detector",
    "build_ganet_lane_detector",
    "build_genlanenet_lane_detector",
    "build_latr_lane_detector",
    "build_laneatt_lane_detector",
    "build_laneaf_lane_detector",
    "build_laneformer_lane_detector",
    "build_lanegcn_lane_detector",
    "build_lanenet_lane_detector",
    "build_lstr_lane_detector",
    "build_o2sformer_lane_detector",
    "build_persformer_lane_detector",
    "build_pinet_lane_detector",
    "build_polylanenet_lane_detector",
    "build_priorlane_lane_detector",
    "build_resa_lane_detector",
    "build_scnn_lane_detector",
    "build_topolane_lane_detector",
    "build_ufld_lane_detector",
]
