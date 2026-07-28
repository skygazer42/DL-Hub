"""NLP utilities and models (no downloads, CPU-friendly).

This module provides a local NLP architecture zoo similar to `dlhub.vision.local_zoo`,
focused on small, readable implementations that can be exercised on synthetic synthetic datasets.
"""

from .local_zoo import BuildConfig, UnknownLocalArch, build_local_model, list_local_arches

__all__ = [
    "BuildConfig",
    "UnknownLocalArch",
    "build_local_model",
    "list_local_arches",
]
