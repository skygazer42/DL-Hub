"""Lesson 14 (Vision): synthetic video MOT basics with local MOT zoo."""

from __future__ import annotations

import warnings

# Keep CLI output clean: importing PyTorch may emit a noisy FutureWarning about `pynvml`.
warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\..*",
    category=FutureWarning,
)
