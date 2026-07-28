"""Lesson 10 (Vision): Synthetic image denoising (compact-first)."""

from __future__ import annotations

import warnings

# Keep CLI output clean: importing PyTorch may emit a noisy FutureWarning about `pynvml`.
warnings.filterwarnings(
    "ignore",
    message=r"The pynvml package is deprecated\..*",
    category=FutureWarning,
)
