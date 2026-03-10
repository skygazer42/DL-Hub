"""Action recognition models (toy-first, pure torch).

This package covers two common modalities:
- Video action recognition: input (B, C, T, H, W)
- Skeleton-based action recognition: input (B, C, T, V) where V is #joints

Conventions:
- One algorithm family per file (variants live in that file via `_VARIANTS`).
- Each family file exposes a `build_*_video_classifier(...)` or
  `build_*_skeleton_classifier(...)` factory and a `__main__` smoke test.
"""

__all__ = []
