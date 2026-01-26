import math
from typing import Dict, List, Tuple

import numpy as np

_PATTERN_CACHE: Dict[int, List[Tuple[int, int, int, float]]] = {}


def _get_pattern(radius: int) -> List[Tuple[int, int, int, float]]:
    if radius in _PATTERN_CACHE:
        return _PATTERN_CACHE[radius]
    pattern: List[Tuple[int, int, int, float]] = []
    radius_sq = radius * radius
    for dx in range(-radius, radius + 1):
        for dy in range(-radius, radius + 1):
            if dx * dx + dy * dy < radius_sq:
                dz = math.sqrt(radius_sq - dx * dx - dy * dy)
                pattern.append((dx, dy, int(dz), dz / radius))
    _PATTERN_CACHE[radius] = pattern
    return pattern


def render_ball(
    height: int,
    width: int,
    show: np.ndarray,
    xyzs: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    radius: int,
) -> None:
    radius = max(int(radius), 1)
    depth = np.full((height, width), -2100000000, dtype=np.int32)
    pattern = _get_pattern(radius)

    if xyzs.size == 0:
        return

    z_values = xyzs[:, 2].astype(np.float64)
    zmin = (z_values - radius).min()
    zmax = (z_values + radius).max()
    zrange = zmax - zmin

    for i, (x, y, z) in enumerate(xyzs.astype(int)):
        for dx, dy, dz, color_scale in pattern:
            x2 = x + dx
            y2 = y + dy
            if x2 < 0 or x2 >= height or y2 < 0 or y2 >= width:
                continue
            z2 = z + dz
            if depth[x2, y2] < z2:
                depth[x2, y2] = z2
                if zrange != 0:
                    intensity = min(1.0, (z2 - zmin) / zrange * 0.7 + 0.3)
                else:
                    intensity = 1.0
                show[x2, y2, 0] = int(color_scale * c2[i] * intensity)
                show[x2, y2, 1] = int(color_scale * c0[i] * intensity)
                show[x2, y2, 2] = int(color_scale * c1[i] * intensity)
