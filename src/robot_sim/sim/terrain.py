"""Heightmap generation via OpenSimplex noise."""

from __future__ import annotations

import numpy as np

try:
    from opensimplex import OpenSimplex  # type: ignore[import-untyped]
    _HAS_NOISE = True
except ImportError:  # pragma: no cover
    _HAS_NOISE = False


def generate_heightmap(seed: int, cells: int, scale: float, height: float) -> np.ndarray:
    """
    Return a (cells, cells) float32 array of terrain heights in [0, height].

    Uses OpenSimplex noise seeded by `seed` – deterministic across platforms.
    Falls back to a flat zero map if opensimplex is unavailable.
    """
    if not _HAS_NOISE:
        return np.zeros((cells, cells), dtype=np.float32)

    gen = OpenSimplex(seed=seed)
    hmap = np.empty((cells, cells), dtype=np.float32)
    for row in range(cells):
        for col in range(cells):
            # noise2 returns values in [-1, 1]; map to [0, height]
            hmap[row, col] = (gen.noise2(col * scale, row * scale) + 1.0) * 0.5 * height

    return hmap


def sample_height(hmap: np.ndarray, x: float, y: float,
                  world_width: float, world_depth: float) -> float:
    """Bilinearly sample the heightmap at world coordinates (x, y)."""
    cells = hmap.shape[0]
    # Map world coords to cell coords
    cx = x / world_width * (cells - 1)
    cy = y / world_depth * (cells - 1)
    cx = max(0.0, min(cells - 1.0, cx))
    cy = max(0.0, min(cells - 1.0, cy))

    col0 = int(cx)
    row0 = int(cy)
    col1 = min(col0 + 1, cells - 1)
    row1 = min(row0 + 1, cells - 1)
    tc = cx - col0
    tr = cy - row0

    h00 = hmap[row0, col0]
    h10 = hmap[row0, col1]
    h01 = hmap[row1, col0]
    h11 = hmap[row1, col1]

    return float(h00 * (1 - tc) * (1 - tr)
                 + h10 * tc * (1 - tr)
                 + h01 * (1 - tc) * tr
                 + h11 * tc * tr)
