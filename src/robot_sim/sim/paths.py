"""Path generation and spatial query utilities.

Paths are static polylines generated deterministically from the world seed
using a dedicated RNG stream (seed + 5000), so they never perturb the main
simulation stream.

Used by:
  - _build_world()       -- biased litter placement
  - Person.step()        -- biased wandering
  - PandaRenderer        -- terrain-following visual lines
"""

from __future__ import annotations

import math
import random

# Type aliases
_Path  = list[tuple[float, float]]
_Paths = list[_Path]


def generate_paths(seed: int, world_width: float, world_depth: float,
                   num_paths: int = 4) -> _Paths:
    """Return `num_paths` polyline paths, each a list of (x, y) world-space waypoints.

    Uses rng(seed + 5000) so path generation never touches the main rng stream.
    """
    rng = random.Random(seed + 5000)
    paths: _Paths = []
    for _ in range(num_paths):
        x = rng.uniform(5.0, world_width  - 5.0)
        y = rng.uniform(5.0, world_depth  - 5.0)
        waypoints: _Path = [(x, y)]
        direction = rng.uniform(0.0, 2.0 * math.pi)
        n_pts = rng.randint(3, 5)
        for _ in range(n_pts - 1):
            direction += rng.uniform(-math.pi / 3.0, math.pi / 3.0)
            length = rng.uniform(8.0, 15.0)
            x = max(2.0, min(world_width  - 2.0, x + math.cos(direction) * length))
            y = max(2.0, min(world_depth  - 2.0, y + math.sin(direction) * length))
            waypoints.append((x, y))
        paths.append(waypoints)
    return paths


# ---------------------------------------------------------------------------
# Spatial queries
# ---------------------------------------------------------------------------

def _closest_on_segment(px: float, py: float,
                         ax: float, ay: float,
                         bx: float, by: float) -> tuple[float, float, float]:
    """Return (t, cx, cy): closest point on segment AB to P, t in [0, 1]."""
    dx, dy = bx - ax, by - ay
    seg_sq = dx * dx + dy * dy
    if seg_sq < 1e-12:
        return 0.0, ax, ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_sq))
    return t, ax + t * dx, ay + t * dy


def nearest_path_info(paths: _Paths, px: float, py: float,
                      ) -> tuple[float, float, float, float]:
    """Return (dist, nx, ny, seg_dir) for the nearest point on any path segment.

    seg_dir is the bearing of the containing segment (radians, 0 = +X axis).
    """
    best_d   = float("inf")
    best_nx  = px
    best_ny  = py
    best_dir = 0.0
    for path in paths:
        for i in range(len(path) - 1):
            ax, ay = path[i]
            bx, by = path[i + 1]
            _, cx, cy = _closest_on_segment(px, py, ax, ay, bx, by)
            d = math.hypot(px - cx, py - cy)
            if d < best_d:
                best_d   = d
                best_nx  = cx
                best_ny  = cy
                best_dir = math.atan2(by - ay, bx - ax)
    return best_d, best_nx, best_ny, best_dir


def sample_near_path(rng: random.Random, paths: _Paths,
                     world_width: float, world_depth: float,
                     spread: float = 2.5) -> tuple[float, float]:
    """Sample a (x, y) point biased toward a random location on a path.

    Consumes exactly 5 RNG calls: randrange, randrange, random, uniform, uniform.
    """
    path = paths[rng.randrange(len(paths))]
    seg  = rng.randrange(len(path) - 1)
    ax, ay = path[seg]
    bx, by = path[seg + 1]
    t  = rng.random()
    x  = ax + t * (bx - ax) + rng.uniform(-spread, spread)
    y  = ay + t * (by - ay) + rng.uniform(-spread, spread)
    return (
        max(0.5, min(world_width  - 0.5, x)),
        max(0.5, min(world_depth  - 0.5, y)),
    )
