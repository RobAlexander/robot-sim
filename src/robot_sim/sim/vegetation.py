"""Trees and bushes — static vegetation obstacles.

Generated from a dedicated RNG stream (seed + 3000) so they never perturb
the main simulation stream.  Trees block everyone; bushes block people and
the robot but the hedgehog may enter them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .paths import nearest_path_info, _Paths


@dataclass
class Tree:
    id: int
    x: float
    y: float
    z: float = 0.0
    radius: float = 0.4  # trunk radius (metres)


@dataclass
class Bush:
    id: int
    x: float
    y: float
    z: float = 0.0
    radius: float = 0.8  # spreading radius (metres)


def generate_vegetation(
    seed: int,
    world_width: float,
    world_depth: float,
    paths: _Paths,
    num_trees: int,
    num_bushes: int,
    tree_radius: float,
    bush_radius: float,
) -> tuple[list[Tree], list[Bush]]:
    """Return trees and bushes placed away from path centrelines."""
    rng = random.Random(seed + 3000)
    margin = 3.0
    path_clearance = 3.0  # keep centre at least this far from nearest path

    def _candidate() -> tuple[float, float]:
        for _ in range(200):
            x = rng.uniform(margin, world_width - margin)
            y = rng.uniform(margin, world_depth - margin)
            dist, _, _, _ = nearest_path_info(paths, x, y)
            if dist >= path_clearance:
                return x, y
        # Fallback: accept whatever we land on if no clear spot found
        return (
            rng.uniform(margin, world_width - margin),
            rng.uniform(margin, world_depth - margin),
        )

    trees = [Tree(id=i, x=x, y=y, radius=tree_radius)
             for i, (x, y) in enumerate(_candidate() for _ in range(num_trees))]
    bushes = [Bush(id=i, x=x, y=y, radius=bush_radius)
              for i, (x, y) in enumerate(_candidate() for _ in range(num_bushes))]

    return trees, bushes
