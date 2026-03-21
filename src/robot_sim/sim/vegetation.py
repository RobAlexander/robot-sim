"""Trees and bushes — static vegetation obstacles.

Generated from a dedicated RNG stream (seed + 3000) so they never perturb
the main simulation stream.  Trees block everyone; bushes block people and
the robot but the hedgehog may enter them.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .paths import nearest_path_info, _Paths
from ..constants import NUM_TREES_MIN, NUM_TREES_MAX, NUM_BUSHES, TREE_RADIUS, BUSH_RADIUS


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
    normal_counts: bool = False,
    num_trees: int | None = None,
) -> tuple[list[Tree], list[Bush]]:
    """Return trees and bushes placed away from path centrelines.

    Tree count is drawn as the first RNG call so that both sim and renderer
    regenerate identical vegetation from the same seed without coupling to any
    other RNG stream.  If num_trees is given explicitly the RNG draw is skipped.
    """
    rng = random.Random(seed + 3000)
    if num_trees is not None:
        pass  # use caller-supplied count; skip RNG draw so positions follow immediately
    elif normal_counts:
        mu = (NUM_TREES_MIN + NUM_TREES_MAX) / 2
        sigma = (NUM_TREES_MAX - NUM_TREES_MIN) / 6
        num_trees = max(NUM_TREES_MIN, min(NUM_TREES_MAX, round(rng.gauss(mu, sigma))))
    else:
        num_trees = rng.randint(NUM_TREES_MIN, NUM_TREES_MAX)
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

    trees = [Tree(id=i, x=x, y=y, radius=TREE_RADIUS)
             for i, (x, y) in enumerate(_candidate() for _ in range(num_trees))]
    bushes = [Bush(id=i, x=x, y=y, radius=BUSH_RADIUS)
              for i, (x, y) in enumerate(_candidate() for _ in range(NUM_BUSHES))]

    return trees, bushes
