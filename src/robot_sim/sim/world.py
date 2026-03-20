"""World state dataclass – pure data, no logic."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .robot import Robot
from .people import Person
from .litter import Litter
from .hedgehog import Hedgehog
from .vegetation import Tree, Bush


@dataclass
class World:
    seed: int
    terrain: np.ndarray          # shape (TERRAIN_CELLS, TERRAIN_CELLS), float32 heights
    robot: Robot
    hedgehogs: list[Hedgehog]
    paths: list[list[tuple[float, float]]]   # static polylines, generated from seed
    people: list[Person] = field(default_factory=list)
    litter: list[Litter] = field(default_factory=list)
    trees: list[Tree] = field(default_factory=list)
    bushes: list[Bush] = field(default_factory=list)
