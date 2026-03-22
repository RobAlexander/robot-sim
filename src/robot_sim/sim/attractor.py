"""Attractor entity -- a crowd-drawing fixture (fountain, event stand, etc.)."""

from __future__ import annotations

import random
from dataclasses import dataclass

from ..constants import (
    NUM_ATTRACTORS_MIN, NUM_ATTRACTORS_MAX,
    ATTRACTOR_BLOCK_RADIUS,
)


@dataclass
class Attractor:
    id: int
    x: float
    y: float
    z: float = 0.0
    radius: float = ATTRACTOR_BLOCK_RADIUS


def generate_attractors(
    seed: int,
    world_width: float,
    world_depth: float,
) -> list[Attractor]:
    """Generate attractors using dedicated RNG stream (seed + 6000)."""
    rng = random.Random(seed + 6000)
    n = rng.randint(NUM_ATTRACTORS_MIN, NUM_ATTRACTORS_MAX)
    margin = 5.0
    return [
        Attractor(
            id=i,
            x=rng.uniform(margin, world_width - margin),
            y=rng.uniform(margin, world_depth - margin),
        )
        for i in range(n)
    ]
