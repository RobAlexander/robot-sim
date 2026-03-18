"""Hedgehog entity - erratic wandering bonus visitor."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass
class Hedgehog:
    x: float
    y: float
    z: float = 0.0
    heading: float = 0.0   # radians
    _rng: random.Random = None  # type: ignore[assignment]
    _turn_timer: int = 0

    def init_rng(self, rng: random.Random) -> None:
        object.__setattr__(self, "_rng", rng)
        self.heading = rng.uniform(0, 2 * math.pi)
        self._turn_timer = rng.randint(0, 59)  # stagger initial turns

    def step(self, dt: float, world_width: float, world_depth: float, speed: float, turn_interval: int) -> None:
        """Advance hedgehog by one sim step."""
        self._turn_timer += 1
        if self._turn_timer >= turn_interval:
            self._turn_timer = 0
            self.heading += self._rng.uniform(-0.75 * math.pi, 0.75 * math.pi)

        dx = math.cos(self.heading) * speed * dt
        dy = math.sin(self.heading) * speed * dt

        new_x = self.x + dx
        new_y = self.y + dy

        # Bounce off world edges
        if new_x < 0 or new_x > world_width:
            self.heading = math.pi - self.heading
            new_x = max(0.0, min(world_width, new_x))
        if new_y < 0 or new_y > world_depth:
            self.heading = -self.heading
            new_y = max(0.0, min(world_depth, new_y))

        self.x = new_x
        self.y = new_y
