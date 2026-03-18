"""Robot entity and FSM."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto


class RobotState(Enum):
    WANDER = auto()
    SEEK_LITTER = auto()
    AVOID_PERSON = auto()


@dataclass
class Robot:
    x: float
    y: float
    z: float = 0.0
    heading: float = 0.0       # radians; 0 = +x axis
    state: RobotState = RobotState.WANDER

    # Target indices (into world.litter / world.people)
    target_litter: int | None = None

    # Wander timer: change direction every N steps
    wander_timer: int = 0

    @property
    def pos2(self) -> tuple[float, float]:
        return (self.x, self.y)

    def move(self, dx: float, dy: float) -> None:
        self.x += dx
        self.y += dy

    def distance_to(self, ox: float, oy: float) -> float:
        return math.hypot(self.x - ox, self.y - oy)
