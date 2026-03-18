"""Person entity – destination-seeking pedestrian with path and obstacle awareness.

Each person picks a distant waypoint (preferring path waypoints 80% of the
time), walks toward it, then picks the next one on arrival.  Vegetation
obstacles cause soft heading deflection; physics provides hard push-back.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from ..constants import PERSON_OBSTACLE_AVOID_DIST, PERSON_ARRIVE_RADIUS
from .paths import _Paths


def _angle_diff(target: float, current: float) -> float:
    """Signed shortest angular difference (target - current) in [-pi, pi]."""
    d = (target - current) % (2 * math.pi)
    if d > math.pi:
        d -= 2 * math.pi
    return d


@dataclass
class Person:
    id: int
    x: float
    y: float
    z: float = 0.0
    heading: float = 0.0
    _rng: random.Random = None       # type: ignore[assignment]
    _dest_x: float = 0.0
    _dest_y: float = 0.0
    _dest_set: bool = False

    @property
    def pos2(self) -> tuple[float, float]:
        return (self.x, self.y)

    def init_rng(self, rng: random.Random,
                 paths: _Paths | None = None,
                 world_width: float = 50.0,
                 world_depth: float = 50.0) -> None:
        self._rng = rng
        self.heading = rng.uniform(0, 2 * math.pi)
        self._pick_destination(paths, world_width, world_depth)

    # ------------------------------------------------------------------
    # Destination picking
    # ------------------------------------------------------------------

    def _pick_destination(self, paths: _Paths | None,
                          world_width: float, world_depth: float) -> None:
        """Choose next waypoint: 80% chance of a path waypoint, else random."""
        if paths and self._rng.random() < 0.80:
            path = paths[self._rng.randrange(len(paths))]
            wp_x, wp_y = path[self._rng.randrange(len(path))]
            self._dest_x = max(1.0, min(world_width  - 1.0,
                                        wp_x + self._rng.uniform(-2.0, 2.0)))
            self._dest_y = max(1.0, min(world_depth - 1.0,
                                        wp_y + self._rng.uniform(-2.0, 2.0)))
        else:
            self._dest_x = self._rng.uniform(3.0, world_width  - 3.0)
            self._dest_y = self._rng.uniform(3.0, world_depth - 3.0)
        self._dest_set = True

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def step(self, dt: float, world_width: float, world_depth: float,
             speed: float, turn_rate: float, arrive_radius: float,
             paths: _Paths | None = None,
             obstacles: list[tuple[float, float, float]] | None = None) -> None:
        """Advance person one sim step toward their current destination."""
        # Arrive / init
        if (not self._dest_set
                or math.hypot(self.x - self._dest_x, self.y - self._dest_y) < arrive_radius):
            self._pick_destination(paths, world_width, world_depth)

        # Desired heading: toward destination
        desired = math.atan2(self._dest_y - self.y, self._dest_x - self.x)

        # Soft vegetation avoidance
        if obstacles:
            desired = self._avoid_obstacles(desired, obstacles)

        # Smoothly turn toward desired heading
        diff = _angle_diff(desired, self.heading)
        max_turn = turn_rate * dt
        self.heading += max(-max_turn, min(max_turn, diff))

        # Move
        new_x = self.x + math.cos(self.heading) * speed * dt
        new_y = self.y + math.sin(self.heading) * speed * dt

        # Bounce off world edges
        if new_x < 0 or new_x > world_width:
            self.heading = math.pi - self.heading
            new_x = max(0.0, min(world_width, new_x))
        if new_y < 0 or new_y > world_depth:
            self.heading = -self.heading
            new_y = max(0.0, min(world_depth, new_y))

        self.x = new_x
        self.y = new_y

    def _avoid_obstacles(self, desired: float,
                         obstacles: list[tuple[float, float, float]]) -> float:
        """Deflect desired heading away from obstacles that are close and ahead."""
        for ox, oy, oradius in obstacles:
            edge_dist = math.hypot(self.x - ox, self.y - oy) - oradius
            if edge_dist > PERSON_OBSTACLE_AVOID_DIST:
                continue
            toward_obs = math.atan2(oy - self.y, ox - self.x)
            rel = _angle_diff(toward_obs, desired)
            if abs(rel) > math.pi / 2:
                continue  # obstacle is mostly behind the desired direction
            t = 1.0 - max(0.0, edge_dist / PERSON_OBSTACLE_AVOID_DIST)
            strength = t * (math.pi / 2)
            # rel > 0: obstacle is to the left of desired → steer right (subtract)
            desired -= math.copysign(strength, rel)
        return desired
