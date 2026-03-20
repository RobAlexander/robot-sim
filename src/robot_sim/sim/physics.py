"""Collision detection, push-back, and terrain snapping."""

from __future__ import annotations

import math

from ..constants import (
    WORLD_WIDTH, WORLD_DEPTH,
    ROBOT_RADIUS, PERSON_RADIUS, HEDGEHOG_RADIUS,
)
from .terrain import sample_height
from .world import World


def _push_back_circle(ax: float, ay: float, ar: float,
                      bx: float, by: float, br: float) -> tuple[float, float]:
    """Return adjusted (ax, ay) so that circle A does not overlap circle B."""
    dx = ax - bx
    dy = ay - by
    dist = math.hypot(dx, dy)
    min_dist = ar + br
    if dist < min_dist and dist > 1e-9:
        overlap = min_dist - dist
        nx, ny = dx / dist, dy / dist
        return ax + nx * overlap, ay + ny * overlap
    return ax, ay


def apply_physics(world: World) -> None:
    """Mutate world in-place: push-back collisions, clamp bounds, snap heights."""
    r = world.robot

    # 1. Clamp robot to world bounds
    r.x = max(ROBOT_RADIUS, min(WORLD_WIDTH  - ROBOT_RADIUS, r.x))
    r.y = max(ROBOT_RADIUS, min(WORLD_DEPTH - ROBOT_RADIUS, r.y))

    # 2. Push robot away from people
    for p in world.people:
        r.x, r.y = _push_back_circle(r.x, r.y, ROBOT_RADIUS, p.x, p.y, PERSON_RADIUS)

    # 3. Push robot away from trees and bushes
    for t in world.trees:
        r.x, r.y = _push_back_circle(r.x, r.y, ROBOT_RADIUS, t.x, t.y, t.radius)
    for b in world.bushes:
        r.x, r.y = _push_back_circle(r.x, r.y, ROBOT_RADIUS, b.x, b.y, b.radius)

    # 4. Push people away from trees and bushes
    for p in world.people:
        for t in world.trees:
            p.x, p.y = _push_back_circle(p.x, p.y, PERSON_RADIUS, t.x, t.y, t.radius)
        for b in world.bushes:
            p.x, p.y = _push_back_circle(p.x, p.y, PERSON_RADIUS, b.x, b.y, b.radius)

    # 5. Push hedgehogs away from trees only (hedgehog may enter bushes)
    for hog in world.hedgehogs:
        for t in world.trees:
            hog.x, hog.y = _push_back_circle(hog.x, hog.y, HEDGEHOG_RADIUS, t.x, t.y, t.radius)

    # 6. Snap heights
    r.z = sample_height(world.terrain, r.x, r.y, WORLD_WIDTH, WORLD_DEPTH)
    for p in world.people:
        p.z = sample_height(world.terrain, p.x, p.y, WORLD_WIDTH, WORLD_DEPTH)
    for lit in world.litter:
        if not lit.collected:
            lit.z = sample_height(world.terrain, lit.x, lit.y, WORLD_WIDTH, WORLD_DEPTH)
