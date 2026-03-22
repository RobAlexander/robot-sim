"""Per-step safety violation detection."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..constants import (
    SAFETY_DISTANCE_M, ROBOT_RADIUS, PERSON_RADIUS,
    HEDGEHOG_RADIUS, HEDGEHOG_SAFETY_DISTANCE_M,
)
from .world import World


@dataclass(frozen=True)
class Violation:
    step: int
    person_id: int | None   # None for non-person violations
    distance: float         # edge-to-edge distance in metres (negative = overlap)
    robot_x: float
    robot_y: float
    person_x: float         # target x
    person_y: float         # target y
    target: str = ""        # "person 0", "hedgehog", "tree 2", "bush 1"


def check_violations(world: World, step: int) -> list[Violation]:
    """Return all violations for this step."""
    violations: list[Violation] = []
    r = world.robot

    for p in world.people:
        edge_dist = math.hypot(r.x - p.x, r.y - p.y) - ROBOT_RADIUS - PERSON_RADIUS
        if edge_dist < SAFETY_DISTANCE_M:
            violations.append(Violation(
                step=step, person_id=p.id, distance=edge_dist,
                robot_x=r.x, robot_y=r.y, person_x=p.x, person_y=p.y,
                target=f"person {p.id}",
            ))

    for hog in world.hedgehogs:
        edge_dist = math.hypot(r.x - hog.x, r.y - hog.y) - ROBOT_RADIUS - HEDGEHOG_RADIUS
        if edge_dist < HEDGEHOG_SAFETY_DISTANCE_M:
            violations.append(Violation(
                step=step, person_id=None, distance=edge_dist,
                robot_x=r.x, robot_y=r.y, person_x=hog.x, person_y=hog.y,
                target="hedgehog",
            ))

    for t in world.trees:
        edge_dist = math.hypot(r.x - t.x, r.y - t.y) - ROBOT_RADIUS - t.radius
        if edge_dist < 0:
            violations.append(Violation(
                step=step, person_id=None, distance=edge_dist,
                robot_x=r.x, robot_y=r.y, person_x=t.x, person_y=t.y,
                target=f"tree {t.id}",
            ))

    for b in world.bushes:
        edge_dist = math.hypot(r.x - b.x, r.y - b.y) - ROBOT_RADIUS - b.radius
        if edge_dist < 0:
            violations.append(Violation(
                step=step, person_id=None, distance=edge_dist,
                robot_x=r.x, robot_y=r.y, person_x=b.x, person_y=b.y,
                target=f"bush {b.id}",
            ))

    for a in world.attractors:
        edge_dist = math.hypot(r.x - a.x, r.y - a.y) - ROBOT_RADIUS - a.radius
        if edge_dist < 0:
            violations.append(Violation(
                step=step, person_id=None, distance=edge_dist,
                robot_x=r.x, robot_y=r.y, person_x=a.x, person_y=a.y,
                target=f"attractor {a.id}",
            ))

    return violations
