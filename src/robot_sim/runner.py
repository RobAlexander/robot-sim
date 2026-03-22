"""Single-run loop: drives simulation steps, renderer, and wall-clock throttle."""

from __future__ import annotations

import math
import time

from .constants import STEP_DT, RUN_STEPS, ROBOT_RADIUS, PERSON_RADIUS, HEDGEHOG_RADIUS, WORLD_WIDTH
from .sim.simulation import Simulation, StepResult
from .sim.safety import Violation
from .renderer.base import Renderer


def _step_proximity(world) -> float:
    """Sum of proximity scores for all safety-relevant entities in one step.

    For each entity, proximity = max(0, 1 - edge_dist / WORLD_WIDTH) where
    edge_dist = max(0, center_dist - ROBOT_RADIUS - entity_radius).
    This is 1 when the robot is touching the entity (violation boundary) and 0
    when the robot is a full world-width away.
    """
    r = world.robot
    total = 0.0
    entities: list[tuple[float, float, float]] = (
        [(p.x, p.y, PERSON_RADIUS)   for p in world.people]    +
        [(h.x, h.y, HEDGEHOG_RADIUS) for h in world.hedgehogs] +
        [(t.x, t.y, t.radius)        for t in world.trees]     +
        [(b.x, b.y, b.radius)        for b in world.bushes]    +
        [(a.x, a.y, a.radius)        for a in world.attractors]
    )
    for ex, ey, er in entities:
        center_dist = math.hypot(r.x - ex, r.y - ey)
        edge_dist = max(0.0, center_dist - ROBOT_RADIUS - er)
        total += max(0.0, 1.0 - edge_dist / WORLD_WIDTH)
    return total


def run_simulation(
    situation,
    renderer: Renderer,
    speed_multiplier: float = 1.0,
    normal_counts: bool = False,
) -> tuple[list[Violation], dict[str, int], float]:
    """
    Run one full simulation (RUN_STEPS ticks).

    ``situation`` is any object with a ``seed`` attribute and optional
    ``num_people``, ``num_hedgehogs``, ``num_trees`` attributes (e.g. Situation
    from generators.py, or any duck-typed equivalent).

    Returns (violations, entity_counts, proximity_total) where entity_counts is
    a dict with keys num_people, num_trees, num_hedgehogs, and proximity_total
    is the sum of per-step proximity scores across all entities.

    The renderer is updated every step.  Wall-clock throttling is delegated to
    renderer.sleep_for_realtime() so that NullRenderer can skip it entirely.
    """
    sim = Simulation(
        situation.seed,
        normal_counts=normal_counts,
        num_people=getattr(situation, 'num_people', None),
        num_hedgehogs=getattr(situation, 'num_hedgehogs', None),
        num_trees=getattr(situation, 'num_trees', None),
        entity_list=getattr(situation, 'entity_list', None),
        paths=getattr(situation, 'paths', None),
    )
    wall_start = time.perf_counter()

    result: StepResult | None = None
    proximity_total = 0.0
    for _ in range(RUN_STEPS):
        result = sim.step()
        proximity_total += _step_proximity(sim.world)
        renderer.update(result)

        # Let renderer override nav mode (no-op for NullRenderer)
        nm = getattr(renderer, 'nav_mode', None)
        if nm is not None:
            sim.nav_mode = nm

        wall_elapsed = time.perf_counter() - wall_start
        sim_elapsed = sim.step_count * STEP_DT
        renderer.sleep_for_realtime(wall_elapsed, sim_elapsed, speed_multiplier)

        if result.sim_complete:
            break

    renderer.play_end_tune()
    return sim.all_violations, sim.entity_counts, proximity_total
