"""Single-run loop: drives simulation steps, renderer, and wall-clock throttle."""

from __future__ import annotations

import time

from .constants import STEP_DT, RUN_STEPS
from .sim.simulation import Simulation, StepResult
from .sim.safety import Violation
from .renderer.base import Renderer


def run_simulation(
    situation,
    renderer: Renderer,
    speed_multiplier: float = 1.0,
    normal_counts: bool = False,
) -> tuple[list[Violation], dict[str, int]]:
    """
    Run one full simulation (RUN_STEPS ticks).

    ``situation`` is any object with a ``seed`` attribute and optional
    ``num_people``, ``num_hedgehogs``, ``num_trees`` attributes (e.g. Situation
    from generators.py, or any duck-typed equivalent).

    Returns (violations, entity_counts) where entity_counts is a dict with
    keys num_people, num_trees, num_hedgehogs.

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
    )
    wall_start = time.perf_counter()

    result: StepResult | None = None
    for _ in range(RUN_STEPS):
        result = sim.step()
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
    return sim.all_violations, sim.entity_counts
