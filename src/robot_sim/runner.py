"""Single-run loop: drives simulation steps, renderer, and wall-clock throttle."""

from __future__ import annotations

import time

from .constants import STEP_DT, RUN_STEPS
from .sim.simulation import Simulation, StepResult
from .sim.safety import Violation
from .renderer.base import Renderer


def run_simulation(
    seed: int,
    renderer: Renderer,
    speed_multiplier: float = 1.0,
) -> list[Violation]:
    """
    Run one full simulation (RUN_STEPS ticks) and return all safety violations.

    The renderer is updated every step.  Wall-clock throttling is delegated to
    renderer.sleep_for_realtime() so that NullRenderer can skip it entirely.
    """
    sim = Simulation(seed)
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
    return sim.all_violations
