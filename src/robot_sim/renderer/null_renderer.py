"""No-op renderer for headless / batch runs."""

from __future__ import annotations

from .base import Renderer
from ..sim.simulation import StepResult


class NullRenderer(Renderer):
    def update(self, result: StepResult) -> None:
        pass

    def sleep_for_realtime(self, wall_elapsed: float, sim_elapsed: float,
                           speed_multiplier: float) -> None:
        # Headless: run as fast as hardware allows
        pass

    def play_end_tune(self) -> None:
        pass

    def shutdown(self) -> None:
        pass
