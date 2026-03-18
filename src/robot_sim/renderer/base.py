"""Abstract renderer interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..sim.simulation import StepResult


class Renderer(ABC):
    @abstractmethod
    def update(self, result: StepResult) -> None:
        """Push new world state to the renderer after each sim step."""

    @abstractmethod
    def sleep_for_realtime(self, wall_elapsed: float, sim_elapsed: float,
                           speed_multiplier: float) -> None:
        """
        Block so that visual runs don't outpace wall-clock time.
        Headless renderers implement this as a no-op.
        """

    @abstractmethod
    def play_end_tune(self) -> None:
        """Play completion audio (no-op in headless mode)."""

    @abstractmethod
    def shutdown(self) -> None:
        """Release any resources (window, audio context, etc.)."""
