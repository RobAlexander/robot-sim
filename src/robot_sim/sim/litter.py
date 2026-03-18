"""Litter entity."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Litter:
    id: int
    x: float
    y: float
    z: float = 0.0
    collected: bool = False

    @property
    def pos2(self) -> tuple[float, float]:
        return (self.x, self.y)
