"""Situation generators -- pluggable strategies for producing (seed, entity-count) configurations."""

from __future__ import annotations

import os
import random
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from .constants import (
    NUM_HEDGEHOGS_MIN, NUM_HEDGEHOGS_MAX,
    NUM_PEOPLE_MIN, NUM_PEOPLE_MAX,
    NUM_TREES_MIN, NUM_TREES_MAX,
)
from .job import generate_seeds


@dataclass
class Situation:
    seed: int
    num_people: int | None = None
    num_hedgehogs: int | None = None
    num_trees: int | None = None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class SituationGenerator(ABC):
    @abstractmethod
    def generate(self, n: int) -> list[Situation]: ...


# ---------------------------------------------------------------------------
# Random generator (wraps existing behaviour)
# ---------------------------------------------------------------------------

class RandomGenerator(SituationGenerator):
    """No explicit count overrides -- counts drawn by sim from main RNG."""

    def __init__(self, normal_counts: bool = False) -> None:
        self.normal_counts = normal_counts

    def generate(self, n: int) -> list[Situation]:
        return [Situation(seed=s) for s in generate_seeds(n)]


# ---------------------------------------------------------------------------
# Hillclimbing evaluation worker (module-level so pickle works on Windows)
# ---------------------------------------------------------------------------

def _eval_worker(seed: int, num_people: int, num_hedgehogs: int, num_trees: int) -> int:
    """Run one headless sim with explicit counts; return violation count."""
    from .runner import run_simulation
    from .renderer.null_renderer import NullRenderer

    situation = Situation(
        seed=seed,
        num_people=num_people,
        num_hedgehogs=num_hedgehogs,
        num_trees=num_trees,
    )
    renderer = NullRenderer()
    violations, _ = run_simulation(situation, renderer)
    return len(violations)


# ---------------------------------------------------------------------------
# Hillclimbing generator
# ---------------------------------------------------------------------------

class HillclimbingGenerator(SituationGenerator):
    """Search (num_people, num_hedgehogs, num_trees) space for maximum violations."""

    def __init__(
        self,
        k_eval: int = 3,
        max_steps: int = 30,
        num_workers: int | None = None,
    ) -> None:
        self.k_eval = k_eval
        self.max_steps = max_steps
        self.num_workers = num_workers

    def generate(self, n: int) -> list[Situation]:
        best_counts = self._search(n)
        seeds = generate_seeds(n)
        return [Situation(seed=s, **best_counts) for s in seeds]

    def _search(self, n: int) -> dict:
        import typer

        num_workers = (
            self.num_workers
            if self.num_workers is not None
            else max(1, (os.cpu_count() or 2) // 2)
        )

        # Start at midpoints of each range
        cur = [
            (NUM_PEOPLE_MIN + NUM_PEOPLE_MAX) // 2,
            (NUM_HEDGEHOGS_MIN + NUM_HEDGEHOGS_MAX) // 2,
            (NUM_TREES_MIN + NUM_TREES_MAX) // 2,
        ]
        bounds = [
            (NUM_PEOPLE_MIN, NUM_PEOPLE_MAX),
            (NUM_HEDGEHOGS_MIN, NUM_HEDGEHOGS_MAX),
            (NUM_TREES_MIN, NUM_TREES_MAX),
        ]

        srng = random.SystemRandom()
        cur_score: int | None = None

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for step in range(1, self.max_steps + 1):
                # Build all valid neighbours (+-1 per dimension)
                neighbours = []
                for dim in range(3):
                    for delta in (+1, -1):
                        nb = cur[:]
                        nb[dim] += delta
                        lo, hi = bounds[dim]
                        if lo <= nb[dim] <= hi:
                            neighbours.append(nb)

                if not neighbours:
                    break

                # Shared seeds for this step: current and all neighbours are
                # evaluated with the same seeds so seed noise cancels out.
                step_seeds = [srng.randint(0, 2**31 - 1) for _ in range(self.k_eval)]

                all_candidates = [cur] + neighbours
                cand_futures = {
                    idx: [
                        pool.submit(_eval_worker, s, c[0], c[1], c[2])
                        for s in step_seeds
                    ]
                    for idx, c in enumerate(all_candidates)
                }
                cand_scores = {
                    idx: sum(f.result() for f in futs)
                    for idx, futs in cand_futures.items()
                }

                cur_score = cand_scores[0]
                nb_scores = {i - 1: cand_scores[i] for i in range(1, len(all_candidates))}

                if step == 1:
                    typer.echo(
                        f"Hillclimbing: start  (people={cur[0]}, hedgehogs={cur[1]},"
                        f" trees={cur[2]})  eval={cur_score} violations"
                    )

                best_nb_idx = max(nb_scores, key=lambda k: nb_scores[k])
                best_nb_score = nb_scores[best_nb_idx]

                if best_nb_score <= cur_score:
                    typer.echo(
                        f"Hillclimbing: converged at (people={cur[0]}, hedgehogs={cur[1]},"
                        f" trees={cur[2]}) after {step - 1} steps"
                    )
                    break

                cur = neighbours[best_nb_idx]
                cur_score = best_nb_score
                typer.echo(
                    f"Hillclimbing: step {step} (people={cur[0]}, hedgehogs={cur[1]},"
                    f" trees={cur[2]})  eval={cur_score} violations"
                )
            else:
                typer.echo(
                    f"Hillclimbing: max steps reached at (people={cur[0]},"
                    f" hedgehogs={cur[1]}, trees={cur[2]})"
                )

        typer.echo(f"Generating {n} run(s) with best configuration...")
        return {"num_people": cur[0], "num_hedgehogs": cur[1], "num_trees": cur[2]}

