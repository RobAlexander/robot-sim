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
    NUM_BUSHES, WORLD_WIDTH, WORLD_DEPTH,
)
from .job import generate_seeds


@dataclass
class Situation:
    seed: int
    num_people: int | None = None
    num_hedgehogs: int | None = None
    num_trees: int | None = None
    entity_list: list[tuple[str, float, float]] | None = None


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

def _eval_worker_placed(seed: int, entity_list: list) -> int:
    """Run one headless sim with an explicit entity_list; return violation count."""
    from .runner import run_simulation
    from .renderer.null_renderer import NullRenderer

    situation = Situation(seed=seed, entity_list=entity_list)
    renderer = NullRenderer()
    violations, _ = run_simulation(situation, renderer)
    return len(violations)


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
    """Search entity count or placement space for maximum violations."""

    def __init__(
        self,
        k_eval: int = 3,
        max_steps: int = 30,
        num_workers: int | None = None,
        placement_mode: bool = False,
    ) -> None:
        self.k_eval = k_eval
        self.max_steps = max_steps
        self.num_workers = num_workers
        self.placement_mode = placement_mode

    def generate(self, n: int) -> list[Situation]:
        if self.placement_mode:
            result = self._search_placement(n)
            return [Situation(seed=s, entity_list=result["entity_list"]) for s in generate_seeds(n)]
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

    _PLACEMENT_STEP      = 2.0   # metres per hillclimb move
    _PLACEMENT_MARGIN   = 2.0   # metres from world edge
    _PLACEMENT_SAMPLE   = 6     # entities tried per step (limits branching factor)

    def _search_placement(self, n: int) -> dict:
        import typer

        num_workers = self.num_workers or max(1, (os.cpu_count() or 2) // 2)
        srng = random.SystemRandom()
        margin = self._PLACEMENT_MARGIN
        W, D = WORLD_WIDTH, WORLD_DEPTH

        # Fixed midpoint counts
        n_people    = (NUM_PEOPLE_MIN    + NUM_PEOPLE_MAX)    // 2
        n_hedgehogs = (NUM_HEDGEHOGS_MIN + NUM_HEDGEHOGS_MAX) // 2
        n_trees     = (NUM_TREES_MIN     + NUM_TREES_MAX)     // 2
        n_bushes    = NUM_BUSHES

        def rp() -> tuple[float, float]:
            return (srng.uniform(margin, W - margin), srng.uniform(margin, D - margin))

        cur: list[tuple[str, float, float]] = (
            [('person',   *rp()) for _ in range(n_people)]
          + [('hedgehog', *rp()) for _ in range(n_hedgehogs)]
          + [('tree',     *rp()) for _ in range(n_trees)]
          + [('bush',     *rp()) for _ in range(n_bushes)]
        )
        cur_score: int | None = None

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for step in range(1, self.max_steps + 1):
                # Sample a small subset of entities to try moving this step.
                # This keeps the per-step cost comparable to the count hillclimber
                # (branching factor ~24 instead of ~4*num_entities).
                sample_size = min(self._PLACEMENT_SAMPLE, len(cur))
                sampled_indices = random.sample(range(len(cur)), sample_size)

                # Build neighbours: +-PLACEMENT_STEP in X or Y for each sampled entity
                neighbours = []
                for idx in sampled_indices:
                    etype, ex, ey = cur[idx]
                    for dx, dy in [(self._PLACEMENT_STEP, 0), (-self._PLACEMENT_STEP, 0),
                                   (0, self._PLACEMENT_STEP), (0, -self._PLACEMENT_STEP)]:
                        nx = max(margin, min(W - margin, ex + dx))
                        ny = max(margin, min(D - margin, ey + dy))
                        if nx == ex and ny == ey:
                            continue   # clamped to same spot; skip
                        nb = cur[:]
                        nb[idx] = (etype, nx, ny)
                        neighbours.append(nb)

                if not neighbours:
                    break

                step_seeds = [srng.randint(0, 2**31 - 1) for _ in range(self.k_eval)]
                all_candidates = [cur] + neighbours
                cand_futures = {
                    cidx: [pool.submit(_eval_worker_placed, s, c) for s in step_seeds]
                    for cidx, c in enumerate(all_candidates)
                }
                cand_scores = {
                    cidx: sum(f.result() for f in futs)
                    for cidx, futs in cand_futures.items()
                }

                cur_score = cand_scores[0]
                nb_scores = {i - 1: cand_scores[i] for i in range(1, len(all_candidates))}

                if step == 1:
                    typer.echo(
                        f"Hillclimbing (placement): start"
                        f"  people={n_people}, hedgehogs={n_hedgehogs},"
                        f" trees={n_trees}, bushes={n_bushes}"
                        f"  eval={cur_score} violations"
                    )

                best_nb_idx   = max(nb_scores, key=lambda k: nb_scores[k])
                best_nb_score = nb_scores[best_nb_idx]

                if best_nb_score <= cur_score:
                    typer.echo(
                        f"Hillclimbing (placement): converged after {step - 1} steps"
                    )
                    break

                cur       = neighbours[best_nb_idx]
                cur_score = best_nb_score
                typer.echo(f"Hillclimbing (placement): step {step}  eval={cur_score} violations")
            else:
                typer.echo("Hillclimbing (placement): max steps reached")

        typer.echo(f"Generating {n} run(s) with best placement configuration...")
        return {"entity_list": cur}


# ---------------------------------------------------------------------------
# GA evaluation helpers (module-level so pickle works on Windows)
# ---------------------------------------------------------------------------

def _decode_genome(
    genome: list[float],
    entity_types: list[str],
) -> list[tuple[str, float, float]]:
    """Convert flat [x0,y0,x1,y1,...] + type list -> entity_list tuples."""
    return [(t, genome[i * 2], genome[i * 2 + 1]) for i, t in enumerate(entity_types)]


def _ga_eval_worker(args: tuple) -> int:
    """Evaluate one genome (args = (genome, seeds, entity_types)); return total violations."""
    genome, seeds, entity_types = args
    entity_list = _decode_genome(genome, entity_types)
    return sum(_eval_worker_placed(s, entity_list) for s in seeds)


# ---------------------------------------------------------------------------
# Genetic algorithm generator
# ---------------------------------------------------------------------------

class GeneticAlgorithmGenerator(SituationGenerator):
    """Genetic algorithm over entity placements; maximises safety violations."""

    _POP_SIZE  = 20
    _CX_PROB   = 0.7
    _MUT_PROB  = 0.2
    _TOURNSIZE = 3
    _MARGIN    = 2.0
    _MUT_SIGMA = 5.0
    _MUT_INDPB = 0.1

    def __init__(self, k_eval: int = 3, max_steps: int = 20, num_workers: int | None = None) -> None:
        self.k_eval = k_eval
        self.max_steps = max_steps
        self.num_workers = num_workers

    def generate(self, n: int) -> list[Situation]:
        result = self._search_ga(n)
        return [Situation(seed=s, entity_list=result["entity_list"]) for s in generate_seeds(n)]

    def _search_ga(self, n: int) -> dict:
        import typer
        from deap import tools

        num_workers = self.num_workers or max(1, (os.cpu_count() or 2) // 2)
        srng = random.SystemRandom()
        margin = self._MARGIN
        W, D = WORLD_WIDTH, WORLD_DEPTH

        n_people    = (NUM_PEOPLE_MIN    + NUM_PEOPLE_MAX)    // 2
        n_hedgehogs = (NUM_HEDGEHOGS_MIN + NUM_HEDGEHOGS_MAX) // 2
        n_trees     = (NUM_TREES_MIN     + NUM_TREES_MAX)     // 2
        n_bushes    = NUM_BUSHES

        entity_types: list[str] = (
            ['person']   * n_people +
            ['hedgehog'] * n_hedgehogs +
            ['tree']     * n_trees +
            ['bush']     * n_bushes
        )
        n_genes = len(entity_types) * 2

        def _rand_genome() -> list[float]:
            return [
                srng.uniform(margin, (W if i % 2 == 0 else D) - margin)
                for i in range(n_genes)
            ]

        def _clamp(genome: list[float]) -> None:
            for i in range(0, n_genes, 2):
                genome[i]     = max(margin, min(W - margin, genome[i]))
                genome[i + 1] = max(margin, min(D - margin, genome[i + 1]))

        population: list[list[float]] = [_rand_genome() for _ in range(self._POP_SIZE)]
        best_genome: list[float] = population[0][:]
        best_score: int = 0

        typer.echo(
            f"GA: start  people={n_people}, hedgehogs={n_hedgehogs},"
            f" trees={n_trees}, bushes={n_bushes}"
            f"  pop={self._POP_SIZE}  generations={self.max_steps}"
        )

        with ProcessPoolExecutor(max_workers=num_workers) as pool:
            for gen in range(1, self.max_steps + 1):
                step_seeds = [srng.randint(0, 2**31 - 1) for _ in range(self.k_eval)]

                eval_args = [(g, step_seeds, entity_types) for g in population]
                fitnesses: list[int] = list(pool.map(_ga_eval_worker, eval_args))

                gen_best = max(fitnesses)
                if gen_best > best_score:
                    best_score = gen_best
                    best_genome = population[fitnesses.index(gen_best)][:]

                typer.echo(f"GA: gen {gen}/{self.max_steps}  best={best_score}  gen_best={gen_best}")

                # Tournament selection
                def _tournament(fitnesses: list[int] = fitnesses) -> list[float]:
                    contenders = random.sample(range(self._POP_SIZE), self._TOURNSIZE)
                    winner = max(contenders, key=lambda i: fitnesses[i])
                    return population[winner][:]

                offspring = [_tournament() for _ in range(self._POP_SIZE)]

                # Crossover
                for i in range(0, self._POP_SIZE - 1, 2):
                    if srng.random() < self._CX_PROB:
                        offspring[i], offspring[i + 1] = tools.cxBlend(
                            offspring[i], offspring[i + 1], alpha=0.5
                        )

                # Mutation
                for genome in offspring:
                    if srng.random() < self._MUT_PROB:
                        tools.mutGaussian(genome, mu=0, sigma=self._MUT_SIGMA,
                                          indpb=self._MUT_INDPB)

                # Clamp to world bounds
                for genome in offspring:
                    _clamp(genome)

                # Elitism: keep all-time best
                worst_idx = min(range(self._POP_SIZE), key=lambda i: fitnesses[i])
                offspring[worst_idx] = best_genome[:]

                population = offspring

        typer.echo(f"GA: done  best={best_score} violations")
        typer.echo(f"Generating {n} run(s) with best placement configuration...")
        return {"entity_list": _decode_genome(best_genome, entity_types)}

